from pathlib import Path
import argparse
import json
import platform
import statistics
import sys
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoar_model import (
    load_emoar_model,
    historical_web_transform,
)


DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
DATASET_REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "publicTest"

VIT_MODEL_ID = "trpakov/vit-face-expression"
VIT_REVISION = "ef0bc6fc34241b6587e7e009e7711357be28c024"

EMOAR_CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

N_IMAGES = 100
WARMUP_RUNS = 10


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def summarize(values):
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


parser = argparse.ArgumentParser()

parser.add_argument(
    "--order",
    required=True,
    choices=[
        "emoar-first",
        "vit-first",
    ],
)

parser.add_argument(
    "--output",
    required=True,
)

args = parser.parse_args()


print("=" * 80)
print("Counterbalanced CPU latency benchmark")
print("=" * 80)

print()
print("Order:", args.order)
print("Images:", N_IMAGES)
print("Warm-up:", WARMUP_RUNS)
print("Batch size: 1")
print("Device: CPU")
print("Torch threads:", torch.get_num_threads())


print()
print("Loading images...")

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=DATASET_REVISION,
)

images = []

for sample in dataset:
    images.append(
        sample["image"].copy()
    )

    if len(images) == N_IMAGES:
        break


print("Loading models...")

emoar_model = load_emoar_model(
    EMOAR_CHECKPOINT
)

emoar_transform = historical_web_transform()

vit_processor = AutoImageProcessor.from_pretrained(
    VIT_MODEL_ID,
    revision=VIT_REVISION,
)

vit_model = AutoModelForImageClassification.from_pretrained(
    VIT_MODEL_ID,
    revision=VIT_REVISION,
)

vit_model.eval()

print("Models ready.")


def warmup_emoar():

    with torch.inference_mode():

        for i in range(WARMUP_RUNS):

            tensor = (
                emoar_transform(
                    images[i % N_IMAGES]
                )
                .unsqueeze(0)
            )

            _ = emoar_model(tensor)


def warmup_vit():

    with torch.inference_mode():

        for i in range(WARMUP_RUNS):

            inputs = vit_processor(
                images=images[i % N_IMAGES],
                return_tensors="pt",
            )

            _ = vit_model(
                **inputs
            ).logits


def benchmark_emoar():

    values = []

    with torch.inference_mode():

        for image in images:

            start = time.perf_counter_ns()

            tensor = (
                emoar_transform(image)
                .unsqueeze(0)
            )

            _ = emoar_model(tensor)

            end = time.perf_counter_ns()

            values.append(
                (end - start)
                / 1_000_000
            )

    return values


def benchmark_vit():

    values = []

    with torch.inference_mode():

        for image in images:

            start = time.perf_counter_ns()

            inputs = vit_processor(
                images=image,
                return_tensors="pt",
            )

            _ = vit_model(
                **inputs
            ).logits

            end = time.perf_counter_ns()

            values.append(
                (end - start)
                / 1_000_000
            )

    return values


if args.order == "emoar-first":

    print()
    print("Warm-up EmoAR...")
    warmup_emoar()

    print("Benchmark EmoAR...")
    emoar_values = benchmark_emoar()

    print("Warm-up ViT...")
    warmup_vit()

    print("Benchmark ViT...")
    vit_values = benchmark_vit()

else:

    print()
    print("Warm-up ViT...")
    warmup_vit()

    print("Benchmark ViT...")
    vit_values = benchmark_vit()

    print("Warm-up EmoAR...")
    warmup_emoar()

    print("Benchmark EmoAR...")
    emoar_values = benchmark_emoar()


emoar = summarize(
    emoar_values
)

vit = summarize(
    vit_values
)

ratio = (
    vit["mean_ms"]
    / emoar["mean_ms"]
)


result = {
    "order":
        args.order,
    "environment": {
        "python":
            sys.version.split()[0],
        "torch":
            torch.__version__,
        "platform":
            platform.platform(),
        "processor":
            platform.processor(),
        "device":
            "cpu",
        "torch_threads":
            torch.get_num_threads(),
    },
    "protocol": {
        "dataset_id":
            DATASET_ID,
        "dataset_revision":
            DATASET_REVISION,
        "split":
            SPLIT,
        "images":
            N_IMAGES,
        "warmup_runs":
            WARMUP_RUNS,
        "batch_size":
            1,
    },
    "emoar":
        emoar,
    "vit":
        vit,
    "vit_to_emoar_mean_latency_ratio":
        ratio,
}


output = Path(
    args.output
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print()
print("Order:", args.order)

print()
print("EmoAR")
print(
    f"  mean:   "
    f"{emoar['mean_ms']:.3f} ms"
)
print(
    f"  median: "
    f"{emoar['median_ms']:.3f} ms"
)
print(
    f"  p95:    "
    f"{emoar['p95_ms']:.3f} ms"
)

print()
print("ViT")
print(
    f"  mean:   "
    f"{vit['mean_ms']:.3f} ms"
)
print(
    f"  median: "
    f"{vit['median_ms']:.3f} ms"
)
print(
    f"  p95:    "
    f"{vit['p95_ms']:.3f} ms"
)

print()
print(
    "ViT / EmoAR mean latency:",
    f"{ratio:.3f}x",
)

print()
print("Saved:")
print(output)
