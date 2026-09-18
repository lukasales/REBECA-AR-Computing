from pathlib import Path
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


# ---------------------------------------------------------------------
# Frozen artifacts
# ---------------------------------------------------------------------

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

OUTPUT = (
    ROOT
    / "results"
    / "computational_benchmark_emoar_vs_vit.json"
)

N_IMAGES = 100
WARMUP_RUNS = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def percentile(values, p):
    ordered = sorted(values)

    if not ordered:
        return None

    position = (len(ordered) - 1) * p

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def summarize_ms(values):
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "std_ms": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "effective_fps_from_mean":
            1000.0 / statistics.mean(values),
    }


def model_tensor_bytes(model):
    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    return {
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "total_tensor_bytes":
            parameter_bytes + buffer_bytes,
        "total_tensor_mb":
            (parameter_bytes + buffer_bytes)
            / (1024 ** 2),
    }


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

print("=" * 80)
print("Controlled computational benchmark: EmoAR vs ViT")
print("=" * 80)

print()
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Platform:", platform.platform())
print("Processor:", platform.processor())
print("Torch threads:", torch.get_num_threads())
print("Torch interop threads:", torch.get_num_interop_threads())

print()
print("Device: CPU")
print("Batch size: 1")
print("Images:", N_IMAGES)
print("Warm-up runs:", WARMUP_RUNS)


# ---------------------------------------------------------------------
# Load images BEFORE timing
# ---------------------------------------------------------------------

print()
print("Loading fixed image set into memory...")

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

if len(images) != N_IMAGES:
    raise RuntimeError(
        f"Expected {N_IMAGES} images, got {len(images)}."
    )

print("Images loaded:", len(images))


# ---------------------------------------------------------------------
# Load models BEFORE timing
# ---------------------------------------------------------------------

print()
print("Loading EmoAR...")

emoar_model = load_emoar_model(
    EMOAR_CHECKPOINT
)

emoar_transform = historical_web_transform()

print("Loading ViT...")

vit_processor = AutoImageProcessor.from_pretrained(
    VIT_MODEL_ID,
    revision=VIT_REVISION,
)

vit_model = AutoModelForImageClassification.from_pretrained(
    VIT_MODEL_ID,
    revision=VIT_REVISION,
)

vit_model.eval()

print("Models loaded successfully.")


# ---------------------------------------------------------------------
# Prepare tensors for inference-only benchmark
# ---------------------------------------------------------------------

print()
print("Preparing tensors outside inference timing...")

emoar_tensors = [
    emoar_transform(image).unsqueeze(0)
    for image in images
]

vit_tensors = [
    vit_processor(
        images=image,
        return_tensors="pt",
    )["pixel_values"]
    for image in images
]


# ---------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------

print()
print("Warm-up...")

with torch.inference_mode():

    for i in range(WARMUP_RUNS):

        _ = emoar_model(
            emoar_tensors[
                i % N_IMAGES
            ]
        )

    for i in range(WARMUP_RUNS):

        _ = vit_model(
            pixel_values=vit_tensors[
                i % N_IMAGES
            ]
        ).logits


# ---------------------------------------------------------------------
# Preprocessing-only timing
# ---------------------------------------------------------------------

print()
print("Benchmarking preprocessing...")

emoar_preprocess_ms = []
vit_preprocess_ms = []

for image in images:

    start = time.perf_counter_ns()

    _ = emoar_transform(image)

    end = time.perf_counter_ns()

    emoar_preprocess_ms.append(
        (end - start) / 1_000_000
    )


for image in images:

    start = time.perf_counter_ns()

    _ = vit_processor(
        images=image,
        return_tensors="pt",
    )

    end = time.perf_counter_ns()

    vit_preprocess_ms.append(
        (end - start) / 1_000_000
    )


# ---------------------------------------------------------------------
# Inference-only timing
# ---------------------------------------------------------------------

print("Benchmarking inference-only...")

emoar_inference_ms = []
vit_inference_ms = []

with torch.inference_mode():

    for tensor in emoar_tensors:

        start = time.perf_counter_ns()

        _ = emoar_model(tensor)

        end = time.perf_counter_ns()

        emoar_inference_ms.append(
            (end - start) / 1_000_000
        )


with torch.inference_mode():

    for tensor in vit_tensors:

        start = time.perf_counter_ns()

        _ = vit_model(
            pixel_values=tensor
        ).logits

        end = time.perf_counter_ns()

        vit_inference_ms.append(
            (end - start) / 1_000_000
        )


# ---------------------------------------------------------------------
# End-to-end timing
# preprocessing + forward pass
# ---------------------------------------------------------------------

print("Benchmarking end-to-end...")

emoar_e2e_ms = []
vit_e2e_ms = []


with torch.inference_mode():

    for image in images:

        start = time.perf_counter_ns()

        tensor = (
            emoar_transform(image)
            .unsqueeze(0)
        )

        _ = emoar_model(tensor)

        end = time.perf_counter_ns()

        emoar_e2e_ms.append(
            (end - start) / 1_000_000
        )


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

        vit_e2e_ms.append(
            (end - start) / 1_000_000
        )


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

emoar = {
    "architecture":
        "DenseNet121 + Linear(1024, 7)",
    "parameters":
        count_parameters(emoar_model),
    "tensor_footprint":
        model_tensor_bytes(emoar_model),
    "preprocessing":
        summarize_ms(
            emoar_preprocess_ms
        ),
    "inference_only":
        summarize_ms(
            emoar_inference_ms
        ),
    "end_to_end":
        summarize_ms(
            emoar_e2e_ms
        ),
}


vit = {
    "architecture":
        "ViTForImageClassification",
    "parameters":
        count_parameters(vit_model),
    "tensor_footprint":
        model_tensor_bytes(vit_model),
    "preprocessing":
        summarize_ms(
            vit_preprocess_ms
        ),
    "inference_only":
        summarize_ms(
            vit_inference_ms
        ),
    "end_to_end":
        summarize_ms(
            vit_e2e_ms
        ),
}


result = {
    "environment": {
        "python":
            sys.version.split()[0],
        "torch":
            torch.__version__,
        "platform":
            platform.platform(),
        "processor":
            platform.processor(),
        "torch_num_threads":
            torch.get_num_threads(),
        "torch_num_interop_threads":
            torch.get_num_interop_threads(),
        "device":
            "cpu",
    },
    "protocol": {
        "dataset_id":
            DATASET_ID,
        "dataset_revision":
            DATASET_REVISION,
        "split":
            SPLIT,
        "number_of_images":
            N_IMAGES,
        "warmup_runs":
            WARMUP_RUNS,
        "batch_size":
            1,
        "dataset_loading_in_timed_region":
            False,
        "model_loading_in_timed_region":
            False,
    },
    "emoar":
        emoar,
    "vit":
        vit,
}


OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)


def show(name, data):

    print()
    print(name)
    print("-" * 80)

    print(
        "Parameters:",
        f"{data['parameters']:,}"
    )

    print(
        "Tensor footprint:",
        f"{data['tensor_footprint']['total_tensor_mb']:.2f} MB"
    )

    print()

    for section_name in [
        "preprocessing",
        "inference_only",
        "end_to_end",
    ]:

        section = data[
            section_name
        ]

        print(section_name)

        print(
            f"  mean: "
            f"{section['mean_ms']:.3f} ms"
        )

        print(
            f"  median: "
            f"{section['median_ms']:.3f} ms"
        )

        print(
            f"  p95: "
            f"{section['p95_ms']:.3f} ms"
        )

        print(
            f"  p99: "
            f"{section['p99_ms']:.3f} ms"
        )

        print(
            f"  effective FPS: "
            f"{section['effective_fps_from_mean']:.3f}"
        )

        print()


show(
    "EmoAR DenseNet121",
    emoar,
)

show(
    "Modern ViT",
    vit,
)


ratio_parameters = (
    vit["parameters"]
    / emoar["parameters"]
)

ratio_memory = (
    vit["tensor_footprint"]["total_tensor_bytes"]
    / emoar["tensor_footprint"]["total_tensor_bytes"]
)

ratio_e2e = (
    vit["end_to_end"]["mean_ms"]
    / emoar["end_to_end"]["mean_ms"]
)


print("=" * 80)
print("RELATIVE COST")
print("=" * 80)

print()
print(
    "ViT / EmoAR parameter ratio:",
    f"{ratio_parameters:.3f}x",
)

print(
    "ViT / EmoAR tensor-memory ratio:",
    f"{ratio_memory:.3f}x",
)

print(
    "ViT / EmoAR mean end-to-end latency ratio:",
    f"{ratio_e2e:.3f}x",
)

print()
print("Saved:")
print(OUTPUT)

