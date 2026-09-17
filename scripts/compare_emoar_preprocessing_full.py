from pathlib import Path
import csv
import json
import math
import sys
import time

import torch
from datasets import load_dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoar_model import load_emoar_model, CLASSES


DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "publicTest"

BATCH_SIZE = 32

CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

JSON_OUTPUT = (
    ROOT
    / "results"
    / "emoar_preprocessing_full_publictest.json"
)

CSV_OUTPUT = (
    ROOT
    / "results"
    / "emoar_preprocessing_full_publictest_predictions.csv"
)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

pipelines = {
    "web_255_crop224": transforms.Compose([
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),

    "training_like_resize256": transforms.Compose([
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),
        transforms.Resize(256),
        transforms.ToTensor(),
        normalize,
    ]),
}


def empty_state():
    return {
        "confusion": [
            [0 for _ in CLASSES]
            for _ in CLASSES
        ],
        "correct": 0,
        "confidence_sum": 0.0,
        "predictions": [],
    }


states = {
    name: empty_state()
    for name in pipelines
}


def update_state(
    state,
    pipeline_name,
    sample_indices,
    true_labels,
    predicted_indices,
    confidences,
):

    for i in range(len(true_labels)):

        true_label = int(true_labels[i])
        predicted_label = int(
            predicted_indices[i].item()
        )

        confidence = float(
            confidences[i].item()
        )

        correct = (
            true_label == predicted_label
        )

        state["confusion"][
            true_label
        ][
            predicted_label
        ] += 1

        state["correct"] += int(correct)
        state["confidence_sum"] += confidence

        state["predictions"].append({
            "pipeline": pipeline_name,
            "sample_index": int(sample_indices[i]),
            "true_index": true_label,
            "true_class": CLASSES[true_label],
            "predicted_index": predicted_label,
            "predicted_class": CLASSES[predicted_label],
            "confidence": confidence,
            "correct": correct,
        })


def calculate_metrics(state):

    cm = state["confusion"]

    total = sum(
        sum(row)
        for row in cm
    )

    precisions = []
    recalls = []
    f1_scores = []

    per_class = {}

    for class_index, class_name in enumerate(CLASSES):

        tp = cm[class_index][class_index]

        actual = sum(
            cm[class_index]
        )

        predicted = sum(
            cm[row][class_index]
            for row in range(len(CLASSES))
        )

        precision = (
            tp / predicted
            if predicted
            else 0.0
        )

        recall = (
            tp / actual
            if actual
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        per_class[class_name] = {
            "support": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total": total,
        "correct": state["correct"],
        "accuracy": state["correct"] / total,
        "balanced_accuracy": (
            sum(recalls) / len(recalls)
        ),
        "macro_precision": (
            sum(precisions) / len(precisions)
        ),
        "macro_recall": (
            sum(recalls) / len(recalls)
        ),
        "macro_f1": (
            sum(f1_scores) / len(f1_scores)
        ),
        "mean_confidence": (
            state["confidence_sum"] / total
        ),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def exact_mcnemar_p_value(b, c):

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    lower_tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2 * lower_tail,
    )


print("=" * 80)
print("EmoAR preprocessing - full FER2013 publicTest comparison")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Revision:", REVISION)
print("Split:", SPLIT)
print("Batch size:", BATCH_SIZE)

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=REVISION,
)

model = load_emoar_model(CHECKPOINT)

print()
print("Model loaded successfully.")
print("Beginning complete validation evaluation...")
print()

batch = []
processed = 0

start_time = time.perf_counter()


def process_batch(batch):

    global processed

    images = [
        item["image"]
        for item in batch
    ]

    labels = [
        item["label"]
        for item in batch
    ]

    indices = [
        item["index"]
        for item in batch
    ]

    batch_outputs = {}

    for pipeline_name, transform in pipelines.items():

        tensor_batch = torch.stack([
            transform(image)
            for image in images
        ])

        with torch.inference_mode():

            logits = model(
                tensor_batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidences, predictions = (
                probabilities.max(dim=1)
            )

        batch_outputs[pipeline_name] = {
            "predictions": predictions,
            "confidences": confidences,
        }

        update_state(
            states[pipeline_name],
            pipeline_name,
            indices,
            labels,
            predictions,
            confidences,
        )

    processed += len(batch)

    if (
        processed % 256 < len(batch)
        or processed == 3589
    ):
        print(
            f"Processed: {processed}"
        )


for sample_index, sample in enumerate(dataset):

    batch.append({
        "index": sample_index,
        "label": int(sample["label"]),
        "image": sample["image"],
    })

    if len(batch) == BATCH_SIZE:

        process_batch(batch)
        batch = []


if batch:
    process_batch(batch)


elapsed = time.perf_counter() - start_time


metrics = {
    name: calculate_metrics(state)
    for name, state in states.items()
}


# ---------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------

web_predictions = {
    row["sample_index"]: row
    for row in states[
        "web_255_crop224"
    ]["predictions"]
}

training_predictions = {
    row["sample_index"]: row
    for row in states[
        "training_like_resize256"
    ]["predictions"]
}


web_correct_training_wrong = 0
web_wrong_training_correct = 0
both_correct = 0
both_wrong = 0


for sample_index in web_predictions:

    web = web_predictions[sample_index]
    training = training_predictions[sample_index]

    if web["correct"] and training["correct"]:
        both_correct += 1

    elif (
        web["correct"]
        and not training["correct"]
    ):
        web_correct_training_wrong += 1

    elif (
        not web["correct"]
        and training["correct"]
    ):
        web_wrong_training_correct += 1

    else:
        both_wrong += 1


mcnemar_p = exact_mcnemar_p_value(
    web_correct_training_wrong,
    web_wrong_training_correct,
)


paired = {
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "web_correct_training_wrong":
        web_correct_training_wrong,
    "web_wrong_training_correct":
        web_wrong_training_correct,
    "discordant_total":
        web_correct_training_wrong
        + web_wrong_training_correct,
    "mcnemar_exact_two_sided_p":
        mcnemar_p,
}


summary = {
    "dataset_id": DATASET_ID,
    "revision": REVISION,
    "split": SPLIT,
    "batch_size": BATCH_SIZE,
    "elapsed_seconds": elapsed,
    "pipelines": metrics,
    "paired_comparison": paired,
}


JSON_OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


with CSV_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "pipeline",
        "sample_index",
        "true_index",
        "true_class",
        "predicted_index",
        "predicted_class",
        "confidence",
        "correct",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for state in states.values():

        for row in state["predictions"]:
            writer.writerow(row)


print()
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)


for name, result in metrics.items():

    print()
    print(name)

    print(
        f"  correct: "
        f"{result['correct']}/{result['total']}"
    )

    print(
        f"  accuracy: "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  balanced_accuracy: "
        f"{result['balanced_accuracy']:.6f}"
    )

    print(
        f"  macro_f1: "
        f"{result['macro_f1']:.6f}"
    )

    print(
        f"  mean_confidence: "
        f"{result['mean_confidence']:.6f}"
    )

    print()
    print("  per-class:")

    for class_name, values in (
        result["per_class"].items()
    ):

        print(
            f"    {class_name:10s} "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f} "
            f"n={values['support']}"
        )


print()
print("=" * 80)
print("PAIRED COMPARISON")
print("=" * 80)

print(
    "Both correct:",
    both_correct,
)

print(
    "Both wrong:",
    both_wrong,
)

print(
    "Web correct / Training-like wrong:",
    web_correct_training_wrong,
)

print(
    "Web wrong / Training-like correct:",
    web_wrong_training_correct,
)

print(
    "Exact McNemar two-sided p:",
    f"{mcnemar_p:.8f}",
)

print()
print(
    "Elapsed seconds:",
    f"{elapsed:.2f}",
)

print()
print("Saved:")
print(JSON_OUTPUT)
print(CSV_OUTPUT)
