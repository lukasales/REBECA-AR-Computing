from pathlib import Path
import csv
import json
import sys
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
DATASET_REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"

MODEL_ID = "trpakov/vit-face-expression"
MODEL_REVISION = "ef0bc6fc34241b6587e7e009e7711357be28c024"

SPLIT = "privateTest"
BATCH_SIZE = 16


FER_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


MODEL_TO_FER = {
    0: 0,  # angry
    1: 1,  # disgust
    2: 2,  # fear
    3: 3,  # happy
    4: 6,  # neutral
    5: 4,  # sad
    6: 5,  # surprise
}


JSON_OUTPUT = (
    ROOT
    / "results"
    / "vit_private_test.json"
)

CSV_OUTPUT = (
    ROOT
    / "results"
    / "vit_private_test_predictions.csv"
)


def calculate_metrics(confusion, confidence_sum):

    total = sum(
        sum(row)
        for row in confusion
    )

    correct = sum(
        confusion[i][i]
        for i in range(len(FER_CLASSES))
    )

    precisions = []
    recalls = []
    f1_scores = []

    per_class = {}

    for class_index, class_name in enumerate(FER_CLASSES):

        tp = confusion[class_index][class_index]

        support = sum(
            confusion[class_index]
        )

        predicted = sum(
            confusion[row][class_index]
            for row in range(len(FER_CLASSES))
        )

        precision = (
            tp / predicted
            if predicted
            else 0.0
        )

        recall = (
            tp / support
            if support
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
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
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
            confidence_sum / total
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


print("=" * 80)
print("Modern ViT - FER2013 privateTest held-out benchmark")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Dataset revision:", DATASET_REVISION)
print("Split:", SPLIT)

print()
print("Model:", MODEL_ID)
print("Model revision:", MODEL_REVISION)
print("Batch size:", BATCH_SIZE)


dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=DATASET_REVISION,
)


processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

model.eval()


print()
print("Model loaded successfully.")
print("Beginning complete privateTest evaluation...")
print()


confusion = [
    [0 for _ in FER_CLASSES]
    for _ in FER_CLASSES
]

confidence_sum = 0.0
predictions_output = []

batch = []
processed = 0

start_time = time.perf_counter()


def process_batch(batch):

    global processed
    global confidence_sum

    images = [
        item["image"]
        for item in batch
    ]

    true_labels = [
        item["label"]
        for item in batch
    ]

    sample_indices = [
        item["index"]
        for item in batch
    ]


    inputs = processor(
        images=images,
        return_tensors="pt",
    )


    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        confidences, raw_predictions = (
            probabilities.max(dim=1)
        )


    for i in range(len(batch)):

        true_index = int(
            true_labels[i]
        )

        raw_index = int(
            raw_predictions[i].item()
        )

        fer_predicted_index = (
            MODEL_TO_FER[raw_index]
        )

        confidence = float(
            confidences[i].item()
        )

        correct = (
            true_index
            == fer_predicted_index
        )

        confusion[
            true_index
        ][
            fer_predicted_index
        ] += 1

        confidence_sum += confidence

        predictions_output.append({
            "sample_index":
                int(sample_indices[i]),
            "true_fer_index":
                true_index,
            "true_class":
                FER_CLASSES[true_index],
            "raw_model_index":
                raw_index,
            "raw_model_class":
                model.config.id2label[
                    raw_index
                ].lower(),
            "predicted_fer_index":
                fer_predicted_index,
            "predicted_class":
                FER_CLASSES[
                    fer_predicted_index
                ],
            "confidence":
                confidence,
            "correct":
                correct,
        })


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

metrics = calculate_metrics(
    confusion,
    confidence_sum,
)


result = {
    "dataset_id":
        DATASET_ID,
    "dataset_revision":
        DATASET_REVISION,
    "split":
        SPLIT,
    "model_id":
        MODEL_ID,
    "model_revision":
        MODEL_REVISION,
    "batch_size":
        BATCH_SIZE,
    "model_to_fer_mapping":
        MODEL_TO_FER,
    "elapsed_seconds":
        elapsed,
    "images_per_second":
        metrics["total"] / elapsed,
    "metrics":
        metrics,
}


JSON_OUTPUT.write_text(
    json.dumps(
        result,
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
        "sample_index",
        "true_fer_index",
        "true_class",
        "raw_model_index",
        "raw_model_class",
        "predicted_fer_index",
        "predicted_class",
        "confidence",
        "correct",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in predictions_output:
        writer.writerow(row)


print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

print()
print(
    f"Correct: "
    f"{metrics['correct']}/{metrics['total']}"
)

print(
    f"Accuracy: "
    f"{metrics['accuracy']:.6f}"
)

print(
    f"Balanced accuracy: "
    f"{metrics['balanced_accuracy']:.6f}"
)

print(
    f"Macro-F1: "
    f"{metrics['macro_f1']:.6f}"
)

print(
    f"Mean confidence: "
    f"{metrics['mean_confidence']:.6f}"
)


print()
print("Per-class:")

for class_name, values in (
    metrics["per_class"].items()
):

    print(
        f"{class_name:10s} "
        f"P={values['precision']:.4f} "
        f"R={values['recall']:.4f} "
        f"F1={values['f1']:.4f} "
        f"n={values['support']}"
    )


print()
print("Confusion matrix:")

print(
    "true\\pred "
    + " ".join(
        f"{name[:3]:>4s}"
        for name in FER_CLASSES
    )
)

for true_index, row in enumerate(
    confusion
):

    print(
        f"{FER_CLASSES[true_index][:3]:>9s} "
        + " ".join(
            f"{value:4d}"
            for value in row
        )
    )


print()
print(
    "Elapsed seconds:",
    f"{elapsed:.2f}",
)

print(
    "Images/second:",
    f"{metrics['total'] / elapsed:.3f}",
)


print()
print("Saved:")
print(JSON_OUTPUT)
print(CSV_OUTPUT)


