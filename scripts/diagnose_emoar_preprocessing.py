from pathlib import Path
import csv
import json
import random
import sys

import torch
from datasets import load_dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoar_model import load_emoar_model, CLASSES


DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "publicTest"

SEED = 20260917
SAMPLES_PER_CLASS = 20
BATCH_SIZE = 16

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
    / "emoar_preprocessing_diagnostic.json"
)

CSV_OUTPUT = (
    ROOT
    / "results"
    / "emoar_preprocessing_predictions.csv"
)


print("=" * 80)
print("EmoAR preprocessing diagnostic")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Revision:", REVISION)
print("Split:", SPLIT)
print("Seed:", SEED)
print("Samples per class:", SAMPLES_PER_CLASS)

rng = random.Random(SEED)

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=REVISION,
)


# ---------------------------------------------------------------------
# Deterministic stratified reservoir sampling
# ---------------------------------------------------------------------

reservoirs = {
    label: []
    for label in range(len(CLASSES))
}

seen = {
    label: 0
    for label in range(len(CLASSES))
}


print()
print("Selecting deterministic stratified sample...")


for sample_index, sample in enumerate(dataset):

    label = int(sample["label"])

    seen[label] += 1

    item = {
        "sample_index": sample_index,
        "true_label": label,
        "image": sample["image"].copy(),
    }

    reservoir = reservoirs[label]

    if len(reservoir) < SAMPLES_PER_CLASS:
        reservoir.append(item)

    else:
        position = rng.randrange(seen[label])

        if position < SAMPLES_PER_CLASS:
            reservoir[position] = item


for label in range(len(CLASSES)):

    if len(reservoirs[label]) != SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Class {CLASSES[label]} has only "
            f"{len(reservoirs[label])} selected samples."
        )


records = []

for label in range(len(CLASSES)):
    records.extend(reservoirs[label])


print("Selected samples:", len(records))


# ---------------------------------------------------------------------
# Candidate preprocessing pipelines
# ---------------------------------------------------------------------

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


pipelines = {

    # Declared by EmoAR web_app/commons.py
    "web_255_crop224": transforms.Compose([
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),

    # Observed in the training notebook valid/test preprocessing
    "training_like_resize256": transforms.Compose([
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),
        transforms.Resize(256),
        transforms.ToTensor(),
        normalize,
    ]),
}


model = load_emoar_model(CHECKPOINT)

print()
print("Model loaded successfully.")


def evaluate_pipeline(name, transform):

    confusion = [
        [0 for _ in CLASSES]
        for _ in CLASSES
    ]

    predictions = []

    correct = 0
    confidence_sum = 0.0

    for start in range(
        0,
        len(records),
        BATCH_SIZE,
    ):

        batch_records = records[
            start:start + BATCH_SIZE
        ]

        tensors = torch.stack([
            transform(item["image"])
            for item in batch_records
        ])

        with torch.inference_mode():

            logits = model(tensors)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        confidences, predicted_indices = (
            probabilities.max(dim=1)
        )

        for i, item in enumerate(batch_records):

            true_label = item["true_label"]

            predicted_label = int(
                predicted_indices[i].item()
            )

            confidence = float(
                confidences[i].item()
            )

            is_correct = (
                true_label == predicted_label
            )

            confusion[
                true_label
            ][
                predicted_label
            ] += 1

            correct += int(is_correct)

            confidence_sum += confidence

            predictions.append({
                "pipeline": name,
                "sample_index": item["sample_index"],
                "true_index": true_label,
                "true_class": CLASSES[true_label],
                "predicted_index": predicted_label,
                "predicted_class": CLASSES[predicted_label],
                "confidence": confidence,
                "correct": is_correct,
            })


    recalls = []

    for class_index in range(len(CLASSES)):

        row = confusion[class_index]

        total_class = sum(row)

        class_recall = (
            row[class_index] / total_class
            if total_class
            else 0.0
        )

        recalls.append(class_recall)


    total = len(records)

    accuracy = correct / total

    balanced_accuracy = (
        sum(recalls) / len(recalls)
    )

    mean_confidence = (
        confidence_sum / total
    )


    return {
        "pipeline": name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mean_confidence": mean_confidence,
        "per_class_recall": {
            CLASSES[index]: recalls[index]
            for index in range(len(CLASSES))
        },
        "confusion_matrix": confusion,
        "predictions": predictions,
    }


all_results = {}


for pipeline_name, transform in pipelines.items():

    print()
    print("=" * 80)
    print("PIPELINE:", pipeline_name)
    print("=" * 80)

    result = evaluate_pipeline(
        pipeline_name,
        transform,
    )

    all_results[pipeline_name] = result

    print()
    print(
        f"Correct: "
        f"{result['correct']}/{result['total']}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{result['balanced_accuracy']:.6f}"
    )

    print(
        f"Mean confidence: "
        f"{result['mean_confidence']:.6f}"
    )

    print()
    print("Per-class recall:")

    for class_name, value in (
        result["per_class_recall"].items()
    ):
        print(
            f"{class_name:10s}: {value:.6f}"
        )

    print()
    print("Confusion matrix:")
    print(
        "true\\pred "
        + " ".join(
            f"{name[:3]:>4s}"
            for name in CLASSES
        )
    )

    for true_index, row in enumerate(
        result["confusion_matrix"]
    ):

        print(
            f"{CLASSES[true_index][:3]:>9s} "
            + " ".join(
                f"{value:4d}"
                for value in row
            )
        )


summary = {
    name: {
        key: value
        for key, value in result.items()
        if key != "predictions"
    }
    for name, result in all_results.items()
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

    for result in all_results.values():

        for prediction in result["predictions"]:

            writer.writerow(prediction)


print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)

for name, result in all_results.items():

    print(
        name,
        "->",
        f"{result['correct']}/{result['total']}",
        "| balanced_accuracy =",
        f"{result['balanced_accuracy']:.6f}",
    )


print()
print("Saved:")
print(JSON_OUTPUT)
print(CSV_OUTPUT)
