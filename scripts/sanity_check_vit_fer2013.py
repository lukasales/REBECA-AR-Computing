from pathlib import Path
import csv
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
DATASET_REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "publicTest"

MODEL_ID = "trpakov/vit-face-expression"
MODEL_REVISION = "ef0bc6fc34241b6587e7e009e7711357be28c024"


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
    / "vit_fer2013_seven_class_sanity.json"
)

CSV_OUTPUT = (
    ROOT
    / "results"
    / "vit_fer2013_seven_class_sanity.csv"
)


print("=" * 80)
print("Modern ViT FER2013 seven-class sanity check")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Dataset revision:", DATASET_REVISION)
print("Split:", SPLIT)

print()
print("Model:", MODEL_ID)
print("Model revision:", MODEL_REVISION)


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


selected = {}

for sample_index, sample in enumerate(dataset):

    true_index = int(sample["label"])

    if true_index not in selected:
        selected[true_index] = {
            "sample_index": sample_index,
            "image": sample["image"].copy(),
        }

    if len(selected) == 7:
        break


if len(selected) != 7:
    raise RuntimeError(
        f"Only {len(selected)} classes found."
    )


results = []


print()
print("-" * 90)
print(
    f"{'TRUE':10s} "
    f"{'MODEL RAW':10s} "
    f"{'FER PRED':10s} "
    f"{'CONFIDENCE':>12s} "
    f"{'CORRECT':>10s}"
)
print("-" * 90)


for true_index in range(7):

    item = selected[true_index]

    inputs = processor(
        images=item["image"],
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

    confidence, raw_prediction = (
        probabilities.max(dim=1)
    )

    raw_index = int(
        raw_prediction.item()
    )

    fer_predicted_index = (
        MODEL_TO_FER[raw_index]
    )

    raw_name = (
        model.config.id2label[
            raw_index
        ].lower()
    )

    true_name = FER_CLASSES[
        true_index
    ]

    predicted_name = FER_CLASSES[
        fer_predicted_index
    ]

    correct = (
        true_index
        == fer_predicted_index
    )

    row = {
        "sample_index":
            item["sample_index"],
        "true_fer_index":
            true_index,
        "true_class":
            true_name,
        "raw_model_index":
            raw_index,
        "raw_model_class":
            raw_name,
        "predicted_fer_index":
            fer_predicted_index,
        "predicted_class":
            predicted_name,
        "confidence":
            float(confidence.item()),
        "correct":
            correct,
    }

    results.append(row)

    print(
        f"{true_name:10s} "
        f"{raw_name:10s} "
        f"{predicted_name:10s} "
        f"{row['confidence']:12.6f} "
        f"{str(correct):>10s}"
    )


correct_count = sum(
    int(row["correct"])
    for row in results
)


print("-" * 90)

print()
print(
    f"Correct examples: "
    f"{correct_count}/7"
)


JSON_OUTPUT.write_text(
    json.dumps(
        {
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
            "model_to_fer_mapping":
                MODEL_TO_FER,
            "results":
                results,
        },
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

    for row in results:
        writer.writerow(row)


print()
print("Saved:")
print(JSON_OUTPUT)
print(CSV_OUTPUT)

