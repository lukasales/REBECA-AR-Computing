from pathlib import Path
import csv
import json
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoar_model import load_emoar_model, predict


DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "publicTest"

CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

CSV_OUTPUT = ROOT / "results" / "emoar_fer2013_seven_class_sanity.csv"
JSON_OUTPUT = ROOT / "results" / "emoar_fer2013_seven_class_sanity.json"


print("=" * 80)
print("EmoAR FER2013 seven-class sanity check")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Revision:", REVISION)
print("Split:", SPLIT)

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=REVISION,
)

model = load_emoar_model(CHECKPOINT)

print()
print("Model loaded successfully.")

selected = {}

for sample_index, sample in enumerate(dataset):
    true_label = int(sample["label"])

    if true_label not in selected:
        selected[true_label] = {
            "sample_index": sample_index,
            "image": sample["image"],
        }

    if len(selected) == 7:
        break


if len(selected) != 7:
    raise RuntimeError(
        f"Could only find {len(selected)} of 7 classes."
    )


results = []

print()
print("-" * 80)
print(
    f"{'TRUE':10s} {'PREDICTED':10s} "
    f"{'CONFIDENCE':>12s} {'CORRECT':>10s}"
)
print("-" * 80)

for true_label in range(7):
    item = selected[true_label]

    prediction = predict(
        model,
        item["image"],
    )

    predicted_label = prediction["class_index"]
    correct = predicted_label == true_label

    row = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "split": SPLIT,
        "sample_index": item["sample_index"],
        "true_index": true_label,
        "true_class": CLASSES[true_label],
        "predicted_index": predicted_label,
        "predicted_class": prediction["class_name"],
        "confidence": prediction["confidence"],
        "correct": correct,
        "probabilities": prediction["probabilities"],
    }

    results.append(row)

    print(
        f"{CLASSES[true_label]:10s} "
        f"{prediction['class_name']:10s} "
        f"{prediction['confidence']:12.6f} "
        f"{str(correct):>10s}"
    )


correct_count = sum(
    int(row["correct"])
    for row in results
)

print("-" * 80)
print()
print(f"Correct examples: {correct_count}/7")
print()

JSON_OUTPUT.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)

with CSV_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sample_index",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "confidence",
            "correct",
        ],
    )

    writer.writeheader()

    for row in results:
        writer.writerow({
            key: row[key]
            for key in writer.fieldnames
        })


print("Saved:")
print(CSV_OUTPUT)
print(JSON_OUTPUT)
