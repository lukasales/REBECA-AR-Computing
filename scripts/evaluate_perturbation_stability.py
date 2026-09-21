from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys

import torch
from datasets import load_dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
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
# Frozen protocol
# ---------------------------------------------------------------------

DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
DATASET_REVISION = "da8d1643649c86bfc2cffba5b744b51fdf329212"
SPLIT = "privateTest"

VIT_MODEL_ID = "trpakov/vit-face-expression"
VIT_REVISION = "ef0bc6fc34241b6587e7e009e7711357be28c024"

EMOAR_CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

SEED = 20260921
SAMPLES_PER_CLASS = 50
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
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 6,
    5: 4,
    6: 5,
}


JSON_OUTPUT = (
    ROOT
    / "results"
    / "perturbation_stability_results.json"
)

CSV_OUTPUT = (
    ROOT
    / "results"
    / "perturbation_stability_predictions.csv"
)


# ---------------------------------------------------------------------
# Deterministic perturbations
# ---------------------------------------------------------------------

def perturbations(image):

    image = image.convert("RGB")

    return {
        "original":
            image,

        "rotation_minus_5":
            TF.rotate(
                image,
                angle=-5,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),

        "rotation_plus_5":
            TF.rotate(
                image,
                angle=5,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),

        "translate_minus_2":
            TF.affine(
                image,
                angle=0,
                translate=[-2, 0],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),

        "translate_plus_2":
            TF.affine(
                image,
                angle=0,
                translate=[2, 0],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),

        "brightness_0_90":
            TF.adjust_brightness(
                image,
                0.90,
            ),

        "brightness_1_10":
            TF.adjust_brightness(
                image,
                1.10,
            ),

        "contrast_0_90":
            TF.adjust_contrast(
                image,
                0.90,
            ),

        "contrast_1_10":
            TF.adjust_contrast(
                image,
                1.10,
            ),
    }


PERTURBED_CONDITIONS = [
    "rotation_minus_5",
    "rotation_plus_5",
    "translate_minus_2",
    "translate_plus_2",
    "brightness_0_90",
    "brightness_1_10",
    "contrast_0_90",
    "contrast_1_10",
]


# ---------------------------------------------------------------------
# Deterministic balanced sampling
# ---------------------------------------------------------------------

def deterministic_score(index):

    value = f"{SEED}:{index}".encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        value
    ).hexdigest()

    return digest


print("=" * 80)
print("FER2013 perturbation stability experiment")
print("=" * 80)

print()
print("Dataset:", DATASET_ID)
print("Revision:", DATASET_REVISION)
print("Split:", SPLIT)
print("Seed:", SEED)
print("Samples per class:", SAMPLES_PER_CLASS)
print("Batch size:", BATCH_SIZE)


print()
print("Loading privateTest...")

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    revision=DATASET_REVISION,
)

print("Dataset rows:", len(dataset))


by_class = {
    class_index: []
    for class_index in range(7)
}

for index in range(len(dataset)):

    label = int(
        dataset[index]["label"]
    )

    by_class[label].append(
        index
    )


selected_indices = []

for class_index in range(7):

    candidates = by_class[
        class_index
    ]

    ranked = sorted(
        candidates,
        key=deterministic_score,
    )

    chosen = ranked[
        :SAMPLES_PER_CLASS
    ]

    if len(chosen) != SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Class {class_index} has "
            f"only {len(chosen)} samples."
        )

    selected_indices.extend(
        chosen
    )

    print(
        FER_CLASSES[class_index],
        "selected:",
        len(chosen),
    )


selected_indices = sorted(
    selected_indices
)

print()
print(
    "Total selected originals:",
    len(selected_indices),
)


# ---------------------------------------------------------------------
# Build all image variants once
# ---------------------------------------------------------------------

print()
print("Generating deterministic perturbations...")

records = []

for sample_index in selected_indices:

    sample = dataset[
        sample_index
    ]

    true_index = int(
        sample["label"]
    )

    variants = perturbations(
        sample["image"]
    )

    for condition, image in (
        variants.items()
    ):

        records.append({
            "sample_index":
                sample_index,
            "true_index":
                true_index,
            "true_class":
                FER_CLASSES[
                    true_index
                ],
            "condition":
                condition,
            "image":
                image,
        })


expected_records = (
    350 * 9
)

if len(records) != expected_records:
    raise RuntimeError(
        f"Expected {expected_records} "
        f"variants, got {len(records)}."
    )

print(
    "Generated image variants:",
    len(records),
)


# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------

print()
print("Loading EmoAR...")

emoar_model = load_emoar_model(
    EMOAR_CHECKPOINT
)

emoar_transform = (
    historical_web_transform()
)

emoar_model.eval()


print("Loading ViT...")

vit_processor = (
    AutoImageProcessor.from_pretrained(
        VIT_MODEL_ID,
        revision=VIT_REVISION,
    )
)

vit_model = (
    AutoModelForImageClassification.from_pretrained(
        VIT_MODEL_ID,
        revision=VIT_REVISION,
    )
)

vit_model.eval()

print("Models loaded.")


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------

def run_emoar(records):

    output = []

    total = len(records)

    print()
    print("Running EmoAR...")

    with torch.inference_mode():

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):

            batch_records = records[
                start:
                start + BATCH_SIZE
            ]

            tensors = torch.stack([
                emoar_transform(
                    row["image"]
                )
                for row in batch_records
            ])

            logits = emoar_model(
                tensors
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidences, predictions = (
                probabilities.max(dim=1)
            )

            for i, row in enumerate(
                batch_records
            ):

                predicted_index = int(
                    predictions[i].item()
                )

                confidence = float(
                    confidences[i].item()
                )

                output.append({
                    "model":
                        "EmoAR",
                    "sample_index":
                        row["sample_index"],
                    "true_index":
                        row["true_index"],
                    "true_class":
                        row["true_class"],
                    "condition":
                        row["condition"],
                    "predicted_index":
                        predicted_index,
                    "predicted_class":
                        FER_CLASSES[
                            predicted_index
                        ],
                    "confidence":
                        confidence,
                    "correct":
                        predicted_index
                        == row["true_index"],
                })

            processed = min(
                start + BATCH_SIZE,
                total,
            )

            if (
                processed % 500 < BATCH_SIZE
                or processed == total
            ):
                print(
                    f"  processed: "
                    f"{processed}/{total}"
                )

    return output


def run_vit(records):

    output = []

    total = len(records)

    print()
    print("Running ViT...")

    with torch.inference_mode():

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):

            batch_records = records[
                start:
                start + BATCH_SIZE
            ]

            images = [
                row["image"]
                for row in batch_records
            ]

            inputs = vit_processor(
                images=images,
                return_tensors="pt",
            )

            logits = vit_model(
                **inputs
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidences, predictions = (
                probabilities.max(dim=1)
            )

            for i, row in enumerate(
                batch_records
            ):

                raw_index = int(
                    predictions[i].item()
                )

                predicted_index = (
                    MODEL_TO_FER[
                        raw_index
                    ]
                )

                confidence = float(
                    confidences[i].item()
                )

                output.append({
                    "model":
                        "ViT",
                    "sample_index":
                        row["sample_index"],
                    "true_index":
                        row["true_index"],
                    "true_class":
                        row["true_class"],
                    "condition":
                        row["condition"],
                    "predicted_index":
                        predicted_index,
                    "predicted_class":
                        FER_CLASSES[
                            predicted_index
                        ],
                    "confidence":
                        confidence,
                    "correct":
                        predicted_index
                        == row["true_index"],
                })

            processed = min(
                start + BATCH_SIZE,
                total,
            )

            if (
                processed % 500 < BATCH_SIZE
                or processed == total
            ):
                print(
                    f"  processed: "
                    f"{processed}/{total}"
                )

    return output


emoar_predictions = run_emoar(
    records
)

vit_predictions = run_vit(
    records
)


# ---------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------

def analyze_model(rows):

    by_sample = {}

    for row in rows:

        by_sample.setdefault(
            row["sample_index"],
            {}
        )[row["condition"]] = row


    original_rows = [
        row
        for row in rows
        if row["condition"]
        == "original"
    ]

    perturbed_rows = [
        row
        for row in rows
        if row["condition"]
        != "original"
    ]


    original_accuracy = (
        sum(
            row["correct"]
            for row in original_rows
        )
        / len(original_rows)
    )

    aggregate_perturbed_accuracy = (
        sum(
            row["correct"]
            for row in perturbed_rows
        )
        / len(perturbed_rows)
    )


    per_condition = {}

    total_stable_pairs = 0
    total_pairs = 0

    absolute_confidence_changes = []

    samples_with_flip = 0
    fully_stable_samples = 0
    unique_prediction_counts = []


    for condition in (
        PERTURBED_CONDITIONS
    ):

        condition_rows = [
            row
            for row in rows
            if row["condition"]
            == condition
        ]

        correct = sum(
            row["correct"]
            for row in condition_rows
        )

        stable = 0

        for row in condition_rows:

            original = (
                by_sample[
                    row["sample_index"]
                ]["original"]
            )

            if (
                row["predicted_index"]
                == original[
                    "predicted_index"
                ]
            ):
                stable += 1

            absolute_confidence_changes.append(
                abs(
                    row["confidence"]
                    - original[
                        "confidence"
                    ]
                )
            )

        total_stable_pairs += stable
        total_pairs += len(
            condition_rows
        )

        per_condition[
            condition
        ] = {
            "n":
                len(condition_rows),
            "accuracy":
                correct
                / len(condition_rows),
            "label_stability":
                stable
                / len(condition_rows),
        }


    for sample_index, conditions in (
        by_sample.items()
    ):

        original_prediction = (
            conditions[
                "original"
            ]["predicted_index"]
        )

        perturbed_predictions = [
            conditions[
                condition
            ]["predicted_index"]
            for condition
            in PERTURBED_CONDITIONS
        ]

        flips = [
            prediction
            != original_prediction
            for prediction
            in perturbed_predictions
        ]

        if any(flips):
            samples_with_flip += 1

        if not any(flips):
            fully_stable_samples += 1

        all_predictions = [
            original_prediction,
            *perturbed_predictions,
        ]

        unique_prediction_counts.append(
            len(
                set(
                    all_predictions
                )
            )
        )


    return {
        "original_accuracy":
            original_accuracy,

        "aggregate_perturbed_accuracy":
            aggregate_perturbed_accuracy,

        "pairwise_label_stability":
            total_stable_pairs
            / total_pairs,

        "stable_pairs":
            total_stable_pairs,

        "total_pairs":
            total_pairs,

        "samples_with_at_least_one_flip":
            samples_with_flip,

        "fraction_with_at_least_one_flip":
            samples_with_flip
            / len(by_sample),

        "fully_stable_samples":
            fully_stable_samples,

        "fraction_fully_stable":
            fully_stable_samples
            / len(by_sample),

        "mean_unique_predicted_categories":
            statistics.mean(
                unique_prediction_counts
            ),

        "mean_absolute_confidence_change":
            statistics.mean(
                absolute_confidence_changes
            ),

        "max_absolute_confidence_change":
            max(
                absolute_confidence_changes
            ),

        "per_condition":
            per_condition,
    }


emoar_summary = analyze_model(
    emoar_predictions
)

vit_summary = analyze_model(
    vit_predictions
)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

result = {
    "protocol": {
        "dataset_id":
            DATASET_ID,
        "dataset_revision":
            DATASET_REVISION,
        "split":
            SPLIT,
        "seed":
            SEED,
        "samples_per_class":
            SAMPLES_PER_CLASS,
        "original_images":
            len(selected_indices),
        "conditions_per_image":
            9,
        "perturbation_pairs_per_model":
            2800,
        "batch_size":
            BATCH_SIZE,
        "selected_indices":
            selected_indices,
    },

    "emoar":
        emoar_summary,

    "vit":
        vit_summary,
}


JSON_OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


all_predictions = (
    emoar_predictions
    + vit_predictions
)

with CSV_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "model",
        "sample_index",
        "true_index",
        "true_class",
        "condition",
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

    for row in all_predictions:
        writer.writerow(row)


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

def show(name, data):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        "Original accuracy:",
        f"{data['original_accuracy']:.6f}"
    )

    print(
        "Aggregate perturbed accuracy:",
        f"{data['aggregate_perturbed_accuracy']:.6f}"
    )

    print(
        "Pairwise label stability:",
        f"{data['pairwise_label_stability']:.6f}"
    )

    print(
        "Stable pairs:",
        f"{data['stable_pairs']}/"
        f"{data['total_pairs']}"
    )

    print(
        "Samples with >=1 flip:",
        data[
            "samples_with_at_least_one_flip"
        ],
        (
            f"("
            f"{data['fraction_with_at_least_one_flip'] * 100:.2f}%"
            f")"
        ),
    )

    print(
        "Fully stable samples:",
        data[
            "fully_stable_samples"
        ],
        (
            f"("
            f"{data['fraction_fully_stable'] * 100:.2f}%"
            f")"
        ),
    )

    print(
        "Mean unique predicted categories:",
        f"{data['mean_unique_predicted_categories']:.6f}"
    )

    print(
        "Mean absolute confidence change:",
        f"{data['mean_absolute_confidence_change']:.6f}"
    )

    print(
        "Max absolute confidence change:",
        f"{data['max_absolute_confidence_change']:.6f}"
    )


    print()
    print("Per-condition results")

    print(
        f"{'CONDITION':24s} "
        f"{'ACCURACY':>10s} "
        f"{'STABILITY':>10s}"
    )

    for condition in (
        PERTURBED_CONDITIONS
    ):

        row = data[
            "per_condition"
        ][condition]

        print(
            f"{condition:24s} "
            f"{row['accuracy']:10.6f} "
            f"{row['label_stability']:10.6f}"
        )


show(
    "EmoAR DenseNet121",
    emoar_summary,
)

show(
    "Modern ViT",
    vit_summary,
)


print()
print("=" * 80)
print("COMPARATIVE SUMMARY")
print("=" * 80)

print(
    "Pairwise stability difference "
    "(ViT - EmoAR):",
    f"{vit_summary['pairwise_label_stability'] - emoar_summary['pairwise_label_stability']:+.6f}"
)

print(
    "Perturbed accuracy difference "
    "(ViT - EmoAR):",
    f"{vit_summary['aggregate_perturbed_accuracy'] - emoar_summary['aggregate_perturbed_accuracy']:+.6f}"
)

print(
    "Fully stable fraction difference "
    "(ViT - EmoAR):",
    f"{vit_summary['fraction_fully_stable'] - emoar_summary['fraction_fully_stable']:+.6f}"
)

print()
print("Saved:")
print(JSON_OUTPUT)
print(CSV_OUTPUT)

