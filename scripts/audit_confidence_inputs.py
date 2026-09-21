from pathlib import Path
import csv
import statistics


ROOT = Path(__file__).resolve().parents[1]

EMOAR_FILE = (
    ROOT
    / "results"
    / "emoar_private_test_predictions.csv"
)

VIT_FILE = (
    ROOT
    / "results"
    / "vit_private_test_predictions.csv"
)


def load_emoar():

    rows = []

    with EMOAR_FILE.open(
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        print("EmoAR columns:")
        print(reader.fieldnames)

        for row in reader:

            if (
                row["pipeline"]
                != "web_255_crop224"
            ):
                continue

            rows.append({
                "sample_index":
                    int(row["sample_index"]),
                "true_class":
                    row["true_class"].lower(),
                "predicted_class":
                    row["predicted_class"].lower(),
                "confidence":
                    float(row["confidence"]),
                "correct":
                    row["correct"].lower()
                    == "true",
            })

    return rows


def load_vit():

    rows = []

    with VIT_FILE.open(
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        print()
        print("ViT columns:")
        print(reader.fieldnames)

        for row in reader:

            rows.append({
                "sample_index":
                    int(row["sample_index"]),
                "true_class":
                    row["true_class"].lower(),
                "predicted_class":
                    row["predicted_class"].lower(),
                "confidence":
                    float(row["confidence"]),
                "correct":
                    row["correct"].lower()
                    == "true",
            })

    return rows


def audit(name, rows):

    indexes = [
        row["sample_index"]
        for row in rows
    ]

    confidence = [
        row["confidence"]
        for row in rows
    ]

    correct_confidence = [
        row["confidence"]
        for row in rows
        if row["correct"]
    ]

    wrong_confidence = [
        row["confidence"]
        for row in rows
        if not row["correct"]
    ]

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("Rows:", len(rows))

    print(
        "Unique sample indices:",
        len(set(indexes))
    )

    print(
        "Correct:",
        sum(
            row["correct"]
            for row in rows
        )
    )

    print(
        "Wrong:",
        sum(
            not row["correct"]
            for row in rows
        )
    )

    print(
        "Confidence range:",
        f"{min(confidence):.6f}",
        "-",
        f"{max(confidence):.6f}",
    )

    print(
        "Mean confidence overall:",
        f"{statistics.mean(confidence):.6f}"
    )

    print(
        "Mean confidence correct:",
        f"{statistics.mean(correct_confidence):.6f}"
    )

    print(
        "Mean confidence wrong:",
        f"{statistics.mean(wrong_confidence):.6f}"
    )

    print(
        "Median confidence wrong:",
        f"{statistics.median(wrong_confidence):.6f}"
    )

    for threshold in [
        0.80,
        0.90,
        0.95,
        0.99,
    ]:

        high_conf_wrong = sum(
            (not row["correct"])
            and row["confidence"] >= threshold
            for row in rows
        )

        print(
            f"Wrong with confidence >= "
            f"{threshold:.2f}:",
            high_conf_wrong,
        )


emoar = load_emoar()
vit = load_vit()

audit(
    "EmoAR DenseNet121",
    emoar,
)

audit(
    "Modern ViT",
    vit,
)


emoar_by_index = {
    row["sample_index"]: row
    for row in emoar
}

vit_by_index = {
    row["sample_index"]: row
    for row in vit
}


print()
print("=" * 80)
print("PAIRING CHECK")
print("=" * 80)

print(
    "Same sample indices:",
    set(emoar_by_index)
    == set(vit_by_index)
)

label_mismatches = sum(
    emoar_by_index[index]["true_class"]
    != vit_by_index[index]["true_class"]
    for index in emoar_by_index
)

print(
    "True-label mismatches:",
    label_mismatches
)

invalid_confidences = sum(
    not (
        0.0
        <= row["confidence"]
        <= 1.0
    )
    for row in emoar + vit
)

print(
    "Invalid confidence values:",
    invalid_confidences
)
