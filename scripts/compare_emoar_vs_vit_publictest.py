from pathlib import Path
import csv
import json
import math


ROOT = Path(__file__).resolve().parents[1]

EMOAR_FILE = (
    ROOT
    / "results"
    / "emoar_preprocessing_full_publictest_predictions.csv"
)

VIT_FILE = (
    ROOT
    / "results"
    / "vit_publictest_predictions.csv"
)

JSON_OUTPUT = (
    ROOT
    / "results"
    / "emoar_vs_vit_publictest.json"
)


def read_emoar_primary():

    rows = {}

    with EMOAR_FILE.open(
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["pipeline"]
                != "web_255_crop224"
            ):
                continue

            index = int(
                row["sample_index"]
            )

            rows[index] = {
                "correct":
                    row["correct"].lower()
                    == "true",
                "true_class":
                    row["true_class"].lower(),
                "predicted_class":
                    row["predicted_class"].lower(),
                "confidence":
                    float(row["confidence"]),
            }

    return rows


def read_vit():

    rows = {}

    with VIT_FILE.open(
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            index = int(
                row["sample_index"]
            )

            rows[index] = {
                "correct":
                    row["correct"].lower()
                    == "true",
                "true_class":
                    row["true_class"].lower(),
                "predicted_class":
                    row["predicted_class"].lower(),
                "confidence":
                    float(row["confidence"]),
            }

    return rows


def exact_mcnemar(b, c):

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


emoar = read_emoar_primary()
vit = read_vit()


print("=" * 80)
print("Paired FER2013 publicTest comparison: EmoAR vs ViT")
print("=" * 80)

print()
print("EmoAR samples:", len(emoar))
print("ViT samples:", len(vit))


if set(emoar) != set(vit):

    raise RuntimeError(
        "Sample indices differ between model outputs."
    )


both_correct = 0
both_wrong = 0
emoar_only = 0
vit_only = 0

label_mismatches = 0


for index in sorted(emoar):

    e = emoar[index]
    v = vit[index]

    if (
        e["true_class"]
        != v["true_class"]
    ):
        label_mismatches += 1

    if e["correct"] and v["correct"]:
        both_correct += 1

    elif (
        e["correct"]
        and not v["correct"]
    ):
        emoar_only += 1

    elif (
        not e["correct"]
        and v["correct"]
    ):
        vit_only += 1

    else:
        both_wrong += 1


p_value = exact_mcnemar(
    emoar_only,
    vit_only,
)


result = {
    "split":
        "publicTest",
    "total_samples":
        len(emoar),
    "label_mismatches":
        label_mismatches,
    "both_correct":
        both_correct,
    "both_wrong":
        both_wrong,
    "emoar_correct_vit_wrong":
        emoar_only,
    "emoar_wrong_vit_correct":
        vit_only,
    "discordant_total":
        emoar_only + vit_only,
    "exact_mcnemar_two_sided_p":
        p_value,
}


JSON_OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("Label mismatches:")
print(label_mismatches)

print()
print("Both correct:")
print(both_correct)

print()
print("Both wrong:")
print(both_wrong)

print()
print("EmoAR correct / ViT wrong:")
print(emoar_only)

print()
print("EmoAR wrong / ViT correct:")
print(vit_only)

print()
print("Discordant predictions:")
print(emoar_only + vit_only)

print()
print("Exact McNemar two-sided p:")
print(f"{p_value:.12g}")

print()
print("Saved:")
print(JSON_OUTPUT)
