from pathlib import Path
import csv
import json
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

JSON_OUTPUT = (
    ROOT
    / "results"
    / "confidence_calibration_analysis.json"
)

BINS_OUTPUT = (
    ROOT
    / "results"
    / "confidence_calibration_bins.csv"
)

SELECTIVE_OUTPUT = (
    ROOT
    / "results"
    / "confidence_selective_prediction.csv"
)


N_BINS = 15

HIGH_CONF_THRESHOLDS = [
    0.80,
    0.90,
    0.95,
    0.99,
]

SELECTIVE_THRESHOLDS = [
    0.00,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]


def load_emoar():

    rows = []

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


def calibration_bins(rows):

    bins = [
        []
        for _ in range(N_BINS)
    ]

    for row in rows:

        confidence = row["confidence"]

        index = min(
            int(confidence * N_BINS),
            N_BINS - 1,
        )

        bins[index].append(row)


    output = []

    total = len(rows)

    ece = 0.0
    mce = 0.0

    for index, members in enumerate(bins):

        lower = index / N_BINS
        upper = (index + 1) / N_BINS

        if not members:

            output.append({
                "bin_index":
                    index,
                "lower_bound":
                    lower,
                "upper_bound":
                    upper,
                "count":
                    0,
                "fraction":
                    0.0,
                "accuracy":
                    None,
                "mean_confidence":
                    None,
                "signed_gap":
                    None,
                "absolute_gap":
                    None,
            })

            continue


        count = len(members)

        accuracy = sum(
            row["correct"]
            for row in members
        ) / count

        mean_confidence = statistics.mean(
            row["confidence"]
            for row in members
        )

        signed_gap = (
            mean_confidence
            - accuracy
        )

        absolute_gap = abs(
            signed_gap
        )

        fraction = (
            count / total
        )

        ece += (
            fraction
            * absolute_gap
        )

        mce = max(
            mce,
            absolute_gap,
        )


        output.append({
            "bin_index":
                index,
            "lower_bound":
                lower,
            "upper_bound":
                upper,
            "count":
                count,
            "fraction":
                fraction,
            "accuracy":
                accuracy,
            "mean_confidence":
                mean_confidence,
            "signed_gap":
                signed_gap,
            "absolute_gap":
                absolute_gap,
        })


    return output, ece, mce


def high_confidence_errors(rows):

    wrong_rows = [
        row
        for row in rows
        if not row["correct"]
    ]

    total_wrong = len(
        wrong_rows
    )

    output = []

    for threshold in (
        HIGH_CONF_THRESHOLDS
    ):

        count = sum(
            row["confidence"]
            >= threshold
            for row in wrong_rows
        )

        output.append({
            "threshold":
                threshold,
            "wrong_count":
                count,
            "proportion_of_all_errors":
                (
                    count / total_wrong
                    if total_wrong
                    else 0.0
                ),
        })

    return output


def selective_prediction(rows):

    total = len(rows)

    output = []

    for threshold in (
        SELECTIVE_THRESHOLDS
    ):

        retained = [
            row
            for row in rows
            if row["confidence"]
            >= threshold
        ]

        retained_count = len(
            retained
        )

        retained_correct = sum(
            row["correct"]
            for row in retained
        )

        retained_errors = (
            retained_count
            - retained_correct
        )

        coverage = (
            retained_count / total
        )

        accuracy = (
            retained_correct
            / retained_count
            if retained_count
            else None
        )

        output.append({
            "threshold":
                threshold,
            "retained":
                retained_count,
            "coverage":
                coverage,
            "accuracy":
                accuracy,
            "errors":
                retained_errors,
        })

    return output


def analyze(name, rows):

    total = len(rows)

    correct = sum(
        row["correct"]
        for row in rows
    )

    wrong = (
        total - correct
    )

    accuracy = (
        correct / total
    )

    confidences = [
        row["confidence"]
        for row in rows
    ]

    correct_confidences = [
        row["confidence"]
        for row in rows
        if row["correct"]
    ]

    wrong_confidences = [
        row["confidence"]
        for row in rows
        if not row["correct"]
    ]

    mean_confidence = (
        statistics.mean(
            confidences
        )
    )

    signed_overall_gap = (
        mean_confidence
        - accuracy
    )


    bins, ece, mce = (
        calibration_bins(rows)
    )

    high_conf = (
        high_confidence_errors(
            rows
        )
    )

    selective = (
        selective_prediction(
            rows
        )
    )


    return {
        "model":
            name,
        "total":
            total,
        "correct":
            correct,
        "wrong":
            wrong,
        "accuracy":
            accuracy,
        "mean_confidence":
            mean_confidence,
        "signed_overall_confidence_gap":
            signed_overall_gap,
        "mean_confidence_correct":
            statistics.mean(
                correct_confidences
            ),
        "mean_confidence_wrong":
            statistics.mean(
                wrong_confidences
            ),
        "ece_15_bins":
            ece,
        "mce_15_bins":
            mce,
        "bins":
            bins,
        "high_confidence_errors":
            high_conf,
        "selective_prediction":
            selective,
    }


emoar_rows = load_emoar()
vit_rows = load_vit()


if len(emoar_rows) != 3589:
    raise RuntimeError(
        "Unexpected EmoAR row count."
    )

if len(vit_rows) != 3589:
    raise RuntimeError(
        "Unexpected ViT row count."
    )


emoar_indices = {
    row["sample_index"]
    for row in emoar_rows
}

vit_indices = {
    row["sample_index"]
    for row in vit_rows
}

if emoar_indices != vit_indices:
    raise RuntimeError(
        "Prediction files do not contain "
        "the same sample indices."
    )


emoar = analyze(
    "EmoAR DenseNet121",
    emoar_rows,
)

vit = analyze(
    "Modern ViT",
    vit_rows,
)


result = {
    "protocol": {
        "split":
            "privateTest",
        "samples":
            3589,
        "ece_bins":
            N_BINS,
        "binning":
            "equal-width",
        "high_confidence_thresholds":
            HIGH_CONF_THRESHOLDS,
        "selective_thresholds":
            SELECTIVE_THRESHOLDS,
    },
    "emoar":
        emoar,
    "vit":
        vit,
}


JSON_OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


with BINS_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "model",
        "bin_index",
        "lower_bound",
        "upper_bound",
        "count",
        "fraction",
        "accuracy",
        "mean_confidence",
        "signed_gap",
        "absolute_gap",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for analysis in [
        emoar,
        vit,
    ]:

        for row in analysis["bins"]:

            writer.writerow({
                "model":
                    analysis["model"],
                **row,
            })


with SELECTIVE_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "model",
        "threshold",
        "retained",
        "coverage",
        "accuracy",
        "errors",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for analysis in [
        emoar,
        vit,
    ]:

        for row in (
            analysis[
                "selective_prediction"
            ]
        ):

            writer.writerow({
                "model":
                    analysis["model"],
                **row,
            })


def print_analysis(data):

    print()
    print("=" * 80)
    print(data["model"])
    print("=" * 80)

    print(
        "Accuracy:",
        f"{data['accuracy']:.6f}"
    )

    print(
        "Mean confidence:",
        f"{data['mean_confidence']:.6f}"
    )

    print(
        "Signed confidence gap:",
        f"{data['signed_overall_confidence_gap']:+.6f}"
    )

    print(
        "ECE (15 bins):",
        f"{data['ece_15_bins']:.6f}"
    )

    print(
        "MCE (15 bins):",
        f"{data['mce_15_bins']:.6f}"
    )

    print(
        "Mean confidence correct:",
        f"{data['mean_confidence_correct']:.6f}"
    )

    print(
        "Mean confidence wrong:",
        f"{data['mean_confidence_wrong']:.6f}"
    )


    print()
    print("High-confidence errors")

    for row in (
        data[
            "high_confidence_errors"
        ]
    ):

        print(
            f"  >= {row['threshold']:.2f}: "
            f"{row['wrong_count']} "
            f"("
            f"{row['proportion_of_all_errors'] * 100:.2f}% "
            f"of errors)"
        )


    print()
    print("Selective prediction")

    print(
        f"{'THRESHOLD':>10s} "
        f"{'RETAINED':>10s} "
        f"{'COVERAGE':>10s} "
        f"{'ACCURACY':>10s} "
        f"{'ERRORS':>8s}"
    )

    for row in (
        data[
            "selective_prediction"
        ]
    ):

        accuracy = (
            f"{row['accuracy']:.6f}"
            if row["accuracy"]
            is not None
            else "NA"
        )

        print(
            f"{row['threshold']:10.2f} "
            f"{row['retained']:10d} "
            f"{row['coverage']:10.6f} "
            f"{accuracy:>10s} "
            f"{row['errors']:8d}"
        )


    print()
    print("Calibration bins")

    print(
        f"{'BIN':>4s} "
        f"{'RANGE':>15s} "
        f"{'N':>6s} "
        f"{'ACC':>9s} "
        f"{'CONF':>9s} "
        f"{'GAP':>9s}"
    )

    for row in data["bins"]:

        if row["count"] == 0:

            print(
                f"{row['bin_index']:4d} "
                f"[{row['lower_bound']:.3f},"
                f"{row['upper_bound']:.3f}] "
                f"{0:6d} "
                f"{'NA':>9s} "
                f"{'NA':>9s} "
                f"{'NA':>9s}"
            )

            continue

        print(
            f"{row['bin_index']:4d} "
            f"[{row['lower_bound']:.3f},"
            f"{row['upper_bound']:.3f}] "
            f"{row['count']:6d} "
            f"{row['accuracy']:9.6f} "
            f"{row['mean_confidence']:9.6f} "
            f"{row['signed_gap']:+9.6f}"
        )


print_analysis(emoar)
print_analysis(vit)


print()
print("=" * 80)
print("COMPARATIVE SUMMARY")
print("=" * 80)

print(
    "ECE difference ViT - EmoAR:",
    f"{vit['ece_15_bins'] - emoar['ece_15_bins']:+.6f}"
)

print(
    "MCE difference ViT - EmoAR:",
    f"{vit['mce_15_bins'] - emoar['mce_15_bins']:+.6f}"
)

print(
    "Overall confidence-gap difference:",
    f"{vit['signed_overall_confidence_gap'] - emoar['signed_overall_confidence_gap']:+.6f}"
)


print()
print("Saved:")
print(JSON_OUTPUT)
print(BINS_OUTPUT)
print(SELECTIVE_OUTPUT)

