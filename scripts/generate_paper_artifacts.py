from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load frozen results
# ---------------------------------------------------------------------

def load_json(name):

    with (
        RESULTS / name
    ).open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


emoar_private = load_json(
    "emoar_private_test.json"
)

vit_private = load_json(
    "vit_private_test.json"
)

latency = load_json(
    "latency_counterbalanced_summary.json"
)

confidence = load_json(
    "confidence_calibration_analysis.json"
)

stability = load_json(
    "perturbation_stability_results.json"
)


emoar_perf = (
    emoar_private[
        "pipelines"
    ][
        "web_255_crop224"
    ]
)

vit_perf = (
    vit_private[
        "metrics"
    ]
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def save_figure(
    fig,
    stem,
):

    png = (
        FIGURES
        / f"{stem}.png"
    )

    pdf = (
        FIGURES
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

table_rows = [
    {
        "model":
            "EmoAR DenseNet121",

        "accuracy":
            emoar_perf[
                "accuracy"
            ],

        "balanced_accuracy":
            emoar_perf[
                "balanced_accuracy"
            ],

        "macro_f1":
            emoar_perf[
                "macro_f1"
            ],

        "mean_latency_ms":
            latency[
                "overall"
            ][
                "emoar_ms"
            ][
                "mean"
            ],

        "latency_std_ms":
            latency[
                "overall"
            ][
                "emoar_ms"
            ][
                "std"
            ],

        "ece_15_bins":
            confidence[
                "emoar"
            ][
                "ece_15_bins"
            ],

        "mce_15_bins":
            confidence[
                "emoar"
            ][
                "mce_15_bins"
            ],

        "confidence_gap":
            confidence[
                "emoar"
            ][
                "signed_overall_confidence_gap"
            ],

        "perturbed_accuracy":
            stability[
                "emoar"
            ][
                "aggregate_perturbed_accuracy"
            ],

        "pairwise_stability":
            stability[
                "emoar"
            ][
                "pairwise_label_stability"
            ],

        "fully_stable_fraction":
            stability[
                "emoar"
            ][
                "fraction_fully_stable"
            ],
    },

    {
        "model":
            "Modern ViT",

        "accuracy":
            vit_perf[
                "accuracy"
            ],

        "balanced_accuracy":
            vit_perf[
                "balanced_accuracy"
            ],

        "macro_f1":
            vit_perf[
                "macro_f1"
            ],

        "mean_latency_ms":
            latency[
                "overall"
            ][
                "vit_ms"
            ][
                "mean"
            ],

        "latency_std_ms":
            latency[
                "overall"
            ][
                "vit_ms"
            ][
                "std"
            ],

        "ece_15_bins":
            confidence[
                "vit"
            ][
                "ece_15_bins"
            ],

        "mce_15_bins":
            confidence[
                "vit"
            ][
                "mce_15_bins"
            ],

        "confidence_gap":
            confidence[
                "vit"
            ][
                "signed_overall_confidence_gap"
            ],

        "perturbed_accuracy":
            stability[
                "vit"
            ][
                "aggregate_perturbed_accuracy"
            ],

        "pairwise_stability":
            stability[
                "vit"
            ][
                "pairwise_label_stability"
            ],

        "fully_stable_fraction":
            stability[
                "vit"
            ][
                "fraction_fully_stable"
            ],
    },
]


TABLE_CSV = (
    RESULTS
    / "paper_summary_table.csv"
)

with TABLE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            table_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        table_rows
    )


TABLE_MD = (
    RESULTS
    / "paper_summary_table.md"
)

with TABLE_MD.open(
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "| Metric | EmoAR DenseNet121 | Modern ViT |\n"
    )

    file.write(
        "|---|---:|---:|\n"
    )

    metrics = [
        (
            "Accuracy",
            "accuracy",
            ".4f",
        ),
        (
            "Balanced accuracy",
            "balanced_accuracy",
            ".4f",
        ),
        (
            "Macro-F1",
            "macro_f1",
            ".4f",
        ),
        (
            "CPU latency (ms)",
            "mean_latency_ms",
            ".2f",
        ),
        (
            "ECE (15 bins)",
            "ece_15_bins",
            ".4f",
        ),
        (
            "MCE (15 bins)",
            "mce_15_bins",
            ".4f",
        ),
        (
            "Confidence gap",
            "confidence_gap",
            ".4f",
        ),
        (
            "Perturbed accuracy",
            "perturbed_accuracy",
            ".4f",
        ),
        (
            "Pairwise stability",
            "pairwise_stability",
            ".4f",
        ),
        (
            "Fully stable fraction",
            "fully_stable_fraction",
            ".4f",
        ),
    ]

    for (
        label,
        key,
        fmt,
    ) in metrics:

        emoar_value = format(
            table_rows[0][key],
            fmt,
        )

        vit_value = format(
            table_rows[1][key],
            fmt,
        )

        file.write(
            f"| {label} | "
            f"{emoar_value} | "
            f"{vit_value} |\n"
        )


print("Saved:", TABLE_CSV)
print("Saved:", TABLE_MD)


# ---------------------------------------------------------------------
# Figure 1: FER2013 classification performance
# ---------------------------------------------------------------------

metrics = [
    "Accuracy",
    "Balanced accuracy",
    "Macro-F1",
]

emoar_values = [
    emoar_perf[
        "accuracy"
    ],
    emoar_perf[
        "balanced_accuracy"
    ],
    emoar_perf[
        "macro_f1"
    ],
]

vit_values = [
    vit_perf[
        "accuracy"
    ],
    vit_perf[
        "balanced_accuracy"
    ],
    vit_perf[
        "macro_f1"
    ],
]


x = np.arange(
    len(metrics)
)

width = 0.36


fig, ax = plt.subplots(
    figsize=(7.2, 4.5)
)

bars_emoar = ax.bar(
    x - width / 2,
    emoar_values,
    width,
    label="EmoAR DenseNet121",
)

bars_vit = ax.bar(
    x + width / 2,
    vit_values,
    width,
    label="Modern ViT",
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "FER2013 privateTest classification performance"
)

ax.set_xticks(
    x,
    metrics,
)

ax.set_ylim(
    0,
    0.85,
)

ax.legend()

ax.bar_label(
    bars_emoar,
    fmt="%.3f",
    padding=3,
)

ax.bar_label(
    bars_vit,
    fmt="%.3f",
    padding=3,
)

fig.tight_layout()

save_figure(
    fig,
    "01_private_test_performance",
)


# ---------------------------------------------------------------------
# Figure 2: CPU latency
# ---------------------------------------------------------------------

models = [
    "EmoAR",
    "ViT",
]

means = [
    latency[
        "overall"
    ][
        "emoar_ms"
    ][
        "mean"
    ],

    latency[
        "overall"
    ][
        "vit_ms"
    ][
        "mean"
    ],
]

stds = [
    latency[
        "overall"
    ][
        "emoar_ms"
    ][
        "std"
    ],

    latency[
        "overall"
    ][
        "vit_ms"
    ][
        "std"
    ],
]


fig, ax = plt.subplots(
    figsize=(5.5, 4.5)
)

bars = ax.bar(
    models,
    means,
    yerr=stds,
    capsize=5,
)

ax.set_ylabel(
    "Mean end-to-end latency (ms)"
)

ax.set_title(
    "Counterbalanced CPU latency benchmark"
)

ax.bar_label(
    bars,
    labels=[
        f"{value:.1f} ms"
        for value in means
    ],
    padding=5,
)

fig.tight_layout()

save_figure(
    fig,
    "02_cpu_latency",
)


# ---------------------------------------------------------------------
# Figure 3: Reliability diagram
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(6.0, 5.2)
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)


for (
    key,
    label,
) in [
    (
        "emoar",
        "EmoAR DenseNet121",
    ),
    (
        "vit",
        "Modern ViT",
    ),
]:

    bins = [
        row
        for row
        in confidence[
            key
        ][
            "bins"
        ]
        if row[
            "count"
        ] > 0
    ]

    x_values = [
        row[
            "mean_confidence"
        ]
        for row in bins
    ]

    y_values = [
        row[
            "accuracy"
        ]
        for row in bins
    ]

    ax.plot(
        x_values,
        y_values,
        marker="o",
        label=label,
    )


ax.set_xlabel(
    "Mean predictive confidence"
)

ax.set_ylabel(
    "Observed accuracy"
)

ax.set_title(
    "Top-label reliability diagram"
)

ax.set_xlim(
    0.2,
    1.0,
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "03_reliability_diagram",
)


# ---------------------------------------------------------------------
# Figure 4: Selective prediction
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(6.2, 5.0)
)


for (
    key,
    label,
) in [
    (
        "emoar",
        "EmoAR DenseNet121",
    ),
    (
        "vit",
        "Modern ViT",
    ),
]:

    rows = confidence[
        key
    ][
        "selective_prediction"
    ]

    rows = sorted(
        rows,
        key=lambda row:
            row["coverage"],
    )

    coverage = [
        row["coverage"]
        for row in rows
    ]

    accuracy = [
        row["accuracy"]
        for row in rows
    ]

    ax.plot(
        coverage,
        accuracy,
        marker="o",
        label=label,
    )

    for row in rows:

        if row[
            "threshold"
        ] not in {
            0.80,
            0.90,
            0.95,
            0.99,
        }:
            continue

        ax.annotate(
            f"{row['threshold']:.2f}",
            (
                row[
                    "coverage"
                ],
                row[
                    "accuracy"
                ],
            ),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )


ax.set_xlabel(
    "Coverage"
)

ax.set_ylabel(
    "Accuracy among retained predictions"
)

ax.set_title(
    "Selective prediction: accuracy versus coverage"
)

ax.set_xlim(
    0,
    1.02,
)

ax.set_ylim(
    0.60,
    1.00,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "04_accuracy_coverage",
)


# ---------------------------------------------------------------------
# Figure 5: Perturbation stability
# ---------------------------------------------------------------------

conditions = [
    (
        "rotation_minus_5",
        "Rot. -5°",
    ),
    (
        "rotation_plus_5",
        "Rot. +5°",
    ),
    (
        "translate_minus_2",
        "Trans. -2 px",
    ),
    (
        "translate_plus_2",
        "Trans. +2 px",
    ),
    (
        "brightness_0_90",
        "Bright. 0.90",
    ),
    (
        "brightness_1_10",
        "Bright. 1.10",
    ),
    (
        "contrast_0_90",
        "Contrast 0.90",
    ),
    (
        "contrast_1_10",
        "Contrast 1.10",
    ),
]


labels = [
    label
    for _, label
    in conditions
]

emoar_stability = [
    stability[
        "emoar"
    ][
        "per_condition"
    ][key][
        "label_stability"
    ]
    for key, _
    in conditions
]

vit_stability = [
    stability[
        "vit"
    ][
        "per_condition"
    ][key][
        "label_stability"
    ]
    for key, _
    in conditions
]


x = np.arange(
    len(labels)
)

width = 0.36


fig, ax = plt.subplots(
    figsize=(9.5, 4.8)
)

ax.bar(
    x - width / 2,
    emoar_stability,
    width,
    label="EmoAR DenseNet121",
)

ax.bar(
    x + width / 2,
    vit_stability,
    width,
    label="Modern ViT",
)

ax.set_ylabel(
    "Pairwise label stability"
)

ax.set_title(
    "Robustness to controlled image perturbations"
)

ax.set_xticks(
    x,
    labels,
    rotation=30,
    ha="right",
)

ax.set_ylim(
    0.65,
    1.0,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "05_perturbation_stability",
)


print()
print("=" * 80)
print("Paper artifacts generated successfully")
print("=" * 80)

print()
print(
    "Summary table:",
    TABLE_CSV,
)

print(
    "Markdown table:",
    TABLE_MD,
)

print(
    "Figures directory:",
    FIGURES,
)

