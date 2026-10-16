from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "temporal_video_summary_aggregate.csv"

OUTPUT_PNG = ROOT / "figures" / "06_temporal_applicability_tradeoff.png"
OUTPUT_PDF = ROOT / "figures" / "06_temporal_applicability_tradeoff.pdf"


with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 2:
    raise RuntimeError(f"Expected 2 models, found {len(rows)}")


data = {}

for row in rows:
    model = row["model"]
    data[model] = {
        "latency": float(row["mean_latency_ms"]),
        "change_rate": float(row["change_rate"]),
        "fps": float(row["effective_classifier_fps"]),
        "confidence_delta_p95": float(row["p95_abs_conf_delta"]),
        "agreement": float(row["consecutive_agreement"]),
    }


fig, ax = plt.subplots(figsize=(8.0, 5.4))

markers = {
    "emoar": "o",
    "vit": "s",
}

names = {
    "emoar": "EmoAR (DenseNet121)",
    "vit": "ViT",
}


xs = []
ys = []

for model in ["emoar", "vit"]:
    d = data[model]
    x = d["latency"]
    y = d["change_rate"] * 100.0

    xs.append(x)
    ys.append(y)

    ax.scatter(
        x,
        y,
        s=140,
        marker=markers[model],
        label=names[model],
    )

    annotation = (
        f"{names[model]}\n"
        f"{d['fps']:.2f} FPS\n"
        f"P95 |Δconf| = {d['confidence_delta_p95']:.3f}"
    )

    if model == "emoar":
        ax.annotate(
            annotation,
            (x, y),
            xytext=(18, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
        )
    else:
        ax.annotate(
            annotation,
            (x, y),
            xytext=(-18, 18),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
        )


ax.set_xlabel("Mean classifier latency (ms)")
ax.set_ylabel("Temporal class-change rate (%)")
ax.set_title("Temporal applicability trade-off on RAVDESS")

ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

x_margin = (max(xs) - min(xs)) * 0.05
y_margin = (max(ys) - min(ys)) * 0.20

ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

ax.text(
    0.02,
    0.02,
    "Lower-left indicates lower latency and lower temporal flickering.",
    transform=ax.transAxes,
    fontsize=8.5,
    va="bottom",
)

fig.tight_layout(rect=(0, 0, 1, 0.96))

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")

plt.close(fig)

print()
print("========== TEMPORAL FIGURE ==========")
for model in ["emoar", "vit"]:
    d = data[model]
    print(
        f"{names[model]:20s} | "
        f"latency={d['latency']:.3f} ms | "
        f"flicker={d['change_rate']:.4%} | "
        f"agreement={d['agreement']:.4%} | "
        f"FPS={d['fps']:.3f} | "
        f"P95 Δconf={d['confidence_delta_p95']:.4f}"
    )

print()
print(f"[OK] {OUTPUT_PNG}")
print(f"[OK] {OUTPUT_PDF}")
print()
print("TEMPORAL FIGURE: OK")
