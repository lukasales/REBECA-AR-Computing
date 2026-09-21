from pathlib import Path
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    ROOT
    / "results"
    / "latency_counterbalanced"
)

OUTPUT = (
    ROOT
    / "results"
    / "latency_counterbalanced_summary.json"
)


runs = []

for i in range(1, 7):

    path = RUN_DIR / f"run_{i}.json"

    with path.open(
        encoding="utf-8"
    ) as file:
        runs.append(
            json.load(file)
        )


def summarize(values):

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


emoar_all = [
    run["emoar"]["mean_ms"]
    for run in runs
]

vit_all = [
    run["vit"]["mean_ms"]
    for run in runs
]

ratios_all = [
    run["vit_to_emoar_mean_latency_ratio"]
    for run in runs
]


groups = {}

for order in [
    "emoar-first",
    "vit-first",
]:

    selected = [
        run
        for run in runs
        if run["order"] == order
    ]

    emoar = [
        run["emoar"]["mean_ms"]
        for run in selected
    ]

    vit = [
        run["vit"]["mean_ms"]
        for run in selected
    ]

    ratios = [
        run["vit_to_emoar_mean_latency_ratio"]
        for run in selected
    ]

    groups[order] = {
        "emoar_ms":
            summarize(emoar),
        "vit_ms":
            summarize(vit),
        "latency_ratio":
            summarize(ratios),
    }


summary = {
    "protocol": {
        "runs": 6,
        "runs_per_order": 3,
        "images_per_run": 100,
        "batch_size": 1,
        "device": "cpu",
        "orders": [
            run["order"]
            for run in runs
        ],
    },
    "overall": {
        "emoar_ms":
            summarize(emoar_all),
        "vit_ms":
            summarize(vit_all),
        "latency_ratio":
            summarize(ratios_all),
        "ratio_of_overall_mean_latencies":
            statistics.mean(vit_all)
            / statistics.mean(emoar_all),
    },
    "by_order":
        groups,
}


OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 80)
print("Counterbalanced latency summary")
print("=" * 80)

print()

print("Overall")

print(
    f"  EmoAR mean:   "
    f"{statistics.mean(emoar_all):.3f} ms"
)

print(
    f"  EmoAR median: "
    f"{statistics.median(emoar_all):.3f} ms"
)

print(
    f"  ViT mean:     "
    f"{statistics.mean(vit_all):.3f} ms"
)

print(
    f"  ViT median:   "
    f"{statistics.median(vit_all):.3f} ms"
)

print(
    f"  Mean ratio:   "
    f"{statistics.mean(ratios_all):.3f}x"
)

print(
    f"  Median ratio: "
    f"{statistics.median(ratios_all):.3f}x"
)

print(
    f"  Ratio range:  "
    f"{min(ratios_all):.3f}x - "
    f"{max(ratios_all):.3f}x"
)

print()

for order in [
    "emoar-first",
    "vit-first",
]:

    data = groups[order]

    print(order)

    print(
        f"  EmoAR mean: "
        f"{data['emoar_ms']['mean']:.3f} ms"
    )

    print(
        f"  ViT mean:   "
        f"{data['vit_ms']['mean']:.3f} ms"
    )

    print(
        f"  Mean ratio: "
        f"{data['latency_ratio']['mean']:.3f}x"
    )

    print()


print("Saved:")
print(OUTPUT)
