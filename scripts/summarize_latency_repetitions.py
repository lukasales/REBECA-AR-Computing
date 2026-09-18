from pathlib import Path
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    ROOT
    / "results"
    / "latency_repetitions"
)

OUTPUT = (
    ROOT
    / "results"
    / "latency_repetitions_summary.json"
)


def summarize(values):

    return {
        "n_runs": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
        "cv_percent": (
            statistics.stdev(values)
            / statistics.mean(values)
            * 100
            if len(values) > 1
            else 0.0
        ),
    }


runs = []

for index in range(1, 6):

    path = (
        RUN_DIR
        / f"run_{index}.json"
    )

    with path.open(
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    runs.append(data)


emoar_e2e = [
    run["emoar"]["end_to_end"]["mean_ms"]
    for run in runs
]

vit_e2e = [
    run["vit"]["end_to_end"]["mean_ms"]
    for run in runs
]

emoar_inference = [
    run["emoar"]["inference_only"]["mean_ms"]
    for run in runs
]

vit_inference = [
    run["vit"]["inference_only"]["mean_ms"]
    for run in runs
]

emoar_preprocessing = [
    run["emoar"]["preprocessing"]["mean_ms"]
    for run in runs
]

vit_preprocessing = [
    run["vit"]["preprocessing"]["mean_ms"]
    for run in runs
]

latency_ratios = [
    vit_e2e[i] / emoar_e2e[i]
    for i in range(len(runs))
]


emoar_parameters = runs[0]["emoar"]["parameters"]
vit_parameters = runs[0]["vit"]["parameters"]

emoar_memory = (
    runs[0]["emoar"]
    ["tensor_footprint"]
    ["total_tensor_mb"]
)

vit_memory = (
    runs[0]["vit"]
    ["tensor_footprint"]
    ["total_tensor_mb"]
)


summary = {
    "protocol": {
        "runs": 5,
        "images_per_run": 100,
        "batch_size": 1,
        "device": "cpu",
        "warmup_runs": 10,
        "note": (
            "Model order was fixed: EmoAR before ViT. "
            "Absolute latency therefore requires "
            "counterbalanced confirmation."
        ),
    },
    "emoar": {
        "parameters": emoar_parameters,
        "tensor_footprint_mb": emoar_memory,
        "preprocessing_mean_ms_across_runs":
            summarize(emoar_preprocessing),
        "inference_mean_ms_across_runs":
            summarize(emoar_inference),
        "end_to_end_mean_ms_across_runs":
            summarize(emoar_e2e),
    },
    "vit": {
        "parameters": vit_parameters,
        "tensor_footprint_mb": vit_memory,
        "preprocessing_mean_ms_across_runs":
            summarize(vit_preprocessing),
        "inference_mean_ms_across_runs":
            summarize(vit_inference),
        "end_to_end_mean_ms_across_runs":
            summarize(vit_e2e),
    },
    "relative_cost": {
        "parameter_ratio":
            vit_parameters
            / emoar_parameters,
        "tensor_memory_ratio":
            vit_memory
            / emoar_memory,
        "end_to_end_latency_ratio_across_runs":
            summarize(latency_ratios),
    },
}


OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 80)
print("Repeated latency benchmark summary")
print("=" * 80)

print()

for name, values in [
    ("EmoAR end-to-end", emoar_e2e),
    ("ViT end-to-end", vit_e2e),
    ("Latency ratio", latency_ratios),
]:

    result = summarize(values)

    print(name)
    print(
        f"  mean:   {result['mean']:.3f}"
    )
    print(
        f"  median: {result['median']:.3f}"
    )
    print(
        f"  std:    {result['std']:.3f}"
    )
    print(
        f"  min:    {result['min']:.3f}"
    )
    print(
        f"  max:    {result['max']:.3f}"
    )
    print(
        f"  CV:     {result['cv_percent']:.2f}%"
    )
    print()


print(
    "EmoAR effective FPS from aggregate mean:",
    f"{1000 / statistics.mean(emoar_e2e):.3f}",
)

print(
    "ViT effective FPS from aggregate mean:",
    f"{1000 / statistics.mean(vit_e2e):.3f}",
)

print()
print(
    "Parameter ratio:",
    f"{vit_parameters / emoar_parameters:.3f}x",
)

print(
    "Tensor-memory ratio:",
    f"{vit_memory / emoar_memory:.3f}x",
)

print()
print("Saved:")
print(OUTPUT)
