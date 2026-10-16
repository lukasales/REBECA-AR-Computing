from pathlib import Path
import csv
import json
import math
import statistics
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "temporal_video_predictions.csv"
)

METADATA = (
    ROOT
    / "results"
    / "temporal_video_run_metadata.json"
)

OUTPUT_BY_VIDEO = (
    ROOT
    / "results"
    / "temporal_video_summary_by_video.csv"
)

OUTPUT_AGGREGATE = (
    ROOT
    / "results"
    / "temporal_video_summary_aggregate.csv"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "temporal_video_analysis.json"
)

OUTPUT_MD = (
    ROOT
    / "results"
    / "temporal_video_summary.md"
)


CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

MODELS = [
    "emoar",
    "vit",
]


def percentile(values, p):
    values = sorted(
        float(v)
        for v in values
    )

    if not values:
        return float("nan")

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        * (1.0 - fraction)
        + values[upper]
        * fraction
    )


def safe_mean(values):
    if not values:
        return float("nan")

    return statistics.mean(values)


def safe_median(values):
    if not values:
        return float("nan")

    return statistics.median(values)


def safe_pstdev(values):
    if not values:
        return float("nan")

    return statistics.pstdev(values)


def run_lengths(labels):
    if not labels:
        return []

    lengths = []

    current = labels[0]
    length = 1

    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            lengths.append(length)
            current = label
            length = 1

    lengths.append(length)

    return lengths


def confidence_deltas(rows):
    return [
        abs(
            rows[i]["confidence"]
            - rows[i - 1]["confidence"]
        )
        for i in range(
            1,
            len(rows)
        )
    ]


def sequence_metrics(rows):
    rows = sorted(
        rows,
        key=lambda r: r["frame_index"],
    )

    labels = [
        row["predicted_class"]
        for row in rows
    ]

    n_frames = len(rows)

    transitions = max(
        n_frames - 1,
        0,
    )

    switches = sum(
        labels[i]
        != labels[i - 1]
        for i in range(
            1,
            len(labels)
        )
    )

    change_rate = (
        switches / transitions
        if transitions
        else 0.0
    )

    agreement = (
        1.0 - change_rate
        if transitions
        else 1.0
    )

    runs = run_lengths(
        labels
    )

    deltas = confidence_deltas(
        rows
    )

    return {
        "labels": labels,
        "n_frames": n_frames,
        "transitions": transitions,
        "switches": switches,
        "change_rate": change_rate,
        "consecutive_agreement": agreement,
        "run_lengths": runs,
        "confidence_deltas": deltas,
    }


def prediction_distribution(rows):
    counts = Counter(
        row["predicted_class"]
        for row in rows
    )

    total = len(rows)

    result = {}

    for label in CLASSES:
        count = counts[label]

        result[
            f"pred_count_{label}"
        ] = count

        result[
            f"pred_prop_{label}"
        ] = (
            count / total
            if total
            else 0.0
        )

    dominant = max(
        CLASSES,
        key=lambda label: counts[label],
    )

    result[
        "dominant_prediction"
    ] = dominant

    result[
        "dominant_prediction_share"
    ] = (
        counts[dominant] / total
        if total
        else 0.0
    )

    return result


def summarize_video(
    model,
    video,
    rows,
):
    rows = sorted(
        rows,
        key=lambda r: r["frame_index"],
    )

    seq = sequence_metrics(
        rows
    )

    confidences = [
        row["confidence"]
        for row in rows
    ]

    latencies = [
        row["latency_ms"]
        for row in rows
    ]

    runs = seq[
        "run_lengths"
    ]

    deltas = seq[
        "confidence_deltas"
    ]

    fps = rows[0]["video_fps"]

    summary = {
        "model": model,
        "video": video,
        "actor": rows[0]["actor"],
        "clip_label": rows[0][
            "expected_expression"
        ],
        "n_frames": seq[
            "n_frames"
        ],
        "n_transitions": seq[
            "transitions"
        ],
        "switches": seq[
            "switches"
        ],
        "change_rate": seq[
            "change_rate"
        ],
        "consecutive_agreement": seq[
            "consecutive_agreement"
        ],
        "n_runs": len(runs),
        "mean_run_frames": (
            safe_mean(runs)
        ),
        "median_run_frames": (
            safe_median(runs)
        ),
        "max_run_frames": (
            max(runs)
            if runs
            else 0
        ),
        "max_run_seconds": (
            max(runs) / fps
            if runs and fps > 0
            else float("nan")
        ),
        "mean_confidence": (
            safe_mean(confidences)
        ),
        "std_confidence": (
            safe_pstdev(confidences)
        ),
        "mean_abs_conf_delta": (
            safe_mean(deltas)
        ),
        "p95_abs_conf_delta": (
            percentile(
                deltas,
                95,
            )
        ),
        "mean_latency_ms": (
            safe_mean(latencies)
        ),
        "median_latency_ms": (
            safe_median(latencies)
        ),
        "p95_latency_ms": (
            percentile(
                latencies,
                95,
            )
        ),
    }

    summary[
        "effective_classifier_fps"
    ] = (
        1000.0
        / summary["mean_latency_ms"]
        if summary["mean_latency_ms"] > 0
        else float("nan")
    )

    summary.update(
        prediction_distribution(
            rows
        )
    )

    return summary


def summarize_aggregate(
    model,
    grouped_video_rows,
):
    all_rows = []

    all_runs = []

    all_conf_deltas = []

    total_switches = 0
    total_transitions = 0

    for video in sorted(
        grouped_video_rows
    ):
        rows = sorted(
            grouped_video_rows[video],
            key=lambda r: r["frame_index"],
        )

        all_rows.extend(rows)

        seq = sequence_metrics(
            rows
        )

        total_switches += (
            seq["switches"]
        )

        total_transitions += (
            seq["transitions"]
        )

        all_runs.extend(
            seq["run_lengths"]
        )

        all_conf_deltas.extend(
            seq["confidence_deltas"]
        )

    confidences = [
        row["confidence"]
        for row in all_rows
    ]

    latencies = [
        row["latency_ms"]
        for row in all_rows
    ]

    change_rate = (
        total_switches
        / total_transitions
        if total_transitions
        else 0.0
    )

    result = {
        "model": model,
        "n_videos": len(
            grouped_video_rows
        ),
        "n_frames": len(
            all_rows
        ),
        "n_transitions": (
            total_transitions
        ),
        "switches": (
            total_switches
        ),
        "change_rate": (
            change_rate
        ),
        "consecutive_agreement": (
            1.0 - change_rate
        ),
        "n_runs": len(
            all_runs
        ),
        "mean_run_frames": (
            safe_mean(
                all_runs
            )
        ),
        "median_run_frames": (
            safe_median(
                all_runs
            )
        ),
        "max_run_frames": (
            max(all_runs)
            if all_runs
            else 0
        ),
        "mean_confidence": (
            safe_mean(
                confidences
            )
        ),
        "std_confidence": (
            safe_pstdev(
                confidences
            )
        ),
        "mean_abs_conf_delta": (
            safe_mean(
                all_conf_deltas
            )
        ),
        "p95_abs_conf_delta": (
            percentile(
                all_conf_deltas,
                95,
            )
        ),
        "mean_latency_ms": (
            safe_mean(
                latencies
            )
        ),
        "median_latency_ms": (
            safe_median(
                latencies
            )
        ),
        "p95_latency_ms": (
            percentile(
                latencies,
                95,
            )
        ),
    }

    result[
        "effective_classifier_fps"
    ] = (
        1000.0
        / result["mean_latency_ms"]
        if result["mean_latency_ms"] > 0
        else float("nan")
    )

    result.update(
        prediction_distribution(
            all_rows
        )
    )

    return result


def clean_number(value):
    if isinstance(
        value,
        float,
    ):
        if math.isnan(value):
            return ""

        return round(
            value,
            6,
        )

    return value


if not INPUT.exists():
    raise FileNotFoundError(
        INPUT
    )

if not METADATA.exists():
    raise FileNotFoundError(
        METADATA
    )


with INPUT.open(
    newline="",
    encoding="utf-8",
) as f:
    raw_rows = list(
        csv.DictReader(f)
    )


rows = []

for row in raw_rows:
    rows.append({
        "video": row["video"],
        "actor": row["actor"],
        "expected_expression":
            row["expected_expression"],
        "frame_index":
            int(row["frame_index"]),
        "timestamp_s":
            float(row["timestamp_s"]),
        "video_fps":
            float(row["video_fps"]),
        "model":
            row["model"],
        "predicted_class":
            row["predicted_class"],
        "confidence":
            float(row["confidence"]),
        "latency_ms":
            float(row["latency_ms"]),
        "frame_model_order":
            row["frame_model_order"],
    })


if len(rows) != 3240:
    raise RuntimeError(
        f"Expected 3240 rows, "
        f"found {len(rows)}"
    )


grouped = defaultdict(
    lambda: defaultdict(list)
)

for row in rows:
    grouped[
        row["model"]
    ][
        row["video"]
    ].append(row)


if set(grouped) != set(MODELS):
    raise RuntimeError(
        f"Unexpected models: "
        f"{sorted(grouped)}"
    )


by_video = []

aggregate = []


for model in MODELS:

    if len(grouped[model]) != 14:
        raise RuntimeError(
            f"{model}: expected 14 "
            f"videos, found "
            f"{len(grouped[model])}"
        )

    for video in sorted(
        grouped[model]
    ):
        by_video.append(
            summarize_video(
                model,
                video,
                grouped[model][video],
            )
        )

    aggregate.append(
        summarize_aggregate(
            model,
            grouped[model],
        )
    )


by_video = [
    {
        key: clean_number(value)
        for key, value
        in row.items()
    }
    for row in by_video
]

aggregate = [
    {
        key: clean_number(value)
        for key, value
        in row.items()
    }
    for row in aggregate
]


OUTPUT_BY_VIDEO.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT_BY_VIDEO.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            by_video[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        by_video
    )


with OUTPUT_AGGREGATE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            aggregate[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        aggregate
    )


metadata = json.loads(
    METADATA.read_text(
        encoding="utf-8"
    )
)


analysis = {
    "source_predictions":
        "results/temporal_video_predictions.csv",

    "source_metadata":
        "results/temporal_video_run_metadata.json",

    "decoded_frames":
        metadata["decoded_frames"],

    "analysis_notes": {
        "flickering_definition":
            "switches / consecutive-frame transitions",

        "consecutive_agreement":
            "1 - change_rate",

        "transition_policy":
            (
                "transitions are counted only "
                "within each video; never across "
                "video boundaries"
            ),

        "confidence_delta":
            (
                "absolute difference in maximum "
                "softmax confidence between "
                "consecutive frames"
            ),

        "effective_classifier_fps":
            (
                "1000 / mean classifier latency_ms; "
                "this excludes video decoding, crop, "
                "color conversion and AR rendering"
            ),

        "clip_labels":
            (
                "RAVDESS clip emotion labels are "
                "retained only as descriptive "
                "metadata and are not treated as "
                "frame-level ground truth"
            ),
    },

    "aggregate":
        aggregate,

    "by_video":
        by_video,
}


OUTPUT_JSON.write_text(
    json.dumps(
        analysis,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


lookup = {
    row["model"]: row
    for row in aggregate
}


def fmt(value, digits=4):
    if value == "":
        return "-"

    if isinstance(
        value,
        (int, float),
    ):
        return f"{value:.{digits}f}"

    return str(value)


md = []

md.append(
    "# Temporal video benchmark summary"
)

md.append("")

md.append(
    "Primary temporal analysis of the "
    "RAVDESS subset. Clip-level emotion "
    "labels are not treated as ground "
    "truth for each individual frame."
)

md.append("")

md.append(
    "| Metric | EmoAR | ViT |"
)

md.append(
    "|---|---:|---:|"
)


metrics = [
    (
        "Frames",
        "n_frames",
        0,
    ),
    (
        "Class switches",
        "switches",
        0,
    ),
    (
        "Change / flickering rate",
        "change_rate",
        4,
    ),
    (
        "Consecutive agreement",
        "consecutive_agreement",
        4,
    ),
    (
        "Prediction runs",
        "n_runs",
        0,
    ),
    (
        "Mean run length (frames)",
        "mean_run_frames",
        2,
    ),
    (
        "Median run length (frames)",
        "median_run_frames",
        2,
    ),
    (
        "Maximum run length (frames)",
        "max_run_frames",
        0,
    ),
    (
        "Mean confidence",
        "mean_confidence",
        4,
    ),
    (
        "Confidence SD",
        "std_confidence",
        4,
    ),
    (
        "Mean |Δ confidence|",
        "mean_abs_conf_delta",
        4,
    ),
    (
        "P95 |Δ confidence|",
        "p95_abs_conf_delta",
        4,
    ),
    (
        "Mean latency (ms)",
        "mean_latency_ms",
        2,
    ),
    (
        "Median latency (ms)",
        "median_latency_ms",
        2,
    ),
    (
        "P95 latency (ms)",
        "p95_latency_ms",
        2,
    ),
    (
        "Effective classifier FPS",
        "effective_classifier_fps",
        2,
    ),
]


for label, key, digits in metrics:

    emoar = lookup[
        "emoar"
    ][key]

    vit = lookup[
        "vit"
    ][key]

    md.append(
        f"| {label} | "
        f"{fmt(emoar, digits)} | "
        f"{fmt(vit, digits)} |"
    )


md.append("")

md.append(
    "Change rate is computed only between "
    "consecutive frames belonging to the "
    "same video."
)

md.append("")

md.append(
    "Latency represents model-specific "
    "preprocessing plus classification; "
    "it does not represent end-to-end "
    "AR application latency."
)


OUTPUT_MD.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)


print(
    "\n========== TEMPORAL ANALYSIS =========="
)


for row in aggregate:

    print()
    print(
        f"MODEL: {row['model'].upper()}"
    )

    print(
        f"Frames                : "
        f"{row['n_frames']}"
    )

    print(
        f"Transitions           : "
        f"{row['n_transitions']}"
    )

    print(
        f"Class switches        : "
        f"{row['switches']}"
    )

    print(
        f"Change rate           : "
        f"{row['change_rate']:.6f}"
    )

    print(
        f"Consecutive agreement : "
        f"{row['consecutive_agreement']:.6f}"
    )

    print(
        f"Runs                   : "
        f"{row['n_runs']}"
    )

    print(
        f"Mean run frames        : "
        f"{row['mean_run_frames']:.3f}"
    )

    print(
        f"Median run frames      : "
        f"{row['median_run_frames']:.3f}"
    )

    print(
        f"Max run frames         : "
        f"{row['max_run_frames']}"
    )

    print(
        f"Mean confidence        : "
        f"{row['mean_confidence']:.6f}"
    )

    print(
        f"Mean |delta conf|      : "
        f"{row['mean_abs_conf_delta']:.6f}"
    )

    print(
        f"P95 |delta conf|       : "
        f"{row['p95_abs_conf_delta']:.6f}"
    )

    print(
        f"Mean latency ms        : "
        f"{row['mean_latency_ms']:.3f}"
    )

    print(
        f"Median latency ms      : "
        f"{row['median_latency_ms']:.3f}"
    )

    print(
        f"P95 latency ms         : "
        f"{row['p95_latency_ms']:.3f}"
    )

    print(
        f"Effective FPS          : "
        f"{row['effective_classifier_fps']:.3f}"
    )

    print(
        f"Dominant prediction    : "
        f"{row['dominant_prediction']} "
        f"({row['dominant_prediction_share']:.4f})"
    )


print()
print(
    f"[OK] {OUTPUT_BY_VIDEO}"
)

print(
    f"[OK] {OUTPUT_AGGREGATE}"
)

print(
    f"[OK] {OUTPUT_JSON}"
)

print(
    f"[OK] {OUTPUT_MD}"
)

print()
print(
    "TEMPORAL ANALYSIS: OK"
)
