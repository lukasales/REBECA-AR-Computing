from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import time

import cv2
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoar_model import (
    CLASSES as EMOAR_CLASSES,
    historical_web_transform,
    load_emoar_model,
)


VIDEO_DIR = (
    ROOT
    / "data"
    / "video"
    / "ravdess"
    / "selected"
)

CROP_MANIFEST = (
    ROOT
    / "experiments"
    / "manifests"
    / "ravdess_fixed_face_crops.csv"
)

EMOAR_CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

VIT_MODEL_ID = "trpakov/vit-face-expression"
VIT_REVISION = (
    "ef0bc6fc34241b6587e7e009e7711357be28c024"
)

CANONICAL_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def normalize_label(label):
    value = str(label).strip().lower()

    aliases = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "happy": "happy",
        "happiness": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "neutral": "neutral",
    }

    if value not in aliases:
        raise RuntimeError(
            f"Unknown model label: {label!r}"
        )

    return aliases[value]


def load_crop_manifest():
    with CROP_MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 14:
        raise RuntimeError(
            f"Expected 14 crop rows, found {len(rows)}"
        )

    return {
        row["filename"]: {
            "x1": int(row["crop_x1"]),
            "y1": int(row["crop_y1"]),
            "x2": int(row["crop_x2"]),
            "y2": int(row["crop_y2"]),
        }
        for row in rows
    }


def build_vit_label_map(model):
    raw = model.config.id2label

    mapping = {}

    for index in range(model.config.num_labels):
        label = raw.get(
            index,
            raw.get(str(index)),
        )

        mapping[index] = normalize_label(
            label
        )

    mapped = set(mapping.values())

    expected = set(CANONICAL_CLASSES)

    if mapped != expected:
        raise RuntimeError(
            "Unexpected ViT labels.\n"
            f"Raw: {raw}\n"
            f"Mapped: {mapping}"
        )

    return mapping


def emoar_predict(
    model,
    transform,
    image,
):
    start = time.perf_counter_ns()

    tensor = (
        transform(image)
        .unsqueeze(0)
    )

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    values = {
        str(name).lower(): float(
            probabilities[index].item()
        )
        for index, name
        in enumerate(EMOAR_CLASSES)
    }

    predicted = max(
        values,
        key=values.get,
    )

    return (
        predicted,
        values[predicted],
        values,
        elapsed_ms,
    )


def vit_predict(
    model,
    processor,
    label_map,
    image,
):
    start = time.perf_counter_ns()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(
            **inputs
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    values = {
        label_map[index]: float(
            probabilities[index].item()
        )
        for index
        in range(len(probabilities))
    }

    predicted = max(
        values,
        key=values.get,
    )

    return (
        predicted,
        values[predicted],
        values,
        elapsed_ms,
    )


def warm_up(
    image,
    emoar_model,
    emoar_transform,
    vit_model,
    vit_processor,
    vit_label_map,
):
    print("Warming up models...")

    for _ in range(3):
        emoar_predict(
            emoar_model,
            emoar_transform,
            image,
        )

        vit_predict(
            vit_model,
            vit_processor,
            vit_label_map,
            image,
        )

    print("Warm-up complete.")


def frame_to_pil(
    frame,
    crop,
):
    face = frame[
        crop["y1"]:crop["y2"],
        crop["x1"]:crop["x2"],
    ]

    if face.size == 0:
        raise RuntimeError(
            "Empty fixed face crop."
        )

    rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def get_expected_expression(
    filename,
):
    parts = Path(filename).stem.split("-")

    emotion_code = parts[2]

    if emotion_code not in RAVDESS_EMOTIONS:
        raise RuntimeError(
            f"Unexpected RAVDESS emotion: "
            f"{emotion_code}"
        )

    return RAVDESS_EMOTIONS[
        emotion_code
    ]


def get_actor(filename):
    return (
        Path(filename)
        .stem
        .split("-")[-1]
    )


def prediction_row(
    *,
    video,
    actor,
    expected,
    frame_index,
    timestamp_s,
    fps,
    model_name,
    predicted,
    confidence,
    probabilities,
    latency_ms,
    order,
):
    row = {
        "video": video,
        "actor": actor,
        "expected_expression": expected,
        "frame_index": frame_index,
        "timestamp_s": round(
            timestamp_s,
            6,
        ),
        "video_fps": round(
            fps,
            6,
        ),
        "model": model_name,
        "predicted_class": predicted,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "frame_model_order": order,
    }

    for label in CANONICAL_CLASSES:
        row[f"prob_{label}"] = (
            probabilities[label]
        )

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Process only the first "
            "3 decoded frames of the "
            "first video."
        ),
    )

    args = parser.parse_args()

    if not EMOAR_CHECKPOINT.exists():
        raise FileNotFoundError(
            EMOAR_CHECKPOINT
        )

    crop_manifest = (
        load_crop_manifest()
    )

    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    if len(videos) != 14:
        raise RuntimeError(
            f"Expected 14 videos, "
            f"found {len(videos)}"
        )

    print(
        "Loading EmoAR..."
    )

    emoar_model = load_emoar_model(
        EMOAR_CHECKPOINT
    )

    emoar_transform = (
        historical_web_transform()
    )

    print(
        "Loading ViT..."
    )

    vit_processor = (
        AutoImageProcessor.from_pretrained(
            VIT_MODEL_ID,
            revision=VIT_REVISION,
        )
    )

    vit_model = (
        AutoModelForImageClassification
        .from_pretrained(
            VIT_MODEL_ID,
            revision=VIT_REVISION,
        )
    )

    vit_model.eval()

    vit_label_map = (
        build_vit_label_map(
            vit_model
        )
    )

    print(
        "ViT label map:",
        vit_label_map,
    )

    # Prepare first frame for warm-up.
    first_video = videos[0]

    first_cap = cv2.VideoCapture(
        str(first_video)
    )

    ok, first_frame = (
        first_cap.read()
    )

    first_cap.release()

    if not ok:
        raise RuntimeError(
            "Could not decode first "
            "warm-up frame."
        )

    first_crop = crop_manifest[
        first_video.name
    ]

    first_image = frame_to_pil(
        first_frame,
        first_crop,
    )

    warm_up(
        first_image,
        emoar_model,
        emoar_transform,
        vit_model,
        vit_processor,
        vit_label_map,
    )

    if args.smoke:
        output_dir = (
            ROOT
            / "data"
            / "video"
            / "ravdess"
            / "qa"
        )

        output_csv = (
            output_dir
            / "temporal_benchmark_smoke.csv"
        )

        output_json = (
            output_dir
            / "temporal_benchmark_smoke.json"
        )

        videos_to_run = [
            videos[0]
        ]

        max_frames = 3

    else:
        output_dir = (
            ROOT
            / "results"
        )

        output_csv = (
            output_dir
            / "temporal_video_predictions.csv"
        )

        output_json = (
            output_dir
            / "temporal_video_run_metadata.json"
        )

        videos_to_run = videos
        max_frames = None

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_decoded = 0

    for video_index, video in enumerate(
        videos_to_run
    ):
        crop = crop_manifest[
            video.name
        ]

        expected = (
            get_expected_expression(
                video.name
            )
        )

        actor = get_actor(
            video.name
        )

        cap = cv2.VideoCapture(
            str(video)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open "
                f"{video.name}"
            )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_index = 0

        print(
            f"Processing "
            f"{video.name}..."
        )

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if (
                max_frames is not None
                and frame_index
                >= max_frames
            ):
                break

            image = frame_to_pil(
                frame,
                crop,
            )

            global_index = (
                total_decoded
            )

            if global_index % 2 == 0:
                model_order = [
                    "emoar",
                    "vit",
                ]
            else:
                model_order = [
                    "vit",
                    "emoar",
                ]

            frame_results = {}

            for model_name in model_order:

                if model_name == "emoar":
                    result = emoar_predict(
                        emoar_model,
                        emoar_transform,
                        image,
                    )

                else:
                    result = vit_predict(
                        vit_model,
                        vit_processor,
                        vit_label_map,
                        image,
                    )

                frame_results[
                    model_name
                ] = result

            order_label = (
                f"{model_order[0]}-first"
            )

            timestamp_s = (
                frame_index / fps
                if fps > 0
                else 0.0
            )

            for model_name in [
                "emoar",
                "vit",
            ]:
                (
                    predicted,
                    confidence,
                    probabilities,
                    latency_ms,
                ) = frame_results[
                    model_name
                ]

                rows.append(
                    prediction_row(
                        video=video.name,
                        actor=actor,
                        expected=expected,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        fps=fps,
                        model_name=model_name,
                        predicted=predicted,
                        confidence=confidence,
                        probabilities=probabilities,
                        latency_ms=latency_ms,
                        order=order_label,
                    )
                )

            total_decoded += 1
            frame_index += 1

        cap.release()

        print(
            f"  decoded frames: "
            f"{frame_index}"
        )

    fieldnames = [
        "video",
        "actor",
        "expected_expression",
        "frame_index",
        "timestamp_s",
        "video_fps",
        "model",
        "predicted_class",
        "confidence",
        "latency_ms",
        "frame_model_order",
    ] + [
        f"prob_{label}"
        for label in CANONICAL_CLASSES
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "smoke_test": bool(args.smoke),
        "decoded_frames": total_decoded,
        "prediction_rows": len(rows),
        "models": {
            "emoar": {
                "checkpoint": (
                    "external/EmoAR/"
                    "web_app/classifier.pt"
                ),
            },
            "vit": {
                "model_id": VIT_MODEL_ID,
                "revision": VIT_REVISION,
                "label_map": vit_label_map,
            },
        },
        "crop_policy": {
            "type": "fixed_per_video",
            "scale": 1.7,
        },
        "timing_scope": (
            "model-specific preprocessing "
            "plus classification; excludes "
            "video decoding, fixed crop, "
            "BGR-to-RGB conversion and "
            "PIL conversion"
        ),
        "execution_order": (
            "alternated between "
            "emoar-first and vit-first "
            "by decoded frame"
        ),
        "torch_threads": (
            torch.get_num_threads()
        ),
        "python": (
            platform.python_version()
        ),
        "opencv": (
            cv2.__version__
        ),
        "torch": (
            torch.__version__
        ),
    }

    output_json.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "========== BENCHMARK COMPLETE =========="
    )

    print(
        f"Decoded frames : {total_decoded}"
    )

    print(
        f"Prediction rows: {len(rows)}"
    )

    print(
        f"CSV            : {output_csv}"
    )

    print(
        f"Metadata       : {output_json}"
    )


if __name__ == "__main__":
    main()
