import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.emoar_model import load_emoar_model, predict


CHECKPOINT = (
    ROOT
    / "external"
    / "EmoAR"
    / "web_app"
    / "classifier.pt"
)

RESULT_FILE = (
    ROOT
    / "results"
    / "emoar_smoke_test.json"
)


def create_deterministic_test_image():
    width = 300
    height = 300

    x = np.linspace(
        0,
        255,
        width,
        dtype=np.uint8,
    )

    y = np.linspace(
        0,
        255,
        height,
        dtype=np.uint8,
    )

    xx, yy = np.meshgrid(x, y)

    image = np.stack(
        [
            xx,
            yy,
            ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8),
        ],
        axis=-1,
    )

    return Image.fromarray(image, mode="RGB")


def main():
    print("=" * 80)
    print("EmoAR modern reproduction - smoke test")
    print("=" * 80)

    print("Loading checkpoint with strict=True...")

    model = load_emoar_model(CHECKPOINT)

    print("Model loaded successfully.")

    image = create_deterministic_test_image()

    result = predict(
        model,
        image,
    )

    RESULT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Input shape:")
    print(result["input_shape"])

    print()
    print("Logits shape:")
    print(result["logits_shape"])

    print()
    print("Predicted class:")
    print(
        result["class_index"],
        result["class_name"],
    )

    print()
    print("Confidence:")
    print(
        f'{result["confidence"]:.6f}'
    )

    print()
    print("Probabilities:")

    total_probability = 0.0

    for class_name, probability in result["probabilities"].items():
        total_probability += probability
        print(
            f"{class_name:10s}: {probability:.6f}"
        )

    print()
    print(
        f"Probability sum: {total_probability:.6f}"
    )

    print()
    print(f"Saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
