from pathlib import Path
import json
import sys

import torch
import transformers
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "trpakov/vit-face-expression"

OUTPUT = (
    ROOT
    / "results"
    / "modern_model_probe.json"
)


print("=" * 80)
print("Modern FER model probe")
print("=" * 80)

print()
print("Model:")
print(MODEL_ID)


# ---------------------------------------------------------------------
# Freeze exact Hugging Face revision
# ---------------------------------------------------------------------

api = HfApi()

info = api.model_info(
    MODEL_ID
)

revision = info.sha


print()
print("Repository revision:")
print(revision)

print()
print("Python:")
print(sys.version.split()[0])

print()
print("PyTorch:")
print(torch.__version__)

print()
print("Transformers:")
print(transformers.__version__)


# ---------------------------------------------------------------------
# Read metadata before model execution
# ---------------------------------------------------------------------

config = AutoConfig.from_pretrained(
    MODEL_ID,
    revision=revision,
)

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    revision=revision,
)


print()
print("=" * 80)
print("CONFIG")
print("=" * 80)

print()
print("Architecture:")
print(config.architectures)

print()
print("Number of labels:")
print(config.num_labels)

print()
print("id2label:")
print(config.id2label)

print()
print("label2id:")
print(config.label2id)


print()
print("=" * 80)
print("IMAGE PROCESSOR")
print("=" * 80)

processor_dict = processor.to_dict()

for key in sorted(processor_dict):
    value = processor_dict[key]

    if key in {
        "size",
        "crop_size",
        "image_mean",
        "image_std",
        "do_resize",
        "do_center_crop",
        "do_normalize",
        "do_rescale",
        "rescale_factor",
    }:
        print(
            f"{key}: {value}"
        )


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("MODEL LOAD")
print("=" * 80)

print()
print("Loading model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    revision=revision,
)

model.eval()

print("Model loaded successfully.")


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()
print("Total parameters:")
print(f"{total_parameters:,}")

print()
print("Trainable parameters:")
print(f"{trainable_parameters:,}")


# ---------------------------------------------------------------------
# Structural output verification
# ---------------------------------------------------------------------

height = 224
width = 224

if isinstance(processor_dict.get("size"), dict):

    size = processor_dict["size"]

    if "height" in size:
        height = size["height"]

    if "width" in size:
        width = size["width"]


dummy = torch.zeros(
    1,
    3,
    height,
    width,
)


with torch.inference_mode():

    outputs = model(
        pixel_values=dummy
    )


logits = outputs.logits


print()
print("Dummy input shape:")
print(list(dummy.shape))

print()
print("Logits shape:")
print(list(logits.shape))


# ---------------------------------------------------------------------
# Save reproducibility manifest
# ---------------------------------------------------------------------

result = {
    "model_id": MODEL_ID,
    "revision": revision,
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "architectures": config.architectures,
    "num_labels": config.num_labels,
    "id2label": {
        str(key): value
        for key, value in config.id2label.items()
    },
    "label2id": config.label2id,
    "processor": {
        key: processor_dict.get(key)
        for key in [
            "size",
            "crop_size",
            "image_mean",
            "image_std",
            "do_resize",
            "do_center_crop",
            "do_normalize",
            "do_rescale",
            "rescale_factor",
        ]
    },
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "dummy_input_shape": list(dummy.shape),
    "logits_shape": list(logits.shape),
}


OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("Saved:")
print(OUTPUT)

