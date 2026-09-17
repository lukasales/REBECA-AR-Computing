from pathlib import Path
import re
import sys

import torch
import torchvision
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "external" / "EmoAR" / "web_app" / "classifier.pt"


print("=" * 80)
print("EmoAR checkpoint structural inspection")
print("=" * 80)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Checkpoint size: {CHECKPOINT.stat().st_size:,} bytes")
print()


# IMPORTANT:
# weights_only=True prevents general arbitrary Python object
# deserialization. Do not change this to False for this audit.
try:
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
except Exception as exc:
    print("SAFE LOAD FAILED")
    print(type(exc).__name__)
    print(exc)
    print()
    print("Do NOT retry with weights_only=False.")
    sys.exit(2)


print("Safe load succeeded.")
print("Loaded object type:", type(checkpoint))


if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("Nested state_dict detected.")
else:
    state_dict = checkpoint
    print("Checkpoint treated directly as state_dict.")


if not isinstance(state_dict, dict):
    print("ERROR: loaded object is not a state dictionary.")
    sys.exit(3)


print(f"State-dict entries: {len(state_dict)}")

tensor_entries = {
    key: value
    for key, value in state_dict.items()
    if isinstance(value, torch.Tensor)
}

print(f"Tensor entries: {len(tensor_entries)}")

total_values = sum(t.numel() for t in tensor_entries.values())

print(f"Stored tensor values: {total_values:,}")
print()


print("=" * 80)
print("FIRST 25 CHECKPOINT KEYS")
print("=" * 80)

for key in list(state_dict.keys())[:25]:
    value = state_dict[key]
    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
    print(f"{key:70s} {shape}")

print()


print("=" * 80)
print("CLASSIFIER-LIKE KEYS")
print("=" * 80)

for key, value in state_dict.items():
    lowered = key.lower()

    if (
        "classifier" in lowered
        or key.endswith(".fc.weight")
        or key.endswith(".fc.bias")
    ):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        print(f"{key:70s} {shape}")

print()


def normalize_legacy_densenet_keys(original):
    """
    Old torchvision DenseNet checkpoints may contain names such as:

        denselayer1.norm.1.weight

    while modern torchvision uses:

        denselayer1.norm1.weight

    Normalize only this known historical naming convention.
    """

    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\."
        r"(weight|bias|running_mean|running_var)$"
    )

    normalized = {}

    for key, value in original.items():
        match = pattern.match(key)

        if match:
            new_key = (
                match.group(1)
                + match.group(2)
                + "."
                + match.group(3)
            )
        else:
            new_key = key

        normalized[new_key] = value

    return normalized


normalized_state = normalize_legacy_densenet_keys(state_dict)


def compare_model(name, model, checkpoint_state):
    print()
    print("=" * 80)
    print(f"CANDIDATE: {name}")
    print("=" * 80)

    model_state = model.state_dict()

    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    common = checkpoint_keys & model_keys
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    shape_mismatches = []

    for key in sorted(common):
        checkpoint_value = checkpoint_state[key]
        model_value = model_state[key]

        if (
            isinstance(checkpoint_value, torch.Tensor)
            and tuple(checkpoint_value.shape) != tuple(model_value.shape)
        ):
            shape_mismatches.append(
                (
                    key,
                    tuple(checkpoint_value.shape),
                    tuple(model_value.shape),
                )
            )

    print(f"Model state keys:      {len(model_keys)}")
    print(f"Checkpoint keys:       {len(checkpoint_keys)}")
    print(f"Matching key names:    {len(common)}")
    print(f"Missing model keys:    {len(missing)}")
    print(f"Unexpected ckpt keys:  {len(unexpected)}")
    print(f"Shape mismatches:      {len(shape_mismatches)}")

    ckpt_overlap = (
        len(common) / len(checkpoint_keys) * 100
        if checkpoint_keys else 0
    )

    model_overlap = (
        len(common) / len(model_keys) * 100
        if model_keys else 0
    )

    print(f"Checkpoint-key overlap: {ckpt_overlap:.2f}%")
    print(f"Model-key overlap:      {model_overlap:.2f}%")

    if shape_mismatches:
        print()
        print("First shape mismatches:")

        for item in shape_mismatches[:10]:
            print(item)

    if unexpected:
        print()
        print("First unexpected checkpoint keys:")

        for key in sorted(unexpected)[:15]:
            print(key)

    if missing:
        print()
        print("First missing model keys:")

        for key in sorted(missing)[:15]:
            print(key)

    loadable_state = {
        key: value
        for key, value in checkpoint_state.items()
        if (
            key in model_state
            and isinstance(value, torch.Tensor)
            and tuple(value.shape) == tuple(model_state[key].shape)
        )
    }

    try:
        result = model.load_state_dict(
            loadable_state,
            strict=False,
        )

        print()
        print("Compatible tensors could be loaded.")
        print("Missing after compatible load:", len(result.missing_keys))
        print("Unexpected after compatible load:", len(result.unexpected_keys))

    except Exception as exc:
        print()
        print("Model load failed:")
        print(type(exc).__name__, exc)


# Candidate 1: DenseNet121 declared by web_app/commons.py

dense = models.densenet121(weights=None)
dense.classifier = nn.Linear(1024, 7)

compare_model(
    "DenseNet121 -> Linear(1024, 7)",
    dense,
    normalized_state,
)


# Candidate 2: MobileNetV2 used in conversion notebooks

mobile = models.mobilenet_v2(weights=None)
mobile.classifier[1] = nn.Linear(
    mobile.last_channel,
    7,
)

compare_model(
    "MobileNetV2 -> Linear(1280, 7)",
    mobile,
    state_dict,
)


print()
print("=" * 80)
print("ARCHITECTURE HINTS")
print("=" * 80)

keys_text = "\n".join(state_dict.keys()).lower()

print(
    "Contains DenseNet-style 'denseblock':",
    "denseblock" in keys_text,
)

print(
    "Contains DenseNet-style 'transition':",
    "transition" in keys_text,
)

print(
    "Contains MobileNet-style classifier.1:",
    "classifier.1" in keys_text,
)

print()
print("Inspection finished.")
