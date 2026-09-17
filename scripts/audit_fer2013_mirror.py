from collections import Counter
from pathlib import Path
import json

import numpy as np
from datasets import load_dataset, get_dataset_split_names
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
SPLIT = "publicTest"

OUTPUT = ROOT / "results" / "fer2013_mirror_audit.json"

EXPECTED_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


print("=" * 80)
print("FER2013 mirror audit")
print("=" * 80)
print()

api = HfApi()
info = api.dataset_info(DATASET_ID)

revision = info.sha

print("Dataset:")
print(DATASET_ID)

print()
print("Repository revision:")
print(revision)

print()

splits = get_dataset_split_names(
    DATASET_ID,
    revision=revision,
)

print("Splits:")
for split in splits:
    print(" -", split)

if SPLIT not in splits:
    raise RuntimeError(
        f"Expected split {SPLIT!r} not found."
    )

print()
print("Opening:")
print(SPLIT)

dataset = load_dataset(
    DATASET_ID,
    split=SPLIT,
    streaming=True,
    revision=revision,
)

label_feature = dataset.features["label"]

class_names = list(label_feature.names)

print()
print("Class names:")
for index, name in enumerate(class_names):
    print(index, name)

if class_names != EXPECTED_CLASSES:
    raise RuntimeError(
        "Class ordering differs from expected FER2013 order."
    )


label_counts = Counter()
image_modes = Counter()
image_sizes = Counter()

rgb_identical_channels = 0
rgb_nonidentical_channels = 0
non_rgb_images = 0

first_sample_by_class = {}

total = 0


print()
print("Auditing complete publicTest split...")

for sample_index, sample in enumerate(dataset):

    label = int(sample["label"])
    image = sample["image"]

    label_counts[label] += 1

    image_modes[image.mode] += 1
    image_sizes[str(image.size)] += 1

    if label not in first_sample_by_class:
        first_sample_by_class[label] = {
            "sample_index": sample_index,
            "class_name": class_names[label],
            "mode": image.mode,
            "size": list(image.size),
        }

    array = np.asarray(image)

    if image.mode == "RGB":

        r = array[..., 0]
        g = array[..., 1]
        b = array[..., 2]

        if (
            np.array_equal(r, g)
            and np.array_equal(r, b)
        ):
            rgb_identical_channels += 1
        else:
            rgb_nonidentical_channels += 1

    else:
        non_rgb_images += 1

    total += 1


audit = {
    "dataset_id": DATASET_ID,
    "revision": revision,
    "split": SPLIT,
    "total_samples": total,
    "classes": class_names,
    "label_counts": {
        class_names[index]: label_counts[index]
        for index in range(len(class_names))
    },
    "image_modes": dict(image_modes),
    "image_sizes": dict(image_sizes),
    "rgb_identical_channels": rgb_identical_channels,
    "rgb_nonidentical_channels": rgb_nonidentical_channels,
    "non_rgb_images": non_rgb_images,
    "first_sample_by_class": {
        str(index): value
        for index, value in first_sample_by_class.items()
    },
}


OUTPUT.write_text(
    json.dumps(
        audit,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print()
print("Total samples:")
print(total)

print()
print("Label counts:")
for index, name in enumerate(class_names):
    print(
        f"{index} {name:10s}: {label_counts[index]}"
    )

print()
print("Image modes:")
for mode, count in image_modes.items():
    print(mode, count)

print()
print("Image sizes:")
for size, count in image_sizes.items():
    print(size, count)

print()
print("RGB images with R=G=B:")
print(rgb_identical_channels)

print()
print("RGB images with different channels:")
print(rgb_nonidentical_channels)

print()
print("Non-RGB images:")
print(non_rgb_images)

print()
print("First sample for each class:")
for index in sorted(first_sample_by_class):
    print(
        index,
        class_names[index],
        first_sample_by_class[index]
    )

print()
print("Saved:")
print(OUTPUT)
