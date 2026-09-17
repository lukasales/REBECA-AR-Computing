from datasets import load_dataset, get_dataset_split_names

DATASET_ID = "Aaryan333/fer2013_train_publicTest_privateTest"

print("=" * 80)
print("FER2013 dataset probe")
print("=" * 80)
print()

print("Dataset:")
print(DATASET_ID)
print()

splits = get_dataset_split_names(DATASET_ID)

print("Available splits:")
for split in splits:
    print(" -", split)

print()

# Prefer the original public test partition for the first inspection.
preferred = None

for candidate in ["publicTest", "publictest", "test", "validation"]:
    if candidate in splits:
        preferred = candidate
        break

if preferred is None:
    raise RuntimeError(
        f"Could not identify a public-test-like split. Found: {splits}"
    )

print("Selected split:")
print(preferred)
print()

print("Opening dataset in streaming mode...")

dataset = load_dataset(
    DATASET_ID,
    split=preferred,
    streaming=True,
)

print("Features:")
print(dataset.features)
print()

sample = next(iter(dataset))

print("First sample keys:")
print(list(sample.keys()))
print()

for key, value in sample.items():
    print("-" * 80)
    print("FIELD:", key)
    print("TYPE:", type(value))

    if hasattr(value, "size"):
        print("SIZE:", value.size)
        print("MODE:", getattr(value, "mode", None))
    else:
        text = repr(value)
        if len(text) > 500:
            text = text[:500] + "... [truncated]"
        print("VALUE:", text)

print()
print("Probe finished.")
