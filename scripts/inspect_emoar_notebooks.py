import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMOAR = ROOT / "external" / "EmoAR"

keywords = [
    "densenet",
    "resnet",
    "vgg",
    "classifier",
    "fer2013",
    "transform",
    "grayscale",
    "rgb",
    "imagefolder",
    "state_dict",
    "torch.save",
    "torch.load",
    "batch_size",
    "optimizer",
    "accuracy",
    "normalize",
    "linear(1024",
]

def cell_text(cell):
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)

results = []

for notebook in sorted(EMOAR.rglob("*.ipynb")):
    try:
        data = json.loads(notebook.read_text(encoding="utf-8"))
    except Exception as exc:
        results.append(f"\n### ERROR: {notebook}\n{exc}\n")
        continue

    results.append(f"\n{'=' * 80}\nNOTEBOOK: {notebook.relative_to(EMOAR)}\n{'=' * 80}\n")

    found = False

    for index, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        text = cell_text(cell)
        lowered = text.lower()

        matched = [keyword for keyword in keywords if keyword in lowered]

        if matched:
            found = True
            results.append(
                f"\n--- CODE CELL {index} | MATCHES: {', '.join(matched)} ---\n"
            )
            results.append(text)
            if not text.endswith("\n"):
                results.append("\n")

    if not found:
        results.append("\nNo matching code cells found.\n")

output = ROOT / "results" / "emoar_notebook_scan.txt"
output.write_text("".join(results), encoding="utf-8")

print(f"Saved: {output}")
print(f"Scanned notebooks: {len(list(EMOAR.rglob('*.ipynb')))}")
