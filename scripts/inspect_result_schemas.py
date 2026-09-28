from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "emoar_private_test.json",
    "vit_private_test.json",
    "latency_counterbalanced_summary.json",
    "confidence_calibration_analysis.json",
    "perturbation_stability_results.json",
]


def show_structure(obj, indent=0, max_depth=3):

    prefix = "  " * indent

    if indent >= max_depth:
        return

    if isinstance(obj, dict):

        for key, value in obj.items():

            print(
                f"{prefix}{key}: "
                f"{type(value).__name__}"
            )

            if isinstance(
                value,
                (dict, list),
            ):
                show_structure(
                    value,
                    indent + 1,
                    max_depth,
                )

    elif isinstance(obj, list):

        print(
            f"{prefix}[list length={len(obj)}]"
        )

        if obj:
            show_structure(
                obj[0],
                indent + 1,
                max_depth,
            )


for name in FILES:

    path = ROOT / "results" / name

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    with path.open(
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    show_structure(data)

