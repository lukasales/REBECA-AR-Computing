from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import torch
import timm
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "external"
    / "emotionram_faces_demo"
    / "enet_b0_8_best_afew.pt"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "emotionram_fer2013_results.json"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "emotionram_fer2013_predictions.csv"
)

OUTPUT_DOC = (
    ROOT
    / "docs"
    / "emotionram_technical_evaluation.md"
)


DATASET = (
    "Aaryan333/"
    "fer2013_train_publicTest_privateTest"
)

REVISION = (
    "da8d1643649c86bfc2cffba5b744b51fdf329212"
)


# EmotionRAM output space.
EMOTIONRAM_CLASSES = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}


# FER2013 canonical order used in our benchmark:
#
# 0 Angry
# 1 Disgust
# 2 Fear
# 3 Happy
# 4 Sad
# 5 Surprise
# 6 Neutral
#
# EmotionRAM contains an additional Contempt class.
#
EMOTIONRAM_TO_FER = {
    0: 0,   # Anger -> Angry
    1: 7,   # Contempt -> extra / incompatible
    2: 1,   # Disgust
    3: 2,   # Fear
    4: 3,   # Happiness -> Happy
    5: 6,   # Neutral
    6: 4,   # Sadness -> Sad
    7: 5,   # Surprise
}


FER_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


print()
print("========== LOAD CHECKPOINT ==========")

print("Checkpoint:", CHECKPOINT)
print("SHA256:", sha256(CHECKPOINT))

# EmotionRAM stores the complete model object,
# rather than only a state_dict.
model = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = model.cpu().eval()

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Model type:", type(model))
print("Parameters:", parameters)
print("Classifier:", model.classifier)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


print()
print("========== DUMMY FORWARD ==========")

dummy = torch.zeros(
    1,
    3,
    224,
    224,
)

with torch.inference_mode():
    output = model(dummy)

print("Output shape:", tuple(output.shape))

if tuple(output.shape) != (1, 8):
    raise RuntimeError(
        f"Unexpected EmotionRAM output: "
        f"{tuple(output.shape)}"
    )

print("DUMMY FORWARD: OK")


print()
print("========== LOAD FER2013 ==========")

dataset = load_dataset(
    DATASET,
    revision=REVISION,
    split="privateTest",
)

print("Rows:", len(dataset))
print("Columns:", dataset.column_names)

if len(dataset) != 3589:
    raise RuntimeError(
        f"Expected 3589 images, "
        f"found {len(dataset)}"
    )


image_col = (
    "image"
    if "image" in dataset.column_names
    else dataset.column_names[0]
)

label_candidates = [
    "label",
    "emotion",
    "labels",
]

label_col = None

for candidate in label_candidates:
    if candidate in dataset.column_names:
        label_col = candidate
        break

if label_col is None:
    raise RuntimeError(
        "Could not identify label column."
    )

print("Image column:", image_col)
print("Label column:", label_col)


print()
print("========== EVALUATE ==========")

batch_size = 32

true_labels = []
pred_labels = []
pred_raw = []
confidences = []
rows = []

start = time.perf_counter()

with torch.inference_mode():

    for begin in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            begin + batch_size,
            len(dataset),
        )

        batch = dataset[
            begin:end
        ]

        images = [
            transform(
                img.convert("RGB")
            )
            for img in batch[image_col]
        ]

        x = torch.stack(images)

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        raw_preds = probabilities.argmax(
            dim=1
        ).cpu().numpy()

        max_conf = probabilities.max(
            dim=1
        ).values.cpu().numpy()

        batch_true = [
            int(v)
            for v in batch[label_col]
        ]

        for offset, (
            true_label,
            raw_pred,
            confidence,
        ) in enumerate(
            zip(
                batch_true,
                raw_preds,
                max_conf,
            )
        ):
            raw_pred = int(raw_pred)

            mapped_pred = (
                EMOTIONRAM_TO_FER[
                    raw_pred
                ]
            )

            index = begin + offset

            true_labels.append(
                true_label
            )

            pred_labels.append(
                mapped_pred
            )

            pred_raw.append(
                raw_pred
            )

            confidences.append(
                float(confidence)
            )

            rows.append({
                "index": index,
                "true_label": true_label,
                "true_class":
                    FER_NAMES[true_label],
                "emotionram_raw_label":
                    raw_pred,
                "emotionram_raw_class":
                    EMOTIONRAM_CLASSES[
                        raw_pred
                    ],
                "mapped_fer_label":
                    mapped_pred,
                "mapped_fer_class":
                    (
                        FER_NAMES[
                            mapped_pred
                        ]
                        if mapped_pred in FER_NAMES
                        else "Contempt"
                    ),
                "confidence":
                    float(confidence),
            })

        if (
            end % 512 == 0
            or end == len(dataset)
        ):
            print(
                f"{end}/{len(dataset)}"
            )


elapsed = (
    time.perf_counter()
    - start
)


accuracy = accuracy_score(
    true_labels,
    pred_labels,
)

balanced_accuracy = (
    balanced_accuracy_score(
        true_labels,
        pred_labels,
    )
)

macro_f1 = f1_score(
    true_labels,
    pred_labels,
    labels=list(range(7)),
    average="macro",
    zero_division=0,
)

contempt_count = sum(
    pred == 1
    for pred in pred_raw
)

contempt_fraction = (
    contempt_count
    / len(pred_raw)
)

mean_confidence = float(
    np.mean(confidences)
)


results = {
    "model":
        "EmotionRAM Faces / EfficientNet-B0",

    "training_dataset":
        "AfeW",

    "evaluation_dataset":
        "FER2013 privateTest",

    "evaluation_revision":
        REVISION,

    "n_images":
        len(dataset),

    "parameters":
        parameters,

    "checkpoint_sha256":
        sha256(CHECKPOINT),

    "output_classes":
        EMOTIONRAM_CLASSES,

    "accuracy":
        accuracy,

    "balanced_accuracy":
        balanced_accuracy,

    "macro_f1":
        macro_f1,

    "mean_confidence":
        mean_confidence,

    "contempt_predictions":
        contempt_count,

    "contempt_fraction":
        contempt_fraction,

    "evaluation_seconds":
        elapsed,

    "interpretation":
        (
            "Cross-dataset/off-the-shelf "
            "evaluation. EmotionRAM was "
            "trained on AfeW and contains "
            "an additional Contempt class."
        ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


doc = f"""# EmotionRAM Faces — avaliação técnica

## Status

O EmotionRAM Faces foi investigado como tecnologia adicional
relacionada à contribuição computacional.

## Modelo

- arquitetura: EfficientNet-B0;
- treinamento declarado pelo repositório: AfeW;
- entrada: RGB 224x224;
- normalização ImageNet;
- classes: Anger, Contempt, Disgust, Fear, Happiness,
  Neutral, Sadness e Surprise;
- número de parâmetros: {parameters:,};
- checkpoint SHA256: `{sha256(CHECKPOINT)}`.

## Compatibilidade com o benchmark

Sete das oito classes possuem correspondência direta com o
espaço de classes FER2013 utilizado nos experimentos anteriores.

A classe adicional `Contempt` não existe no protocolo FER2013.

Por isso, esta avaliação não deve ser apresentada como uma
comparação perfeitamente equivalente com EmoAR e ViT.

Ela é uma avaliação cross-dataset/off-the-shelf, pois o
EmotionRAM foi treinado em AfeW e avaliado aqui no FER2013.

## FER2013 privateTest

- imagens: {len(dataset)};
- accuracy: {accuracy:.4f};
- balanced accuracy: {balanced_accuracy:.4f};
- macro-F1: {macro_f1:.4f};
- mean confidence: {mean_confidence:.4f};
- predições Contempt: {contempt_count};
- fração Contempt: {contempt_fraction:.4%}.

Predições `Contempt` são contabilizadas como incorretas no
protocolo de sete classes.

## Capacidade live

O repositório original contém `webcam.py`.

O pipeline utiliza:

camera -> OpenCV -> MediaPipe face detection -> face ROI ->
EfficientNet-B0 -> classe de expressão -> overlay.

A presença dessa implementação torna o EmotionRAM uma opção
relevante para investigação futura do protótipo live/demo.

A existência do código não significa que a execução da webcam
tenha sido validada neste experimento.

## Interpretação

Os resultados quantitativos do EmotionRAM devem permanecer
separados da comparação principal EmoAR x ViT ou serem
explicitamente identificados como avaliação cross-dataset.

Sua principal relevância adicional para o projeto está na
disponibilidade de um pipeline de webcam já implementado.
"""

OUTPUT_DOC.write_text(
    doc + "\n",
    encoding="utf-8",
)


print()
print("========== RESULTS ==========")

print(
    f"Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Balanced accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro-F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Mean confidence   : "
    f"{mean_confidence:.4f}"
)

print(
    f"Contempt          : "
    f"{contempt_count}/"
    f"{len(dataset)} "
    f"({contempt_fraction:.2%})"
)

print(
    f"Parameters        : "
    f"{parameters:,}"
)

print(
    f"Elapsed           : "
    f"{elapsed:.1f} s"
)

print()
print("[OK]", OUTPUT_JSON)
print("[OK]", OUTPUT_CSV)
print("[OK]", OUTPUT_DOC)

print()
print(
    "EMOTIONRAM FER2013 EVALUATION: OK"
)
