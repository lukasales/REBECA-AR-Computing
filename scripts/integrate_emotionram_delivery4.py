from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

RESULT = ROOT / "results" / "emotionram_fer2013_results.json"
HANDOFF = ROOT / "docs" / "implementation_handoff.md"
COMP = ROOT / "docs" / "computational_results_for_paper.md"
TABLE = ROOT / "results" / "model_comparison_delivery4.md"

r = json.loads(
    RESULT.read_text(encoding="utf-8")
)

accuracy = r["accuracy"]
balanced = r["balanced_accuracy"]
macro = r["macro_f1"]
params = r["parameters"]
contempt_n = r["contempt_predictions"]
contempt_frac = r["contempt_fraction"]


table = f"""# Comparative computational table — Delivery 4

| Aspect | EmoAR | ViT | EmotionRAM-faces |
|---|---|---|---|
| Architecture | DenseNet121 | Vision Transformer | EfficientNet-B0 |
| Declared training dataset | FER2013-based baseline | FER2013 | AfeW |
| Output classes | 7 | 7 | 8 |
| FER2013 privateTest Accuracy | 66.51% | 71.75% | {accuracy*100:.2f}% |
| FER2013 Balanced Accuracy | 63.82% | 71.01% | {balanced*100:.2f}% |
| FER2013 Macro-F1 | 62.45% | 71.07% | {macro*100:.2f}% |
| Parameters | 6.96 M | 85.80 M | {params/1e6:.2f} M |
| Live webcam implementation | Not part of our evaluated pipeline | Not part of our evaluated pipeline | Yes |
| Live face detection | Not part of our evaluated pipeline | Not part of our evaluated pipeline | MediaPipe |
| Main role in project | Baseline | Modern comparison | Additional cross-dataset model + live reference |

## Methodological note

The EmotionRAM-faces result is not perfectly equivalent to the
EmoAR and ViT comparison.

EmotionRAM was trained on AfeW and exposes eight classes:

Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness and
Surprise.

FER2013 contains seven categories and does not contain Contempt.

Seven EmotionRAM categories were mapped directly to the
corresponding FER2013 categories. Predictions of Contempt were
counted as incorrect.

EmotionRAM predicted Contempt for {contempt_n} of 3,589 images
({contempt_frac*100:.2f}%).

Therefore, its FER2013 result should be described as an
off-the-shelf cross-dataset evaluation rather than a perfectly
controlled head-to-head comparison.

## Interpretation

- ViT achieved the highest FER2013 classification performance.
- EmoAR obtained intermediate classification performance with much
  lower computational cost than ViT.
- EmotionRAM was the smallest model by parameter count.
- EmotionRAM obtained lower FER2013 performance, but it was trained
  on a different dataset and has an additional output category.
- EmotionRAM provides an existing webcam pipeline with MediaPipe,
  making it particularly relevant as a technical reference for a
  future live demonstration.
"""

TABLE.write_text(
    table + "\n",
    encoding="utf-8"
)


emotionram_section = f"""
## EmotionRAM-faces

Após a solicitação da equipe, o EmotionRAM-faces também foi
investigado.

Características identificadas:

- arquitetura EfficientNet-B0;
- treinamento declarado no AfeW;
- entrada RGB 224x224;
- normalização ImageNet;
- 8 classes, incluindo Contempt;
- {params:,} parâmetros;
- checkpoint original auditado;
- pipeline de webcam disponível;
- detecção facial live com MediaPipe.

Foi realizada uma avaliação off-the-shelf no mesmo FER2013
privateTest de 3.589 imagens.

Resultados:

| Métrica | EmotionRAM |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Balanced accuracy | {balanced:.4f} |
| Macro-F1 | {macro:.4f} |
| Mean confidence | {r["mean_confidence"]:.4f} |
| Parameters | {params:,} |
| Contempt predictions | {contempt_n} ({contempt_frac*100:.2f}%) |

Sete das oito classes apresentam correspondência direta com o
protocolo FER2013. A classe adicional Contempt foi contabilizada
como previsão incorreta.

Essa avaliação deve ser descrita como cross-dataset/off-the-shelf,
pois o EmotionRAM foi treinado no AfeW. Portanto, o valor de
accuracy não deve ser interpretado como uma comparação
perfeitamente equivalente com EmoAR e ViT.

O EmotionRAM também é particularmente relevante para o protótipo
futuro porque seu repositório inclui uma implementação de webcam
com OpenCV, MediaPipe, inferência EfficientNet-B0 e overlay da
categoria prevista.
""".strip()


for path in [HANDOFF, COMP]:

    text = path.read_text(
        encoding="utf-8"
    )

    if "## EmotionRAM-faces" not in text:
        text = (
            text.rstrip()
            + "\n\n"
            + emotionram_section
            + "\n"
        )

        path.write_text(
            text,
            encoding="utf-8"
        )


print("========== DELIVERY 4 COMPARISON ==========")
print()
print(table)
print()
print("[OK]", TABLE)
print("[OK]", HANDOFF)
print("[OK]", COMP)
