# Comparative computational table — Delivery 4

| Aspect | EmoAR | ViT | EmotionRAM-faces |
|---|---|---|---|
| Architecture | DenseNet121 | Vision Transformer | EfficientNet-B0 |
| Declared training dataset | FER2013-based baseline | FER2013 | AfeW |
| Output classes | 7 | 7 | 8 |
| FER2013 privateTest Accuracy | 66.51% | 71.75% | 55.70% |
| FER2013 Balanced Accuracy | 63.82% | 71.01% | 52.20% |
| FER2013 Macro-F1 | 62.45% | 71.07% | 50.92% |
| Parameters | 6.96 M | 85.80 M | 4.02 M |
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

EmotionRAM predicted Contempt for 52 of 3,589 images
(1.45%).

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

