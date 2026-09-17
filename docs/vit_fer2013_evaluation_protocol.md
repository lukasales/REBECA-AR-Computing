# Modern ViT FER2013 Held-Out Evaluation Protocol

## Status

This protocol was frozen before evaluating the modern ViT model on
the FER2013 privateTest split.

No model, preprocessing, class mapping, metric, or comparison rule
will be changed based on privateTest results.

## Dataset

Dataset mirror:
Aaryan333/fer2013_train_publicTest_privateTest

Frozen dataset revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Evaluation split:
privateTest

## Modern model

Model:
trpakov/vit-face-expression

Frozen model revision:
ef0bc6fc34241b6587e7e009e7711357be28c024

Architecture:
ViTForImageClassification

Parameters:
85,804,039

Number of output classes:
7

## Model-native class order

0 - angry
1 - disgust
2 - fear
3 - happy
4 - neutral
5 - sad
6 - surprise

## FER2013 evaluation order

0 - angry
1 - disgust
2 - fear
3 - happy
4 - sad
5 - surprise
6 - neutral

## Frozen class mapping

Model -> FER2013

0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 6
5 -> 4
6 -> 5

This mapping was verified before the complete publicTest benchmark.

## Preprocessing

The exact AutoImageProcessor distributed with the frozen model
revision will be used.

Observed processor configuration:

- resize: enabled
- input size: 224 x 224
- rescale: enabled
- rescale factor: 1/255
- normalization: enabled
- mean: [0.5, 0.5, 0.5]
- standard deviation: [0.5, 0.5, 0.5]

No alternative preprocessing will be selected after viewing
privateTest results.

## Metrics frozen before privateTest

Primary descriptive metrics:

- accuracy
- balanced accuracy
- macro-F1

Additional metrics:

- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- mean prediction confidence

Mean prediction confidence is descriptive and must not be interpreted
as calibration performance.

## Paired comparison with EmoAR

The modern ViT predictions will be paired with the previously frozen
EmoAR historical baseline predictions for the same FER2013 examples.

The comparison will report:

- both models correct
- both models wrong
- EmoAR correct / ViT wrong
- EmoAR wrong / ViT correct
- exact two-sided McNemar test

## publicTest evidence available before privateTest

EmoAR historical baseline:

- accuracy: 0.647813
- balanced accuracy: 0.620968
- macro-F1: 0.605343

Modern ViT:

- accuracy: 0.711897
- balanced accuracy: 0.682312
- macro-F1: 0.689131

Absolute difference in favor of ViT:

- accuracy: +0.064084
- balanced accuracy: +0.061344
- macro-F1: +0.083788

Paired publicTest comparison:

- both correct: 2056
- both wrong: 765
- EmoAR correct / ViT wrong: 269
- EmoAR wrong / ViT correct: 499
- discordant cases: 768
- exact two-sided McNemar p: 8.3848233573e-17

## Interpretation constraint

The models classify FER2013 facial-expression categories.

The results must not be interpreted as direct evidence of:

- recognition of true internal emotional states
- clinical validity
- effectiveness with autistic children
- performance specifically on child faces
- effectiveness of an Augmented Reality intervention

## Important provenance limitation

The modern ViT was developed using FER2013.

Therefore, privateTest is held out from decisions made in this
reproduction study, but it cannot be assumed to constitute an
independent external dataset that was unseen during the original
development of the pretrained model.

The privateTest experiment will therefore be described as a
reproducible comparative benchmark rather than independent external
validation.
