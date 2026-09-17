# FER2013 Held-Out Evaluation Protocol

## Status

This protocol was frozen before evaluating the EmoAR baseline on
the FER2013 privateTest split.

## Dataset

Dataset mirror:
Aaryan333/fer2013_train_publicTest_privateTest

Frozen revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Class order:

0 - Angry
1 - Disgust
2 - Fear
3 - Happy
4 - Sad
5 - Surprise
6 - Neutral

## Dataset roles

train:
original training partition.

publicTest:
used for preprocessing investigation and validation.

privateTest:
held-out evaluation partition.

No preprocessing decision will be changed based on privateTest results.

## Primary preprocessing

web_255_crop224

Pipeline:

1. explicit RGB conversion;
2. Resize(255);
3. CenterCrop(224);
4. ToTensor();
5. ImageNet normalization.

Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]

Rationale:

This preprocessing directly reproduces the pipeline implemented in
EmoAR web_app/commons.py and is therefore the primary historical
baseline.

## Sensitivity preprocessing

training_like_resize256

Pipeline:

1. explicit RGB conversion;
2. Resize(256);
3. ToTensor();
4. ImageNet normalization.

This preprocessing reflects the validation/test transformation
identified in the EmoAR training notebook.

It will be reported only as a sensitivity analysis and will not
replace the primary historical baseline based on privateTest results.

## Validation evidence

On FER2013 publicTest:

Primary web pipeline:
- accuracy: 0.647813
- balanced accuracy: 0.620968
- macro-F1: 0.605343

Training-like sensitivity pipeline:
- accuracy: 0.654778
- balanced accuracy: 0.619238
- macro-F1: 0.624630

Paired comparison:
- web correct / training-like wrong: 154
- web wrong / training-like correct: 179
- exact two-sided McNemar p: 0.18836499

The validation comparison does not provide sufficient evidence to
replace the historically faithful web preprocessing.

## Primary held-out metrics

The following metrics are defined before opening privateTest:

- accuracy;
- balanced accuracy;
- macro-F1;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix.

Mean prediction confidence will also be recorded descriptively.

## Interpretation

The evaluation measures classification of FER2013 facial-expression
categories.

It must not be interpreted as measurement of true internal emotion,
clinical validity, or performance specifically for autistic children.

