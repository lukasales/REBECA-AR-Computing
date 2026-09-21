# Predictive Confidence and Calibration Analysis Protocol

## Status

This protocol defines the confidence-analysis procedure before
calculating calibration metrics.

The analysis uses only predictions already produced for the frozen
FER2013 privateTest evaluation. No model inference will be repeated.

## Dataset

Dataset:
Aaryan333/fer2013_train_publicTest_privateTest

Revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Split:
privateTest

Number of samples:
3,589

## Models

### EmoAR

Primary historical pipeline:

web_255_crop224

Architecture:

DenseNet121 + Linear(1024, 7)

### Modern ViT

Model:

trpakov/vit-face-expression

Revision:

ef0bc6fc34241b6587e7e009e7711357be28c024

## Definition of confidence

Predictive confidence is defined as the maximum softmax probability
associated with the top-1 predicted facial-expression category.

This quantity must not be interpreted as:

- certainty about a person's true emotional state;
- epistemic uncertainty;
- clinical confidence;
- probability that the inferred emotion is psychologically true.

It is only the model's top-1 predictive confidence under its
softmax output.

## Input integrity

Before this protocol, input auditing confirmed:

- 3,589 prediction rows per model;
- 3,589 unique sample indices per model;
- identical sample-index sets;
- zero true-label mismatches between model files;
- all recorded confidence values between 0 and 1.

The audit also descriptively inspected confidence in correct and
incorrect predictions.

## Calibration metric

Primary calibration metric:

Top-label Expected Calibration Error (ECE)

The confidence interval [0, 1] will be divided into 15 fixed,
equal-width bins.

For each non-empty bin:

- bin accuracy will be calculated;
- mean predictive confidence will be calculated;
- calibration gap will be calculated as:
  absolute(mean confidence - accuracy).

ECE will be the sample-weighted mean of these absolute bin gaps.

## Additional calibration summaries

The analysis will also report:

- Maximum Calibration Error (MCE);
- overall accuracy;
- overall mean confidence;
- signed overall confidence gap:
  mean confidence - accuracy;
- confidence mean for correct predictions;
- confidence mean for incorrect predictions.

A positive signed confidence gap indicates aggregate overconfidence,
but does not alone establish perfect or imperfect calibration across
the complete confidence range.

## High-confidence errors

Incorrect predictions will be counted at the following confidence
thresholds:

- >= 0.80
- >= 0.90
- >= 0.95
- >= 0.99

For each threshold, both count and proportion of all model errors
will be reported.

## Selective prediction analysis

The following fixed confidence thresholds will be evaluated:

- 0.00
- 0.50
- 0.60
- 0.70
- 0.80
- 0.90
- 0.95
- 0.99

For each threshold, the analysis will report:

- number of retained predictions;
- coverage;
- accuracy among retained predictions;
- number of retained errors.

These thresholds are descriptive and are not optimized using
privateTest results.

## Interpretation constraint

Higher confidence must not automatically be interpreted as better
model reliability.

A model may achieve higher classification accuracy while also making
high-confidence incorrect predictions.

Likewise, a confidence threshold may increase accuracy among retained
predictions while reducing coverage.

## Applicability to REBECA-AR

This analysis is intended to assess whether raw classifier confidence
could safely support downstream interaction logic in a future
REBECA-AR prototype.

No confidence threshold will be recommended for deployment solely
from FER2013.

Performance on FER2013 does not establish calibration for:

- autistic children;
- child faces;
- real-world camera streams;
- AR usage conditions;
- clinical decision-making.

Any future deployment-oriented threshold would require validation in
the intended population and context.
