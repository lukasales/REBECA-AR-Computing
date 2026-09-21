# Predictive Confidence and Calibration Results

## Scope

This analysis evaluates the predictive confidence associated with
the frozen FER2013 privateTest predictions of:

1. the historical EmoAR DenseNet121 classifier;
2. the modern ViT classifier.

No model inference was repeated.

Predictive confidence corresponds to the maximum softmax probability
of the top-1 facial-expression category.

It must not be interpreted as certainty about a person's true
emotional state.

## Dataset

Dataset:
Aaryan333/fer2013_train_publicTest_privateTest

Revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Split:
privateTest

Samples:
3,589

## EmoAR results

Accuracy:

- 0.665088

Mean predictive confidence:

- 0.748708

Overall signed confidence gap:

- +0.083621

Top-label ECE with 15 equal-width bins:

- 0.083886

MCE:

- 0.147681

Mean confidence:

- correct predictions: 0.820119
- incorrect predictions: 0.606897

### High-confidence errors

Among 1,202 incorrect predictions:

- confidence >= 0.80: 211 (17.55%)
- confidence >= 0.90: 97 (8.07%)
- confidence >= 0.95: 49 (4.08%)
- confidence >= 0.99: 15 (1.25%)

## Modern ViT results

Accuracy:

- 0.717470

Mean predictive confidence:

- 0.901042

Overall signed confidence gap:

- +0.183572

Top-label ECE with 15 equal-width bins:

- 0.183572

MCE:

- 0.375048

Mean confidence:

- correct predictions: 0.938420
- incorrect predictions: 0.806121

### High-confidence errors

Among 1,014 incorrect predictions:

- confidence >= 0.80: 619 (61.05%)
- confidence >= 0.90: 463 (45.66%)
- confidence >= 0.95: 323 (31.85%)
- confidence >= 0.99: 32 (3.16%)

## Comparative calibration result

Although the ViT achieved higher classification accuracy, its raw
softmax confidence was substantially more overconfident on the
FER2013 privateTest split.

Compared with EmoAR:

- ECE increased by 0.099685;
- MCE increased by 0.227367;
- the overall confidence-minus-accuracy gap increased by 0.099951.

Therefore, higher classification performance did not correspond to
better calibrated predictive confidence.

## Selective prediction

Confidence thresholds produced different accuracy-coverage
trade-offs between the models.

At confidence >= 0.90:

EmoAR:

- coverage: 0.358874
- retained accuracy: 0.924689
- retained errors: 97

ViT:

- coverage: 0.732516
- retained accuracy: 0.823887
- retained errors: 463

At confidence >= 0.95:

EmoAR:

- coverage: 0.273335
- retained accuracy: 0.950051
- retained errors: 49

ViT:

- coverage: 0.637225
- retained accuracy: 0.858767
- retained errors: 323

At confidence >= 0.99:

EmoAR:

- coverage: 0.138757
- retained accuracy: 0.969880
- retained errors: 15

ViT:

- coverage: 0.251602
- retained accuracy: 0.964563
- retained errors: 32

## Interpretation for REBECA-AR

The modern ViT presents a relevant trade-off:

- higher FER2013 classification performance;
- substantially higher computational cost;
- substantially stronger raw predictive confidence;
- but worse top-label calibration.

Consequently, raw softmax confidence should not be treated as a
directly reliable control signal for a future adaptive AR experience.

For example, a prediction with confidence above 0.90 cannot be
assumed to be highly reliable solely because of the numerical
softmax value.

A future REBECA-AR implementation could investigate calibrated
confidence, abstention, temporal aggregation or other uncertainty-
aware interaction strategies.

## Limitations

These results characterize calibration only on FER2013.

They do not establish calibration for:

- autistic children;
- child faces;
- live camera input;
- real-world AR conditions;
- clinical interpretation.

No deployment threshold is recommended from these results.

ECE is also dependent on the chosen binning scheme. The reported
value uses the pre-specified 15 equal-width bins and should be
interpreted together with the bin-level results, high-confidence
error counts and selective-prediction analysis.
