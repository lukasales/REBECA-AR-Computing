# Frame-Level Perturbation Stability Protocol

## Status

This protocol is frozen before performing the perturbation-stability
experiment.

No perturbation, sampling rule, model configuration or primary metric
will be changed after observing the results.

## Motivation

A future REBECA-AR experience would operate on camera frames rather
than isolated FER2013 images.

Small frame-to-frame changes in pose, illumination and image position
should ideally not cause arbitrary changes in the predicted facial-
expression category.

FER2013 does not contain temporal video sequences. Therefore, this
experiment does not measure true temporal stability.

Instead, it evaluates deterministic robustness to small image
perturbations as a proxy for frame-level prediction stability.

## Dataset

Dataset:
Aaryan333/fer2013_train_publicTest_privateTest

Revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Split:
privateTest

## Sampling

A balanced subset will be used to keep the experiment computationally
feasible on CPU.

Sampling rule:

- 50 images per FER2013 class;
- 7 classes;
- 350 original images total;
- deterministic pseudo-random sampling;
- seed: 20260921.

All classes have sufficient support for this sampling size.

The same original images and the same perturbed variants will be used
for both models.

## Models

### EmoAR

Architecture:
DenseNet121 + Linear(1024, 7)

Checkpoint:
historical EmoAR web classifier

Preprocessing:
web_255_crop224

### Modern ViT

Model:
trpakov/vit-face-expression

Revision:
ef0bc6fc34241b6587e7e009e7711357be28c024

The previously validated model-to-FER2013 class mapping will be used.

## Perturbations

Each selected image will be evaluated in its original form and under
eight deterministic small perturbations.

Conditions:

1. original
2. rotation -5 degrees
3. rotation +5 degrees
4. horizontal translation -2 pixels
5. horizontal translation +2 pixels
6. brightness factor 0.90
7. brightness factor 1.10
8. contrast factor 0.90
9. contrast factor 1.10

No random perturbation will be introduced during inference.

The perturbations are intentionally small and are not intended to
simulate every possible real-world AR condition.

## Primary stability metric

Pairwise label stability will be defined as:

the proportion of perturbed predictions whose predicted FER2013
category is identical to the prediction produced for the original
image.

The original image itself is not counted as a perturbation pair.

There will therefore be:

350 original images x 8 perturbations = 2,800 comparison pairs per
model.

## Additional stability metrics

The analysis will also report:

- stability separately for each perturbation;
- percentage of original images with at least one prediction flip;
- percentage of original images stable across all eight perturbations;
- mean number of unique predicted categories per original image;
- mean absolute change in top-1 confidence relative to the original;
- maximum absolute confidence change;
- classification accuracy for the original images;
- classification accuracy under each perturbation;
- aggregate accuracy across all perturbed images.

## Correctness versus consistency

Prediction stability and prediction correctness are different
properties.

A model can be:

- consistently correct;
- consistently wrong;
- unstable but sometimes correct;
- unstable and wrong.

Therefore, label stability will not be interpreted as accuracy.

## Comparative interpretation

The experiment will compare EmoAR and ViT on exactly the same image
variants.

A model with higher FER2013 accuracy may still exhibit greater
prediction instability under small input changes.

Conversely, a highly stable model may simply preserve an incorrect
classification.

Both properties will therefore be reported separately.

## REBECA-AR interpretation

Prediction instability under small image perturbations could be
relevant to a future AR interface because rapid category changes
could produce unstable or confusing visual feedback.

However, this experiment does not establish performance in a live AR
camera stream.

It does not include:

- actual temporal video;
- face detection;
- tracking;
- motion blur;
- occlusion;
- large head-pose changes;
- changing distance to the camera;
- real-world illumination dynamics;
- autistic children or child-specific data.

The results should be described as controlled perturbation robustness,
not true temporal validation.

## No deployment claim

No smoothing rule, temporal voting scheme or deployment threshold will
be selected using this experiment alone.

Any future live-camera REBECA-AR implementation would require separate
validation under realistic usage conditions.
