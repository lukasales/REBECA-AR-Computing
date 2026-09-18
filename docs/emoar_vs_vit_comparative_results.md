# EmoAR vs Modern ViT: FER2013 Comparative Results

## Experimental scope

This experiment compares:

1. the historical EmoAR DenseNet121 facial-expression classifier;
2. a modern pretrained Vision Transformer:
   trpakov/vit-face-expression.

Both models were evaluated using the same frozen FER2013 dataset
revision:

Aaryan333/fer2013_train_publicTest_privateTest

Revision:

da8d1643649c86bfc2cffba5b744b51fdf329212

The primary held-out comparison uses the privateTest split with
3,589 images.

## Models

### EmoAR

Architecture:
DenseNet121 + Linear(1024, 7)

Primary preprocessing:
web_255_crop224

### Modern model

Architecture:
ViTForImageClassification

Model:
trpakov/vit-face-expression

Revision:
ef0bc6fc34241b6587e7e009e7711357be28c024

Parameters:
85,804,039

The model-native class order was explicitly mapped to the FER2013
class order before evaluation.

## publicTest results

### EmoAR

- accuracy: 0.647813
- balanced accuracy: 0.620968
- macro-F1: 0.605343

### ViT

- accuracy: 0.711897
- balanced accuracy: 0.682312
- macro-F1: 0.689131

### Absolute differences

- accuracy: +0.064084
- balanced accuracy: +0.061344
- macro-F1: +0.083788

### Paired comparison

- both correct: 2056
- both wrong: 765
- EmoAR correct / ViT wrong: 269
- EmoAR wrong / ViT correct: 499
- discordant cases: 768
- exact two-sided McNemar p: 8.3848233573e-17

## privateTest results

### EmoAR

- correct: 2387 / 3589
- accuracy: 0.665088
- balanced accuracy: 0.638232
- macro-F1: 0.624476

### ViT

- correct: 2575 / 3589
- accuracy: 0.717470
- balanced accuracy: 0.710095
- macro-F1: 0.710671

### Absolute differences

- correct predictions: +188
- accuracy: +0.052382
- balanced accuracy: +0.071863
- macro-F1: +0.086195

## privateTest paired comparison

- both correct: 2088
- both wrong: 715
- EmoAR correct / ViT wrong: 299
- EmoAR wrong / ViT correct: 487
- discordant cases: 786
- exact two-sided McNemar p: 2.06332616174e-11

Among discordant cases, the ViT was correct in approximately 62%
of cases.

The paired comparison therefore shows a clear asymmetry in favor of
the ViT on this FER2013 partition.

## Interpretation

The modern ViT outperformed the reproduced EmoAR DenseNet121
baseline on accuracy, balanced accuracy and macro-F1 on both
FER2013 publicTest and privateTest.

The advantage is also supported by paired per-image comparisons:
the ViT corrected substantially more EmoAR errors than the reverse.

This result supports the conclusion that a more recent pretrained
facial-expression classifier can provide stronger FER2013
classification performance than the historical model used in EmoAR.

## Limitations

The comparison concerns classification of FER2013 facial-expression
categories.

It does not establish:

- recognition of true emotional states;
- clinical validity;
- performance with autistic children;
- performance specifically with child faces;
- effectiveness in an Augmented Reality intervention.

The ViT itself was developed using FER2013. Therefore, the
privateTest comparison is held out from decisions made during this
reproduction study, but it must not be described as independent
external validation of the pretrained model.

## Implication for REBECA-AR

The results suggest that replacing the historical EmoAR classifier
with a more recent pretrained model could improve facial-expression
category classification.

However, classification performance alone is insufficient for an AR
application.

Computational cost, latency, temporal stability and uncertainty must
also be evaluated before determining practical applicability to a
future REBECA-AR system.
