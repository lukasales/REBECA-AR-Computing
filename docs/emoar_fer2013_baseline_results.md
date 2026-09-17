# EmoAR FER2013 Baseline Results

## Experimental status

The EmoAR web baseline was evaluated on the held-out FER2013
privateTest split after the evaluation protocol had been frozen.

Dataset:
Aaryan333/fer2013_train_publicTest_privateTest

Dataset revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Split:
privateTest

Samples:
3589

Model:
DenseNet121 with Linear(1024, 7)

Checkpoint:
EmoAR web_app/classifier.pt

Checkpoint SHA-256:
01DC84D15D836DC8C6552277D7AC0C376086BAA00F1490FD5ACB63CFEF72E5C9


## Primary historical baseline

Preprocessing:
web_255_crop224

Pipeline:

1. RGB conversion
2. Resize(255)
3. CenterCrop(224)
4. ToTensor
5. ImageNet normalization

Results:

- correct: 2387 / 3589
- accuracy: 0.665088
- balanced accuracy: 0.638232
- macro-F1: 0.624476
- mean prediction confidence: 0.748708


### Per-class performance

Angry:
- precision: 0.5593
- recall: 0.6151
- F1: 0.5858
- support: 491

Disgust:
- precision: 0.4638
- recall: 0.5818
- F1: 0.5161
- support: 55

Fear:
- precision: 0.5673
- recall: 0.3352
- F1: 0.4214
- support: 528

Happy:
- precision: 0.8658
- recall: 0.8953
- F1: 0.8803
- support: 879

Sad:
- precision: 0.4950
- recall: 0.5791
- F1: 0.5337
- support: 594

Surprise:
- precision: 0.7385
- recall: 0.8077
- F1: 0.7715
- support: 416

Neutral:
- precision: 0.6716
- recall: 0.6534
- F1: 0.6623
- support: 626


## Sensitivity analysis

Preprocessing:
training_like_resize256

Results:

- correct: 2388 / 3589
- accuracy: 0.665366
- balanced accuracy: 0.632438
- macro-F1: 0.642523
- mean prediction confidence: 0.713154


## Paired comparison

- both correct: 2216
- both wrong: 1030
- web correct / training-like wrong: 171
- web wrong / training-like correct: 172
- exact two-sided McNemar p: 1.00000000

The paired result does not indicate a meaningful difference in
overall correctness between the two preprocessing variants.

The historically faithful web preprocessing remains the primary
baseline because this decision was frozen before privateTest
evaluation.


## Interpretation

The reproduced EmoAR model achieves approximately 66.5% accuracy
on the held-out FER2013 privateTest partition.

Performance varies substantially across expression categories.

Happy and Surprise show comparatively stronger classification
performance, while Fear presents the lowest per-class F1 score in
the primary baseline.

The evaluation concerns FER2013 facial-expression categories only.

These results must not be interpreted as evidence of:

- recognition of a person's true internal emotional state;
- clinical validity;
- effectiveness with autistic children;
- performance on child faces;
- effectiveness of an Augmented Reality intervention.


## Next experimental direction

This result establishes the historical EmoAR baseline.

Subsequent experiments may compare it against a modern
pretrained facial-expression model using the same FER2013
evaluation protocol and may investigate computational cost,
latency, uncertainty and temporal stability.
