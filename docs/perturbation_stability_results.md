# Controlled Perturbation Stability Results

## Scope

This experiment evaluates prediction stability under small,
deterministic image perturbations as a controlled proxy for
frame-level robustness.

It does not measure true temporal stability or live AR performance.

## Dataset and sample

Dataset:
Aaryan333/fer2013_train_publicTest_privateTest

Revision:
da8d1643649c86bfc2cffba5b744b51fdf329212

Split:
privateTest

Balanced sample:

- 50 images per class
- 7 classes
- 350 original images
- deterministic seed: 20260921

Each image was evaluated in its original form and under eight
pre-specified perturbations.

This produced:

- 3,150 image variants per model
- 2,800 original-versus-perturbed comparison pairs per model

## EmoAR results

Original accuracy:

- 0.631429

Aggregate accuracy across perturbed images:

- 0.625714

Pairwise label stability:

- 0.908929

Stable prediction pairs:

- 2,545 / 2,800

Images with at least one prediction flip:

- 109 / 350
- 31.14%

Images stable across all eight perturbations:

- 241 / 350
- 68.86%

Mean number of unique predicted categories per original image:

- 1.368571

Mean absolute change in top-1 confidence:

- 0.051590

Maximum absolute confidence change:

- 0.548228

## Modern ViT results

Original accuracy:

- 0.720000

Aggregate accuracy across perturbed images:

- 0.686786

Pairwise label stability:

- 0.876429

Stable prediction pairs:

- 2,454 / 2,800

Images with at least one prediction flip:

- 156 / 350
- 44.57%

Images stable across all eight perturbations:

- 194 / 350
- 55.43%

Mean number of unique predicted categories per original image:

- 1.520000

Mean absolute change in top-1 confidence:

- 0.062393

Maximum absolute confidence change:

- 0.720841

## Comparative results

Relative to EmoAR, the ViT achieved:

- +0.061071 aggregate perturbed accuracy;
- -0.032500 pairwise label stability;
- -0.134286 fraction of fully stable images.

Therefore, the ViT preserved its classification-performance advantage
on the controlled perturbed images while showing lower consistency
with its original predicted category.

## Perturbation-specific behavior

### Rotation

EmoAR label stability:

- -5 degrees: 0.828571
- +5 degrees: 0.811429

ViT label stability:

- -5 degrees: 0.742857
- +5 degrees: 0.731429

Rotation was among the most disruptive perturbations for both models,
with a larger stability reduction for the ViT.

### Horizontal translation

EmoAR:

- -2 pixels: 0.922857
- +2 pixels: 0.925714

ViT:

- -2 pixels: 0.845714
- +2 pixels: 0.817143

The ViT was also more sensitive to the tested small translations.

### Brightness

EmoAR:

- factor 0.90: 0.940000
- factor 1.10: 0.942857

ViT:

- factor 0.90: 0.960000
- factor 1.10: 0.948571

Both models were comparatively stable to the tested brightness
changes.

### Contrast

EmoAR:

- factor 0.90: 0.945714
- factor 1.10: 0.954286

ViT:

- factor 0.90: 0.982857
- factor 1.10: 0.982857

The ViT showed particularly high stability under the tested contrast
changes.

## Interpretation

The modern ViT remained more accurate than the reproduced EmoAR
baseline under the evaluated perturbations.

However, higher accuracy did not correspond to greater prediction
consistency.

Under the controlled perturbation protocol, the ViT changed its
predicted facial-expression category more frequently than EmoAR,
especially under small rotations and translations.

This distinction is relevant to a future AR application because
classification accuracy and frame-level consistency are different
system properties.

A classifier may be more accurate overall while still producing more
frequent label changes under small variations of the input.

## Important limitations

This experiment must not be described as temporal validation.

FER2013 consists of static images and the perturbations were generated
artificially.

The geometric perturbations also introduce image-border effects due
to the specified fill behavior.

The experiment does not reproduce:

- natural head movement;
- live camera noise;
- motion blur;
- face tracking;
- occlusion;
- changing camera distance;
- realistic illumination transitions;
- child faces;
- autistic children;
- complete AR rendering and interaction.

The results therefore characterize controlled perturbation robustness,
not real-world AR stability.
