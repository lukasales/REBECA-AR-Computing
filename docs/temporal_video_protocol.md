# Temporal Video Evaluation Protocol

## Objective

Evaluate the temporal behavior of the EmoAR and ViT facial-expression classifiers on video sequences.

This experiment extends the previous static-image perturbation analysis by evaluating consecutive frames.

It does not constitute validation with children, children with ASD, clinical validation, or evaluation of a complete Augmented Reality application.

## Research purpose

The experiment investigates whether model predictions remain sufficiently stable over consecutive video frames to support discussion about future interactive use.

## Models

- EmoAR DenseNet121 baseline;
- trpakov/vit-face-expression ViT;
- same frozen model revisions and class mapping used in previous experiments.

## Input

Video files processed frame by frame.

Videos will not be committed to the repository.

Expected local directory:

data/video/

## Frame processing

- frames decoded sequentially;
- RGB conversion before model preprocessing;
- both models receive the same decoded frame;
- original model-specific preprocessing is preserved;
- no temporal smoothing will be applied in the primary experiment.

The primary analysis therefore measures raw frame-level behavior.

## Metrics

### Computational performance

- number of processed frames;
- mean inference latency per frame;
- median inference latency;
- p95 inference latency;
- effective classifier FPS.

### Temporal label stability

- consecutive-frame label agreement;
- consecutive-frame label change rate;
- total number of label switches;
- number of stable prediction runs;
- mean stable-run length;
- median stable-run length;
- maximum stable-run length.

### Confidence behavior

- mean confidence;
- standard deviation of confidence;
- mean absolute confidence change between consecutive frames;
- p95 absolute confidence change.

### Prediction distribution

- number of frames predicted for each class;
- proportion of frames predicted for each class.

## Flickering

Prediction flickering will be operationalized as frequent label changes across consecutive frames.

Primary flicker measure:

label_switches / (processed_frames - 1)

Lower values indicate greater raw temporal label stability.

## Latency scope

Reported latency will measure model preprocessing and classification only unless explicitly stated otherwise.

Video decoding, camera capture, face detection, tracking and AR rendering must not be interpreted as included in classifier latency.

## Experimental comparison

Both models must process the same frame sequence.

Results will be reported per video and in aggregate.

## Primary outputs

- results/temporal_video_predictions.csv;
- results/temporal_video_summary.csv;
- results/temporal_video_results.json;
- docs/temporal_video_results.md;
- figures/temporal_video_stability.png.

## Important limitations

- the experiment measures temporal prediction behavior, not recognition of true internal emotion;
- video subjects may not represent children with ASD;
- stability does not imply semantic correctness;
- no temporal smoothing is used in the primary analysis;
- no face tracking is assumed in the initial benchmark;
- results must not be generalized to a complete AR system.

## Protocol freeze

This protocol must be committed before final temporal-video results are generated.
