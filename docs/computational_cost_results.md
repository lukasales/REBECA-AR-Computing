# Computational Cost and Latency Results

## Scope

This experiment evaluates the computational trade-off between the
historical EmoAR DenseNet121 classifier and the modern ViT facial-
expression classifier used in the FER2013 comparison.

The purpose is not to benchmark a complete Augmented Reality system.
The measurements cover image preprocessing and facial-expression
classification only.

Camera capture, face detection, face tracking, AR rendering, user
interface processing and other application components are outside
the timed region.

## Environment

- device: CPU
- batch size: 1
- images per run: 100
- warm-up iterations: 10
- PyTorch CPU execution
- same machine and experimental environment for both models

## Structural computational cost

### EmoAR DenseNet121

- parameters: 6,961,031
- tensor footprint: approximately 26.87 MB

### Modern ViT

- parameters: 85,804,039
- tensor footprint: approximately 327.32 MB

### Relative structural cost

The ViT contains approximately:

- 12.326x more parameters
- 12.180x larger tensor-memory footprint

than the reproduced EmoAR classifier.

## Initial repeated benchmark

Five repeated runs were initially performed with a fixed execution
order in which EmoAR was evaluated before ViT.

Mean end-to-end latency across these runs:

- EmoAR: 72.145 ms
- ViT: 164.590 ms

Mean ViT/EmoAR latency ratio:

- 2.285x

Because model order was fixed, these measurements were followed by a
counterbalanced experiment.

## Counterbalanced benchmark

Six additional runs were performed.

Execution order:

1. EmoAR first
2. ViT first
3. ViT first
4. EmoAR first
5. EmoAR first
6. ViT first

Each run evaluated the same number of images under the same batch-1
CPU protocol.

## Overall counterbalanced results

### EmoAR

- mean latency: 70.595 ms
- median latency: 65.322 ms

### ViT

- mean latency: 151.343 ms
- median latency: 149.374 ms

The ratio between the aggregate mean latencies was approximately:

- 2.144x

The mean of the six within-run ViT/EmoAR latency ratios was:

- 2.171x

Median within-run ratio:

- 2.240x

Observed within-run ratio range:

- 1.768x to 2.341x

## Results by execution order

### EmoAR first

- EmoAR mean: 67.831 ms
- ViT mean: 149.238 ms
- mean within-run ratio: 2.205x

### ViT first

- EmoAR mean: 73.360 ms
- ViT mean: 153.448 ms
- mean within-run ratio: 2.138x

Reversing model order therefore did not remove the substantial
latency difference between the two classifiers.

The small number of runs per ordering condition does not justify a
formal claim that execution order has no effect.

## Interpretation

Under the evaluated CPU-only protocol, the modern ViT provides
stronger FER2013 classification performance but requires
substantially greater computational resources.

Relative to EmoAR, the ViT:

- achieved higher classification performance;
- contains approximately 12.3x more parameters;
- has approximately 12.2x greater tensor-memory footprint;
- required approximately 2.1x the end-to-end processing time in the
  counterbalanced CPU benchmark.

This represents an accuracy-versus-computational-cost trade-off that
is relevant when considering integration into a future REBECA-AR
experience.

## AR applicability limitation

These latency values must not be interpreted as the frame rate of a
complete AR application.

A practical AR system may also require:

- image acquisition;
- face detection;
- face tracking;
- temporal processing;
- rendering;
- interaction logic;
- visual feedback.

Conversely, facial-expression classification does not necessarily
need to execute at the same frequency as the AR rendering loop.

The results should therefore be interpreted as classifier-level
computational evidence rather than complete-system real-time
performance.
