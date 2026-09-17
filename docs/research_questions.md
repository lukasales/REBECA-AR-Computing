# Research Questions - Computational Applicability

This document defines the technical questions investigated by the
computational component of the REBECA-AR project.

## Main technical question

To what extent can existing facial-expression classification
approaches support a future Augmented Reality experience within
REBECA considering predictive performance, computational cost,
latency, temporal stability and uncertainty?

## RQ-C1 - Reproducibility and baseline feasibility

Can an existing facial-expression classification model be executed
in a reproducible pipeline and evaluated using a standardized
benchmark?

Initial evidence:

- successful model inference;
- documented preprocessing;
- accuracy;
- macro-F1;
- balanced accuracy;
- per-class metrics;
- confusion matrix.

## RQ-C2 - Computational applicability

What computational trade-offs exist between candidate models for
a future interactive REBECA-AR application?

Possible evidence:

- model size;
- number of parameters;
- inference latency;
- latency p50 and p95;
- throughput;
- memory requirements;
- predictive performance.

## RQ-C3 - Temporal stability and uncertainty

Can temporal smoothing and confidence-based abstention reduce
unstable changes in expression-category feedback during video
inference?

Possible evidence:

- label changes per second;
- prediction confidence;
- coverage;
- selective performance;
- additional latency introduced by smoothing.

## RQ-C4 - Child-domain generalization

If an appropriately licensed and authorized child facial-expression
dataset becomes available, how does a model evaluated on FER2013
behave when applied without retraining to child faces?

This question is conditional on authorized dataset access.

A child facial-expression dataset must not be interpreted as evidence
of performance specifically on autistic children unless the dataset
actually represents that population.

## Scope limitation

The computational component does not attempt to infer a person's
true internal emotional state.

Models are treated as classifiers that assign observed facial
configurations to predefined expression categories.

The current stage evaluates technical applicability and does not
establish therapeutic or educational effectiveness for children
with autism.
