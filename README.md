# REBECA-AR Computing

Computational and experimental engineering component of the
REBECA-AR course project.

Course:

2026.2 - IN1123 - Realidade Virtual e Aumentada  
Centro de Informática - Universidade Federal de Pernambuco

## Objective

This repository investigates the technical applicability of existing
facial-expression classification approaches to a future Augmented
Reality extension of REBECA.

The project focuses on reproducibility, quantitative evaluation,
computational performance, temporal stability and uncertainty.

The goal is not to develop a new emotion-recognition algorithm.

## Current research workflow

Definition -> Development -> Evaluation

### Definition

- identify technical research questions;
- inspect prior systems and baselines;
- define models, datasets and metrics;
- document limitations.

### Development

- build a reproducible inference pipeline;
- implement experimental instrumentation;
- implement temporal smoothing and uncertainty handling;
- develop a technical demonstrator;
- investigate AR integration.

### Evaluation

- predictive performance;
- error analysis;
- inference latency;
- computational cost;
- temporal stability;
- uncertainty and abstention;
- external-domain evaluation when possible.

## Repository structure

    data/          Dataset documentation and local data
    demo/          Interactive demonstrators
    docs/          Research and reproducibility documentation
    experiments/   Experiment configurations
    figures/       Paper-ready figures
    models/        Model documentation and local weights
    notebooks/     Jupyter/Google Colab experiments
    results/       Experimental results
    scripts/       Utility scripts
    src/           Reusable source code

## Execution environments

Local development:

- AMD Ryzen 7 5700X
- 15.93 GB RAM
- AMD Radeon RX 7600
- Windows 11 Pro
- Python 3.11.9

GPU-intensive experiments will preferentially use Google Colab.

See:

- docs/environment.md
- docs/research_questions.md
- docs/computational_contribution.md

## Current status

Phase 1 - Research environment and reproducibility structure.
