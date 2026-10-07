# Reproducibility

This document describes the environment and main steps required to reproduce the computational experiments.

## Tested environment

- Python 3.11.9
- Windows 11 Pro
- AMD Ryzen 7 5700X
- 15.93 GB RAM
- AMD Radeon RX 7600
- CPU inference for the reported latency benchmark
- PyTorch 2.14.0+cpu
- torchvision 0.29.0+cpu

## Environment setup

From the repository root:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

The reported latency results were obtained using CPU builds of PyTorch and torchvision.

## FER2013

Dataset mirror:

Aaryan333/fer2013_train_publicTest_privateTest

Frozen revision:

da8d1643649c86bfc2cffba5b744b51fdf329212

The dataset is not stored in this repository.

## EmoAR baseline

Original repository:

https://github.com/kshntn/EmoAR

Revision used:

33d0c6b46775a830744625ee97f736b3255b8060

Expected local location:

external/EmoAR

Clone and freeze the revision with:

    git clone https://github.com/kshntn/EmoAR.git external/EmoAR
    git -C external/EmoAR checkout 33d0c6b46775a830744625ee97f736b3255b8060

The external repository and its checkpoint are not redistributed by this repository.

## Main experiment scripts

- scripts/evaluate_emoar_private_test.py
- scripts/evaluate_vit_private_test.py
- scripts/compare_emoar_vs_vit_private_test.py
- scripts/benchmark_latency_counterbalanced.py
- scripts/analyze_confidence_calibration.py
- scripts/evaluate_perturbation_stability.py
- scripts/generate_paper_artifacts.py

Detailed experimental protocols and limitations are available in docs/.

## Limitations

FER2013 evaluates facial-expression categories. The results must not be interpreted as direct evidence of a person's internal emotional state.

The current experiments do not validate performance for children, children with ASD, clinical use, temporal video behavior, or a complete Augmented Reality application.

The perturbation experiment is a controlled robustness proxy and not a temporal-stability experiment.

The CPU latency benchmark measures preprocessing and classification, not a complete AR pipeline.
