# Computational Environment

## Local workstation

- Operating system: Microsoft Windows 11 Pro 64-bit
- Windows version: 10.0.26200
- CPU: AMD Ryzen 7 5700X
- Physical CPU cores: 8
- Logical processors: 16
- RAM: 15.93 GB
- GPU: AMD Radeon RX 7600
- NVIDIA CUDA: unavailable
- Python: 3.11.9
- pip: 26.1.2
- Git: 2.55.0.windows.2

## Execution strategy

The local workstation will be used primarily for:

- source-code development;
- Git/version control;
- lightweight preprocessing;
- lightweight inference;
- analysis of experimental results;
- visualization;
- webcam-based prototypes;
- future AR integration experiments when feasible.

Computationally intensive experiments will preferentially be
executed in Google Colab using a GPU runtime.

This strategy avoids making the experimental pipeline dependent
on NVIDIA CUDA support on the local Windows workstation.

## Reproducibility

Software versions, model identifiers, dataset versions, experiment
configurations and random seeds should be recorded whenever
applicable.

Generated datasets and model weights must not be committed
directly to the repository.
