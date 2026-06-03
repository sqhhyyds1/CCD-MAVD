# CCD-MAVD

Causal-inspired Context-Debiased Multimodal Video Anomaly Detection.

This repository starts from a clean implementation while using MAVD-style alignment-first multimodal VAD as a protocol reference and baseline target. The first reproducible path uses public pre-extracted features.

## Project Layout

- `configs/`: dataset and experiment YAML configs.
- `src/ccd_mavd/`: main implementation.
- `tests/`: unit and smoke tests.
- `scripts/`: runnable entry points for data preparation, training, and evaluation.
- `baselines/`: external baseline checkouts or notes.
- `third_party/`: read-only third-party references.
- `data/`: lists, features, and generated context labels.
- `experiments/` and `logs/`: local run artifacts.

## First Protocol

- XD-Violence: I3D RGB/Flow + VGGish audio; frame-level AP.
- ShanghaiTech weak split: I3D features; frame-level ROC-AUC and cross-scene checks.
- UCF-Crime: I3D features; frame-level ROC-AUC with AP/PR-AUC as supplemental.
