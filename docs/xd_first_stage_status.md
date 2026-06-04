# XD-Violence First Stage Status

Date: 2026-06-04

## Scope

This document records the first-stage execution state for XD-Violence before any full CCD-MAVD training.

## Completed

- XD protocol files from `/home/han/CQAF-VAD/list` are connected under `data/lists/xd_violence/`.
- `data/lists/xd_violence/protocol_manifest.json` records file sizes and SHA-256 checksums.
- `scripts/audit_xd.py` generates `experiments/audit/xd_manifest.json` and `experiments/audit/xd_audit_report.md`.
- M0 audit passed:
  - train videos: 3954
  - test videos: 800
  - RGB train/test files: 19770 / 4000
  - Flow train/test files: 19770 / 4000
  - Audio train/test files: 3954 / 800
  - five-crop grouping pass: true
  - modality alignment pass: true
  - GT frames: 2330384
  - test snippets: 145649
  - GT/snippet ratio: 16.0
- `scripts/smoke_xd_metric.py` ran dummy score-to-frame expansion over the full XD test set.
- Metric smoke result:
  - frame scores: 2330384
  - frame labels: 2330384
  - dummy frame AP: 0.2307795625098696
- XD loader returns RGB, Flow, Audio tensors resampled to `T = 32`.
- A0 MIL-only tiny training completed for 2 epochs.

## Latest A0 Tiny Run

Run directory:

```text
experiments/xd_violence/xd_a0_mil_tiny_seed0_20260604T011150Z
```

Command:

```text
/home/han/miniconda3/envs/hh/bin/python scripts/train.py --config configs/xd_a0_mil_tiny.yaml
```

Run artifacts:

- `config.yaml`
- `command.txt`
- `env.txt`
- `train.log`
- `metrics.jsonl`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `predictions/test_scores.npz`
- `reports/summary.md`

Prediction export contains:

- `video_ids`
- `snippet_scores`
- `frame_scores`
- `video_labels`
- `frame_labels`

Latest metric row:

```json
{"epoch": 2, "loss_total": 0.004018645238829777, "val_frame_ap": 0.5454868893436664, "lr": 0.0003}
```

This tiny result is a pipeline smoke signal only. It is not a paper result. Feature dimensions are inferred from loader samples before model construction, not hard-coded in the model.

## Next Gate

Proceed to A1 only after reviewing A0 tiny outputs for obvious protocol errors.

Recommended next order:

1. A1 RTFM-style top-k/ranking on a small run.
2. A2/A3 fusion baselines.
3. A4 MAVD-style gated fusion + local alignment.
4. Context label generation and context audit only after A4 is stable.
5. A5/A6 context debiasing only after context audit passes.
