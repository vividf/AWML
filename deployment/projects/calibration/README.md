# Calibration Status Classification deployment

Deploys the camera–LiDAR calibration-status classifier (mmpretrain ResNet-18, **5-channel** input)
from a trained checkpoint to ONNX and TensorRT, with cross-backend verification and classification
metrics (accuracy / precision / recall / F1 + confusion matrix).

## Quick start

```bash
python -m deployment.cli.main calibration \
    deployment/projects/calibration/config/deploy_config.py \
    [model_cfg.py]   # optional; defaults to the deploy config's top-level model_cfg
```

## How it works

- **Input**: a 5-channel fused image `[B, G, R, depth, intensity]` at native camera resolution
  (dynamic H/W), built by `CalibrationClassificationTransform` (normalized by `/255`; no ImageNet
  mean/std).
- **Model**: mmpretrain `ImageClassifier` (ResNet-18, `in_channels=5`), loaded with `get_model`. Run
  in `mode='tensor'`, so it emits raw logits `[1, 2]`; the ONNX graph exports logits
  (`IdentityWrapper`) and softmax is applied in postprocess.
- **Synthetic ground truth**: a sample is *miscalibrated* by perturbing the LiDAR→camera extrinsic
  before projection, else *calibrated*. The [data loader](io/data_loader.py) expands each base
  sample into **two frames** (`num_samples = 2 × base`, even = calibrated, odd = miscalibrated) and
  **seeds each index**, so the synthetic perturbation is reproducible across evaluation and
  verification. This replaces the old evaluator's dual-loader override — the evaluator stays a pure
  metrics adapter.
- **Metrics**: the shared `ClassificationMetricsInterface` (`autoware_perception_evaluation`).

> `num_samples` in the deploy config counts **frames**; use an even number (or `-1`) for a
> class-balanced run.

## Layout

| File | Role |
| --- | --- |
| [config/deploy_config.py](config/deploy_config.py) | Reference deploy config (single `model` component, `class_names`) |
| [io/model_loader.py](io/model_loader.py) | mmpretrain `get_model` loader |
| [io/data_loader.py](io/data_loader.py) | Dual-variant, seeded synthetic-GT loader |
| [inference/](inference/) | PyTorch / ONNX / TensorRT pipelines (logits → softmax) |
| [evaluation/executor.py](evaluation/executor.py) | `BackendExecutor` (pipeline creation + input prep) |
| [runner.py](runner.py) / [entrypoint.py](entrypoint.py) | Wiring to the shared runtime and CLI |

The classification evaluator and metrics live in the shared framework
([`deployment/evaluation/classification_evaluator.py`](../../evaluation/classification_evaluator.py),
[`deployment/metrics/classification_metrics.py`](../../metrics/classification_metrics.py)) — they are
task-level, not calibration-specific.

See the shared docs for framework structure ([../../docs/architecture.md](../../docs/architecture.md)),
config fields ([../../docs/configuration.md](../../docs/configuration.md)) and run behavior
([../../docs/runbook.md](../../docs/runbook.md)).
