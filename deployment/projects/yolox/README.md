# YOLOX deployment

Deploys a YOLOX / YOLOX-opt 2D object detector (mmdet) from a trained checkpoint to ONNX and
TensorRT, with cross-backend verification and 2D-detection mAP evaluation. The reference deploy
config targets **YOLOX-opt-ELAN (960×960, 8-class)**, but the bundle is config-driven: classes,
score/NMS thresholds, strides and input size are read from the model config, so the same bundle
deploys any YOLOX variant (e.g. the 416×416 traffic-light detector) by pointing at its config.

## Quick start

```bash
python -m deployment.cli.main yolox \
    deployment/projects/yolox/config/deploy_config.py \
    [model_cfg.py]   # optional; defaults to the deploy config's top-level model_cfg
```

## How it works

- **Model**: loaded with mmdet `init_detector`; ReLU6 activations are swapped for ReLU for ONNX.
- **Export**: single whole-model ONNX/TensorRT via the shared default export pipeline. The
  [`YOLOXONNXWrapper`](export/model_wrappers.py) emits the Tier4 layout
  `[batch, anchors, 4 + 1 + num_classes]` (sigmoid objectness/class, **raw** bbox) — decode and NMS
  run in postprocess, not in-graph.
- **Postprocess**: mmdet `YOLOXHead` prior generation + `_bbox_decode` + `_bbox_post_process`
  (NMS + rescale to original image space), identical across PyTorch/ONNX/TensorRT.
- **Metrics**: IoU-2D mAP via the shared `Detection2DMetricsInterface`
  (`autoware_perception_evaluation`), consistent with training-time evaluation.

## Layout

| File | Role |
| --- | --- |
| [config/deploy_config.py](config/deploy_config.py) | Reference deploy config (single `model` component) |
| [io/model_loader.py](io/model_loader.py) | mmdet load + custom-module registration + ReLU6→ReLU |
| [io/data_loader.py](io/data_loader.py) | Runs the mmdet test pipeline; original-space ground truth |
| [export/model_wrappers.py](export/model_wrappers.py) | ONNX output-layout wrapper |
| [inference/](inference/) | PyTorch / ONNX / TensorRT pipelines (`run_model` per backend) |
| [evaluation/executor.py](evaluation/executor.py) | `BackendExecutor` (pipeline creation + input prep) |
| [runner.py](runner.py) / [entrypoint.py](entrypoint.py) | Wiring to the shared runtime and CLI |

The 2D evaluator and metrics live in the shared framework
([`deployment/evaluation/detection_2d_evaluator.py`](../../evaluation/detection_2d_evaluator.py),
[`deployment/metrics/detection_2d_metrics.py`](../../metrics/detection_2d_metrics.py)) — they are
task-level, not YOLOX-specific.

See the shared docs for framework structure ([../../docs/architecture.md](../../docs/architecture.md)),
config fields ([../../docs/configuration.md](../../docs/configuration.md)) and run behavior
([../../docs/runbook.md](../../docs/runbook.md)).
