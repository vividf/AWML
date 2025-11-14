# Autoware ML Deployment Framework

A unified, task-agnostic deployment framework for exporting, verifying, and evaluating machine learning models across multiple backends (PyTorch, ONNX Runtime, TensorRT). The package provides reusable building blocks so individual projects only need to implement task-specific data loading, evaluation, and optional pipeline overrides.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Directory Layout](#directory-layout)
- [Workflow](#workflow)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Extending the Framework](#extending-the-framework)

## Overview
The deployment package standardizes the process of
1. Loading project-specific datasets and checkpoints
2. Exporting trained PyTorch models to ONNX and TensorRT
3. Verifying numerical parity across backends (optional)
4. Evaluating accuracy and latency metrics across backends

Projects consume the framework through lightweight entry scripts (see `projects/*/deploy/main.py`) that glue project implementations to the shared deployment workflow.

## Key Features
- **Unified Runner** – `DeploymentRunner` orchestrates export → verification → evaluation with pluggable hooks.
- **Config Driven** – `BaseDeploymentConfig` parses `mmengine` configs to control export modes, runtime IO, backend settings, and verification/evaluation scenarios.
- **Backend Exporters** – `ONNXExporter` and `TensorRTExporter` expose a common interface for exporting PyTorch models and ONNX graphs.
- **Task Pipelines** – Base pipelines in `core/` implement reusable preprocessing/inference/postprocessing logic for classification, 2D detection, and 3D detection; model-specific pipelines live under `pipelines/`.
- **Verification Support** – Optional cross-backend verification compares PyTorch, ONNX, and TensorRT outputs using project evaluators.
- **Evaluation & Benchmarking** – Evaluators compute task metrics and latency statistics, enabling fair comparisons across backends.

## Directory Layout
```
autoware_ml/deployment/
├── core/                  # Task-agnostic base classes
│   ├── base_config.py     # Export/runtime/backend config helpers
│   ├── base_data_loader.py
│   ├── base_evaluator.py
│   ├── base_pipeline.py   # Template method for preprocess → run_model → postprocess
│   ├── classification_pipeline.py
│   ├── detection_2d_pipeline.py
│   └── detection_3d_pipeline.py
├── exporters/             # Backend exporters
│   ├── base_exporter.py
│   ├── onnx_exporter.py
│   └── tensorrt_exporter.py
├── pipelines/             # Model-specific pipeline stacks
│   ├── calibration/
│   ├── centerpoint/
│   └── yolox/
├── runners/
│   └── deployment_runner.py
├── core/
│   └── preprocessing_builder.py  # Preprocessing pipeline builder
└── docs/                  # Additional design notes and tutorials
```

## Workflow
`DeploymentRunner` executes the following stages:
1. **Model Loading** – Uses project-provided `load_model_fn` (or subclass override) to restore a PyTorch checkpoint.
2. **Export** – Calls `ONNXExporter` and/or `TensorRTExporter` according to `export.mode` (`onnx`, `trt`, `both`, `none`).
3. **Verification** (optional) – If `export.verify=True`, project `BaseEvaluator.verify()` implementations perform cross-backend checks.
4. **Evaluation** – When `evaluation.enabled=True`, the runner iterates through configured backends and calls `BaseEvaluator.evaluate()` to report metrics and latency statistics.
5. **Results Summary** – Consolidates export paths, verification summaries, and evaluation metrics.

Each stage can be customized per project via dependency injection (passing callables to the runner) or subclassing `DeploymentRunner` (see `projects/CenterPoint/deploy/main.py`).

## Quick Start
Example: deploy a calibration status classification model.
```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/deploy/configs/deploy_config.py \
    projects/CalibrationStatusClassification/configs/model_config.py \
    checkpoints/calib_classifier.pth \
    --work-dir work_dirs/calibration_deploy
```

General steps for all projects:
1. **Prepare configs** – Copy or edit a deployment config under `projects/<Project>/deploy/configs/`.
2. **Run the entry script** – Provide `deploy_config`, `model_cfg`, and (when exporting) a checkpoint path.
3. **Inspect outputs** – Exported models and reports are stored under `export.work_dir`.

### Common CLI overrides
```
--work-dir    Override export.work_dir from the config
--device      Override export.device (e.g., "cuda:0", "cpu")
--log-level   Adjust logging verbosity
```

## Configuration Reference
Deployment configs are standard `mmengine.Config` files. Key sections include:

### `task_type`
Task type for pipeline building (required when using `build_preprocessing_pipeline`):
```python
task_type = "detection3d"  # Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
```

### `export`
```python
export = dict(
    mode='both',          # 'onnx', 'trt', 'both', 'none'
    verify=True,          # Enable cross-backend verification
    device='cuda:0',      # Device used during export/inference
    work_dir='work_dirs'  # Output directory for artifacts
)
```

### `runtime_io`
Holds dataset-specific paths and overrides (e.g., info files, sample indices).

### `model_io`
Describes model inputs/outputs used for export:
- `input_name`, `input_shape`, `input_dtype`
- `additional_inputs` / `additional_outputs`
- `batch_size` or `dynamic_axes`

### `onnx_config`
ONNX exporter options such as `opset_version`, `keep_initializers_as_inputs`, `save_file`, `decode_in_inference`.

### `backend_config`
TensorRT-specific settings:
```python
backend_config = dict(
    common_config=dict(
        precision_policy='auto',     # 'auto', 'fp16', 'fp32_tf32', 'explicit_int8', 'strongly_typed'
        max_workspace_size=1 << 30   # Bytes of workspace memory
    ),
    model_inputs=[ ... ]            # Optional shape dictionaries for TensorRT optimization profiles
)
```

### `verification`
Controls cross-backend checks (consumed by project evaluators):
```python
verification = dict(
    num_verify_samples=3,
    tolerance=1e-3
)
```

### `evaluation`
Configures evaluation runs:
```python
evaluation = dict(
    enabled=True,
    num_samples=100,
    verbose=False,
    models=dict(
        pytorch='path/to/model.pth',
        onnx='work_dirs/model.onnx',
        tensorrt='work_dirs/model.engine'
    )
)
```
If `models` omit a backend, the runner will fall back to freshly exported artifacts when available.

## Extending the Framework
1. **Create project-specific components**
   - Implement `BaseDataLoader` and `BaseEvaluator` subclasses under `projects/<Project>/deploy/`.
   - Optionally provide backend-specific pipelines by extending classes in `pipelines/<model>/`.

2. **Author a deployment script**
   - Instantiate project components, parse arguments via `parse_base_args`, and create a `DeploymentRunner` (or subclass).
   - Pass custom callables (`load_model_fn`, `export_onnx_fn`, `export_tensorrt_fn`) when project logic diverges from the defaults.

3. **Update documentation/configs**
   - Add configuration examples in `projects/<Project>/deploy/configs/`.
   - Document project-specific flags in the project README.

Refer to `projects/CenterPoint`, `projects/YOLOX_opt_elan`, and `projects/CalibrationStatusClassification` for complete examples covering multi-stage pipelines, ONNX/TensorRT overrides, and verification workflows.
