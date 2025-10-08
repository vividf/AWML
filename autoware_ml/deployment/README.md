# Autoware ML Deployment Framework

A unified, task-agnostic deployment framework for exporting, verifying, and evaluating machine learning models across different backends (ONNX, TensorRT) with comprehensive support for model validation and performance benchmarking.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Current Support](#current-support)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Basic Export](#basic-export)
  - [Export with Verification](#export-with-verification)
  - [Export with Full Evaluation](#export-with-full-evaluation)
  - [Evaluation Only Mode](#evaluation-only-mode)
- [Configuration Reference](#configuration-reference)

## Overview

The Autoware ML Deployment Framework provides a standardized pipeline for deploying trained models to production-ready inference backends. It handles the complete deployment workflow from model export to validation and performance analysis, with a focus on ensuring model quality and correctness across different deployment targets.

### Key Capabilities

- **Multi-Backend Export**: Export models to ONNX and TensorRT formats
- **Precision Policy Support**: Flexible precision policies (FP32, FP16, TF32, INT8)
- **Automated Verification**: Cross-backend output validation to ensure correctness
- **Performance Benchmarking**: Comprehensive latency and throughput analysis
- **Full Evaluation**: Complete model evaluation with metrics and confusion matrices
- **Modular Design**: Easy to extend for new tasks and backends



## Current Support

### Detection 3D
* [ ] BEVFusion
* [ ] CenterPoint
* [ ] TransFusion
* [ ] StreamPETR

### Detection 2D
* [ ] YOLOX
* [ ] YOLOX_opt (Traffic Light Detection)
* [ ] FRNet
* [ ] GLIP (Grounded Language-Image Pre-training)

### Classification
* [X] CalibrationStatusClassification
* [ ] MobileNetv2 (Traffic Light Classification)

### Backbones & Components
* [ ] SwinTransformer
* [ ] ConvNeXt_PC (Point Cloud)
* [ ] SparseConvolution

### Multimodal
* [ ] BLIP-2 (Vision-Language Model)

> **Note**: Currently, only **CalibrationStatusClassification** has full deployment framework support. Other models may have custom deployment scripts in their respective project directories but are not yet integrated with the unified deployment framework.

### Supported Backends

| Backend | Export | Inference | Verification |
|---------|--------|-----------|--------------|
| **ONNX** | ✅ | ✅ | ✅ |
| **TensorRT** | ✅ | ✅ | ✅ |

## Architecture

The deployment framework follows a modular architecture:

```
autoware_ml/deployment/
├── core/                          # Core abstractions
│   ├── base_config.py            # Configuration management
│   ├── base_data_loader.py       # Data loading interface
│   ├── base_evaluator.py         # Evaluation interface
│   └── verification.py           # Cross-backend verification
├── backends/                      # Backend implementations
│   ├── pytorch_backend.py        # PyTorch inference
│   ├── onnx_backend.py           # ONNX Runtime inference
│   └── tensorrt_backend.py       # TensorRT inference
└── exporters/                     # Export implementations
    ├── onnx_exporter.py          # ONNX export
    └── tensorrt_exporter.py      # TensorRT export
```

### Design Principles

1. **Task-Agnostic Core**: Base classes are independent of specific tasks
2. **Backend Abstraction**: Unified interface across different inference backends
3. **Extensibility**: Easy to add new tasks, backends, or exporters
4. **Configuration-Driven**: All settings managed through Python config files
5. **Comprehensive Validation**: Built-in verification at every step


## Quick Start

Here's a minimal example to export and verify a calibration classification model:

```bash
# Export to both ONNX and TensorRT with verification
python projects/CalibrationStatusClassification/deploy/main.py \
    deploy_config.py \
    model_config.py \
    checkpoint.pth \
    --work-dir work_dirs/deployment
```


## Usage Guide

### Basic Export

Export a model to ONNX format:

**1. Create deployment config** (`deploy_config_onnx.py`):

```python
export = dict(
    mode='onnx',           # Export mode: 'onnx', 'trt', 'both', 'none'
    verify=False,          # Skip verification
    device='cuda:0',       # Device for export
    work_dir='work_dirs/deployment'
)

# Runtime I/O settings
runtime_io = dict(
    info_pkl='path/to/info.pkl',  # Dataset info file
    sample_idx=0                   # Sample index for export
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    save_file='model.onnx',
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        precision_policy='auto',      # Options: 'auto', 'fp16', 'fp32_tf32', 'int8'
        max_workspace_size=1 << 30    # 1 GB for TensorRT
    )
)
```

**2. Run export**:

```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    deploy_config_onnx.py \
    path/to/model_config.py \
    path/to/checkpoint.pth
```

### Export with Verification

Verify that exported models produce correct outputs:

**Update config**:

```python
export = dict(
    mode='both',          # Export to both ONNX and TensorRT
    verify=True,          # Enable verification
    device='cuda:0',
    work_dir='work_dirs/deployment'
)

# ... rest of config ...
```

**Run with verification**:

```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    deploy_config_verify.py \
    path/to/model_config.py \
    path/to/checkpoint.pth
```

### Export with Full Evaluation

Perform complete model evaluation on a validation dataset:

**Update config** (`deploy_config_eval.py`):

```python
export = dict(
    mode='both',
    verify=True,
    device='cuda:0',
    work_dir='work_dirs/deployment'
)

# Enable evaluation
evaluation = dict(
    enabled=True,
    num_samples=1000,        # Number of samples to evaluate
    verbose=False,           # Set True for detailed per-sample output
    models_to_evaluate=[
        'pytorch',           # Evaluate PyTorch model
        'onnx',             # Evaluate ONNX model
        'tensorrt'          # Evaluate TensorRT model
    ]
)

# ... rest of config ...
```

**Run with evaluation**:

```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    deploy_config_eval.py \
    path/to/model_config.py \
    path/to/checkpoint.pth
```

**Output includes**:
- Per-model accuracy and performance metrics
- Confusion matrices for each backend
- Latency statistics (min, max, mean, median, p95, p99)
- Per-class accuracy breakdown

### Evaluation Only Mode

Run evaluation without exporting (useful for testing existing deployments):

**Config** (`deploy_config_eval_only.py`):

```python
export = dict(
    mode='none',           # Skip export
    device='cuda:0',
    work_dir='work_dirs/deployment'
)

evaluation = dict(
    enabled=True,
    num_samples=1000,
    models_to_evaluate=['onnx', 'tensorrt']  # Evaluate existing models
)

runtime_io = dict(
    info_pkl='path/to/info.pkl',
    onnx_file='work_dirs/deployment/model.onnx'  # Path to existing ONNX
)

# ... rest of config ...
```

**Run**:

```bash
# No checkpoint needed in eval-only mode
python projects/CalibrationStatusClassification/deploy/main.py \
    deploy_config_eval_only.py \
    path/to/model_config.py
```

## Configuration Reference

### Export Configuration

```python
export = dict(
    mode='both',          # 'onnx', 'trt', 'both', 'none'
    verify=True,          # Enable cross-backend verification
    device='cuda:0',      # Device for export/inference
    work_dir='work_dirs'  # Output directory
)
```

### Runtime I/O Configuration

```python
runtime_io = dict(
    info_pkl='path/to/dataset/info.pkl',      # Required: dataset info file
    sample_idx=0,                              # Sample index for export
    onnx_file='path/to/existing/model.onnx'   # Optional: use existing ONNX
)
```

### ONNX Configuration

```python
onnx_config = dict(
    opset_version=16,                # ONNX opset version
    do_constant_folding=True,        # Enable constant folding optimization
    input_names=['input'],           # Input tensor names
    output_names=['output'],         # Output tensor names
    save_file='model.onnx',          # Output filename
    export_params=True,              # Export model parameters
    dynamic_axes={                   # Dynamic dimensions
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    keep_initializers_as_inputs=False  # ONNX optimization
)
```

### Backend Configuration

```python
backend_config = dict(
    common_config=dict(
        precision_policy='fp16',        # Precision policy (see below)
        max_workspace_size=1 << 30      # TensorRT workspace size (bytes)
    ),
    model_inputs=[                      # Optional: input specifications
        dict(
            name='input',
            shape=(1, 5, 512, 512),
            dtype='float32'
        )
    ]
)
```

### Precision Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `auto` | Let TensorRT decide | Default, balanced performance |
| `fp16` | Half precision (FP16) | 2x faster, ~same accuracy |
| `fp32_tf32` | TensorFlow 32 (TF32) | Good balance for Ampere+ GPUs |
| `strongly_typed` | Strict type enforcement | For debugging |

### Evaluation Configuration

```python
evaluation = dict(
    enabled=True,                    # Enable evaluation
    num_samples=1000,                # Number of samples to evaluate
    verbose=False,                   # Detailed per-sample output
    models_to_evaluate=[             # Backends to evaluate
        'pytorch',
        'onnx',
        'tensorrt'
    ]
)
```
