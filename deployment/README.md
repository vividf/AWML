# AWML Deployment Framework

A unified, task-agnostic deployment framework for exporting PyTorch models to ONNX and TensorRT, with comprehensive verification and evaluation capabilities.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project-Specific Implementations](#project-specific-implementations)
- [Pipeline Architecture](#pipeline-architecture)
- [Export Workflow](#export-workflow)
- [Verification & Evaluation](#verification--evaluation)

---

## Overview

The AWML Deployment Framework provides a standardized approach to model deployment across different projects (CenterPoint, YOLOX, CalibrationStatusClassification). It abstracts common deployment workflows while allowing project-specific customizations.

### Design Principles

1. **Unified Interface**: Shared base runner (`BaseDeploymentRunner`) with project-specific subclasses
2. **Task-Agnostic Core**: Base classes that work across detection, classification, and segmentation
3. **Backend Flexibility**: Support for PyTorch, ONNX, and TensorRT backends
4. **Pipeline Architecture**: Shared preprocessing/postprocessing with backend-specific inference
5. **Configuration-Driven**: All settings controlled via config files
6. **Dependency Injection**: Explicit creation and injection of exporters and wrappers for better clarity and type safety

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                           │
│  - CenterPoint, YOLOX-ELAN, Calibration                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ BaseDeploymentRunner + Project Runners                  │
│  - Coordinates export → verification → evaluation      │
│  - Each project extends the base class for custom logic│
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│   Exporters    │   │   Evaluators    │
│  - ONNX        │   │  - Task-specific│
│  - TensorRT    │   │  - Metrics      │
│  - Wrappers    │   │                 │
│  (Unified      │   │                 │
│   structure)   │   │                 │
└────────────────┘   └─────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Pipeline Architecture                      │
│  - BaseDeploymentPipeline                               │
│  - Task-specific pipelines (Detection2D/3D, Classify)   │
│  - Backend-specific implementations                     │
└──────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **BaseDeploymentRunner & Project Runners**
`BaseDeploymentRunner` orchestrates the complete deployment workflow, while each project provides a thin subclass (`CenterPointDeploymentRunner`, `YOLOXDeploymentRunner`, `CalibrationDeploymentRunner`) that plugs in model-specific logic.

- **Model Loading**: Implemented by each project runner to load PyTorch checkpoints
- **Export**: Uses injected ONNX/TensorRT exporters that encapsulate wrapper logic
- **Verification**: Scenario-based verification across backends
- **Evaluation**: Performance metrics and latency statistics

**Required Parameters:**
- `onnx_exporter`: ONNX exporter instance injected per runner (typically the generic `ONNXExporter` with a wrapper)
- `tensorrt_exporter`: TensorRT exporter instance injected per runner (typically the generic `TensorRTExporter` with a wrapper)

Exporters receive their corresponding `model_wrapper` during construction. Runners never implicitly create exporters/wrappers—everything is injected for clarity and testability.

#### 2. **Core Components** (in `core/`)

- **`BaseDeploymentConfig`**: Configuration container for deployment settings
- **`Backend`**: Enum for supported backends (PyTorch, ONNX, TensorRT)
- **`Artifact`**: Dataclass representing deployment artifacts (ONNX/TensorRT outputs)
- **`BaseEvaluator`**: Abstract interface for task-specific evaluation
- **`BaseDataLoader`**: Abstract interface for data loading
- **`build_preprocessing_pipeline`**: Utility to extract preprocessing pipelines from MMDet/MMDet3D configs
- **`BaseDeploymentPipeline`**: Abstract pipeline for inference (in `pipelines/base/`)
- **`Detection2DPipeline`**: Base pipeline for 2D detection tasks
- **`Detection3DPipeline`**: Base pipeline for 3D detection tasks
- **`ClassificationPipeline`**: Base pipeline for classification tasks

#### 3. **Exporters**

**Unified Architecture**:
- Simple projects rely directly on the base exporters with optional wrappers.
- Complex projects (CenterPoint) assemble workflows (`onnx_workflow.py`, `tensorrt_workflow.py`) that orchestrate multiple single-file exports using the base exporters via composition.

- **Base Exporters** (in `exporters/base/`):
  - **`BaseExporter`**: Abstract base class for all exporters
  - **`ONNXExporter`**: Standard ONNX export with model wrapping support
  - **`TensorRTExporter`**: TensorRT engine building with precision policies
  - **`BaseModelWrapper`**: Abstract base class for model wrappers
  - **`IdentityWrapper`**: Provided wrapper that doesn't modify model output
  - **`configs.py`**: Typed configuration classes:
    - **`ONNXExportConfig`**: Typed schema for ONNX exporter configuration
    - **`TensorRTExportConfig`**: Typed schema for TensorRT exporter configuration
    - **`TensorRTModelInputConfig`**: Configuration for TensorRT input shapes
    - **`TensorRTProfileConfig`**: Optimization profile configuration for dynamic shapes

- **Project-Specific Exporters / Workflows**:
  - **YOLOX** (`exporters/yolox/`):
    - **`ONNXExporter`**: Generic exporter instantiated with `YOLOXONNXWrapper` to adapt outputs
    - **`TensorRTExporter`**: Generic exporter sharing the same wrapper for consistency
    - **`YOLOXONNXWrapper`**: Transforms YOLOX output to Tier4-compatible format
  - **CenterPoint** (`exporters/centerpoint/`):
    - **`CenterPointONNXExportWorkflow`**: Composes the generic `ONNXExporter` to emit multiple ONNX files
    - **`CenterPointTensorRTExportWorkflow`**: Composes the generic `TensorRTExporter` to build multiple engines
    - **`CenterPointONNXWrapper`**: Identity wrapper (no transformation needed)
  - **Calibration** (`exporters/calibration/`):
    - **`ONNXExporter`**: Generic exporter instantiated with the identity `CalibrationONNXWrapper`
    - **`TensorRTExporter`**: Generic exporter instantiated with the same wrapper
    - **`CalibrationONNXWrapper`**: Identity wrapper (no transformation needed)

**Architecture Pattern**:
- **Simple models** (YOLOX, Calibration): Instantiate the generic base exporters with project wrappers; no subclassing required
- **Complex models** (CenterPoint): Keep base exporters generic and layer workflows for multi-file orchestration, still using wrappers as needed

#### 4. **Pipelines**

- **`BaseDeploymentPipeline`**: Abstract base with `preprocess() → run_model() → postprocess()`
- **Task-specific pipelines**: `Detection2DPipeline`, `Detection3DPipeline`, `ClassificationPipeline`
- **Backend implementations**: PyTorch, ONNX, TensorRT variants for each pipeline

---

## Key Features

### 1. Unified Deployment Workflow

All projects follow the same workflow:
```
Load Model → Export ONNX → Export TensorRT → Verify → Evaluate
```

### 2. Scenario-Based Verification

Flexible verification system that compares outputs between backends:

```python
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cpu"},
            {"ref_backend": "onnx", "ref_device": "cpu",
             "test_backend": "tensorrt", "test_device": "cuda:0"},
        ]
    }
)
```

### 3. Multi-Backend Evaluation

Evaluate models across multiple backends with consistent metrics:

```python
evaluation = dict(
    enabled=True,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

### 4. Pipeline Architecture

Shared preprocessing/postprocessing with backend-specific inference:

- **Preprocessing**: Image resize, normalization, voxelization (shared)
  - Can be built from MMDet/MMDet3D configs using `build_preprocessing_pipeline`
  - Used in data loaders to prepare input data
- **Model Inference**: Backend-specific (PyTorch/ONNX/TensorRT)
- **Postprocessing**: NMS, coordinate transform, decoding (shared)

### 5. Flexible Export Modes

- `mode="onnx"`: Export PyTorch → ONNX only
- `mode="trt"`: Build TensorRT from existing ONNX
- `mode="both"`: Full export pipeline
- `mode="none"`: Skip export (evaluation only)

### 6. Precision Policies for TensorRT

Support for different TensorRT precision modes:

- `auto`: TensorRT decides automatically
- `fp16`: FP16 precision
- `fp32_tf32`: FP32 with TF32 acceleration
- `strongly_typed`: Enable strongly-typed TensorRT networks (all tensor dtypes must be explicitly defined; no implicit casting).

---

## Usage

### Basic Usage

```bash
# CenterPoint deployment
python projects/CenterPoint/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py

# YOLOX deployment
python projects/YOLOX_opt_elan/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py

# Calibration deployment
python projects/CalibrationStatusClassification/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py
```

### Creating a Project Runner

All projects follow the dependency injection pattern: explicitly create exporters (with their wrappers) and pass them to a project-specific runner subclass of `BaseDeploymentRunner`. Example (YOLOX):

```python
from deployment.exporters import ONNXExporter, TensorRTExporter
from deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
from deployment.runners import YOLOXDeploymentRunner

# Create exporters with project wrapper
onnx_settings = config.get_onnx_settings()
trt_settings = config.get_tensorrt_settings()

onnx_exporter = ONNXExporter(
    onnx_settings,
    model_wrapper=YOLOXONNXWrapper,
    logger=logger,
)
tensorrt_exporter = TensorRTExporter(
    trt_settings,
    model_wrapper=YOLOXONNXWrapper,
    logger=logger,
)

# Instantiate the project runner
runner = YOLOXDeploymentRunner(
    data_loader=data_loader,
    evaluator=evaluator,
    config=config,
    model_cfg=model_cfg,
    logger=logger,
    onnx_exporter=onnx_exporter,        # Required
    tensorrt_exporter=tensorrt_exporter, # Required
)
```

**Key Points:**
- Exporters (and their wrappers) must be explicitly created in the entry point
- `onnx_exporter` and `tensorrt_exporter` are **required** arguments for every runner
- Each project uses its own specific exporter, wrapper, data loader, evaluator, and runner class
- This explicit wiring keeps dependencies clear and improves testability

### Command-Line Arguments

```bash
python deploy/main.py \
    <deploy_cfg> \          # Deployment configuration file
    <model_cfg> \           # Model configuration file
    [checkpoint] \          # Optional: checkpoint path (can be in config)
    --work-dir <dir> \      # Optional: override work directory
    --device <device> \     # Optional: override device
    --log-level <level>     # Optional: logging level (DEBUG, INFO, WARNING, ERROR)
```

### Export Modes

#### Export ONNX Only
```python
export = dict(
    mode="onnx",
    checkpoint_path="model.pth",
    work_dir="work_dirs/deployment",
)
```

#### Build TensorRT from Existing ONNX
```python
export = dict(
    mode="trt",
    onnx_path="work_dirs/deployment/onnx/model.onnx",
    work_dir="work_dirs/deployment",
)
```

#### Full Export Pipeline
```python
export = dict(
    mode="both",
    checkpoint_path="model.pth",
    work_dir="work_dirs/deployment",
)
```

#### Evaluation Only (No Export)
```python
export = dict(
    mode="none",
    work_dir="work_dirs/deployment",
)
```

---

## Configuration

### Configuration Structure

```python
# Task type
task_type = "detection3d"  # or "detection2d", "classification"

# Export configuration
export = dict(
    mode="both",  # "onnx", "trt", "both", "none"
    work_dir="work_dirs/deployment",
    checkpoint_path="model.pth",
    onnx_path=None,  # Optional: for mode="trt"
)

# Runtime I/O settings
runtime_io = dict(
    info_file="data/info.pkl",  # Dataset info file
    sample_idx=0,  # Sample index for export
)

# Model I/O configuration
model_io = dict(
    input_name="input",
    input_shape=(3, 960, 960),  # (C, H, W)
    input_dtype="float32",
    output_name="output",
    batch_size=1,  # or None for dynamic
    dynamic_axes={...},  # When batch_size=None
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    save_file="model.onnx",
    multi_file=False,  # True for multi-file ONNX (e.g., CenterPoint)
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        precision_policy="auto",  # "auto", "fp16", "fp32_tf32", "strongly_typed"
        max_workspace_size=1 << 30,  # 1 GB
    ),
)

# Verification configuration
verification = dict(
    enabled=True,
    num_verify_samples=3,
    tolerance=0.1,
    devices={
        "cpu": "cpu",
        "cuda": "cuda:0",
    },
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cpu"},
        ]
    }
)

# Evaluation configuration
evaluation = dict(
    enabled=True,
    num_samples=100,  # or -1 for all samples
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

#### Backend Enum

To avoid backend name typos, `deployment.core.Backend` enumerates the supported values:

```python
from deployment.core import Backend

evaluation = dict(
    backends={
        Backend.PYTORCH: {"enabled": True, "device": "cpu"},
        Backend.ONNX: {"enabled": True, "device": "cpu"},
        Backend.TENSORRT: {"enabled": True, "device": "cuda:0"},
    }
)
```

Configuration dictionaries accept either raw strings or `Backend` enum members, so teams can adopt the enum incrementally without breaking existing configs.

#### Typed Exporter Configurations

The framework provides typed configuration classes in `deployment.exporters.base.configs` for better type safety and validation:

```python
from deployment.exporters.base.configs import (
    ONNXExportConfig,
    TensorRTExportConfig,
    TensorRTModelInputConfig,
    TensorRTProfileConfig,
)

# ONNX configuration with typed schema
onnx_config = ONNXExportConfig(
    input_names=("input",),
    output_names=("output",),
    opset_version=16,
    do_constant_folding=True,
    simplify=True,
    save_file="model.onnx",
    batch_size=1,
)

# TensorRT configuration with typed schema
trt_config = TensorRTExportConfig(
    precision_policy="auto",
    max_workspace_size=1 << 30,
    model_inputs=(
        TensorRTModelInputConfig(
            input_shapes={
                "input": TensorRTProfileConfig(
                    min_shape=(1, 3, 960, 960),
                    opt_shape=(1, 3, 960, 960),
                    max_shape=(1, 3, 960, 960),
                )
            }
        ),
    ),
)
```

These typed configs can be created from dictionaries using `from_mapping()` or `from_dict()` class methods, providing a bridge between configuration files and type-safe code.

### Configuration Examples

See project-specific configs:
- `projects/CenterPoint/deploy/configs/deploy_config.py`
- `projects/YOLOX_opt_elan/deploy/configs/deploy_config.py`
- `projects/CalibrationStatusClassification/deploy/configs/deploy_config.py`

---

## Project-Specific Implementations

### CenterPoint (3D Detection)

**Features:**
- Multi-file ONNX export (voxel encoder + backbone/head) orchestrated via workflows
- ONNX-compatible model configuration
- Composed exporters for complex model structure

**Workflows and Wrapper:**
- `CenterPointONNXExportWorkflow`: Drives multiple ONNX exports using the generic `ONNXExporter`
- `CenterPointTensorRTExportWorkflow`: Converts each ONNX file with the generic `TensorRTExporter`
- `CenterPointONNXWrapper`: Identity wrapper (no output transformation)

**Key Files:**
- `projects/CenterPoint/deploy/main.py`
- `projects/CenterPoint/deploy/evaluator.py`
- `deployment/pipelines/centerpoint/`
- `deployment/exporters/centerpoint/onnx_workflow.py`
- `deployment/exporters/centerpoint/tensorrt_workflow.py`
- `deployment/exporters/centerpoint/model_wrappers.py`

**Pipeline Structure:**
```
preprocess() → run_voxel_encoder() → process_middle_encoder() →
run_backbone_head() → postprocess()
```

### YOLOX (2D Detection)

**Features:**
- Standard single-file ONNX export
- Model wrapper for ONNX-compatible output format
- ReLU6 → ReLU replacement for ONNX compatibility

**Export + Wrapper:**
- `ONNXExporter`: Generic exporter instantiated with `YOLOXONNXWrapper`
- `TensorRTExporter`: Generic exporter instantiated with the same wrapper
- `YOLOXONNXWrapper`: Transforms output from `(1, 8, 120, 120)` to `(1, 18900, 13)` format

**Key Files:**
- `projects/YOLOX_opt_elan/deploy/main.py`
- `projects/YOLOX_opt_elan/deploy/evaluator.py`
- `deployment/pipelines/yolox/`
- `deployment/exporters/yolox/model_wrappers.py`

**Pipeline Structure:**
```
preprocess() → run_model() → postprocess()
```

### CalibrationStatusClassification

**Features:**
- Binary classification task
- Simple single-file ONNX export
- Calibrated/miscalibrated data loader variants

**Export + Wrapper:**
- `ONNXExporter`: Generic exporter instantiated with `CalibrationONNXWrapper`
- `TensorRTExporter`: Generic exporter instantiated with the same wrapper
- `CalibrationONNXWrapper`: Identity wrapper (no output transformation)

**Key Files:**
- `projects/CalibrationStatusClassification/deploy/main.py`
- `projects/CalibrationStatusClassification/deploy/evaluator.py`
- `deployment/pipelines/calibration/`
- `deployment/exporters/calibration/model_wrappers.py`

**Pipeline Structure:**
```
preprocess() → run_model() → postprocess()
```

---

## Pipeline Architecture

### Base Pipeline

All pipelines inherit from `BaseDeploymentPipeline` (located in `pipelines/base/base_pipeline.py`):

```python
class BaseDeploymentPipeline(ABC):
    @abstractmethod
    def preprocess(self, input_data, **kwargs) -> Any:
        """Preprocess input data"""
        pass

    @abstractmethod
    def run_model(self, preprocessed_input) -> Any:
        """Backend-specific model inference"""
        pass

    @abstractmethod
    def postprocess(self, model_output, metadata) -> Any:
        """Postprocess model output"""
        pass

    def infer(self, input_data, **kwargs):
        """Complete inference pipeline"""
        preprocessed = self.preprocess(input_data, **kwargs)
        model_output = self.run_model(preprocessed)
        predictions = self.postprocess(model_output, metadata)
        return predictions
```

### Task-Specific Base Pipelines

Located in `pipelines/base/`, these provide task-specific abstractions:

#### Detection2DPipeline (`pipelines/base/detection_2d_pipeline.py`)
- Shared preprocessing: image resize, normalization, padding
- Shared postprocessing: bbox decoding, NMS, coordinate transform
- Backend-specific: model inference

#### Detection3DPipeline (`pipelines/base/detection_3d_pipeline.py`)
- Shared preprocessing: voxelization, feature extraction
- Shared postprocessing: 3D bbox decoding, NMS
- Backend-specific: voxel encoder, backbone/head inference

#### ClassificationPipeline (`pipelines/base/classification_pipeline.py`)
- Shared preprocessing: image normalization
- Shared postprocessing: softmax, top-k selection
- Backend-specific: model inference

### Backend Implementations

Each pipeline has three backend implementations:

1. **PyTorch Pipeline**: Direct PyTorch model inference
2. **ONNX Pipeline**: ONNX Runtime inference
3. **TensorRT Pipeline**: TensorRT engine inference

Example for YOLOX:
- `YOLOXPyTorchPipeline`: Uses PyTorch model directly
- `YOLOXONNXPipeline`: Uses ONNX Runtime
- `YOLOXTensorRTPipeline`: Uses TensorRT engine

---

## Export Workflow

### ONNX Export

1. **Model Preparation**: Load PyTorch model, apply model wrapper if needed
2. **Input Preparation**: Get sample input from data loader
3. **Export**: Call `torch.onnx.export()` with configured settings
4. **Simplification**: Optional ONNX model simplification
5. **Save**: Save to `work_dir/onnx/`

### TensorRT Export

1. **ONNX Validation**: Verify ONNX model exists
2. **Network Creation**: Parse ONNX, create TensorRT network
3. **Precision Configuration**: Apply precision policy flags
4. **Optimization Profile**: Configure input shape ranges
5. **Engine Building**: Build and save TensorRT engine
6. **Save**: Save to `work_dir/tensorrt/`

### Multi-File Export (CenterPoint)

CenterPoint uses a multi-file ONNX structure:
- `voxel_encoder.onnx`: Voxel feature extraction
- `backbone_head.onnx`: Backbone and detection head

The exporter handles:
- Sequential export of each component
- Proper input/output linking
- Directory-based organization

---

## Verification & Evaluation

### Verification

Policy-based verification compares outputs between backends:

```python
# Verification scenarios example
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {
                "ref_backend": "pytorch",
                "ref_device": "cpu",
                "test_backend": "onnx",
                "test_device": "cpu"
            }
        ]
    },
    tolerance=0.1,  # Maximum allowed difference
    num_verify_samples=3
)
```

**Verification Process:**
1. Load reference model (PyTorch or ONNX)
2. Load test model (ONNX or TensorRT)
3. Run inference on same samples
4. Compare outputs with tolerance
5. Report pass/fail for each sample

### Evaluation

Task-specific evaluation with consistent metrics:

**Detection Tasks:**
- mAP (mean Average Precision)
- Per-class AP
- Latency statistics

**Classification Tasks:**
- Accuracy
- Precision, Recall
- Per-class metrics
- Confusion matrix
- Latency statistics

**Evaluation Configuration:**
```python
evaluation = dict(
    enabled=True,
    num_samples=100,  # or -1 for all
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

---

## File Structure

```
deployment/
├── core/                          # Core base classes and utilities
│   ├── artifacts.py               # Artifact descriptors (ONNX/TensorRT outputs)
│   ├── backend.py                 # Backend enum (PyTorch, ONNX, TensorRT)
│   ├── base_config.py             # Configuration management
│   ├── base_data_loader.py        # Data loader interface
│   ├── base_evaluator.py          # Evaluator interface
│   └── preprocessing_builder.py   # Preprocessing pipeline builder
│
├── exporters/                     # Model exporters (unified structure)
│   ├── base/                      # Base exporter classes
│   │   ├── base_exporter.py       # Exporter base class
│   │   ├── configs.py             # Typed configuration classes (ONNXExportConfig, TensorRTExportConfig)
│   │   ├── onnx_exporter.py       # ONNX exporter base class
│   │   ├── tensorrt_exporter.py   # TensorRT exporter base class
│   │   └── model_wrappers.py      # Base model wrappers (BaseModelWrapper, IdentityWrapper)
│   ├── centerpoint/               # CenterPoint exporters (extends base)
│   │   ├── onnx_exporter.py       # CenterPoint ONNX exporter (multi-file export)
│   │   ├── tensorrt_exporter.py   # CenterPoint TensorRT exporter (multi-file export)
│   │   └── model_wrappers.py      # CenterPoint model wrappers (IdentityWrapper)
│   ├── yolox/                     # YOLOX exporters (inherits base)
│   │   ├── onnx_exporter.py       # YOLOX ONNX exporter (inherits base)
│   │   ├── tensorrt_exporter.py   # YOLOX TensorRT exporter (inherits base)
│   │   └── model_wrappers.py      # YOLOX model wrappers (YOLOXONNXWrapper)
│   └── calibration/               # CalibrationStatusClassification exporters (inherits base)
│       ├── onnx_exporter.py       # Calibration ONNX exporter (inherits base)
│       ├── tensorrt_exporter.py   # Calibration TensorRT exporter (inherits base)
│       └── model_wrappers.py      # Calibration model wrappers (IdentityWrapper)
│
├── pipelines/                     # Task-specific pipelines
│   ├── base/                      # Base pipeline classes
│   │   ├── base_pipeline.py       # Pipeline base class
│   │   ├── detection_2d_pipeline.py   # 2D detection pipeline
│   │   ├── detection_3d_pipeline.py   # 3D detection pipeline
│   │   └── classification_pipeline.py # Classification pipeline
│   ├── centerpoint/               # CenterPoint pipelines
│   │   ├── centerpoint_pipeline.py
│   │   ├── centerpoint_pytorch.py
│   │   ├── centerpoint_onnx.py
│   │   └── centerpoint_tensorrt.py
│   ├── yolox/                     # YOLOX pipelines
│   │   ├── yolox_pipeline.py
│   │   ├── yolox_pytorch.py
│   │   ├── yolox_onnx.py
│   │   └── yolox_tensorrt.py
│   └── calibration/               # Calibration pipelines
│       ├── calibration_pipeline.py
│       ├── calibration_pytorch.py
│       ├── calibration_onnx.py
│       └── calibration_tensorrt.py
│
└── runners/                       # Deployment runners
    ├── deployment_runner.py       # BaseDeploymentRunner
    ├── centerpoint_runner.py      # CenterPointDeploymentRunner
    ├── yolox_runner.py            # YOLOXDeploymentRunner
    └── calibration_runner.py      # CalibrationDeploymentRunner

projects/
├── CenterPoint/deploy/
│   ├── main.py                    # Entry point
│   ├── evaluator.py               # CenterPoint evaluator
│   ├── data_loader.py             # CenterPoint data loader
│   └── configs/
│       └── deploy_config.py       # Deployment configuration
│
├── YOLOX_opt_elan/deploy/
│   ├── main.py
│   ├── evaluator.py
│   ├── data_loader.py
│   └── configs/
│       └── deploy_config.py
│
└── CalibrationStatusClassification/deploy/
    ├── main.py
    ├── evaluator.py
    ├── data_loader.py
    └── configs/
        └── deploy_config.py
```

---

## Best Practices

### 1. Configuration Management

- Keep deployment configs separate from model configs
- Use relative paths for data files
- Document all configuration options

### 2. Model Export

- Always explicitly create project-specific exporters in `main.py`
- Always provide required `model_wrapper` parameter when constructing exporters
- Use project-specific wrapper classes (e.g., `YOLOXONNXWrapper`, `CenterPointONNXWrapper`)
- Follow the unified architecture pattern: simple models keep `{model}/onnx_exporter.py`, `{model}/tensorrt_exporter.py`, `{model}/model_wrappers.py`; complex models add workflow modules that compose the base exporters.
- Simple models: inherit base exporters, use custom wrappers if needed
- Complex models: build workflow classes that drive multiple calls into the generic exporters, use IdentityWrapper if no transformation needed
- Always verify ONNX export before TensorRT conversion
- Use appropriate precision policies for TensorRT
- Test with multiple samples during export

### 2.1. Unified Architecture Pattern

All projects follow a unified structure, with simple models sticking to exporter modules and complex models layering workflows on top:

```
exporters/{model}/
├── model_wrappers.py      # Project-specific model wrapper
├── [optional] onnx_workflow.py       # Workflow orchestrating base exporter calls
├── [optional] tensorrt_workflow.py   # Workflow orchestrating base exporter calls
```

**Pattern 1: Simple Models** (YOLOX, Calibration)
- Instantiate the generic base exporters (no subclassing needed)
- Use custom wrappers if output format transformation is required
- Example: YOLOX uses `ONNXExporter` + `YOLOXONNXWrapper`

**Pattern 2: Complex Models** (CenterPoint)
- Keep base exporters generic but introduce workflow classes for special requirements (e.g., multi-file export)
- Use IdentityWrapper if no output transformation needed
- Example: `CenterPointONNXExportWorkflow` composes `ONNXExporter` to produce multiple ONNX files

### 2.2. Dependency Injection Pattern

All projects should follow this pattern:

```python
# 1. Import exporters, wrappers, and runner
from deployment.exporters import ONNXExporter, TensorRTExporter
from deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
from deployment.runners import YOLOXDeploymentRunner

# 2. Create exporters with settings
onnx_exporter = ONNXExporter(
    onnx_settings,
    model_wrapper=YOLOXONNXWrapper,
    logger=logger,
)
tensorrt_exporter = TensorRTExporter(
    trt_settings,
    model_wrapper=YOLOXONNXWrapper,
    logger=logger,
)

# 3. Pass exporters to the project runner (all required)
runner = YOLOXDeploymentRunner(
    ...,
    onnx_exporter=onnx_exporter,        # Required
    tensorrt_exporter=tensorrt_exporter, # Required
)
```

**Benefits:**
- Clear dependencies: All components are visible in `main.py`
- Type safety: IDE can provide better type hints
- No hidden dependencies: No global registry or string-based lookups
- Easy testing: Can inject mock objects for testing
- Unified structure: All models follow the same architectural pattern

### 3. Verification

- Start with small tolerance (0.01) and increase if needed
- Verify on representative samples
- Check both accuracy and numerical differences

### 4. Evaluation

- Use consistent evaluation settings across backends
- Report latency statistics (mean, std, min, max)
- Compare metrics across backends

### 5. Pipeline Development

- Inherit from appropriate base pipeline
- Share preprocessing/postprocessing logic
- Keep backend-specific code minimal

---

## Troubleshooting

### Common Issues

1. **ONNX Export Fails**
   - Check model compatibility (unsupported ops)
   - Verify input shapes match model expectations
   - Try different opset versions

2. **TensorRT Build Fails**
   - Verify ONNX model is valid
   - Check input shape configuration
   - Reduce workspace size if memory issues

3. **Verification Fails**
   - Check tolerance settings
   - Verify same preprocessing for all backends
   - Check device compatibility

4. **Evaluation Errors**
   - Verify data loader paths
   - Check model output format
   - Ensure correct task type in config

---

## Future Enhancements

- [ ] Support for more task types (segmentation, etc.)
- [ ] Automatic precision tuning for TensorRT
- [ ] Distributed evaluation support
- [ ] Integration with MLOps pipelines
- [ ] Performance profiling tools

---

## Contributing

When adding a new project:

1. **Create project-specific evaluator and data loader**
   - Implement `BaseEvaluator` for task-specific metrics
   - Implement `BaseDataLoader` for data loading

2. **Create exporters following unified architecture pattern**
   - Create `exporters/{model}/onnx_exporter.py` (inherit or extend `ONNXExporter`)
   - Create `exporters/{model}/tensorrt_exporter.py` (inherit or extend `TensorRTExporter`)
   - Create `exporters/{model}/model_wrappers.py` (use `IdentityWrapper` or implement custom wrapper)
   - **Simple models**: Inherit base exporters, use custom wrapper if output transformation needed
   - **Complex models**: Extend base exporters for special logic (e.g., multi-file export)

3. **Implement task-specific pipeline** (if needed)
   - Inherit from appropriate base pipeline (`Detection2DPipeline`, `Detection3DPipeline`, `ClassificationPipeline`)
   - Implement backend-specific variants (PyTorch, ONNX, TensorRT)

4. **Create deployment configuration**
   - Add `projects/{project}/deploy/configs/deploy_config.py`
   - Configure export, verification, and evaluation settings

5. **Add entry point script**
   - Create `projects/{project}/deploy/main.py`
   - Follow dependency injection pattern: explicitly create exporters and wrappers
   - Pass exporters to the appropriate project runner (inherits `BaseDeploymentRunner`)

6. **Update documentation**
   - Add project to README's "Project-Specific Implementations" section
   - Document any special requirements or configurations

---

## License

See LICENSE file in project root.
