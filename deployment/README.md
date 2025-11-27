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
- [Core Contract](#core-contract)

---

## Overview

The AWML Deployment Framework provides a standardized approach to model deployment across different projects (CenterPoint, YOLOX, CalibrationStatusClassification). It abstracts common deployment workflows while allowing project-specific customizations.

### Design Principles

1. **Unified Interface**: Shared base runner (`BaseDeploymentRunner`) with project-specific subclasses
2. **Task-Agnostic Core**: Base classes that work across detection, classification, and segmentation
3. **Backend Flexibility**: Support for PyTorch, ONNX, and TensorRT backends
4. **Pipeline Architecture**: Shared preprocessing/postprocessing with backend-specific inference
5. **Configuration-Driven**: All settings controlled via config files plus typed dataclasses for defaults
6. **Dependency Injection**: Explicit creation and injection of exporters and wrappers for better clarity and type safety
7. **Type-Safe Building Blocks** *(new)*: Task configs, runtime configs, and typed result objects ensure IDE support and prevent runtime surprises
8. **Extensible Verification** *(new)*: Verification mixin compares arbitrary nested outputs, keeping evaluators thin and future-proof

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
│  - Coordinates load → export → verify → evaluate        │
│  - Delegates to helper orchestrators (see below)        │
│  - Each project extends the base class for custom logic │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
┌───────▼────────┐     ┌────────▼───────────────┐
│   Exporters    │     │  Helper Orchestrators │
│  - ONNX / TRT  │     │  - ArtifactManager     │
│  - Wrappers    │     │  - VerificationOrch.   │
│  - Workflows   │     │  - EvaluationOrch.     │
└────────────────┘     └────────┬───────────────┘
                                │
┌───────────────────────────────▼─────────────────────────┐
│                    Evaluators & Pipelines               │
│  - BaseDeploymentPipeline + task-specific variants      │
│  - Backend-specific implementations (PyTorch/ONNX/TRT)  │
└────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **BaseDeploymentRunner & Project Runners**
`BaseDeploymentRunner` orchestrates the complete deployment workflow, while each project provides a thin subclass (`CenterPointDeploymentRunner`, `YOLOXOptElanDeploymentRunner`, `CalibrationDeploymentRunner`) that plugs in model-specific logic.

- **ArtifactManager**: registers and resolves PyTorch/ONNX/TensorRT artifacts
- **VerificationOrchestrator**: runs scenario-based verification (pytorch ↔ onnx ↔ tensorrt)
- **EvaluationOrchestrator**: runs cross-backend evaluation and summarizes metrics

This keeps project runners lightweight—they primarily:

- Implement **Model Loading** for their project
- Plug in optional **export workflows** (e.g., CenterPoint multi-file export)
- Supply **wrapper classes** when ONNX outputs need reshaping

**Required Parameters:**
- `onnx_wrapper_cls`: Optional model wrapper class for ONNX export (required unless a workflow performs the export)
- `onnx_workflow` / `tensorrt_workflow`: Optional workflow objects for specialized multi-file exports

Runners still own exporter initialization via `ExporterFactory`, but orchestrators keep the run loop simple and testable.

**Directory layout**
- `deployment/runners/common/`: shared runner infrastructure (`BaseDeploymentRunner`, orchestrators, `ArtifactManager`)
- `deployment/runners/projects/`: thin adapters for each project (CenterPoint, YOLOX, Calibration, …)

#### 2. **Core Components** (in `core/`)

- **`BaseDeploymentConfig`**: Configuration container for deployment settings
- **`Backend`**: Enum for supported backends (PyTorch, ONNX, TensorRT)
- **`Artifact`**: Dataclass representing deployment artifacts (ONNX/TensorRT outputs)
- **`BaseEvaluator`**: Abstract interface for task-specific evaluation
- **`VerificationMixin`**: Shared verification workflow with recursive comparison that supports any future output structure
- **`BaseDataLoader`**: Abstract interface for data loading
- **`build_preprocessing_pipeline`**: Utility to extract preprocessing pipelines from MMDet/MMDet3D configs
- **Typed configuration/value objects**:
  - `constants.py`: Centralized defaults for export/evaluation/task metadata
  - `runtime_config.py`: `Detection3D/2D/ClassificationRuntimeConfig` dataclasses for strongly-typed runtime I/O
  - `task_config.py`: Immutable `TaskConfig` describing per-task pipeline knobs
  - `results.py`: Typed prediction/evaluation/latency structures for consistent reporting
- **`BaseDeploymentPipeline`**: Abstract pipeline for inference (in `pipelines/common/`)
- **`PipelineFactory`**: Centralized creation of backend-specific pipelines to avoid per-evaluator duplication
- **`Detection2DPipeline`**: Base pipeline for 2D detection tasks
- **`Detection3DPipeline`**: Base pipeline for 3D detection tasks
- **`ClassificationPipeline`**: Base pipeline for classification tasks

#### 3. **Exporters & Workflows**

**Unified Architecture**:
- Exporters are created lazily by `ExporterFactory`, so project entry points only declare wrappers/workflows and never wire exporters manually.
- Base workflow interfaces live in `exporters/workflows/base.py`, enabling complex projects to orchestrate multi-stage exports without forking the base exporters.
- Simple projects rely directly on the base exporters with optional wrappers.
- Complex projects (CenterPoint) assemble workflows (`onnx_workflow.py`, `tensorrt_workflow.py`) that orchestrate multiple single-file exports using the base exporters via composition.

- **Base Exporters** (in `exporters/common/`):
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

- **Factory & Workflow Interfaces**:
- **`ExporterFactory`** (`exporters/common/factory.py`): Builds `ONNXExporter`/`TensorRTExporter` instances using `BaseDeploymentConfig` settings, ensuring consistent logging and configuration.
  - **`OnnxExportWorkflow` / `TensorRTExportWorkflow`** (`exporters/workflows/base.py`): Abstract contracts for orchestrating complex, multi-artifact exports.

- **Project-Specific Wrappers & Workflows**:
  - **YOLOX** (`exporters/yolox/`):
    - **`YOLOXOptElanONNXWrapper`**: Transforms YOLOX output to Tier4-compatible format; paired with base exporters created by the factory.
  - **CenterPoint** (`exporters/centerpoint/`):
    - **`CenterPointONNXExportWorkflow`**: Composes the generic `ONNXExporter` to emit multiple ONNX files
    - **`CenterPointTensorRTExportWorkflow`**: Composes the generic `TensorRTExporter` to build multiple engines
    - **`CenterPointONNXWrapper`**: Identity wrapper (no transformation needed)
  - **Calibration** (`exporters/calibration/`):
    - **`CalibrationONNXWrapper`**: Identity wrapper (no transformation needed); paired with the base exporters from the factory

**Architecture Pattern**:
- **Simple models** (YOLOX, Calibration): Instantiate the generic base exporters via `ExporterFactory` and supply custom wrappers when needed; no subclassing required.
- **Complex models** (CenterPoint): Keep base exporters generic and layer workflows for multi-file orchestration, still using wrappers as needed.

#### 4. **Pipelines**

- **`BaseDeploymentPipeline`**: Abstract base with `preprocess() → run_model() → postprocess()`
- **`PipelineFactory`** *(new)*: Single entry point for building CenterPoint / YOLOX / Calibration pipelines across backends
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

Flexible verification system that compares outputs between backends. The shared `VerificationMixin` now:

- Normalizes devices per backend (PyTorch ↔ CPU, TensorRT ↔ `cuda:0`, etc.)
- Builds reference/test pipelines through `PipelineFactory`
- Recursively compares arbitrary nested outputs (dicts, lists, tensors, scalars) with per-node logging
- Allows evaluators to optionally name multi-head outputs via `_get_output_names()`

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

Evaluate models across multiple backends with consistent, typed metrics. Evaluation outputs now share the same dataclasses (`Detection3DEvaluationMetrics`, `Detection2DEvaluationMetrics`, `ClassificationEvaluationMetrics`) ensuring stable JSON structure.

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

Projects now pass lightweight configuration objects (wrapper classes and optional workflows) into the runner. The runner owns exporter construction through `ExporterFactory` and creates the exporters lazily. Example (YOLOX):

```python
from deployment.exporters.yolox.model_wrappers import YOLOXOptElanONNXWrapper
from deployment.runners import YOLOXOptElanDeploymentRunner

# Instantiate the project runner
runner = YOLOXOptElanDeploymentRunner(
    data_loader=data_loader,
    evaluator=evaluator,
    config=config,
    model_cfg=model_cfg,
    logger=logger,
    onnx_wrapper_cls=YOLOXOptElanONNXWrapper,
)
```

**Key Points:**
- Pass wrapper classes (and optional workflows) instead of exporter instances
- Exporters are constructed lazily inside `BaseDeploymentRunner` via `ExporterFactory`
- Projects still control model-specific behavior by choosing wrappers/workflows
- Entry points remain simple while keeping dependencies explicit

### Typed Context Objects

The framework uses typed context objects for passing parameters through the deployment workflow. This provides:

- **Type Safety**: Catches mismatches at type-check time
- **Discoverability**: IDE autocomplete shows available parameters
- **Refactoring Safety**: Renamed fields are caught by type checkers

```python
from deployment.core import ExportContext, YOLOXExportContext, CenterPointExportContext

# Base context (for simple projects)
ctx = ExportContext(sample_idx=0)

# YOLOX-specific context
ctx = YOLOXExportContext(
    sample_idx=0,
    model_cfg_path="/path/to/config.py",
)

# CenterPoint-specific context
ctx = CenterPointExportContext(
    sample_idx=0,
    rot_y_axis_reference=True,
)

# Run deployment with typed context
results = runner.run(context=ctx)
```

**Available Contexts:**
- `ExportContext`: Base context with `sample_idx` and `extra` dict for ad-hoc options
- `YOLOXExportContext`: Adds `model_cfg_path` for YOLOX model configuration
- `CenterPointExportContext`: Adds `rot_y_axis_reference` for rotation format
- `CalibrationExportContext`: For calibration models (extends base)

**Default Context:**
If no context is provided, a default `ExportContext()` is created automatically:
```python
# These are equivalent:
results = runner.run()
results = runner.run(context=ExportContext())
```

**Creating Custom Contexts:**
```python
from dataclasses import dataclass
from deployment.core import ExportContext

@dataclass(frozen=True)
class MyProjectExportContext(ExportContext):
    """My project-specific export context."""
    my_custom_param: bool = False
    another_param: str = "default"
```

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
# Top-level checkpoint path - single source of truth
checkpoint_path = "model.pth"

export = dict(
    mode="onnx",
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
# Top-level checkpoint path - single source of truth
checkpoint_path = "model.pth"

export = dict(
    mode="both",
    work_dir="work_dirs/deployment",
)
```

#### Evaluation Only (No Export)
```python
# Top-level checkpoint path - used for PyTorch evaluation
checkpoint_path = "model.pth"

export = dict(
    mode="none",
    work_dir="work_dirs/deployment",
)
```

---

## Configuration

### Configuration Structure

Configuration stays dictionary-driven for flexibility, but the framework now layers typed dataclasses on top:

 - **Defaults** pulled from `deployment.core.config.constants`
- **Runtime I/O** captured through `runtime_config.py`
- **Task behavior** modeled via immutable `TaskConfig`
- **Results & metrics** expressed as structured dataclasses (`results.py`)

```python
# Task type
task_type = "detection3d"  # or "detection2d", "classification"

# ============================================================================
# Checkpoint Path - Single source of truth for PyTorch model
# ============================================================================
# This is the main checkpoint path used by:
# - Export workflow: to load the PyTorch model for ONNX conversion
# - Evaluation: for PyTorch backend evaluation
# - Verification: when PyTorch is used as reference or test backend
checkpoint_path = "model.pth"

# Export configuration
export = dict(
    mode="both",  # "onnx", "trt", "both", "none"
    work_dir="work_dirs/deployment",
    onnx_path=None,  # Optional: for mode="trt" when using existing ONNX
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
# Note: PyTorch backend uses top-level checkpoint_path (no need to specify here)
evaluation = dict(
    enabled=True,
    num_samples=100,  # or -1 for all samples
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},  # Uses checkpoint_path
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

The framework provides typed configuration classes in `deployment.exporters.common.configs` for better type safety and validation:

```python
from deployment.exporters.common.configs import (
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
- `projects/CenterPoint/deploy/evaluator.py` *(now inherits `VerificationMixin` and uses `PipelineFactory`)*
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
- `ONNXExporter`: Generic exporter instantiated with `YOLOXOptElanONNXWrapper`
- `TensorRTExporter`: Generic exporter instantiated with the same wrapper
- `YOLOXOptElanONNXWrapper`: Transforms output from `(1, 8, 120, 120)` to `(1, 18900, 13)` format

**Key Files:**
- `projects/YOLOX_opt_elan/deploy/main.py`
- `projects/YOLOX_opt_elan/deploy/evaluator.py` *(shares verification/pipeline creation through mixin + factory)*
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
- `projects/CalibrationStatusClassification/deploy/evaluator.py` *(mix-in verification, typed metrics)*
- `deployment/pipelines/calibration/`
- `deployment/exporters/calibration/model_wrappers.py`

**Pipeline Structure:**
```
preprocess() → run_model() → postprocess()
```

---

## Pipeline Architecture

### Base Pipeline

All pipelines inherit from `BaseDeploymentPipeline` (located in `pipelines/common/base_pipeline.py`). Most projects now obtain backend-specific pipeline instances from `PipelineFactory`, keeping evaluator logic minimal:

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

Located in `pipelines/common/`, these provide task-specific abstractions:

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

Policy-based verification compares outputs between backends using the shared mixin infrastructure:

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

---

## Core Contract

The responsibilities and allowed dependencies between runners, evaluators, pipelines, the `PipelineFactory`, and metrics adapters are defined in [`deployment/CORE_CONTRACT.md`](./CORE_CONTRACT.md). Adhering to that contract keeps refactors safe and makes it clear where new logic belongs.

**Verification Process:**
1. Load reference model (PyTorch or ONNX)
2. Load test model (ONNX or TensorRT)
3. Run inference on same samples via pipelines resolved by `PipelineFactory`
4. Recursively compare outputs with per-node tolerance
5. Report pass/fail for each sample with structured stats

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
│   ├── contexts.py                # Typed context objects (ExportContext, etc.)
│   ├── config/                    # Configuration subpackage
│   │   ├── base_config.py         # Deployment config container + CLI helpers
│   │   ├── constants.py           # Framework-wide defaults (evaluation, export, task)
│   │   ├── runtime_config.py      # Typed runtime I/O configurations
│   │   └── task_config.py         # Immutable per-task configuration
│   ├── evaluation/                # Evaluation subpackage
│   │   ├── base_evaluator.py      # Evaluator base class + TaskProfile
│   │   ├── evaluator_types.py     # Shared type definitions
│   │   ├── results.py             # Typed prediction/metrics data structures
│   │   └── verification_mixin.py  # Shared verification workflow
│   └── io/                        # Data loading + preprocessing utilities
│       ├── base_data_loader.py    # Data loader interface
│       └── preprocessing_builder.py   # Preprocessing pipeline builder
│
├── exporters/                     # Model exporters (unified structure)
│   ├── common/                    # Shared exporter classes
│   │   ├── base_exporter.py       # Exporter base class
│   │   ├── configs.py             # Typed configuration classes (ONNXExportConfig, TensorRTExportConfig)
│   │   ├── factory.py             # ExporterFactory that builds ONNX/TensorRT exporters
│   │   ├── onnx_exporter.py       # ONNX exporter base class
│   │   ├── tensorrt_exporter.py   # TensorRT exporter base class
│   │   └── model_wrappers.py      # Base model wrappers (BaseModelWrapper, IdentityWrapper)
│   ├── workflows/                 # Workflow interfaces
│   │   └── base.py                # OnnxExportWorkflow & TensorRTExportWorkflow ABCs
│   ├── centerpoint/               # CenterPoint-specific helpers (compose base exporters)
│   │   ├── model_wrappers.py      # CenterPoint model wrappers (IdentityWrapper)
│   │   ├── onnx_workflow.py       # CenterPoint multi-file ONNX workflow
│   │   └── tensorrt_workflow.py   # CenterPoint multi-file TensorRT workflow
│   ├── yolox/                     # YOLOX wrappers (paired with base exporters)
│   │   └── model_wrappers.py      # YOLOX model wrappers (YOLOXOptElanONNXWrapper)
│   └── calibration/               # CalibrationStatusClassification wrappers
│       └── model_wrappers.py      # Calibration model wrappers (IdentityWrapper)
│
├── pipelines/                     # Task-specific pipelines
│   ├── common/                    # Shared pipeline classes
│   │   ├── base_pipeline.py       # Pipeline base class
│   │   └── factory.py             # PipelineFactory (backend-agnostic construction)
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
    ├── common/                    # Shared orchestration logic
    │   ├── deployment_runner.py   # BaseDeploymentRunner
    │   ├── artifact_manager.py    # Artifact registration/lookup
    │   ├── evaluation_orchestrator.py
    │   └── verification_orchestrator.py
    └── projects/                  # Thin adapters per project
        ├── centerpoint_runner.py  # CenterPointDeploymentRunner
        ├── yolox_runner.py        # YOLOXOptElanDeploymentRunner
        └── calibration_runner.py  # CalibrationDeploymentRunner

projects/
├── CenterPoint/deploy/
│   ├── main.py                    # Entry point
│   ├── evaluator.py               # CenterPoint evaluator (mix-in + typed metrics)
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

- Pass wrapper classes (and optional workflows) into project runners; `ExporterFactory` constructs ONNX/TensorRT exporters using deployment configs.
- Keep wrapper definitions in `exporters/{model}/model_wrappers.py`; reuse `IdentityWrapper` when no transformation is needed.
- Introduce workflow modules (`exporters/{model}/onnx_workflow.py`, `tensorrt_workflow.py`) only when orchestration beyond single-file export is required.
- Simple models: rely on generic base exporters + wrappers; no subclassing or custom exporters.
- Complex models: implement workflow classes that drive multiple calls into the generic exporters while keeping exporter logic centralized.
- Always verify ONNX export before TensorRT conversion and prefer multiple samples to validate stability.
- Use appropriate precision policies for TensorRT (auto/fp16/fp32_tf32/strongly_typed) based on deployment constraints.

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
- Example: YOLOX uses `ONNXExporter` + `YOLOXOptElanONNXWrapper`

**Pattern 2: Complex Models** (CenterPoint)
- Keep base exporters generic but introduce workflow classes for special requirements (e.g., multi-file export)
- Use IdentityWrapper if no output transformation needed
- Example: `CenterPointONNXExportWorkflow` composes `ONNXExporter` to produce multiple ONNX files

### 2.2. Dependency Injection Pattern

All projects should follow this pattern:

```python
# 1. Import wrappers/workflows and runner
from deployment.exporters.yolox.model_wrappers import YOLOXOptElanONNXWrapper
from deployment.runners import YOLOXOptElanDeploymentRunner

# 2. Instantiate the runner with wrapper classes (TensorRT uses base exporter directly)
runner = YOLOXOptElanDeploymentRunner(
    ...,
    onnx_wrapper_cls=YOLOXOptElanONNXWrapper,
)
```

Complex projects (e.g., CenterPoint) can additionally provide workflow instances, which the runner will use before falling back to the standard exporter flow.

**Benefits:**
- Clear dependencies: All components and hooks are visible in `main.py`
- Lazy exporter creation: Avoids redundant exporter wiring across projects
- No hidden dependencies: No global registry or string-based lookups
- Easy testing: Provide mock wrappers/workflows if needed
- Unified structure: All models follow the same architectural pattern while supporting workflows

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
   - Add `exporters/{model}/model_wrappers.py` (reuse `IdentityWrapper` or implement custom wrapper)
   - Introduce `exporters/{model}/onnx_workflow.py` / `tensorrt_workflow.py` only if you need multi-stage orchestration; otherwise rely on the base exporters created by `ExporterFactory`
   - Prefer composition over inheritance—extend the workflows, not the base exporters, unless a new backend capability is required

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
