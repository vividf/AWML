# AWML Deployment Framework - Code Review Report

**Review Date:** November 2025  
**Reviewer:** AI Code Review Assistant  
**Scope:** Complete deployment framework with focus on CenterPoint deployment workflow

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Issues](#critical-issues)
3. [Major Issues](#major-issues)
4. [Minor Issues](#minor-issues)
5. [Architecture Analysis](#architecture-analysis)
6. [Code Quality Assessment](#code-quality-assessment)
7. [Anti-Patterns](#anti-patterns)
8. [Recommendations](#recommendations)
9. [Strengths](#strengths)
10. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Overall Assessment

The deployment framework demonstrates **solid architectural thinking** with clear separation of concerns and good abstraction patterns. The codebase is well-documented and uses modern Python practices (type hints, dataclasses, dependency injection).

However, the framework suffers from:
- **Over-engineering** with too many layers of abstraction
- **Tight coupling** between CenterPoint-specific code and the deployment framework
- **Responsibility confusion** between workflows, exporters, and runners
- **God objects** (BaseDeploymentRunner at 856 lines)
- **Missing validation** and error handling consistency

### Risk Level: MODERATE

While the code functions correctly, maintenance and extensibility are at risk due to complexity and tight coupling.

### Key Metrics

- **Lines of Code (Deployment):** ~5,000+ lines
- **Number of Classes:** 50+
- **Cyclomatic Complexity:** High in BaseDeploymentRunner
- **Test Coverage:** Unknown (no tests visible)

---

## Critical Issues

### 1. Circular Dependency Risk in CenterPoint Workflow

**Severity:** 🔴 CRITICAL  
**Location:** `deployment/exporters/centerpoint/onnx_workflow.py:51`

**Problem:**
```python
# In CenterPointDeploymentRunner.__init__
self._onnx_workflow = CenterPointONNXExportWorkflow(
    exporter=self._get_onnx_exporter,  # ⚠️ Passing runner's method
    logger=self.logger
)
```

The workflow receives a callable that returns the runner's exporter, creating circular reference:
- Runner creates workflow
- Workflow needs exporter
- Exporter is owned by runner
- Workflow caches exporter instance

**Impact:**
- Unclear ownership and lifecycle
- Hard to test in isolation
- Memory leak potential if not careful
- Violates single responsibility principle

**Fix:**
```python
# Pass factory, not runner's method
class CenterPointONNXExportWorkflow:
    def __init__(
        self,
        exporter_factory: ExporterFactory,
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.exporter_factory = exporter_factory
        self.config = config
        self.logger = logger

    def export(self, ...):
        # Create fresh exporter when needed
        exporter = self.exporter_factory.create_onnx_exporter(
            config=self.config,
            wrapper_cls=CenterPointONNXWrapper,
            logger=self.logger
        )
        # Use exporter, no caching needed
```

---

### 2. Wrong Dependency Direction: deployment/ → projects/

**Severity:** 🔴 CRITICAL  
**Location:** `deployment/exporters/centerpoint/onnx_workflow.py:127`

**Problem:**
```python
# In deployment framework code
from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
```

The deployment framework (generic infrastructure) imports from projects (specific models). This is **backwards** and breaks the layered architecture.

**Correct Dependency Flow:**
```
projects/ → deployment/ → core/
    ↓           ↓           ↓
 (uses)     (uses)      (base)
```

**Current (WRONG):**
```
deployment/ ←──┐
    ↓          │
projects/ ─────┘  (circular!)
```

**Impact:**
- Deployment framework cannot be reused independently
- Adding new models requires modifying deployment code
- Violates Open/Closed Principle
- Creates maintenance nightmare

**Fix:**

Create an abstraction layer:

```python
# deployment/exporters/workflows/base.py
class ModelComponentExtractor(ABC):
    """Interface for extracting exportable model components"""

    @abstractmethod
    def extract_voxel_encoder(self, model) -> Tuple[torch.nn.Module, Any]:
        """Extract voxel encoder and its input"""
        pass

    @abstractmethod
    def create_backbone_head_module(self, model) -> torch.nn.Module:
        """Create combined backbone+neck+head module"""
        pass

# projects/CenterPoint/deploy/component_extractor.py
class CenterPointComponentExtractor(ModelComponentExtractor):
    def create_backbone_head_module(self, model):
        # Import only in project code
        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

# deployment/exporters/centerpoint/onnx_workflow.py
class CenterPointONNXExportWorkflow:
    def __init__(self, component_extractor: ModelComponentExtractor, ...):
        self.component_extractor = component_extractor  # Inject from outside
```

---

### 3. Hard-Coded Model Structure Knowledge in Framework

**Severity:** 🔴 CRITICAL  
**Location:** `deployment/exporters/centerpoint/onnx_workflow.py:81,95,127`

**Problem:**

The deployment framework workflow knows intimate details about CenterPoint model structure:

```python
# Workflow calls private model method
input_features, voxel_dict = model._extract_features(data_loader, sample_idx)

# Workflow accesses model internals
model.pts_voxel_encoder(input_features)
model.pts_middle_encoder(voxel_features, coors, batch_size)
model.pts_backbone
model.pts_neck
model.pts_bbox_head
```

**Impact:**
- Workflow cannot work with other 3D detection models
- Changes to CenterPoint require framework changes
- Testing requires real CenterPoint model
- Violates encapsulation

**Fix:**

Use the Component Extractor pattern (see issue #2 above).

---

### 4. TensorRT Profile Configuration Edge Case

**Severity:** 🔴 CRITICAL  
**Location:** `deployment/exporters/centerpoint/tensorrt_workflow.py:73`

**Problem:**
```python
artifact = self._get_exporter().export(
    model=None,
    sample_input=None,  # ⚠️ No sample input!
    output_path=trt_path,
    onnx_path=onnx_file_path,
)
```

TensorRT export is called with `sample_input=None`. This only works because the config has explicit shape profiles. If shapes are missing from config, it will crash.

**Also in:** `deployment/exporters/base/tensorrt_exporter.py:293`

```python
def _configure_input_shapes(self, profile, sample_input, network=None):
    model_inputs_cfg = self.config.model_inputs

    if not model_inputs_cfg:
        raise ValueError("model_inputs is not set in the config")
        # ⚠️ Should also check sample_input here
```

**Fix:**
```python
def _configure_input_shapes(self, profile, sample_input, network=None):
    model_inputs_cfg = self.config.model_inputs

    # Validate we have enough information
    if not model_inputs_cfg or not model_inputs_cfg[0].input_shapes:
        if sample_input is None:
            raise ValueError(
                "TensorRT export requires either:\n"
                "  1. Explicit model_inputs with input_shapes in config, OR\n"
                "  2. sample_input for automatic shape inference\n"
                "Neither was provided."
            )
        # Infer shapes from sample_input
        return self._infer_shapes_from_sample(profile, sample_input, network)

    # Use explicit config
    ...
```

---

## Major Issues

### 5. God Object: BaseDeploymentRunner ✅ FIXED

**Severity:** 🟡 MAJOR  
**Location:** `deployment/runners/deployment_runner.py` (~~856~~ → **568 lines**)  
**Status:** ✅ **RESOLVED** - Extracted 3 orchestrator classes

**Problem:**

BaseDeploymentRunner has too many responsibilities:

1. Workflow orchestration (export → verify → evaluate)
2. Lazy exporter creation and caching
3. Artifact management
4. Device validation and normalization
5. Model path resolution (pytorch, onnx, tensorrt)
6. Scenario-based verification logic
7. Cross-backend evaluation
8. Model loading coordination

**Metrics:**
- **856 lines** in single class
- **20+ methods**
- **Multiple concerns** (export, verification, evaluation, artifact management)

**Impact:**
- Hard to understand
- Hard to test
- Hard to modify without breaking things
- Violates Single Responsibility Principle

**Fix:**

Split into focused classes:

```python
# 1. Artifact Manager
class ArtifactManager:
    """Manages model artifacts and path resolution"""
    def __init__(self, config: BaseDeploymentConfig):
        self.config = config
        self.artifacts: Dict[Backend, Artifact] = {}

    def register_artifact(self, backend: Backend, artifact: Artifact):
        self.artifacts[backend] = artifact

    def resolve_artifact(self, backend: Backend) -> Optional[Artifact]:
        # Consolidate all resolution logic here
        ...

# 2. Verification Orchestrator
class VerificationOrchestrator:
    """Orchestrates verification across backends"""
    def __init__(self, config: VerificationConfig, evaluator: BaseEvaluator):
        self.config = config
        self.evaluator = evaluator

    def run(self, artifact_manager: ArtifactManager) -> Dict[str, Any]:
        # All verification logic here
        ...

# 3. Evaluation Orchestrator
class EvaluationOrchestrator:
    """Orchestrates evaluation across backends"""
    def __init__(self, config: EvaluationConfig, evaluator: BaseEvaluator):
        self.config = config
        self.evaluator = evaluator

    def run(self, artifact_manager: ArtifactManager) -> Dict[str, Any]:
        # All evaluation logic here
        ...

# 4. Simplified BaseDeploymentRunner
class BaseDeploymentRunner:
    """Coordinates export workflow"""
    def __init__(self, ...):
        self.artifact_manager = ArtifactManager(config)
        self.verification = VerificationOrchestrator(config.verification_config, evaluator)
        self.evaluation = EvaluationOrchestrator(config.evaluation_config, evaluator)
        self.exporter_factory = ExporterFactory()

    def run(self, checkpoint_path=None):
        # Just coordinate high-level flow
        model = self.load_pytorch_model(checkpoint_path)

        onnx_artifact = self.export_onnx(model)
        self.artifact_manager.register_artifact(Backend.ONNX, onnx_artifact)

        trt_artifact = self.export_tensorrt(onnx_artifact)
        self.artifact_manager.register_artifact(Backend.TENSORRT, trt_artifact)

        verification_results = self.verification.run(self.artifact_manager)
        evaluation_results = self.evaluation.run(self.artifact_manager)

        return self._build_results(verification_results, evaluation_results)
```

**Benefits:**
- Each class has single responsibility
- Easier to test (mock ArtifactManager in tests)
- Easier to understand
- Easier to extend

**Implementation (✅ Completed):**

Three orchestrator classes were successfully extracted:

1. **`ArtifactManager`** (`deployment/runners/artifact_manager.py` - 163 lines)
   - Handles all artifact registration and resolution
   - Methods: `register_artifact()`, `resolve_artifact()`, `resolve_pytorch_artifact()`, etc.

2. **`VerificationOrchestrator`** (`deployment/runners/verification_orchestrator.py` - 227 lines)
   - Handles all verification workflows
   - Methods: `run()`, `_resolve_device()`, `_resolve_backend_path()`, `_create_artifact()`

3. **`EvaluationOrchestrator`** (`deployment/runners/evaluation_orchestrator.py` - 198 lines)
   - Handles all evaluation workflows
   - Methods: `run()`, `_get_models_to_evaluate()`, `_normalize_device_for_backend()`, `_print_cross_backend_comparison()`

**Results:**
- BaseDeploymentRunner reduced from **856 → 568 lines (33.6% reduction)**
- Now focused purely on orchestration: load → export → delegate to orchestrators
- Each orchestrator has single, testable responsibility
- 100% backward compatible - no API changes

---

### 6. Workflow/Exporter Responsibility Confusion

**Severity:** 🟡 MAJOR  
**Location:** `deployment/exporters/centerpoint/`

**Problem:**

The boundary between "workflow" and "exporter" is unclear:

- **Workflows** orchestrate multiple exports BUT also cache exporters
- **Workflows** know about model internals (voxel encoder, backbone structure)
- **Exporters** are generic BUT workflows are model-specific

```python
class CenterPointONNXExportWorkflow:
    def __init__(self, exporter: Callable[[], ONNXExporter], ...):
        self._exporter_provider = exporter
        self._exporter_cache = None  # ⚠️ Why is workflow caching exporter?

    def export(self, model, ...):
        # Workflow runs model inference
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)

        # Workflow knows model structure
        backbone_neck_head = CenterPointHeadONNX(
            model.pts_backbone,  # ⚠️ Knows internals
            model.pts_neck,
            model.pts_bbox_head
        )
```

**Better Separation:**

**Workflows**: Pure orchestration, no state
**Exporters**: Stateless, single-purpose ONNX/TensorRT export
**Component Extractors**: Know model structure

```python
class CenterPointONNXExportWorkflow:
    def __init__(
        self,
        exporter_factory: ExporterFactory,
        component_extractor: CenterPointComponentExtractor,
        config: BaseDeploymentConfig,
    ):
        self.factory = exporter_factory  # No caching
        self.extractor = component_extractor  # Knows model
        self.config = config

    def export(self, model, sample_data, output_dir):
        # Extract components (delegates to extractor)
        components = self.extractor.extract_components(model, sample_data)

        # Export each component (creates fresh exporter each time)
        for name, component, cfg in components:
            exporter = self.factory.create_onnx_exporter(self.config, ...)
            exporter.export(component.module, component.input, output_path, cfg)
```

---

### 7. Model Injection Pattern is Inconsistent

**Severity:** 🟡 MAJOR  
**Location:** `deployment/runners/centerpoint_runner.py:86-100`

**Problem:**

```python
# Update runner's internal model_cfg
self.model_cfg = onnx_cfg  # ⚠️ Mutating runner state

# Inject config to evaluator via setter
if hasattr(self.evaluator, "set_onnx_config"):  # ⚠️ hasattr check
    self.evaluator.set_onnx_config(onnx_cfg)

# Inject PyTorch model to evaluator via setter
if hasattr(self.evaluator, "set_pytorch_model"):  # ⚠️ hasattr check
    self.evaluator.set_pytorch_model(model)
```

**Issues:**
1. Uses `hasattr` checks - fragile, no type safety
2. Mutates runner's state during model loading
3. Side effects hidden in `load_pytorch_model()`
4. Not clear when evaluator has model vs when it doesn't
5. Evaluator's state changes externally

**Better Approach:**

**Option 1: Constructor Injection (Recommended)**
```python
class CenterPointEvaluator:
    def __init__(
        self,
        model_cfg: Config,
        pytorch_model: Optional[torch.nn.Module] = None,
    ):
        self.model_cfg = model_cfg
        self.pytorch_model = pytorch_model
        self._pipelines: Dict[Backend, BaseDeploymentPipeline] = {}

    def set_pytorch_model(self, model: torch.nn.Module):
        """Explicit setter with clear intent"""
        self.pytorch_model = model
        # Invalidate cached pipelines
        self._pipelines.clear()

# In runner
evaluator = CenterPointEvaluator(model_cfg=original_cfg)
model = self.load_pytorch_model(checkpoint_path)
evaluator.set_pytorch_model(model)  # Explicit, no hasattr
```

**Option 2: Builder Pattern**
```python
class CenterPointEvaluatorBuilder:
    def __init__(self):
        self._model_cfg = None
        self._pytorch_model = None

    def with_model_config(self, cfg: Config):
        self._model_cfg = cfg
        return self

    def with_pytorch_model(self, model: torch.nn.Module):
        self._pytorch_model = model
        return self

    def build(self) -> CenterPointEvaluator:
        if self._model_cfg is None:
            raise ValueError("model_cfg is required")
        return CenterPointEvaluator(self._model_cfg, self._pytorch_model)

# Usage
evaluator = (CenterPointEvaluatorBuilder()
    .with_model_config(onnx_cfg)
    .with_pytorch_model(model)
    .build())
```

---

### 8. Feature Envy: Workflow Does Model Operations

**Severity:** 🟡 MAJOR  
**Location:** `deployment/exporters/centerpoint/onnx_workflow.py:116-146`

**Problem:**

Workflow is running model inference:

```python
def _export_backbone_neck_head(self, model, input_features, voxel_dict, output_dir):
    # Workflow running model operations ⚠️
    with torch.no_grad():
        voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
        coors = voxel_dict["coors"]
        batch_size = coors[-1, 0] + 1
        spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)

    # Workflow creating model components ⚠️
    from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
    backbone_neck_head = CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)
```

This is **Feature Envy** - workflow is more interested in model's data than its own.

**Fix:**

Move to Component Extractor:

```python
class CenterPointComponentExtractor:
    def prepare_backbone_input(self, model, input_features, voxel_dict):
        """Prepare input tensor for backbone export"""
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            coors = voxel_dict["coors"]
            batch_size = coors[-1, 0] + 1
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
        return spatial_features

    def create_backbone_head_module(self, model):
        """Create combined backbone+neck+head for ONNX export"""
        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

# Workflow just orchestrates
class CenterPointONNXExportWorkflow:
    def export(self, ...):
        # Get prepared components
        backbone_module = self.extractor.create_backbone_head_module(model)
        backbone_input = self.extractor.prepare_backbone_input(model, input_features, voxel_dict)

        # Just export (no model knowledge)
        exporter = self.factory.create_onnx_exporter(...)
        exporter.export(backbone_module, backbone_input, output_path)
```

---

### 9. Race Condition in Stage Latency Tracking

**Severity:** 🟡 MAJOR  
**Location:** `deployment/pipelines/centerpoint/centerpoint_pipeline.py:290-307`

**Problem:**

```python
def run_model(self, preprocessed_input):
    # Stage 1
    start = time.time()
    voxel_features = self.run_voxel_encoder(...)
    self._stage_latencies["voxel_encoder_ms"] = (time.time() - start) * 1000  # ⚠️ Instance variable

    # Stage 2
    start = time.time()
    spatial_features = self.process_middle_encoder(...)
    self._stage_latencies["middle_encoder_ms"] = (time.time() - start) * 1000  # ⚠️ Instance variable
```

**Base class clears it:**
```python
# deployment/pipelines/base/base_pipeline.py:171-174
if hasattr(self, "_stage_latencies") and isinstance(self._stage_latencies, dict):
    latency_breakdown.update(self._stage_latencies)
    self._stage_latencies = {}  # ⚠️ Cleared after each infer()
```

**Issues:**
1. If pipeline is reused across threads → race condition
2. If `infer()` is called before `run_model()` completes → wrong results
3. Relies on clearing side effects
4. Fragile: depends on base class clearing instance variable

**Fix:**

Don't use instance variables for per-request data:

```python
def run_model(self, preprocessed_input) -> Tuple[Any, Dict[str, float]]:
    """
    Run model and return outputs + stage latencies.

    Returns:
        Tuple of (model_output, stage_latencies)
    """
    stage_latencies = {}  # Local variable ✅

    # Stage 1
    start = time.perf_counter()
    voxel_features = self.run_voxel_encoder(...)
    stage_latencies["voxel_encoder_ms"] = (time.perf_counter() - start) * 1000

    # Stage 2
    start = time.perf_counter()
    spatial_features = self.process_middle_encoder(...)
    stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

    # Stage 3
    start = time.perf_counter()
    head_outputs = self.run_backbone_head(...)
    stage_latencies["backbone_head_ms"] = (time.perf_counter() - start) * 1000

    return head_outputs, stage_latencies  # Return, don't store ✅

# Base class handles it
def infer(self, ...):
    ...
    model_output, stage_latencies = self.run_model(model_input)
    latency_breakdown.update(stage_latencies)
    ...
```

---

## Minor Issues

### 10. Inconsistent Error Handling

**Severity:** 🟢 MINOR  
**Locations:** Throughout

**Examples:**

```python
# Example 1: Catch and re-raise with generic error
except Exception:
    self.logger.error("ONNX export workflow failed")
    raise

# Example 2: Catch and wrap with context
except Exception as e:
    self.logger.error(f"ONNX export failed: {e}")
    import traceback
    self.logger.error(traceback.format_exc())
    raise RuntimeError("ONNX export failed") from e

# Example 3: Let it propagate (no catch)
```

**Recommendation:**

Establish consistent pattern:

```python
# Define custom exceptions
class DeploymentError(Exception):
    """Base exception for deployment errors"""
    pass

class ExportError(DeploymentError):
    """ONNX/TensorRT export failed"""
    pass

class VerificationError(DeploymentError):
    """Verification failed"""
    pass

# Use consistently
try:
    self._do_onnx_export(...)
except (IOError, OSError) as e:
    self.logger.error(f"ONNX export failed: {e}", exc_info=True)
    raise ExportError(f"Failed to export ONNX to {output_path}") from e
except torch.onnx.CheckerError as e:
    self.logger.error(f"ONNX model validation failed: {e}", exc_info=True)
    raise ExportError(f"Invalid ONNX model: {e}") from e
```

---

### 11. Magic Strings and Numbers

**Severity:** 🟢 MINOR  
**Location:** `deployment/exporters/centerpoint/onnx_workflow.py:151`

**Problem:**

```python
output_names = ["reg", "height", "dim", "rot", "vel", "heatmap"]  # ⚠️ Magic list
```

**Fix:**

```python
class CenterPointConstants:
    """CenterPoint model constants"""
    DEFAULT_HEAD_OUTPUTS = ("reg", "height", "dim", "rot", "vel", "heatmap")
    EXPECTED_NUM_OUTPUTS = 6
    VOXEL_ENCODER_FILE = "pts_voxel_encoder.onnx"
    BACKBONE_HEAD_FILE = "pts_backbone_neck_head.onnx"

def _get_output_names(self, model):
    if hasattr(model.pts_bbox_head, "output_names"):
        names = tuple(model.pts_bbox_head.output_names)
        if len(names) != CenterPointConstants.EXPECTED_NUM_OUTPUTS:
            self.logger.warning(
                f"Expected {CenterPointConstants.EXPECTED_NUM_OUTPUTS} outputs, "
                f"got {len(names)}: {names}"
            )
        return names
    return CenterPointConstants.DEFAULT_HEAD_OUTPUTS
```

---

### 12. Primitive Obsession

**Severity:** 🟢 MINOR  
**Locations:** Throughout

**Problem:**

Using raw strings for structured data:

```python
def _normalize_device_for_backend(
    self,
    backend: Union[str, Backend],  # ⚠️ Still accepts string
    device: Optional[str]  # ⚠️ String like "cuda:0"
) -> str:  # ⚠️ Returns string
```

**Better:**

```python
@dataclass(frozen=True)
class DeviceSpec:
    """Strongly-typed device specification"""
    device_type: str  # 'cpu' or 'cuda'
    device_id: Optional[int] = None

    @classmethod
    def parse(cls, device_str: str) -> "DeviceSpec":
        """Parse 'cpu', 'cuda', 'cuda:0', etc."""
        normalized = device_str.strip().lower()
        if normalized == "cpu":
            return cls(device_type="cpu")
        if normalized.startswith("cuda"):
            parts = normalized.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0
            return cls(device_type="cuda", device_id=device_id)
        raise ValueError(f"Invalid device: {device_str}")

    def __str__(self) -> str:
        if self.device_type == "cpu":
            return "cpu"
        return f"cuda:{self.device_id}"

    def to_torch_device(self) -> torch.device:
        return torch.device(str(self))

# Usage
device = DeviceSpec.parse("cuda:1")
device.device_id  # 1 (typed!)
str(device)  # "cuda:1"
```

---

## Architecture Analysis

### Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                           │
│  - CenterPoint, YOLOX-ELAN, Calibration                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ BaseDeploymentRunner + Project Runners                  │
│  - 856 lines (too large!)                               │
│  - Too many responsibilities                            │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│   Exporters    │   │   Evaluators    │
│  - ONNX        │   │  - Task-specific│
│  - TensorRT    │   │  - Metrics      │
│  - Workflows   │   │                 │
│    (coupled!)  │   │                 │
└────────────────┘   └─────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Pipeline Architecture                      │
│  - BaseDeploymentPipeline                               │
│  - Backend-specific implementations                     │
└──────────────────────────────────────────────────────────┘
```

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                           │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ DeploymentOrchestrator (Simplified Runner)              │
│  - Coordinates high-level flow only                     │
│  - Delegates to specialized components                  │
└──────────┬───────────────┬──────────────┬───────────────┘
           │               │              │
┌──────────▼──────┐ ┌──────▼──────┐ ┌────▼─────────┐
│ ArtifactManager │ │ Verification│ │ Evaluation   │
│                 │ │ Orchestrator│ │ Orchestrator │
└─────────────────┘ └─────────────┘ └──────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│           Export Strategy (Pluggable)                    │
│  - SingleFileExportStrategy (YOLOX, Calibration)        │
│  - MultiFileExportStrategy (CenterPoint)                │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│        Exporters (Generic, No Model Knowledge)           │
│  - ONNXExporter                                          │
│  - TensorRTExporter                                      │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│    Component Extractors (Model-Specific Knowledge)       │
│  - Implemented in projects/                              │
│  - Extract exportable components from models             │
└──────────────────────────────────────────────────────────┘
```

**Key Improvements:**

1. **Clear Separation**: Each component has single responsibility
2. **Correct Dependencies**: projects/ → deployment/, not reverse
3. **Pluggable Strategies**: Easy to add new export patterns
4. **Testable**: Each component can be tested in isolation
5. **Extensible**: Adding new model doesn't require framework changes

---

## Recommendations

### Priority 1: Fix Critical Issues (1-2 weeks)

1. **Refactor CenterPoint workflow** to remove `projects/` imports
   - Create `ModelComponentExtractor` interface
   - Implement `CenterPointComponentExtractor` in `projects/`
   - Inject extractor into workflow

2. **Remove circular dependency** in workflow initialization
   - Pass `ExporterFactory` instead of runner method
   - Remove exporter caching from workflows

3. **Add TensorRT shape validation**
   - Validate config has shapes OR sample_input provided
   - Add helpful error messages

4. **Fix model injection pattern**
   - Use explicit setters or constructor injection
   - Remove `hasattr` checks

### Priority 2: Refactor Runner (2-3 weeks)

1. **Split BaseDeploymentRunner**
   - Extract `ArtifactManager`
   - Extract `VerificationOrchestrator`
   - Extract `EvaluationOrchestrator`
   - Reduce runner to < 300 lines

2. **Introduce Strategy Pattern** for export
   - Create `ExportStrategy` interface
   - Implement `SingleFileExportStrategy`
   - Implement `MultiFileExportStrategy`

3. **Fix stage latency tracking**
   - Return latencies from `run_model()`, don't use instance variable
   - Update base pipeline to handle returned latencies

### Priority 3: Code Quality (1-2 weeks)

1. **Consistent error handling**
   - Define custom exception hierarchy
   - Use consistent catch-wrap-rethrow pattern
   - Add helpful error messages

2. **Remove magic strings/numbers**
   - Define constants classes
   - Use enums where appropriate

3. **Add validation layer**
   - Validate configs before export
   - Validate ONNX models after export
   - Validate TensorRT engines

4. **Improve logging**
   - Consistent log levels
   - Structured logging (consider using structlog)

### Priority 4: Testing (2-3 weeks)

1. **Add unit tests**
   - Test config parsing
   - Test artifact management
   - Test device normalization
   - Test backend resolution

2. **Add integration tests**
   - Test ONNX export with dummy model
   - Test TensorRT export
   - Test pipeline execution

3. **Add E2E tests**
   - Test full CenterPoint deployment
   - Test full YOLOX deployment

---

## Strengths

The framework has many positive aspects:

✅ **Well-documented**: Extensive README, docstrings, type hints  
✅ **Typed configurations**: Use of dataclasses for type safety  
✅ **Dependency injection**: Explicit dependencies, no globals  
✅ **Clear layering**: Core/Exporters/Pipelines/Runners separation  
✅ **Extensible**: Pattern for adding new models  
✅ **Multi-backend**: Clean abstraction over PyTorch/ONNX/TensorRT  
✅ **Comprehensive**: Handles export, verification, evaluation  
✅ **Modern Python**: Type hints, dataclasses, pathlib  

---

## Testing Strategy

### Recommended Test Structure

```
tests/
├── unit/
│   ├── test_artifact.py              # Artifact dataclass
│   ├── test_backend.py               # Backend enum
│   ├── test_configs.py               # Config parsing
│   ├── test_device_spec.py           # Device parsing
│   ├── test_exporter_factory.py      # Factory (mocked)
│   └── test_component_extractor.py   # Component extraction
│
├── integration/
│   ├── test_onnx_export.py           # Real ONNX export
│   ├── test_tensorrt_export.py       # Real TensorRT export
│   ├── test_pipeline.py              # Pipeline execution
│   └── test_verification.py          # Cross-backend verification
│
└── e2e/
    ├── test_centerpoint_deploy.py    # Full CenterPoint flow
    ├── test_yolox_deploy.py          # Full YOLOX flow
    └── test_calibration_deploy.py    # Full Calibration flow
```

### Make Code More Testable

**Before:**
```python
class ONNXExporter:
    def export(self, model, input, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Direct filesystem call
        torch.onnx.export(model, input, output_path)
```

**After:**
```python
class FileSystem(ABC):
    @abstractmethod
    def makedirs(self, path: str): pass

    @abstractmethod
    def exists(self, path: str) -> bool: pass

class RealFileSystem(FileSystem):
    def makedirs(self, path: str):
        os.makedirs(path, exist_ok=True)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

class ONNXExporter:
    def __init__(self, config, wrapper, logger, filesystem: FileSystem = None):
        self.filesystem = filesystem or RealFileSystem()

    def export(self, model, input, output_path):
        self.filesystem.makedirs(os.path.dirname(output_path))
        torch.onnx.export(model, input, output_path)

# Now testable!
def test_onnx_export():
    mock_fs = MockFileSystem()
    exporter = ONNXExporter(..., filesystem=mock_fs)
    exporter.export(dummy_model, dummy_input, "output.onnx")
    assert "output.onnx" in mock_fs.files_written
```

---

## Appendix A: Code Examples

### Example: Component Extractor Pattern

```python
# deployment/exporters/workflows/interfaces.py
class ModelComponentExtractor(ABC):
    """Interface for extracting exportable model components"""

    @abstractmethod
    def extract_components(
        self,
        model: torch.nn.Module,
        sample_data: Any
    ) -> List[ExportableComponent]:
        """Extract all components that need to be exported"""
        pass

@dataclass
class ExportableComponent:
    """A model component ready for ONNX export"""
    name: str
    module: torch.nn.Module
    sample_input: Any
    config_override: Optional[ONNXExportConfig] = None

# projects/CenterPoint/deploy/component_extractor.py
class CenterPointComponentExtractor(ModelComponentExtractor):
    def extract_components(self, model, sample_data):
        # Prepare inputs
        input_features, voxel_dict = self._prepare_inputs(model, sample_data)

        # Component 1: Voxel Encoder
        voxel_component = ExportableComponent(
            name="pts_voxel_encoder",
            module=model.pts_voxel_encoder,
            sample_input=input_features,
            config_override=ONNXExportConfig(
                input_names=("input_features",),
                output_names=("pillar_features",),
                dynamic_axes={
                    "input_features": {0: "num_voxels", 1: "num_max_points"},
                    "pillar_features": {0: "num_voxels"},
                },
            )
        )

        # Component 2: Backbone+Neck+Head
        backbone_input = self._prepare_backbone_input(model, input_features, voxel_dict)
        backbone_module = self._create_backbone_module(model)

        backbone_component = ExportableComponent(
            name="pts_backbone_neck_head",
            module=backbone_module,
            sample_input=backbone_input,
            config_override=ONNXExportConfig(
                input_names=("spatial_features",),
                output_names=self._get_output_names(model),
                dynamic_axes={
                    "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                },
            )
        )

        return [voxel_component, backbone_component]

    def _create_backbone_module(self, model):
        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

# deployment/exporters/centerpoint/onnx_workflow.py
class CenterPointONNXExportWorkflow:
    def __init__(
        self,
        exporter_factory: ExporterFactory,
        component_extractor: ModelComponentExtractor,
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.factory = exporter_factory
        self.extractor = component_extractor
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def export(self, model, sample_data, output_dir, **kwargs) -> Artifact:
        os.makedirs(output_dir, exist_ok=True)

        # Extract components (all model knowledge is in extractor)
        components = self.extractor.extract_components(model, sample_data)

        # Export each component (workflow is generic!)
        for component in components:
            exporter = self.factory.create_onnx_exporter(
                config=self.config,
                wrapper_cls=IdentityWrapper,
                logger=self.logger
            )

            output_path = os.path.join(output_dir, f"{component.name}.onnx")
            exporter.export(
                model=component.module,
                sample_input=component.sample_input,
                output_path=output_path,
                config_override=component.config_override
            )

            self.logger.info(f"Exported {component.name}: {output_path}")

        return Artifact(path=output_dir, multi_file=True)
```

---

## Appendix B: Refactoring Checklist

### Phase 1: Critical Fixes (Week 1-2)
- [ ] Create `ModelComponentExtractor` interface
- [ ] Implement `CenterPointComponentExtractor` in `projects/`
- [ ] Remove `projects/` imports from `deployment/exporters/centerpoint/`
- [ ] Remove exporter caching from workflows
- [ ] Pass `ExporterFactory` instead of callable to workflows
- [ ] Add TensorRT shape validation with helpful errors
- [ ] Remove `hasattr` checks in model injection
- [ ] Add explicit model setter to evaluator

### Phase 2: Refactor Runner (Week 3-5)
- [ ] Extract `ArtifactManager` class
- [ ] Extract `VerificationOrchestrator` class
- [ ] Extract `EvaluationOrchestrator` class
- [ ] Reduce `BaseDeploymentRunner` to coordination logic only
- [ ] Fix stage latency tracking (return from method, not instance var)
- [ ] Update base pipeline to handle returned latencies

### Phase 3: Code Quality (Week 6-7)
- [ ] Define custom exception hierarchy
- [ ] Standardize error handling across all modules
- [ ] Extract magic strings/numbers to constants
- [ ] Create `DeviceSpec` value object
- [ ] Add validation layer for configs
- [ ] Add ONNX model validation post-export
- [ ] Improve logging consistency
- [ ] Add structured logging

### Phase 4: Testing (Week 8-10)
- [ ] Set up pytest infrastructure
- [ ] Write unit tests for core classes
- [ ] Write integration tests for exporters
- [ ] Write E2E tests for full workflows
- [ ] Add CI/CD pipeline for tests
- [ ] Measure and track code coverage

---

## Conclusion

The AWML deployment framework is architecturally sound but needs refactoring to reduce complexity and improve maintainability. The main issues are:

1. **Circular dependencies** between runner and workflows
2. **Wrong dependency direction** (deployment → projects)
3. **God objects** (BaseDeploymentRunner too large)
4. **Tight coupling** (CenterPoint workflow knows model internals)
5. **Inconsistent patterns** (error handling, model injection)

**Recommended approach:**

1. **Start with critical fixes** (1-2 weeks) - Fix dependency issues and critical bugs
2. **Refactor runner** (2-3 weeks) - Split into focused components
3. **Improve code quality** (1-2 weeks) - Consistent patterns, validation, logging
4. **Add tests** (2-3 weeks) - Unit, integration, E2E tests

**Total estimated effort:** 6-10 weeks

The refactoring will result in:
- ✅ Cleaner architecture with proper dependency flow
- ✅ Easier to test and maintain
- ✅ Easier to extend with new models
- ✅ Better error handling and validation
- ✅ More confidence in code correctness

---

**End of Code Review Report**
