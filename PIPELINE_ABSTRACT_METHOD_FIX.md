# Pipeline Abstract Method Fix

**Date**: 2025-10-27  
**Issue**: `Can't instantiate abstract class CenterPointPyTorchPipeline with abstract method run_model`

## Problem

After refactoring the pipeline architecture to introduce unified base classes (`BaseDeploymentPipeline`, `Detection3DPipeline`), the CenterPoint pipelines could not be instantiated:

```
ERROR: Can't instantiate abstract class CenterPointPyTorchPipeline with abstract method run_model
ERROR: Can't instantiate abstract class CenterPointONNXPipeline with abstract method run_model
ERROR: Can't instantiate abstract class CenterPointTensorRTPipeline with abstract method run_model
```

## Root Cause

The issue occurred because:

1. **`BaseDeploymentPipeline`** defines `run_model()` as an abstract method for simple single-stage inference
2. **`Detection3DPipeline`** inherited from `BaseDeploymentPipeline` and kept `run_model()` as abstract
3. **3D detection pipelines** (like CenterPoint) use a **multi-stage inference pattern**:
   - Voxel Encoder (backend-specific)
   - Middle Encoder (PyTorch)
   - Backbone + Head (backend-specific)
   - Postprocessing (PyTorch)

4. **Concrete implementations** (`CenterPointPyTorchPipeline`, etc.) implement:
   - `run_voxel_encoder()`
   - `run_backbone_head()`
   
   But they do **NOT** implement `run_model()` because it doesn't fit the multi-stage pattern.

## Solution

Modified `Detection3DPipeline.run_model()` to be **non-abstract** and raise `NotImplementedError` with a clear message:

```python
# detection_3d_pipeline.py

def run_model(self, preprocessed_input: Any) -> Any:
    """
    Run 3D detection model (backend-specific).
    
    NOTE: For 3D detection, most models use a multi-stage pipeline
    (voxel encoder → middle encoder → backbone/head) instead of a single
    run_model() call. Therefore, this method is typically not used.
    
    Instead, 3D detection pipelines override infer() and implement
    stage-specific methods like:
    - run_voxel_encoder()
    - process_middle_encoder()
    - run_backbone_head()
    """
    raise NotImplementedError(
        "run_model() is not used for 3D detection pipelines. "
        "3D detection uses a multi-stage inference pipeline. "
        "See CenterPointDeploymentPipeline.infer() for implementation."
    )
```

## Architecture Overview

### Base Pipeline Hierarchy

```
BaseDeploymentPipeline (abstract)
├── run_model() [ABSTRACT] - for simple single-stage inference
├── preprocess() [ABSTRACT]
├── postprocess() [ABSTRACT]
└── infer() [CONCRETE] - orchestrates preprocess → run_model → postprocess

Detection3DPipeline (abstract, extends BaseDeploymentPipeline)
├── run_model() [OVERRIDDEN, NON-ABSTRACT] - raises NotImplementedError
├── preprocess() [ABSTRACT]
├── postprocess() [ABSTRACT]
└── infer() [INHERITED, will be overridden by specific implementations]

CenterPointDeploymentPipeline (abstract, extends Detection3DPipeline)
├── run_voxel_encoder() [ABSTRACT] - stage 1
├── run_backbone_head() [ABSTRACT] - stage 2
├── process_middle_encoder() [CONCRETE]
├── preprocess() [CONCRETE]
├── postprocess() [CONCRETE]
└── infer() [OVERRIDDEN] - custom multi-stage orchestration

CenterPointPyTorchPipeline (concrete, extends CenterPointDeploymentPipeline)
├── run_voxel_encoder() [IMPLEMENTED]
└── run_backbone_head() [IMPLEMENTED]

CenterPointONNXPipeline (concrete, extends CenterPointDeploymentPipeline)
├── run_voxel_encoder() [IMPLEMENTED]
└── run_backbone_head() [IMPLEMENTED]

CenterPointTensorRTPipeline (concrete, extends CenterPointDeploymentPipeline)
├── run_voxel_encoder() [IMPLEMENTED]
└── run_backbone_head() [IMPLEMENTED]
```

## Key Insights

1. **Single-stage models** (2D detection, classification):
   - Use `run_model()` for inference
   - Use default `infer()` from `BaseDeploymentPipeline`

2. **Multi-stage models** (3D detection like CenterPoint):
   - Override `infer()` to orchestrate multiple stages
   - Implement stage-specific methods (`run_voxel_encoder`, `run_backbone_head`)
   - Do NOT use `run_model()`

3. **Design principle**: 
   - Base classes provide sensible defaults
   - Specific task types can override methods to fit their patterns
   - Abstract methods should only be required when truly necessary

## Verification

Test script confirms:
- ✓ `Detection3DPipeline.run_model()` is NOT abstract
- ✓ `CenterPointDeploymentPipeline` has correct abstract methods: `run_voxel_encoder()`, `run_backbone_head()`
- ✓ Concrete pipelines can now be instantiated without errors

## Files Modified

- `/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/pipelines/detection_3d_pipeline.py`
  - Changed `run_model()` from `@abstractmethod` to concrete method with `NotImplementedError`

## Impact

- ✓ CenterPoint evaluation and verification now work
- ✓ No changes needed to concrete pipeline implementations
- ✓ Clear error message if someone tries to use `run_model()` in 3D detection
- ✓ Architecture supports both single-stage and multi-stage inference patterns

