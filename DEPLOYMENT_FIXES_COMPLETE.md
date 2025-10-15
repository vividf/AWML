# Complete Deployment Fixes Summary

This document summarizes all the fixes applied to resolve deployment issues for YOLOX_opt_elan model export.

## Issues Fixed

1. ✅ Pipeline Builder Registry Mismatch
2. ✅ DataLoader Type Conversion  
3. ✅ TensorRT API Compatibility
4. ✅ PyTorch Backend Output Handling
5. ✅ TensorRT Dynamic Shapes Support

---

## Fix 1: Pipeline Builder Registry Mismatch

### Problem
```
KeyError: 'PackDetInputs is not in the mmengine::transform registry'
```

### Root Cause
MMDetection 3.x registers transforms in `mmdet.registry.TRANSFORMS`, but `Compose` was using the base `mmengine.registry.TRANSFORMS` registry.

### Solution
Use `init_default_scope()` to set the correct registry scope before building pipelines.

**File**: `/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/utils/pipeline_builder.py`

```python
# Import mmdet transforms
import mmdet.datasets.transforms
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

# Set default scope to 'mmdet'
init_default_scope('mmdet')
return Compose(pipeline_cfg)
```

**Changes**: Updated all 4 pipeline building functions (`_build_detection2d_pipeline`, `_build_detection3d_pipeline`, `_build_classification_pipeline`, `_build_segmentation_pipeline`)

---

## Fix 2: DataLoader Type Conversion

### Problem
```
RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

### Root Cause
MMDetection pipeline returns images as `uint8` tensors, but the model expects `float32` tensors.

### Solution
Add type conversion in the data loader preprocessing.

**File**: `/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/data_loader.py`

```python
# Convert to float32 if still in uint8 (ByteTensor)
if tensor.dtype == torch.uint8:
    tensor = tensor.float()
```

**Changes**: Modified `_preprocess_with_pipeline()` method (line 197-200)

---

## Fix 3: TensorRT API Compatibility

### Problem
```
AttributeError: 'IBuilderConfig' object has no attribute 'max_workspace_size'
```

### Root Cause
TensorRT 8.4+ replaced `max_workspace_size` with `set_memory_pool_limit()`, and TensorRT 8.5+ replaced `build_engine()` with `build_serialized_network()`.

### Solution
Use runtime API detection to support both old and new TensorRT versions.

**File**: `/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/main.py`

```python
# Workspace size (TensorRT 8.4+ compatibility)
if hasattr(config_trt, 'max_workspace_size'):
    config_trt.max_workspace_size = workspace_size  # < 8.4
else:
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)  # >= 8.4

# Engine building (TensorRT 8.5+ compatibility)
if hasattr(builder, 'build_serialized_network'):
    serialized_engine = builder.build_serialized_network(network, config_trt)  # >= 8.5
else:
    engine = builder.build_engine(network, config_trt)  # < 8.5
    serialized_engine = engine.serialize()
```

**Changes**: Modified `export_tensorrt()` function (lines 185-227)

---

## Fix 4: PyTorch Backend Output Handling

### Problem
```
ValueError: Unexpected PyTorch output type: <class 'tuple'>
```

### Root Cause
MMDetection models return tuples/lists in test mode, but the backend expected only tensor outputs.

### Solution
Handle tuple/list outputs by extracting the first element (predictions).

**File**: `/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/backends/pytorch_backend.py`

```python
# Handle different output formats
# MMDetection models return tuples/lists in test mode
if isinstance(output, (tuple, list)):
    if len(output) == 1:
        output = output[0]
    else:
        output = output[0]  # Use first element (predictions)
```

**Changes**: Modified `infer()` method (lines 64-70)

---

## Fix 5: TensorRT Dynamic Shapes Support

### Problem
```
Error Code 4: API Usage Error (Network has dynamic or shape inputs, but no optimization profile has been defined.)
```

### Root Cause
The ONNX model has dynamic batch size axes, but TensorRT requires an optimization profile for dynamic shapes.

### Solution
Automatically detect dynamic shapes and create optimization profiles.

**File**: `/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/main.py`

```python
# Check if network has dynamic inputs
has_dynamic_shapes = False
for i in range(network.num_inputs):
    input_shape = network.get_input(i).shape
    if -1 in input_shape:
        has_dynamic_shapes = True
        break

if has_dynamic_shapes:
    profile = builder.create_optimization_profile()

    # Get model inputs from config
    model_inputs = config.backend_config.model_inputs

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name

        # Find config and set min/opt/max shapes
        input_config = next((inp for inp in model_inputs if inp["name"] == input_name), None)
        if input_config and "shape" in input_config:
            shape = input_config["shape"]
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    config_trt.add_optimization_profile(profile)
```

**Changes**: Modified `export_tensorrt()` function (lines 141-183)

---

## Files Modified

1. **`/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/utils/pipeline_builder.py`**
   - Lines 161-441: Updated all pipeline building functions

2. **`/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/data_loader.py`**
   - Lines 197-200: Added uint8→float32 conversion

3. **`/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/main.py`**
   - Lines 141-227: Added dynamic shapes support and TensorRT API compatibility

4. **`/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/backends/pytorch_backend.py`**
   - Lines 64-85: Added tuple/list output handling

---

## Testing

To verify all fixes work together:

```bash
cd /workspace/projects/YOLOX_opt_elan/deploy
python main.py \
    --deploy-cfg deploy_config.py \
    --model-cfg <path_to_model_config> \
    --checkpoint <path_to_checkpoint>
```

Expected output:
```
✅ ONNX export successful: work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx
✅ TensorRT export successful: work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.engine
✅ All verifications passed!
```

---

## Summary

These fixes address:
- **Registry System**: Proper MMEngine/MMDetection registry handling
- **Type Safety**: Correct tensor type conversions
- **API Compatibility**: Support for multiple TensorRT versions
- **Output Handling**: Robust handling of various model output formats
- **Dynamic Shapes**: Automatic optimization profile creation for TensorRT

All changes maintain backward compatibility and include proper error handling.

---

## Fix 6: Deployment Configuration Standardization

### Problem
Different project deployment scripts used inconsistent evaluation configuration formats:
- Some used `models_to_evaluate` (list format)
- Some used `models` (dict format)
- Model paths tracked separately instead of being read from config

This led to:
- Confusing configuration patterns
- PyTorch model evaluation failures when not in export mode
- Manual path tracking instead of declarative configuration

### Root Cause
Early deployment scripts evolved separately without a unified standard, leading to divergent patterns.

### Solution
Standardized all deployment configurations to use the **`models` dict format** (matching CalibrationStatusClassification pattern).

**Standard Configuration Format** (`deploy_config.py`):
```python
evaluation = dict(
    enabled=True,
    num_samples=10,
    verbose=False,
    # Specify models to evaluate (comment out paths you don't want)
    models=dict(
        onnx="/path/to/model.onnx",
        tensorrt="/path/to/model.engine",
        # pytorch="/path/to/checkpoint.pth",  # Optional
    ),
)
```

**Standard Evaluation Pattern** (`main.py`):
```python
def get_models_to_evaluate(eval_config: dict, logger: logging.Logger) -> list:
    """Extract models from config and check if files exist."""
    models_config = eval_config.get("models", {})
    models_to_evaluate = []

    for backend_key, model_path in models_config.items():
        if model_path and os.path.exists(model_path):
            models_to_evaluate.append((backend_key, model_path))
            logger.info(f"  - {backend_key}: {model_path}")
        else:
            logger.warning(f"  - {backend_key}: {model_path} (not found, skipping)")

    return models_to_evaluate

def run_evaluation(data_loader, config, model_cfg, logger):
    """Run evaluation using models from config."""
    models_to_evaluate = get_models_to_evaluate(eval_config, logger)

    for backend, model_path in models_to_evaluate:
        results = evaluator.evaluate(
            model_path=model_path,
            backend=backend,
            # ...
        )
```

### Files Modified

1. **YOLOX_opt_elan**:
   - `projects/YOLOX_opt_elan/deploy/deploy_config.py`
   - `projects/YOLOX_opt_elan/deploy/main.py`

2. **CenterPoint**:
   - `projects/CenterPoint/deploy/deploy_config.py`
   - `projects/CenterPoint/deploy/main.py`

### Benefits

✅ **Declarative Configuration**: Model paths specified in config, not tracked programmatically
✅ **Eval-Only Mode**: Can evaluate existing models without export/checkpoint
✅ **Consistent Pattern**: All projects follow CalibrationStatusClassification standard
✅ **File Validation**: Automatically checks if model files exist before evaluation
✅ **Flexible**: Easy to enable/disable specific backends by commenting out paths

### Usage Example

**Evaluation-only mode** (no export needed):
```python
# deploy_config.py
export = dict(mode="none", ...)

evaluation = dict(
    enabled=True,
    models=dict(
        onnx="/workspace/work_dirs/model.onnx",
        tensorrt="/workspace/work_dirs/model.engine",
    ),
)
```

```bash
# No checkpoint needed!
python main.py \
    --deploy-cfg deploy_config.py \
    --model-cfg model_config.py
```

---
