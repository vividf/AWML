# TensorRT API Compatibility Fix

## Problem

After successfully exporting to ONNX, the TensorRT export failed with:

```
AttributeError: 'tensorrt_bindings.tensorrt.IBuilderConfig' object has no attribute 'max_workspace_size'
```

## Root Cause

TensorRT has undergone several API changes:

1. **TensorRT 8.4+**: `max_workspace_size` attribute was deprecated and replaced with `set_memory_pool_limit()`
2. **TensorRT 8.5+**: `build_engine()` was replaced with `build_serialized_network()` which returns serialized bytes directly

The original code used the old API which is no longer supported in newer TensorRT versions.

## Solution

Updated the TensorRT export code to support both old and new API versions using runtime detection:

### Workspace Size Configuration

```python
# Set workspace size (API changed in TensorRT 8.4+)
workspace_size = trt_settings["max_workspace_size"]
if hasattr(config_trt, 'max_workspace_size'):
    # TensorRT < 8.4
    config_trt.max_workspace_size = workspace_size
else:
    # TensorRT >= 8.4
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
```

### Engine Building

```python
# Build engine (API changed in TensorRT 8.5+)
if hasattr(builder, 'build_serialized_network'):
    # TensorRT >= 8.5: build_serialized_network returns serialized engine directly
    serialized_engine = builder.build_serialized_network(network, config_trt)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)
else:
    # TensorRT < 8.5: build_engine returns engine object
    engine = builder.build_engine(network, config_trt)
    with open(output_path, "wb") as f:
        f.write(engine.serialize())
```

## Changes Made

Updated `/home/yihsiangfang/ml_workspace/AWML/projects/YOLOX_opt_elan/deploy/main.py`:
- Modified `export_tensorrt()` function (lines 138-184)
- Added runtime API version detection using `hasattr()`
- Supports both TensorRT < 8.4 and >= 8.4 for workspace size configuration
- Supports both TensorRT < 8.5 and >= 8.5 for engine building
- Added informative logging messages indicating which API version is being used

## Why This Fix Works

1. **Backward Compatibility**: By checking for attribute/method existence at runtime, the code works with both old and new TensorRT versions.

2. **Forward Compatibility**: The code will continue to work even if you upgrade TensorRT in the future.

3. **Clear Logging**: The code logs which API is being used, making it easier to debug version-specific issues.

## TensorRT API Changes Summary

| TensorRT Version | Workspace Size API | Engine Building API |
|-----------------|-------------------|---------------------|
| < 8.4 | `config.max_workspace_size = size` | `builder.build_engine(network, config)` |
| 8.4 - 8.4.x | `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size)` | `builder.build_engine(network, config)` |
| >= 8.5 | `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size)` | `builder.build_serialized_network(network, config)` |

## Expected Behavior After Fix

When running the deployment script with TensorRT >= 8.4, you should see:

```
INFO:deployment:================================================================================
INFO:deployment:Exporting to TensorRT
INFO:deployment:================================================================================
INFO:deployment:Set WORKSPACE memory pool limit to <size> bytes (TensorRT >= 8.4)
INFO:deployment:Enabled FP16 precision
INFO:deployment:Building TensorRT engine (this may take a while)...
INFO:deployment:✅ TensorRT export successful: work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.engine
```

The engine building process should complete without API errors.

## Testing

To verify the fix works, run the full deployment pipeline:
```bash
cd /workspace/projects/YOLOX_opt_elan/deploy
python main.py --deploy-cfg <config> --model-cfg <config> --checkpoint <checkpoint>
```

Both ONNX and TensorRT exports should now complete successfully.
