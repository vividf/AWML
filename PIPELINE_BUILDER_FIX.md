# Pipeline Builder Fix Summary

## Problem

The error occurred when trying to build the preprocessing pipeline for YOLOX_opt_elan deployment:

```
KeyError: 'PackDetInputs is not in the mmengine::transform registry. Please check whether the value of `PackDetInputs` is correct or it was registered as expected.'
```

## Root Cause

The issue was in `/home/yihsiangfang/ml_workspace/AWML/autoware_ml/deployment/utils/pipeline_builder.py`:

1. **Registry Mismatch**: The code was importing `mmdet.datasets.transforms` to register MMDetection-specific transforms like `PackDetInputs`, but then using `mmcv.transforms.Compose` which defaults to the base `mmengine::transform` registry.

2. **MMDetection 3.x Architecture**: In MMDetection 3.x, transforms are registered in `mmdet.registry.TRANSFORMS`, not in the base `mmengine.registry.TRANSFORMS`. When `Compose` is instantiated without specifying a registry, it uses the base registry which doesn't have MMDetection-specific transforms.

## Solution

Updated all pipeline building functions to use `init_default_scope()` to set the correct registry scope before building the pipeline:

### For 2D Detection (MMDetection)
```python
# Import mmdet transforms to register them
import mmdet.datasets.transforms  # noqa: F401
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

# Set default scope to 'mmdet' so Compose uses mmdet's TRANSFORMS registry
init_default_scope('mmdet')
return Compose(pipeline_cfg)
```

### For 3D Detection (MMDetection3D)
```python
# Import mmdet3d transforms to register them
import mmdet3d.datasets.transforms  # noqa: F401
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

# Set default scope to 'mmdet3d' so Compose uses mmdet3d's TRANSFORMS registry
init_default_scope('mmdet3d')
return Compose(pipeline_cfg)
```

### For Classification (MMPretrain)
```python
# Import mmpretrain transforms to register them
import mmpretrain.datasets.transforms  # noqa: F401
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

# Set default scope to 'mmpretrain' so Compose uses mmpretrain's TRANSFORMS registry
init_default_scope('mmpretrain')
return Compose(pipeline_cfg)
```

### For Segmentation (MMSegmentation)
```python
# Import mmseg transforms to register them
import mmseg.datasets.transforms  # noqa: F401
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

# Set default scope to 'mmseg' so Compose uses mmseg's TRANSFORMS registry
init_default_scope('mmseg')
return Compose(pipeline_cfg)
```

## Changes Made

Updated 4 functions in `pipeline_builder.py`:
1. `_build_detection2d_pipeline()` - Lines 161-221
2. `_build_detection3d_pipeline()` - Lines 224-284
3. `_build_classification_pipeline()` - Lines 287-353
4. `_build_segmentation_pipeline()` - Lines 356-419

Each function now:
1. **First attempts**: Import the appropriate framework's transforms, call `init_default_scope()` to set the registry scope, then build Compose
2. **Fallback attempts**: Try without `init_default_scope()` for older versions
3. **Legacy attempts**: Try older import paths for backward compatibility
4. **Error handling**: Provides detailed error messages if all attempts fail

## Why This Fix Works

1. **Default Scope**: `init_default_scope()` sets the default registry scope in mmengine, which tells `Compose` which registry to use when looking up transforms.

2. **Proper Registration**: MMDetection's `PackDetInputs`, `LoadImageFromFile`, and other detection-specific transforms are registered in `mmdet.registry.TRANSFORMS`. By calling `init_default_scope('mmdet')`, we tell mmengine to use mmdet's registry by default.

3. **How mmengine Works**: The `Compose` class from `mmengine.dataset` uses `TRANSFORMS.build()` which looks up transforms in the registry. The registry used depends on the current default scope set by `init_default_scope()`.

4. **Backward Compatibility**: The code maintains fallback mechanisms for different MMDetection/MMCV versions.

## Expected Behavior After Fix

When running the deployment script, you should now see:
```
INFO:autoware_ml.deployment.utils.pipeline_builder:Found test pipeline at: test_dataloader.dataset.pipeline
INFO:autoware_ml.deployment.utils.pipeline_builder:Building 2D detection pipeline with mmengine.dataset.Compose (default_scope='mmdet')
```

The pipeline should build successfully without the KeyError.

## Testing

To verify the fix works, run your deployment script:
```bash
cd /workspace/projects/YOLOX_opt_elan/deploy
python main.py --deploy-cfg <config> --model-cfg <config> --checkpoint <checkpoint>
```

The error should no longer occur at the pipeline building stage.
