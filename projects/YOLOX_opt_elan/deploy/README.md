# YOLOX_opt_elan Deployment

Complete deployment pipeline for YOLOX_opt_elan traffic object detection using the unified Autoware ML deployment framework.

## Features

- ✅ Export to ONNX and TensorRT
- ✅ Cross-backend verification
- ✅ Full evaluation with mAP metrics
- ✅ Latency benchmarking
- ✅ Uses MMDet pipeline for consistency with training
- ✅ Supports T4Dataset format
- ✅ Object detection with 8 classes
## Model Details

**YOLOX_opt_elan** is an optimized YOLOX variant for traffic object detection with:
- **Backbone**: ELAN-Darknet
- **Neck**: YOLOX-PAFPN with ELAN blocks
- **Input Size**: 960x960
- **Classes**: 8 traffic object categories
  1. unknown
  2. car
  3. truck
  4. bus
  5. trailer
  6. motorcycle
  7. pedestrian
  8. bicycle

## Quick Start

### 1. Prepare Data

Make sure you have T4Dataset format annotations:
```
data/t4dataset/
└── samrat/
    ├── yolox_infos_train.json
    ├── yolox_infos_val.json
    └── images/
```

### 2. Export and Evaluate

```bash
# Export to both ONNX and TensorRT with evaluation
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    path/to/checkpoint.pth \
    --work-dir work_dirs/yolox_opt_elan_deployment
```

### 3. Evaluation Only

If you already have exported models:

```python
# In deploy_config.py, set:
export = dict(
    mode='none',  # Skip export
    ...
)

evaluation = dict(
    enabled=True,
    models_to_evaluate=['onnx', 'tensorrt']
)

runtime_io = dict(
    onnx_file='work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx',
    ...
)
```

Then run:
```bash
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py
```

## Configuration

### Export Settings

```python
export = dict(
    mode='both',      # 'onnx', 'trt', 'both', 'none'
    verify=True,      # Cross-backend verification
    device='cuda:0',  # Device
    work_dir='work_dirs/yolox_opt_elan_deployment'
)
```

### Evaluation Settings

```python
evaluation = dict(
    enabled=True,
    num_samples=100,  # -1 for all samples
    verbose=False,
    models_to_evaluate=['pytorch', 'onnx', 'tensorrt']
)
```

### ONNX Settings

```python
onnx_config = dict(
    opset_version=16,
    input_names=['images'],
    output_names=['outputs'],
    dynamic_axes={
        'images': {0: 'batch_size'},
        'outputs': {0: 'batch_size'}
    }
)
```

### TensorRT Settings

```python
backend_config = dict(
    common_config=dict(
        precision_policy='fp16',  # Use FP16 for better performance
        max_workspace_size=1 << 30  # 1 GB
    ),
    model_inputs=[
        dict(
            name='images',
            shape=(1, 3, 960, 960),  # 960x960 input size
            dtype='float32'
        )
    ]
)
```

## Output Example

```
================================================================================
YOLOX_opt_elan Object Detection - Evaluation Results
================================================================================

Detection Metrics:
  mAP (0.5:0.95): 0.7523
  mAP @ IoU=0.50: 0.8912
  mAP @ IoU=0.75: 0.7834

Per-Class AP (Object Classes):
  green                    : 0.8456
  yellow                   : 0.8123
  red                      : 0.8678
  green_arrow_straight     : 0.7234
  yellow_arrow_straight    : 0.6987
  red_arrow_straight       : 0.7123
  green_arrow_left         : 0.6890
  yellow_arrow_left        : 0.6693

Latency Statistics:
  Mean:   8.45 ms
  Std:    0.52 ms
  Min:    7.82 ms
  Max:    11.23 ms
  Median: 8.41 ms
  P95:    9.34 ms
  P99:    10.12 ms

Total Samples: 100
================================================================================
```

## Comparison with Legacy Deployment

### Legacy Approach (`deploy_yolox_s_opt.py`)

The old deployment script:
- Downloads and installs Tier4 YOLOX repository
- Converts checkpoint format from MMDet to Tier4 YOLOX format
- Exports to ONNX using Tier4's export script
- Adds EfficientNMS_TRT plugin manually
- **No evaluation or verification**
- **No latency benchmarking**
- **Complex checkpoint conversion process**

### New Unified Framework Approach

The new deployment:
- ✅ Uses unified deployment framework across all models
- ✅ Direct MMDet model export (no conversion needed)
- ✅ Multi-backend support (PyTorch, ONNX, TensorRT)
- ✅ Cross-backend verification
- ✅ Complete evaluation with mAP metrics
- ✅ Latency benchmarking
- ✅ Consistent pipeline with training
- ✅ Better maintainability and extensibility

## Architecture

```
projects/YOLOX_opt_elan/deploy/
├── __init__.py              # Module exports
├── data_loader.py           # YOLOXOptElanDataLoader (T4Dataset support)
├── evaluator.py             # YOLOXOptElanEvaluator (Object detection metrics)
├── deploy_config.py         # Example deployment configuration
├── main.py                  # Main deployment pipeline
└── README.md                # This file
```

## Key Differences from Standard YOLOX

1. **Input Size**: 960x960 instead of 640x640
2. **Backbone**: ELAN-Darknet instead of CSPDarknet
3. **Neck**: YOLOX-PAFPN with ELAN blocks
4. **Dataset**: T4Dataset (COCO format) for dynamic objects
5. **Classes**: 8 object classes instead of 80 COCO classes

## Integration with Autoware

After deployment, the exported models can be used with:
- Autoware perception stack
- ROS2 object detection nodes
- Real-time object detection in autonomous vehicles

## Troubleshooting

### Issue: Cannot find T4Dataset annotations

**Solution**: Ensure the `ann_file` path in `deploy_config.py` points to the correct T4Dataset JSON file:
```python
runtime_io = dict(
    ann_file="data/t4dataset/samrat/yolox_infos_val.json",
    ...
)
```

### Issue: Image paths not found

**Solution**: If images have full paths in annotations, set `img_prefix=""` in config. Otherwise, set it to the image directory.

### Issue: Model export fails

**Solution**: Make sure all custom modules are imported:
```python
custom_imports = dict(
    imports=[
        'projects.YOLOX_opt_elan.yolox',
        'projects.YOLOX_opt_elan.yolox.models',
    ],
    allow_failed_imports=False,
)
```

## References

- [Autoware ML Deployment Framework](../../../autoware_ml/deployment/README.md)
- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [T4Dataset Documentation](../../../docs/design/architecture_dataset.md)
