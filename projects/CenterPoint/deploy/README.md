# CenterPoint Deployment

Complete deployment pipeline for CenterPoint 3D object detection using the unified deployment framework.

## Features

- ✅ Export to ONNX and TensorRT (multi-file architecture)
- ✅ Full evaluation with 3D detection metrics (autoware_perception_evaluation)
- ✅ Latency benchmarking
- ✅ Uses MMDet3D pipeline for consistency with training
- ✅ Unified runner architecture with composition-based design

## Quick Start

### 1. Prepare Data

Make sure you have T4Dataset or similar 3D detection dataset:
```
data/t4dataset/
├── centerpoint_infos_train.pkl
├── centerpoint_infos_val.pkl
└── lidar/
    └── *.bin
```


### 2. Export and Evaluate

```bash
# Export to ONNX and TensorRT with evaluation
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/configs/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py
```

### 3. Export Modes

The pipeline supports different export modes configured in `deploy_config.py`:

```bash
# ONNX only
# Set export.mode = "onnx" in deploy_config.py

# TensorRT only (requires existing ONNX files)
# Set export.mode = "trt" and export.onnx_path = "path/to/onnx/dir"

# Both ONNX and TensorRT
# Set export.mode = "both"

# Evaluation only (no export)
# Set export.mode = "none"
```

## Configuration

All configuration is done through `deploy_config.py`. Key sections:

### Checkpoint Path

```python
# Single source of truth for PyTorch model
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"
```

### Export Settings

```python
export = dict(
    mode="both",      # 'onnx', 'trt', 'both', 'none'
    work_dir="work_dirs/centerpoint_deployment",
    onnx_path=None,   # Required when mode='trt'
)
```

### ONNX Settings

```python
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    save_file="centerpoint.onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,   # Set to True to run onnx-simplifier
    multi_file=True,  # CenterPoint uses multi-file ONNX
)
```

### Evaluation Settings

```python
evaluation = dict(
    enabled=True,
    num_samples=1,    # Number of samples to evaluate
    verbose=True,
    backends=dict(
        pytorch=dict(enabled=True, device="cuda:0"),
        onnx=dict(enabled=True, device="cuda:0", model_dir="..."),
        tensorrt=dict(enabled=True, device="cuda:0", engine_dir="..."),
    ),
)
```

### TensorRT Settings

```python
backend_config = dict(
    common_config=dict(
        precision_policy="auto",  # 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        max_workspace_size=2 << 30,  # 2 GB
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input_features=dict(
                    min_shape=[1000, 32, 11],
                    opt_shape=[20000, 32, 11],
                    max_shape=[64000, 32, 11],
                ),
                spatial_features=dict(
                    min_shape=[1, 32, 760, 760],
                    opt_shape=[1, 32, 760, 760],
                    max_shape=[1, 32, 760, 760],
                ),
            )
        )
    ],
)
```

### Verification Settings

```python
verification = dict(
    enabled=True,
    tolerance=1e-1,
    num_verify_samples=1,
    devices=devices,  # Reference to top-level devices dict
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu",
                 test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda",
                 test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[...],
        trt=[...],
        none=[],
    ),
)
```

## Architecture

CenterPoint uses a multi-file ONNX/TensorRT architecture:

```
CenterPoint Model
├── pts_voxel_encoder     → pts_voxel_encoder.onnx / .engine
└── pts_backbone_neck_head → pts_backbone_neck_head.onnx / .engine
```

### Component Extractor

The `CenterPointComponentExtractor` handles model-specific logic:
- Extracts voxel encoder and backbone+neck+head components
- Prepares sample inputs for each component
- Configures per-component ONNX export settings

### Deployment Runner

`CenterPointDeploymentRunner` orchestrates the export pipeline:
- Loads ONNX-compatible CenterPoint model
- Injects model and config to evaluator
- Delegates export to `CenterPointONNXExportPipeline` and `CenterPointTensorRTExportPipeline`

## Output Structure

After deployment:

```
work_dirs/centerpoint_deployment/
├── onnx/
│   ├── pts_voxel_encoder.onnx
│   └── pts_backbone_neck_head.onnx
└── tensorrt/
    ├── pts_voxel_encoder.engine
    └── pts_backbone_neck_head.engine
```

## Command Line Options

```bash
python projects/CenterPoint/deploy/main.py \
    <deploy_config.py> \
    <model_config.py> \
    [--rot-y-axis-reference]  # Convert rotation to y-axis clockwise reference
    [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

## Troubleshooting

### TensorRT Build Issues

1. **Memory Issues**: Increase `max_workspace_size` in `backend_config`
2. **Shape Issues**: Verify `model_inputs` shapes match your data
3. **Precision Issues**: Try different `precision_policy` settings

### Verification Failures

1. **Tolerance**: Increase `tolerance` in verification config
2. **Samples**: Reduce `num_verify_samples` for faster testing


## File Structure

```
projects/CenterPoint/deploy/
├── main.py                 # Entry point
├── configs/
│   └── deploy_config.py    # Deployment configuration
├── component_extractor.py  # Model-specific component extraction
├── data_loader.py          # CenterPoint data loader
├── evaluator.py            # CenterPoint evaluator
├── utils.py                # Utility functions
└── README.md               # This file
```

## References

- [Deployment Framework Documentation](../../../deployment/README.md)
- [CenterPoint Paper](https://arxiv.org/abs/2006.11275)
