# CenterPoint Deployment

Deployment pipeline for CenterPoint 3D object detection using the unified deployment framework.

## Features

- Export to ONNX and TensorRT (multi-file architecture)
- Full evaluation with 3D detection metrics (autoware_perception_evaluation)
- Latency benchmarking
- Uses MMDet3D pipeline for consistency with training
- Unified runner architecture with composition-based design
- **INT8 Quantization Support**: PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training) for improved inference speed

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

## Quantization Support

CenterPoint deployment supports INT8 quantization through PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training) to improve inference speed by 1.5x-2x with minimal accuracy loss (<1% mAP drop).

### Installation

Install the quantization dependencies:

```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

### Quick Start: PTQ (Post-Training Quantization)

PTQ is the fastest way to quantize a pre-trained model without retraining.

#### Method 1: Using CLI Tool

```bash
# Quantize a pre-trained model
python tools/detection3d/centerpoint_quantization.py ptq \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    --checkpoint work_dirs/centerpoint/best.pth \
    --calibrate-batches 100 \
    --output work_dirs/centerpoint_ptq.pth
```

#### Method 2: Using Python API

```python
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.runner import Runner
from projects.CenterPoint.quantization import quantize_ptq
import torch

# Load model
cfg = Config.fromfile("projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py")
model = init_model(cfg, "work_dirs/centerpoint/best.pth", device="cuda:0")
model.eval()

# Build calibration dataloader
dataloader = Runner.build_dataloader(cfg.val_dataloader)

# Apply PTQ
quantized_model = quantize_ptq(
    model,
    dataloader,
    num_calibration_batches=100,
    amax_method="mse",
    fuse_bn=True,
)

# Save quantized model
torch.save({'state_dict': quantized_model.state_dict()}, 'work_dirs/centerpoint_ptq.pth')
```

### Sensitivity Analysis

Before quantizing, you can identify which layers are sensitive to quantization:

```bash
python tools/detection3d/centerpoint_quantization.py sensitivity \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    --checkpoint work_dirs/centerpoint/best.pth \
    --calibrate-batches 100 \
    --output sensitivity_report.csv
```

This generates a CSV report showing which layers cause the most accuracy drop when quantized. You can then skip these layers:

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --config ... \
    --checkpoint ... \
    --skip-layers pts_backbone.blocks.0.0 pts_voxel_encoder.pfn_layers.0 \
    --output ...
```

### QAT (Quantization-Aware Training)

For better accuracy, use QAT to fine-tune the model with quantization:

#### Step 1: Add QAT Hook to Config

Modify your training config to include the QAT hook:

```python
# In your config file (e.g., second_secfpn_4xb16_121m_base_amp.py)
custom_hooks = [
    dict(
        type='QATHook',
        calibration_batches=100,      # Number of batches for initial calibration
        calibration_epoch=0,           # Epoch to run calibration
        freeze_bn=True,                # Fuse BatchNorm layers
        sensitive_layers=[],           # Layers to skip quantization
        amax_method='mse',             # Method for computing amax
    ),
]

# Reduce learning rate for fine-tuning
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),  # 10x smaller than original
)

# Shorter training for fine-tuning
train_cfg = dict(max_epochs=10)
```

#### Step 2: Run Training

```bash
python tools/train.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    --work-dir work_dirs/centerpoint_qat
```

Or use the CLI tool:

```bash
python tools/detection3d/centerpoint_quantization.py qat \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    --checkpoint work_dirs/centerpoint/best.pth \
    --calibrate-batches 100 \
    --epochs 10 \
    --lr 0.0001 \
    --output work_dirs/centerpoint_qat.pth
```

### Deployment with Quantized Model

To deploy a quantized model, use the INT8 deployment config:

```bash
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/configs/deploy_config_int8.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py
```

The `deploy_config_int8.py` config includes quantization settings:

```python
quantization = dict(
    enabled=True,
    mode="ptq",  # or "qat"
    calibration=dict(
        num_batches=100,
        method="histogram",
        amax_method="mse",
    ),
    fusion=dict(
        fuse_bn=True,
    ),
    sensitive_layers=[],  # Add layers from sensitivity analysis
)

backend_config = dict(
    common_config=dict(
        precision_policy="int8",  # Use INT8 for TensorRT
        max_workspace_size=4 << 30,
    ),
)
```

### Quantization Module Structure

The quantization implementation is organized as follows:

```
projects/CenterPoint/quantization/
├── modules/
│   ├── quant_conv.py          # QuantConv2d, QuantConvTranspose2d
│   ├── quant_linear.py        # QuantLinear for PillarFeatureNet
│   └── quant_add.py          # QuantAdd for skip connections
├── calibration/
│   └── calibrator.py         # CalibrationManager for PTQ
├── fusion/
│   └── bn_fusion.py         # BatchNorm fusion utilities
├── hooks/
│   └── qat_hook.py          # QATHook for MMEngine training
├── ptq.py                   # PTQ pipeline functions
├── replace.py               # Module replacement functions
├── sensitivity.py           # Layer sensitivity analysis
└── utils.py                 # Utility functions
```

### API Reference

#### Core Functions

```python
from projects.CenterPoint.quantization import (
    # PTQ pipeline
    quantize_ptq,
    save_ptq_model,
    load_ptq_model,

    # Module replacement
    quant_model,
    quant_conv_module,
    quant_linear_module,

    # Calibration
    CalibrationManager,

    # Layer fusion
    fuse_model_bn,

    # Sensitivity analysis
    build_sensitivity_profile,
    get_sensitive_layers,

    # Utilities
    disable_quantization,
    enable_quantization,
    print_quantizer_status,
)
```

#### Example: Custom PTQ Pipeline

```python
from projects.CenterPoint.quantization import (
    quant_model,
    fuse_model_bn,
    CalibrationManager,
    disable_quantization,
)

# 1. Fuse BatchNorm
model.eval()
fuse_model_bn(model)

# 2. Insert Q/DQ nodes
quant_model(model, skip_names={'pts_backbone.blocks.0.0'})

# 3. Calibrate
calibrator = CalibrationManager(model)
calibrator.calibrate(dataloader, num_batches=100, method='mse')

# 4. Disable sensitive layers
disable_quantization(model.pts_backbone.blocks[0][0]).apply()
```

### Performance Expectations

Based on CUDA-CenterPoint results:

| Model | Validation mAP | Validation NDS | Speedup |
|-------|----------------|----------------|---------|
| FP16 Baseline | ~59.55% | ~66.75% | 1x |
| PTQ INT8 | ~59.08% | ~66.45% | 1.5x-2x |
| QAT INT8 | ~59.20% | ~66.53% | 1.5x-2x |

### Troubleshooting

#### Calibration Issues

1. **Out of Memory**: Reduce `calibrate_batches` or use smaller batch size
2. **Poor Accuracy**: Run sensitivity analysis and skip sensitive layers
3. **Calibration Cache**: Save/load calibration cache to avoid re-calibration:
   ```python
   calibrator.save_calib_cache('calib_cache.pth')
   calibrator.load_calib_cache('calib_cache.pth')
   ```

#### QAT Training Issues

1. **NaN Loss**: Reduce learning rate or disable quantization for more layers
2. **Slow Training**: Enable fast histogram mode (done automatically)
3. **No Improvement**: Increase training epochs or adjust calibration

#### TensorRT INT8 Build Issues

1. **Engine Build Fails**: Ensure ONNX has Q/DQ nodes (check with Netron)
2. **Accuracy Mismatch**: Verify calibration cache was loaded correctly
3. **Performance Not Improved**: Check that `precision_policy="int8"` is set

### References

- [Quantization Integration Plan](../docs/AWML_integrate_plan.md) - Detailed implementation plan
- [Deployment Framework Documentation](../../../deployment/README.md)
- [CenterPoint Paper](https://arxiv.org/abs/2006.11275)
- [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
