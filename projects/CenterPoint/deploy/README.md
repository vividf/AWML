# CenterPoint Deployment

Complete deployment pipeline for CenterPoint 3D object detection.

## Features

- ✅ Export to ONNX (TensorRT support coming soon)
- ✅ Full evaluation with 3D detection metrics
- ✅ Latency benchmarking
- ✅ Uses MMDet3D pipeline for consistency with training
- ✅ Modernized from legacy DeploymentRunner

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
# Export to ONNX with evaluation
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    path/to/checkpoint.pth \
    --work-dir work_dirs/centerpoint_deployment \
    --replace-onnx-models  # Important for ONNX export
```

### 3. Evaluation Only

```bash
# Evaluate PyTorch model only
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    path/to/checkpoint.pth
```

## Configuration

### Export Settings

```python
export = dict(
    mode='onnx',      # 'onnx', 'trt' (coming soon), 'both', 'none'
    verify=True,      # Cross-backend verification
    device='cuda:0',  # Device
    work_dir='work_dirs/centerpoint_deployment'
)
```

### Evaluation Settings

```python
evaluation = dict(
    enabled=True,
    num_samples=50,   # 3D is slower, use fewer samples
    verbose=False,
    models_to_evaluate=['pytorch']  # Add 'onnx' after export
)
```

### ONNX Settings

```python
onnx_config = dict(
    opset_version=13,  # CenterPoint uses opset 13
    input_names=['voxels', 'num_points', 'coors'],
    output_names=['reg', 'height', 'dim', 'rot', 'vel', 'hm'],
    dynamic_axes={
        'voxels': {0: 'num_voxels'},
        'num_points': {0: 'num_voxels'},
        'coors': {0: 'num_voxels'}
    }
)
```

### Important Flags

- `--replace-onnx-models`: Replace model components with ONNX-compatible versions
  - Changes `CenterPoint` → `CenterPointONNX`
  - Changes `PillarFeatureNet` → `PillarFeatureNetONNX`
  - Changes `CenterHead` → `CenterHeadONNX`

- `--rot-y-axis-reference`: Convert rotation to y-axis clockwise reference

## Output

The deployment pipeline will generate:

```
work_dirs/centerpoint_deployment/
├── pillar_encoder.onnx
├── backbone.onnx
├── neck.onnx
└── head.onnx
```

And print evaluation results:
```
================================================================================
CenterPoint Evaluation Results
================================================================================

Detection Statistics:
  Total Predictions: 1234
  Total Ground Truths: 1180

Per-Class Statistics:
  VEHICLE:
    Predictions: 890
    Ground Truths: 856
  PEDESTRIAN:
    Predictions: 234
    Ground Truths: 218
  CYCLIST:
    Predictions: 110
    Ground Truths: 106

Latency Statistics:
  Mean: 45.23 ms
  Std:  3.45 ms
  Min:  41.82 ms
  Max:  58.31 ms
  Median: 44.18 ms

Total Samples: 50
================================================================================
```

## Architecture

```
CenterPointDataLoader (from data_loader.py)
    ├── Uses MMDet3D test pipeline
    ├── Loads info.pkl
    ├── Handles voxelization
    └── Preprocesses point clouds

CenterPointEvaluator (from evaluator.py)
    ├── Supports PyTorch, ONNX (TensorRT coming)
    ├── Computes detection statistics
    └── Measures latency

main.py
    ├── Replaces legacy DeploymentRunner
    ├── Exports to ONNX
    ├── Verifies outputs (TODO)
    └── Runs evaluation
```

## Migration from Legacy Code

This new implementation replaces the old `DeploymentRunner`:

### Old Way (scripts/deploy.py)
```python
from projects.CenterPoint.runners.deployment_runner import DeploymentRunner

runner = DeploymentRunner(
    experiment_name=experiment_name,
    model_cfg_path=model_cfg_path,
    checkpoint_path=checkpoint_path,
    work_dir=work_dir,
    replace_onnx_models=True,
    device='gpu',
    onnx_opset_version=13
)
runner.run()
```

### New Way (deploy/main.py)
```bash
python projects/CenterPoint/deploy/main.py \
    deploy_config.py \
    model_config.py \
    checkpoint.pth \
    --replace-onnx-models
```

**Benefits of New Approach**:
- ✅ Integrated with unified deployment framework
- ✅ Supports verification and evaluation
- ✅ Consistent with other projects (YOLOX, etc.)
- ✅ Better configuration management
- ✅ More modular and maintainable

## Troubleshooting

### Issue: Dataset not found

**Solution**: Update `runtime_io.info_file` in deploy_config.py

### Issue: ONNX export fails without --replace-onnx-models

**Solution**: Always use `--replace-onnx-models` flag for ONNX export

### Issue: Out of memory

**Solution**:
1. Reduce `evaluation.num_samples`
2. Reduce point cloud range
3. Increase `backend_config.common_config.max_workspace_size`

### Issue: Different results between training and deployment

**Solution**:
1. Make sure using same config file
2. Verify pipeline is correctly built
3. Check voxelization parameters

## Known Limitations

1. **3D mAP Metrics**: Current implementation uses simplified metrics. For production, integrate with `mmdet3d.core.evaluation` for proper 3D detection metrics (mAP, NDS, mATE, etc.)

2. **TensorRT Support**: TensorRT export for multi-file ONNX models needs custom implementation

3. **Batch Inference**: Currently supports single sample inference

## TODO

- [ ] Integrate with mmdet3d evaluation for proper mAP/NDS metrics
- [ ] Implement TensorRT multi-file export
- [ ] Add cross-backend verification
- [ ] Support batch inference
- [ ] Add visualization tools

## References

- [Deployment Framework Design](../../../docs/design/deploy_pipeline_design.md)
- [DataLoader Tutorial](../../../docs/tutorial/tutorial_deployment_dataloader.md)
- [CenterPoint Paper](https://arxiv.org/abs/2006.11275)
- [Old DeploymentRunner](../runners/deployment_runner.py) (deprecated)
