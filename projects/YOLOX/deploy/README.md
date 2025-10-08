# YOLOX Deployment

Complete deployment pipeline for YOLOX 2D object detection.

## Features

- ✅ Export to ONNX and TensorRT
- ✅ Cross-backend verification
- ✅ Full evaluation with mAP metrics
- ✅ Latency benchmarking
- ✅ Uses MMDet pipeline for consistency with training

## Quick Start

### 1. Prepare Data

Make sure you have COCO format dataset:
```
data/coco/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

### 2. Export and Evaluate

```bash
# Export to both ONNX and TensorRT with evaluation
python projects/YOLOX/deploy/main.py \
    projects/YOLOX/deploy/deploy_config.py \
    projects/YOLOX/configs/yolox_s_8xb8-300e_coco.py \
    path/to/checkpoint.pth \
    --work-dir work_dirs/yolox_deployment
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
    onnx_file='work_dirs/yolox_deployment/yolox.onnx',
    ...
)
```

Then run:
```bash
python projects/YOLOX/deploy/main.py \
    projects/YOLOX/deploy/deploy_config.py \
    projects/YOLOX/configs/yolox_s_8xb8-300e_coco.py
```

## Configuration

### Export Settings

```python
export = dict(
    mode='both',      # 'onnx', 'trt', 'both', 'none'
    verify=True,      # Cross-backend verification
    device='cuda:0',  # Device
    work_dir='work_dirs/yolox_deployment'
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
        precision_policy='fp16',  # 'auto', 'fp16', 'fp32_tf32'
        max_workspace_size=1 << 30  # 1 GB
    )
)
```

## Output

The deployment pipeline will generate:

```
work_dirs/yolox_deployment/
├── yolox.onnx           # ONNX model
└── yolox.engine         # TensorRT engine
```

And print evaluation results:
```
================================================================================
YOLOX Evaluation Results
================================================================================

Detection Metrics:
  mAP (0.5:0.95): 0.3742
  mAP @ IoU=0.50: 0.5683
  mAP @ IoU=0.75: 0.3912

Latency Statistics:
  Mean: 5.23 ms
  Std:  0.45 ms
  Min:  4.82 ms
  Max:  7.31 ms
  Median: 5.18 ms

Total Samples: 100
================================================================================
```

## Architecture

```
YOLOXDataLoader (from data_loader.py)
    ├── Uses MMDet test pipeline
    ├── Loads COCO annotations
    └── Preprocesses images

YOLOXEvaluator (from evaluator.py)
    ├── Supports PyTorch, ONNX, TensorRT
    ├── Computes mAP metrics
    └── Measures latency

main.py
    ├── Exports to ONNX/TensorRT
    ├── Verifies outputs
    └── Runs evaluation
```

## Troubleshooting

### Issue: COCO dataset not found

**Solution**: Update `runtime_io.ann_file` and `runtime_io.img_prefix` in deploy_config.py

### Issue: TensorRT export fails

**Solution**:
1. Make sure TensorRT is installed
2. Check ONNX model is valid
3. Try different precision policy

### Issue: Out of memory

**Solution**:
1. Reduce `evaluation.num_samples`
2. Use smaller input size
3. Set `backend_config.common_config.max_workspace_size` to smaller value

## Advanced Usage

### Custom Dataset

Modify `YOLOXDataLoader` in `data_loader.py` to support your dataset format.

### Custom Metrics

Extend `YOLOXEvaluator` in `evaluator.py` to add custom metrics.

### Batch Inference

Currently supports batch size = 1. For batch inference, modify:
1. ONNX dynamic axes
2. DataLoader to return batches
3. Evaluator to handle batched outputs

## References

- [Deployment Framework Design](../../../docs/design/deploy_pipeline_design.md)
- [DataLoader Tutorial](../../../docs/tutorial/tutorial_deployment_dataloader.md)
- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
