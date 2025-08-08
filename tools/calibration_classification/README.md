# Calibration Classification Model Evaluation

This directory contains scripts for evaluating CalibrationStatusClassification models in different formats (ONNX and TensorRT).

## Scripts Overview

### 1. `test_onnx.py` - ONNX Model Evaluation
Evaluates ONNX models using ONNX Runtime backend.

**Usage:**
```bash
python test_onnx.py
```

**Features:**
- Evaluates ONNX models on the entire test dataset
- Calculates accuracy, per-class accuracy, and confidence
- Shows confusion matrix
- Uses CPU for inference

### 2. `test_tensorrt.py` - TensorRT Model Evaluation
Evaluates TensorRT models using TensorRT backend.

**Usage:**
```bash
python test_tensorrt.py
```

**Features:**
- Evaluates TensorRT models on the entire test dataset
- Calculates accuracy, per-class accuracy, and confidence
- Shows confusion matrix and latency statistics
- Uses CUDA for inference

### 3. `test_tensorrt_simple.py` - TensorRT with Fallback
Evaluates TensorRT models with automatic fallback to ONNX Runtime if TensorRT fails.

**Usage:**
```bash
python test_tensorrt_simple.py
```

**Features:**
- Attempts TensorRT evaluation first
- Automatically falls back to ONNX Runtime if TensorRT fails
- Handles library path issues automatically
- Provides detailed error messages

### 4. `compare_onnx_tensorrt.py` - Model Comparison
Compares ONNX and TensorRT models side by side.

**Usage:**
```bash
# Basic comparison
python compare_onnx_tensorrt.py

# Custom model paths
python compare_onnx_tensorrt.py \
    --onnx-model /path/to/onnx/model.onnx \
    --tensorrt-model /path/to/tensorrt/model.engine \
    --device cuda \
    --max-samples 100
```

**Features:**
- Compares accuracy, latency, and confidence between models
- Checks prediction consistency
- Provides detailed performance analysis

## Configuration Files

### ONNX Configuration
- **File:** `projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_onnxruntime.py`
- **Backend:** ONNX Runtime
- **Device:** CPU
- **Format:** ONNX (.onnx)

### TensorRT Configuration
- **File:** `projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_tensorrt.py`
- **Backend:** TensorRT
- **Device:** CUDA
- **Format:** TensorRT Engine (.engine)

## Model Conversion

Before running the evaluation scripts, you need to convert your trained model to ONNX and TensorRT formats.

### Convert to ONNX
```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    --deploy-cfg projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_onnxruntime.py \
    --model-cfg projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py \
    --checkpoint your_checkpoint.pth
```

### Convert to TensorRT
```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    --deploy-cfg projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_tensorrt.py \
    --model-cfg projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py \
    --checkpoint your_checkpoint.pth
```

## Output Examples

### ONNX Evaluation Output
```
==================================================
ONNX Model Evaluation Results
==================================================
Total samples: 100
Correct predictions: 85
Accuracy: 0.8500 (85.00%)

Per-class accuracy:
  Class 0: 42/50 = 0.8400 (84.00%)
  Class 1: 43/50 = 0.8600 (86.00%)

Average confidence: 0.8234

Confusion Matrix:
Predicted ->
Actual    0    1
  0      42    8
  1       7   43

ONNX model evaluation completed successfully!
Model accuracy: 0.8500 (85.00%)
```

### TensorRT Evaluation Output
```
==================================================
TensorRT Model Evaluation Results
==================================================
Total samples: 100
Correct predictions: 85
Accuracy: 0.8500 (85.00%)

Per-class accuracy:
  Class 0: 42/50 = 0.8400 (84.00%)
  Class 1: 43/50 = 0.8600 (86.00%)

Average confidence: 0.8234

Latency Statistics:
  Average latency: 12.34 ms
  Min latency: 10.12 ms
  Max latency: 15.67 ms
  Std latency: 1.23 ms

Confusion Matrix:
Predicted ->
Actual    0    1
  0      42    8
  1       7   43

TensorRT model evaluation completed successfully!
Model accuracy: 0.8500 (85.00%)
Average inference latency: 12.34 ms
```

### Comparison Output
```
============================================================
ONNX vs TensorRT Model Comparison
============================================================

Accuracy Comparison:
  ONNX:     0.8500 (85.00%)
  TensorRT: 0.8500 (85.00%)
  Difference: 0.0000

Latency Comparison:
  ONNX:     45.67 ms
  TensorRT: 12.34 ms
  Speedup:  3.70x

Confidence Comparison:
  ONNX:     0.8234
  TensorRT: 0.8234
  Difference: 0.0000

Prediction Consistency:
  Predictions match: True

Comparison completed successfully!
```

## Requirements

- Python 3.8+
- PyTorch
- MMDeploy
- ONNX Runtime
- TensorRT (for TensorRT evaluation)
- CUDA (for TensorRT evaluation)

## Notes

1. **Device Requirements:**
   - ONNX evaluation runs on CPU
   - TensorRT evaluation requires CUDA GPU

2. **Model Paths:**
   - Update the model paths in the scripts if your models are in different locations
   - Default paths are set to `/workspace/work_dirs/`

3. **Performance:**
   - TensorRT models typically provide better inference speed
   - ONNX models are more portable and easier to deploy

4. **Accuracy:**
   - Both models should produce similar accuracy results
   - Small differences may occur due to numerical precision differences

## Troubleshooting

### Common Issues

1. **Model not found:**
   - Ensure the model files exist at the specified paths
   - Check if the model conversion was successful

2. **CUDA out of memory:**
   - Reduce batch size or number of samples
   - Use `--max-samples` parameter to limit evaluation

3. **TensorRT errors:**
   - Ensure TensorRT is properly installed
   - Check CUDA version compatibility
   - Verify GPU memory availability

4. **Configuration errors:**
   - Ensure all config files exist
   - Check file paths in the scripts

### TensorRT Library Issues

If you encounter `libnvinfer.so.8: cannot open shared object file` errors:

#### Solution 1: Use the Simple Script
```bash
python test_tensorrt_simple.py
```
This script automatically handles library path issues and falls back to ONNX Runtime if TensorRT fails.

#### Solution 2: Set Library Path Manually
```bash
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
python test_tensorrt.py
```

#### Solution 3: Check TensorRT Installation
```bash
# Check TensorRT version
python -c "import tensorrt as trt; print(trt.__version__)"

# Find TensorRT libraries
find /opt/conda -name "libnvinfer.so*" 2>/dev/null
```

#### Solution 4: Use ONNX Runtime Instead
If TensorRT continues to have issues, use ONNX Runtime evaluation:
```bash
python test_onnx.py
```

### Recommended Workflow

1. **Start with ONNX evaluation** (most reliable):
   ```bash
   python test_onnx.py
   ```

2. **Try TensorRT with fallback**:
   ```bash
   python test_tensorrt_simple.py
   ```

3. **Compare both models**:
   ```bash
   python compare_onnx_tensorrt.py
   ```

4. **Use TensorRT directly** (if library issues are resolved):
   ```bash
   python test_tensorrt.py
   ```
