# INT8 Sparse Encoder TensorRT Deployment — Gap Analysis

## 1. Problem Statement

After successful PyTorch INT8 PTQ (mAP 0.3556 with NVIDIA `pytorch_quantization` + histogram/MSE calibration), deploying through the standard AWML TensorRT pipeline yields **no speed improvement** for the sparse encoder (~6.14 ms, identical to FP16).

## 2. Root Cause

The current AWML deployment path cannot run INT8 sparse convolution for two independent reasons:

### 2.1 ONNX Export Gap

The sparse encoder is exported via `sparse_encoder_float_shadow.py`, which builds a clean **FP32** encoder and runs `torch.onnx.export`. This is necessary because the FX-quantized graph contains `aten::_empty_affine_quantized` ops that the standard ONNX exporter cannot handle.

**Result**: the ONNX model contains standard `SparseConvolution` ops with FP32/FP16 weights and **no quantization metadata** (no scales, no dynamic ranges).

### 2.2 TensorRT Plugin Gap

The Autoware `IndiceConvPlugin` (`indice_conv_plugin.cpp`) only accepts `kFLOAT` and `kHALF`:

```cpp
// indice_conv_plugin.cpp line ~146
bool IndiceConvPlugin::supportsFormatCombination(int pos, ...) {
  return in[pos].type == nvinfer1::DataType::kFLOAT ||
         in[pos].type == nvinfer1::DataType::kHALF;
}
```

Even if INT8 tensors reached the plugin, it would reject them.

### 2.3 spconv Author's Warning

From `spconv/docs/TENSORRT_INT8_GUIDE.md`:

> "There is an important drawback in tensorrt int8: tensorrt won't fuse QDQ for custom int8 plugins. So we must fuse QDQ by ourself (in pytorch)"

TensorRT's automatic Q/DQ fusion only works for its built-in layers. Custom plugins (like `IndiceConvPlugin`) require manual scale/bias handling in the plugin's `enqueue()` method.

## 3. How CUDA-BEVFusion Achieves Real INT8

NVIDIA's CUDA-BEVFusion uses a **completely different runtime** for the sparse encoder:

```
PyTorch PTQ/QAT
    │
    ▼  custom exptool.py (NOT torch.onnx.export)
Custom ONNX with SparseConvolution nodes
    │  attributes: precision="int8", input_dynamic_range, weight_dynamic_ranges
    │  weights: FP16 initializers
    ▼
libspconv ONNX parser (lidar-scn-onnx-parser.cpp)
    │  reads precision, dynamic_ranges → EngineBuilder::push_sparse_conv(Precision::Int8)
    ▼
libspconv Engine (cumm INT8 kernels)
    │  input: FP16 features + INT32 indices
    │  output: FP16 BEV feature map
    ▼
TensorRT Engine (dense backbone/neck/head, FP16)
```

Key characteristics:
- Sparse encoder runs on **libspconv**, NOT on TensorRT
- Dense backbone/neck/head runs on **TensorRT** (FP16)
- The two engines connect at the BEV feature map boundary
- `libspconv.so` is a prebuilt proprietary library with cumm INT8 implicit_gemm kernels

### 3.1 Custom ONNX Format

The custom ONNX uses these non-standard op types:

| Op Type | Description | INT8 Attributes |
|---------|-------------|-----------------|
| `SparseConvolution` | Sparse conv + metadata | `precision`, `output_precision`, `input_dynamic_range`, `weight_dynamic_ranges` |
| `Relu` | Standard ReLU on features | — |
| `Add` / `QuantAdd` | Residual add | `input0_dynamic_range`, `input1_dynamic_range`, `precision`, `output_precision` |
| `ScatterDense` | Sparse → dense scatter | `format` (xyz/zyx) |
| `Reshape` | Dense tensor reshape | `dims` |
| `Transpose` | Dense tensor permute | `dims` |

### 3.2 Precision Assignment

From CUDA-BEVFusion `export-scn.py`:

```python
for name, module in model.named_modules():
    module.precision = "int8"
    module.output_precision = "int8"

model.conv_input.precision = "fp16"         # first layer input stays FP16
model.conv_out.output_precision = "fp16"    # last layer output back to FP16
```

The `conv_input` layer takes FP16 features from voxelization and keeps input precision as FP16. All middle layers run INT8. The `conv_out` output is FP16 so it connects to the dense TensorRT engine.

### 3.3 Dynamic Range Mapping

Our PTQ `_amax` values map directly to libspconv's attributes:

| PTQ Value | libspconv Attribute | Description |
|-----------|-------------------|-------------|
| `_input_quantizer._amax` (scalar) | `input_dynamic_range` (float) | Activation clipping range |
| `_weight_quantizer._amax` (per-channel) | `weight_dynamic_ranges` (float[]) | Per-output-channel weight range |

### 3.4 INT8 Kernel Constraints

From `spconv/docs/INT8_GUIDE.md`:
- `input_channel % 32 == 0 && output_channel % 32 == 0` for INT8 kernels
- Speed advantage when `C >= 64 && K >= 64`

Our encoder channel progression: **5 → 16 → 32 → 64 → 64 → 128**

| Layer | In Ch | Out Ch | INT8 eligible | Expected speedup |
|-------|-------|--------|---------------|-----------------|
| conv_input | 5 | 16 | No (5%32≠0) | None — runs FP16 |
| encoder_layer1 | 16 | 16 | No (16%32≠0) | None |
| encoder_layer2 | 16→32 | 32 | Partial | Minimal |
| encoder_layer3 | 32→64 | 64 | Yes | Moderate |
| encoder_layer4 | 64→64 | 64 | Yes | Moderate |
| conv_out | 64 | 128 | Yes | Good |

Layers 3-4 and conv_out benefit most from INT8 (~40-60% of encoder compute).

## 4. Available Assets

### 4.1 Prebuilt libspconv.so

Located at `Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/`:

| Platform | CUDA | Path |
|----------|------|------|
| x86_64 | 11.4 | `x86_64_cuda11.4/libspconv.so` |
| x86_64 | 12.8 | `x86_64_cuda12.8/libspconv.so` |
| x86_64 | 13.0 | `x86_64_cuda13.0/libspconv.so` |
| aarch64 | 11.4 | `aarch64_cuda11.4/libspconv.so` |
| aarch64 | 12.8 | `aarch64_cuda12.8/libspconv.so` |

### 4.2 Reference Code

| File | Purpose |
|------|---------|
| `CUDA-BEVFusion/qat/lean/exptool.py` | Custom ONNX exporter (monkey-patching tracer) |
| `CUDA-BEVFusion/qat/export-scn.py` | Export script (precision setup + export call) |
| `CUDA-BEVFusion/src/bevfusion/lidar-scn-onnx-parser.cpp` | C++ ONNX parser for libspconv |
| `CUDA-BEVFusion/src/bevfusion/lidar-scn.cpp` | C++ runtime (libspconv + voxelization) |
| `CUDA-CenterPoint/qat/onnx_export/export-scn.py` | CenterPoint variant with per-layer FP16 overrides |

### 4.3 Our PTQ Calibration

Checkpoint: `work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth`

Contains 40 `_amax` buffers (20 input quantizers + 20 weight quantizers) from NVIDIA `pytorch_quantization` with histogram + MSE calibration. These become the `input_dynamic_range` and `weight_dynamic_ranges` in the custom ONNX.

## 5. Implementation Plan

### Step 1: Custom ONNX Exporter

New file: `deployment/projects/bevfusion/experimental/libspconv_onnx_exporter.py`

Adapts CUDA-BEVFusion's `exptool.py` for spconv v2 and our BEVFusionSparseEncoder:
- Hooks `spconv.pytorch.conv.SparseConvolution.forward`
- Hooks `spconv.pytorch.SparseReLU.forward`
- Hooks `SparseConvTensor.dense`, `Tensor.permute`, `Tensor.reshape`
- Writes custom ONNX with `SparseConvolution` nodes including precision + dynamic_range attributes
- Weights stored as FP16 initializers

### Step 2: Export CLI Script

New file: `deployment/projects/bevfusion/experimental/export_sparse_encoder_int8.py`

1. Build fresh BEVFusionSparseEncoder from config
2. Fuse BN, load FP32 weights from PTQ checkpoint
3. Extract `_amax` → set `_input_dynamic_range` / `_weight_dynamic_ranges` on each module
4. Set `precision` / `output_precision` attributes
5. Convert to FP16
6. Call `libspconv_onnx_exporter.export_onnx()`

### Step 3: C++ Inference Bridge

New files under `deployment/projects/bevfusion/cpp/`:
- `libspconv_trt_bridge.hpp` / `.cpp`: loads libspconv engine (INT8 sparse) + TRT engine (FP16 dense), connects at BEV boundary
- `CMakeLists.txt`: build against libspconv.so + TensorRT

### Step 4: Benchmark

Measure sparse encoder latency: FP16 (current Autoware plugin) vs INT8 (libspconv).
Validate end-to-end mAP matches PyTorch INT8 evaluation.
