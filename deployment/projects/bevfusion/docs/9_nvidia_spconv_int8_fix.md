# BEVFusion Sparse Encoder INT8: NVIDIA Approach Fix

## Result Summary

| Metric | FP32 Baseline | INT8 (NVIDIA) | Retention |
|--------|:------------:|:-------------:|:---------:|
| **mAP** | **0.4041** | **0.3556** | **88.0%** |
| mAPH | 0.3522 | 0.2922 | 83.0% |
| car AP@4m | 0.6413 | 0.6306 | 98.3% |
| truck AP@4m | 0.9320 | 0.9347 | **100.3%** |
| sparse_encoder max | 8.21 | 8.06 | 98.2% |
| head_heatmap max | +3.37 | +3.43 | 101.8% |

## Problem History

### Phase 1: FX Approach — mAP = 0.0000

The original implementation used **spconv's FX graph mode quantization**:

```
prepare_fx → calibrate → convert_fx → transform_qdq → remove_conv_add_dq
```

Custom `SparseConvolution` modules from `projects/SparseConvolution/` were not in
spconv's `DEFAULT_SPARSE_CONV_TYPES`, causing FX to trace inside `_conv_forward`.
This led to:

1. **Incorrect spatial shape** — convolution stride was skipped during FX tracing,
   producing `[1440, 1440, 41]` output instead of `[720, 720, 21]`
2. **Channel mismatch** — `RuntimeError: expected 256 channels, got 5248`
3. **GPU OOM** — FX attached observers to every intermediate tensor inside
   `_conv_forward`, causing O(N²) memory

### Phase 2: non_traceable Fix — mAP = 0.0002

Adding custom SparseConv types to `prepare_custom_config.non_traceable_module_classes`
fixed the spatial shape and OOM issues, but created a new problem:

- `transform_qdq` and `remove_conv_add_dq` expect the FX graph to contain traced
  spconv ops (e.g., `SparseConvAddReLU`). With non-traceable modules, the graph
  only has opaque `call_module` nodes — the transforms cannot match the expected
  patterns and produce an incorrect Q/DQ graph.
- Activation peaks clipped from 8.21 → 2.45 (70% loss), destroying detection
  confidence: `head_heatmap max` flipped from +3.37 to −1.02.

### Phase 3: Observer Swap (MinMax) — mAP = 0.0007

Changing from `SparseHistogramObserver` to `SparseMinMaxObserver` barely helped
(max 2.45 → 2.81), confirming the issue was not in the observer/scale selection
but in the FX graph structure itself.

### Phase 4: NVIDIA Approach — mAP = 0.3556 ✓

Adopted the approach used by **CUDA-BEVFusion** and **CUDA-CenterPoint** in NVIDIA's
Lidar AI Solution repository.

## Root Cause Analysis

The spconv FX quantization pipeline was fundamentally incompatible with
`non_traceable_module_classes`:

```
FX Graph (with non_traceable):
  input → [quantize] → [call_module: conv_input] → [quantize] → [call_module: layer1] → ...

Expected by transform_qdq / remove_conv_add_dq:
  input → [quantize] → [spconv_conv_op] → [bn_fuse] → [relu] → [dequantize] → ...
```

The spconv graph transforms look for specific node patterns
(`SparseConvAddReLU`, `torch.quantize_per_tensor`, etc.) that only appear when
FX traces **inside** the spconv operations. Non-traceable modules are opaque
`call_module` nodes with no internal structure for the transforms to match.

## Solution: NVIDIA `pytorch_quantization` Approach

### How CUDA-BEVFusion Does It

Reference: `Lidar_AI_Solution/CUDA-BEVFusion/qat/lean/quantize.py`

```python
class SparseConvolutionQunat(spconv.conv.SparseConvolution, QuantMixin):
    default_quant_desc_input  = QuantDescriptor(num_bits=8, calib_method='histogram')
    default_quant_desc_weight = QuantDescriptor(num_bits=8, axis=(4))

    def forward(self, input):
        input.features = self._input_quantizer(input.features)
        quant_weight = self._weight_quantizer(self.weight)
        self.weight = Parameter(quant_weight)
        return super().forward(input)
```

Calibration: histogram collection + `compute_amax(method="mse")`.

### Our Implementation

Three new functions in `spconv_int8.py`:

#### 1. `apply_nvidia_spconv_int8(encoder)`

Adds `_input_quantizer` and `_weight_quantizer` (`TensorQuantizer`) to each
`SparseConvolution` module in-place. Overrides `forward` to fake-quantize
`input.features` and `self.weight` before calling the original forward.

```python
def _nvidia_quantized_forward(self, input):
    if input is not None and hasattr(input, "features"):
        input = input.replace_feature(self._input_quantizer(input.features))
    if self.weight is not None:
        quant_weight = self._weight_quantizer(self.weight)
        self.weight = Parameter(quant_weight)
    return self._original_forward(input)
```

Key differences from FX approach:
- Quantization happens **inside** each module's `forward()`, not at graph boundaries
- Both **activations** and **weights** are quantized (FX non-traceable only did activations)
- No FX tracing, no `SPCONV_FX_TRACE_MODE`, no graph transforms
- `conv_out` excluded (stays FP32) to preserve spatial dimension handling

#### 2. `calibrate_spconv_nvidia(encoder, data)`

```
1. Enable calibration mode on all TensorQuantizers (disable_quant + enable_calib)
2. Forward pass: quantizers collect activation/weight histograms
3. compute_amax(method="mse") — finds optimal clipping from histogram
4. Re-enable fake-quantization (enable_quant + disable_calib)
```

MSE minimises mean squared error between original and quantized values.
Unlike KL-divergence (HistogramObserver), MSE penalises large errors from
clipping, preserving the rare but critical detection peaks.

#### 3. `_report_nvidia_quantizer_stats(encoder)`

Prints per-module amax summary after calibration for verification.

### Checkpoint Format

Old FX format:
```
_input_scale_0 = tensor([1.9422])
_scale_1 = tensor([11.2532])
conv_input_0_scale_0 = tensor([0.0409])
```

New NVIDIA format:
```
conv_input.0._input_quantizer._amax = [250.58]
conv_input.0._weight_quantizer._amax = [0.011, 0.010, 0.013, ...]   # per-channel
encoder_layers.encoder_layer1.0.conv1._input_quantizer._amax = [2.97]
```

- 40 `_amax` keys total (20 conv modules × 2 quantizers each)
- Activation amax: per-tensor scalar
- Weight amax: per-channel (axis=4), shape `(1, 1, 1, 1, C_in)`

## Modified Files

### `deployment/projects/bevfusion/quantization/spconv_int8.py`

- **Added**: `apply_nvidia_spconv_int8()` — adds TensorQuantizer to sparse convs
- **Added**: `calibrate_spconv_nvidia()` — histogram + MSE calibration
- **Added**: `_report_nvidia_quantizer_stats()` — prints amax summary
- **Added**: `_nvidia_quantized_forward()` — quantized forward method
- **Added**: `_get_sparse_conv_types()` — collects both spconv and custom conv types
- Existing FX functions kept as legacy (not called by default paths)

### `deployment/quantization/bevfusion_quantization.py`

- **Changed**: `_calibrate_spconv()` now calls `apply_nvidia_spconv_int8` +
  `calibrate_spconv_nvidia` instead of FX `prepare_fx` → `calibrate` → `convert_fx`
- **Changed**: Save-check now looks for `_amax` keys instead of `scale/zero_point`
- Removed FX-specific `SparseBasicBlockFX` upgrade step

### `deployment/projects/bevfusion/io/model_loader.py`

- **Added**: `_prepare_encoder_for_nvidia_int8(model)` — adds TensorQuantizer
  so checkpoint `_amax` keys load via `load_state_dict`
- **Changed**: Evaluation path calls `_prepare_encoder_for_nvidia_int8` instead
  of `_replace_encoder_with_fx_converted_structure`
- **Updated**: `_verify_spconv_scale_buffers()` handles both NVIDIA `_amax` and
  legacy FX `scale/zero_point` keys

### `deployment/projects/bevfusion/runner.py`

- **Added**: NVIDIA TensorQuantizer detection in `_apply_spconv_int8()` — if the
  encoder already has `_input_quantizer` submodules, skip the runner's FX
  calibration path (previously this second FX pass overwrote the NVIDIA quantizers)

## Execution Commands

### PTQ Calibration

```bash
python deployment/quantization/bevfusion_quantization.py ptq \
    --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
    --checkpoint work_dirs/bevfusion/bevfusion_epoch_30.pth \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    --sparse-int8-only \
    --calibrate-samples 256 --batch-size 1 --calib-seed 0 \
    --output work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth
```

### Evaluation

```bash
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

## Per-Stage Activation Comparison

### Frame 1

| Stage | FP32 | INT8 (NVIDIA) | Retention |
|-------|:----:|:-------------:|:---------:|
| sparse_encoder mean | 0.0089 | 0.0089 | 100% |
| sparse_encoder max | 8.21 | 8.06 | 98% |
| sparse_encoder nonzero% | 3.91% | 3.91% | 100% |
| backbone_out[0] mean | 0.1324 | 0.1324 | 100% |
| backbone_out[0] max | 6.80 | 6.73 | 99% |
| neck_out mean | 0.0091 | 0.0091 | 100% |
| neck_out max | 1.50 | 1.49 | 99% |
| head_heatmap max | +3.37 | +3.43 | 102% |
| head_heatmap mean | −5.76 | −5.76 | 100% |

The NVIDIA approach preserves activation statistics almost identically to FP32
at every stage of the pipeline.

## Architecture Dataflow

```
voxel_features (FP32)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ pts_middle_encoder (BEVFusionSparseEncoder)     │
│                                                 │
│  conv_input ──► SparseConv3d + TensorQuantizer  │  ← INT8 (fake-quant)
│       │                                         │
│  encoder_layer1 ──► SparseBasicBlock × 2        │  ← INT8
│       │              + strided SparseConv3d      │
│  encoder_layer2 ──► SparseBasicBlock × 2        │  ← INT8
│       │              + strided SparseConv3d      │
│  encoder_layer3 ──► SparseBasicBlock × 2        │  ← INT8
│       │              + strided SparseConv3d      │
│  encoder_layer4 ──► SparseBasicBlock × 2        │  ← INT8
│       │                                         │
│  conv_out ──► SparseConv3d (FP32, excluded)     │  ← FP32
│       │                                         │
│  dense() + BEV reshape                          │
└─────────────────────────────────────────────────┘
    │
    ▼
spatial_features (FP32, shape: 1×256×180×180)
    │
    ▼
pts_backbone (SECOND, FP32)
    │
    ▼
pts_neck (SECONDFPN, FP32)
    │
    ▼
bbox_head (TransFusionHead, FP32)
    │
    ▼
3D bounding box predictions
```
