# BEVFusion Optimization Summary (Mapping, Feature Computation, Sort, Fusion, ImplicitGEMM INT8)

This document summarizes what optimizations were applied based on:

- `deployment/projects/bevfusion/docs/21_README_IMPLICITGEMM_FP_TRT_PLUGIN_ISSUES.md`
- `deployment/projects/bevfusion/docs/22_README_ONNX_SPCONV_TRT_PLUGIN_AND_INFERENCE.md`

It focuses on the requested topics:

- Mapping
- Feature Computation
- Close Sort (disable sorting)
- Fuse ReLU
- Fuse BatchNorm
- ImplicitGEMM INT8

---

## 1) Mapping Optimization

### What was optimized

1. **Clear operator-to-plugin mapping in split deployment**
   - Sparse ONNX nodes are mapped to custom TensorRT plugins:
     - `autoware::ImplicitGemm` (FP path)
     - `autoware::ImplicitGemmInt8` (INT8 path)
   - Split execution maps the pipeline into:
     - Sparse engine (`bevfusion_sparse.engine`)
     - Dense engine (`bevfusion_dense.engine`)

2. **Input contract mapping was standardized for INT8 plugin**
   - `ImplicitGemmInt8` uses a fixed 7-input contract:
     - 5 sparse tensors (`features`, `filters`, `pair_fwd`, `pair_mask_fwd`, `mask_argsort_fwd`)
     - `channel_scale`
     - `bias_scaled`
   - This prevents ambiguous input interpretation during build/runtime.

3. **FP plugin mapping expanded from only 5-input to 5/6-input**
   - After ONNX fusion (`ImplicitGemm -> Add(const) -> ReLU`), FP `ImplicitGemm` may have an extra 6th bias input.
   - Plugin behavior was updated to support both 5 and 6 inputs, so fused models can build engines reliably.

4. **Index mapping policy includes `do_sort` control in `GetIndicePairsImplicitGemm`**
   - `spconv_do_sort` maps to `GetIndicePairsImplicitGemm.do_sort_i` in exported ONNX.
   - This is an index-generation (mapping) optimization, not a GEMM math-kernel optimization.

### Why it helps

- Reduces parser/build fragility.
- Keeps ONNX graph semantics aligned with plugin runtime semantics.
- Avoids engine build aborts caused by mismatched input expectations.

---

## 2) Feature Computation Optimization

### What was optimized

1. **Feature quantization pipeline in `enqueue`**
   - FP16 input features are quantized to INT8 in workspace memory before GEMM.
   - Weights are quantized and cached (constant-only cache mode), then reused across inferences.

2. **Scale fusion inside compute path**
   - Per-channel scales and output scale are fused into GEMM scale/bias preparation.
   - This reduces repeated scale handling overhead in later steps.

3. **Dataflow kept in sparse plugin path**
   - Pair/mask/index tensors produced upstream are passed directly into sparse GEMM.
   - Runtime avoids extra graph-level detours and redundant tensor conversions.

### Why it helps

- Reduces per-frame overhead by reusing cached constants.
- Improves throughput by minimizing repeated quantization work.
- Keeps sparse compute efficient and deterministic.

---

## 3) Close Sort Optimization (`spconv_do_sort = false`, mapping/index stage)

### What was optimized

1. **Sort behavior became configurable from export/deploy config**
   - `do_sort` in `GetIndicePairsImplicitGemm` can be controlled by ONNX attributes/config.
   - This happens in the index mapping stage before `ImplicitGemm` compute.

2. **INT8 path commonly disables sorting**
   - In the documented INT8 deployment flow, sorting is typically turned off.
   - This is referred to as "close sort" (disable sort).

### Why it helps

- Removes sorting overhead from sparse index preparation.
- Reduces kernel launch and memory traffic in index generation stages.
- Improves end-to-end latency in many INT8 sparse scenarios.

---

## 4) Fuse ReLU Optimization

### What was optimized

1. **FP path fusion**
   - `ImplicitGemm -> ReLU` is fused into plugin `act_type`.
   - `ImplicitGemm -> Add(const) -> ReLU` is fused as:
     - `act_type` + optional 6th bias input.

2. **INT8 path fusion**
   - `ImplicitGemmInt8 -> Add(const) -> ReLU` is fused into:
     - `bias_scaled` + `act_type`.

3. **Transform integration**
   - The sparse INT8 ONNX transform applies fusion in the conversion flow to keep fused behavior consistent between FP and INT8 deployment.

### Where it happens in the pipeline (steps)

1. **Sparse ONNX transform step (Path B core)**
   - File: `deployment/projects/bevfusion/export/sparse_int8_onnx_transform.py`
   - During FP `ImplicitGemm` -> `ImplicitGemmInt8` conversion, it runs ReLU/Add fusion logic and writes fused attributes/inputs.

2. **ONNX graph fusion functions**
   - File: `deployment/projects/bevfusion/export/onnx_fuse_implicit_gemm_activation.py`
   - FP path:
     - `ImplicitGemm -> ReLU`
     - `ImplicitGemm -> Add(const) -> ReLU`
   - INT8 path:
     - `ImplicitGemmInt8 -> Add(const) -> ReLU`

3. **Resulting runtime stage**
   - At TensorRT runtime, fused activation is executed inside plugin compute (`act_type` and fused bias path), not as separate ONNX nodes.

### Why it helps

- Removes standalone post-conv activation/add nodes.
- Lowers memory round-trips and kernel fragmentation.
- Improves both latency and numerical consistency across paths.

---

## 5) Fuse BatchNorm Optimization

### What was optimized

1. **Dense BN fusion is explicitly implemented**
   - Dense submodules (`pts_backbone`, `pts_neck`, `bbox_head`) apply BN fusion before/with quantization flow.
   - Main entry: `deployment/projects/bevfusion/io/model_loader.py` (`_fuse_dense_bn`).

2. **Sparse BN fusion is explicitly implemented**
   - Sparse encoder (`pts_middle_encoder`) fuses each `SparseConvolution + BatchNorm1d` pair.
   - Main entries:
     - `deployment/projects/bevfusion/io/model_loader.py` (`_fuse_spconv_bn`)
     - `deployment/projects/bevfusion/quantization/spconv_int8.py` (`_fuse_spconv_bn_in_encoder`)

3. **Config-level control is provided**
   - Dense/sparse PTQ path uses `quantization.fuse_bn=True`.
   - Some split configs also expose `fuse_spconv_bn=True` for explicit sparse-BN fusion behavior in deployment/export flow.

### Where it happens in the pipeline (steps)

1. **Model load / quantization-prepare step**
   - File: `deployment/projects/bevfusion/io/model_loader.py`
   - In `_load_with_quantization(...)`, when `fuse_bn=True`, it runs:
     - `_fuse_dense_bn(model)` for dense modules
     - `_fuse_spconv_bn(model)` for sparse encoder

2. **Sparse INT8 PTQ preparation step**
   - File: `deployment/projects/bevfusion/quantization/spconv_int8.py`
   - `_fuse_spconv_bn_in_encoder(...)` performs the actual sparse conv+BN fusion operation used by PTQ/deploy alignment.

3. **Export/deploy config stage**
   - Files under `deployment/projects/bevfusion/config/*.py`
   - BN fusion is enabled by config flags before ONNX/TRT export, so exported graphs/weights already reflect fused BN behavior.

---

## 6) ImplicitGEMM INT8 Optimization

### What was optimized

1. **ONNX conversion to INT8 sparse plugin**
   - `sparse_int8_onnx_transform.py` replaces FP `ImplicitGemm` nodes with `ImplicitGemmInt8`.
   - Builds and injects `channel_scale` and `bias_scaled` initializers from PTQ/checkpoint metadata.

2. **6-input FP fused bias preservation**
   - When FP node has 6 inputs (extra fused bias), transform merges that extra constant into INT8 `bias_scaled`:
     - `bias_scaled += extra_bias / output_scale`
   - Prevents silent accuracy mismatch between FP fused and INT8 deployed behavior.

3. **Plugin robustness improvements**
   - `supportsFormatCombination` and configure paths avoid hard abort patterns and use reject/return-code behavior.
   - Better boundary checks on `pos`/input-output ranges.

4. **Runtime dtype alignment**
   - Bias dtype handling is aligned with activation/output dtype assumptions to avoid runtime dtype assertion failures.

### Why it helps

- Makes INT8 sparse deployment reproducible and stable.
- Preserves fused-graph semantics across precision modes.
- Reduces runtime crashes and engine build failures.

---

## Quick Table

| Topic | Main Optimization | Status |
|---|---|---|
| Mapping | Standardized operator/input contracts; 5/6-input FP support | Implemented |
| Feature Computation | Quantize-cache-fuse scale path in plugin runtime | Implemented |
| Close Sort | Configurable `do_sort`, often disabled for INT8 | Implemented |
| Fuse ReLU | FP/INT8 add+activation fused into plugin attrs/inputs | Implemented |
| Fuse BatchNorm | Explicit dense + sparse BN fusion in model loading/PTQ path | Implemented |
| ImplicitGEMM INT8 | FP->INT8 transform + bias preservation + robust plugin behavior | Implemented |
