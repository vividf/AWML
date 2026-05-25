# BEVFusion Optimization Summary (Mapping, Feature Computation, Sort, Fusion, ImplicitGEMM INT8)

This document summarizes what optimizations were applied based on:

- `deployment/projects/bevfusion/docs/21_README_IMPLICITGEMM_FP_TRT_PLUGIN_ISSUES.md`
- `deployment/projects/bevfusion/docs/22_README_ONNX_SPCONV_TRT_PLUGIN_AND_INFERENCE.md`

It focuses on the requested topics:

- **Mapping** — index-generation / sorting policy only
- **Feature Computation** — runtime compute path and its sub-optimizations:
  - Close Sort (disable sorting)
  - Fuse ReLU
  - Fuse BatchNorm
  - ImplicitGEMM INT8

---

## 1) Mapping Optimization (Sorting / Index Generation)

Mapping optimization here means **how sparse indices are prepared before GEMM**, specifically the sort policy in `GetIndicePairsImplicitGemm`. It is **not** operator-to-plugin name mapping, ONNX input-contract mapping, or FP 5/6-input plugin wiring (those belong to deployment/plugin docs 21–22).

### What was optimized

1. **`do_sort` control in `GetIndicePairsImplicitGemm`**
   - `spconv_do_sort` maps to `GetIndicePairsImplicitGemm.do_sort_i` in exported ONNX.
   - Sort behavior is configurable from export/deploy config at the **index mapping** stage, before `ImplicitGemm` compute.

2. **Close sort (`spconv_do_sort = false`)**
   - When sorting is disabled, index generation skips the sort step (“close sort”).
   - The INT8 deployment flow commonly sets `do_sort=false` to avoid sort overhead in sparse index preparation.

### Why it helps

- Removes sorting overhead from sparse index preparation when disabled.
- Reduces kernel launch and memory traffic in index-generation stages.
- Improves end-to-end latency in many INT8 sparse scenarios.

### Related config / code

- Export/deploy: `spconv_do_sort` → ONNX attribute `do_sort_i` on `GetIndicePairsImplicitGemm`
- Plugin: `GetIndicePairsImplicitGemm` (`do_sort` / `do_sort_i`)

---

## 2) Feature Computation Optimization

Feature computation covers **what happens inside or immediately around sparse/dense compute**: quantization in the plugin path, fusion of post-conv ops into plugins, BN folding before export, and the INT8 sparse GEMM plugin path.

### 2.1) Core feature compute path (plugin runtime)

#### What was optimized

1. **Feature quantization pipeline in `enqueue`**
   - FP16 input features are quantized to INT8 in workspace memory before GEMM.
   - Weights are quantized and cached (constant-only cache mode), then reused across inferences.

2. **Scale fusion inside compute path**
   - Per-channel scales and output scale are fused into GEMM scale/bias preparation.
   - This reduces repeated scale handling overhead in later steps.

3. **Dataflow kept in sparse plugin path**
   - Pair/mask/index tensors produced upstream are passed directly into sparse GEMM.
   - Runtime avoids extra graph-level detours and redundant tensor conversions.

#### Why it helps

- Reduces per-frame overhead by reusing cached constants.
- Improves throughput by minimizing repeated quantization work.
- Keeps sparse compute efficient and deterministic.

---

### 2.2) Close Sort (`spconv_do_sort = false`)

This is the **feature-pipeline consequence** of the mapping sort policy in §1: with sorting off, downstream `ImplicitGemm` receives unsorted (or differently ordered) pair/mask tensors, and the compute path must remain consistent with that choice.

#### What was optimized

1. **End-to-end alignment with `do_sort=false`**
   - Index tensors (`pair_*`, `mask_*`, `mask_argsort_*`) are produced without the sort step when disabled.
   - INT8 sparse deployment typically keeps this setting through export → engine build → runtime.

2. **No extra sort inside GEMM plugin**
   - Sorting is not reintroduced in `ImplicitGemm` / `ImplicitGemmInt8`; the plugin consumes whatever index layout upstream produced.

#### Why it helps

- Same latency benefits as §1, but stated from the **compute/dataflow** side: fewer dependencies and less work between index generation and GEMM.

#### See also

- §1 for sort policy and export attribute mapping.

---

### 2.3) Fuse ReLU

#### What was optimized

1. **FP path fusion**
   - `ImplicitGemm -> ReLU` is fused into plugin `act_type`.
   - `ImplicitGemm -> Add(const) -> ReLU` is fused as:
     - `act_type` + optional 6th bias input.

2. **INT8 path fusion**
   - `ImplicitGemmInt8 -> Add(const) -> ReLU` is fused into:
     - `bias_scaled` + `act_type`.

3. **Transform integration**
   - The sparse INT8 ONNX transform applies fusion in the conversion flow to keep fused behavior consistent between FP and INT8 deployment.

#### Where it happens in the pipeline (steps)

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

#### Why it helps

- Removes standalone post-conv activation/add nodes.
- Lowers memory round-trips and kernel fragmentation.
- Improves both latency and numerical consistency across paths.

---

### 2.4) Fuse BatchNorm

#### What was optimized

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

#### Where it happens in the pipeline (steps)

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

#### Why it helps

- Folds BN into conv weights before export, reducing ops and improving numeric stability in quantized graphs.

---

### 2.5) ImplicitGEMM INT8

#### What was optimized

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

5. **Fixed 7-input INT8 plugin contract**
   - `ImplicitGemmInt8` uses: 5 sparse tensors (`features`, `filters`, `pair_fwd`, `pair_mask_fwd`, `mask_argsort_fwd`), `channel_scale`, `bias_scaled`.
   - Keeps build/runtime input interpretation unambiguous (see docs 21–22 for full plugin mapping).

#### Why it helps

- Makes INT8 sparse deployment reproducible and stable.
- Preserves fused-graph semantics across precision modes.
- Reduces runtime crashes and engine build failures.

---

## Quick Table

| Section | Topic | Main optimization | Status |
|---|---|---|---|
| **1** | Mapping (sorting) | `do_sort` / close sort in `GetIndicePairsImplicitGemm` | Implemented |
| **2.1** | Feature compute (core) | Quantize-cache-fuse scale path in plugin runtime | Implemented |
| **2.2** | Close sort (compute alignment) | Pipeline consistent with `do_sort=false`; no re-sort in GEMM | Implemented |
| **2.3** | Fuse ReLU | FP/INT8 add+activation fused into plugin attrs/inputs | Implemented |
| **2.4** | Fuse BatchNorm | Dense + sparse BN fusion in model loading/PTQ | Implemented |
| **2.5** | ImplicitGEMM INT8 | FP→INT8 transform + bias preservation + robust plugin | Implemented |

**Note:** Operator/plugin name mapping, split sparse/dense engines, and FP 5/6-input contracts are documented in `21_README_IMPLICITGEMM_FP_TRT_PLUGIN_ISSUES.md` and `22_README_ONNX_SPCONV_TRT_PLUGIN_AND_INFERENCE.md`, not in §1 of this summary.
