# BEVFusion Sparse Encoder — Data-Dependent-Shape (trainStation) Optimization

> Profiling source: `/home/yihsiangfang/bevfusion_2_7/bevfusion_profile_kambe.nsys-rep`
> Reference optimization (same class of problem, already solved for PTv3):
> - [tier4/AWML#206](https://github.com/tier4/AWML/pull/206) — ONNX export side
> - [autowarefoundation/autoware_universe#12727](https://github.com/autowarefoundation/autoware_universe/pull/12727) — runtime side

## 1. Problem Statement

In the BEVFusion TensorRT engine, the sparse middle encoder (spconv) shows repeated
`[trainStationN]` markers between sparse-conv blocks in the Nsight Systems timeline. These are
TensorRT **execution-segment boundaries** forced by **data-dependent shapes (DDS)**: the number of
active output sites after a downsampling sparse convolution is not known until the GPU has computed
it, so TensorRT must copy that shape back to the host (`DeviceToShapeHostCopy`) before it can
configure and launch the next segment. Each such boundary breaks pipelining and leaves the GPU idle.

## 2. Profiling Evidence (61 inferences, averaged per inference)

| Metric | Value |
|--------|-------|
| Single inference (`ExecutionContext::enqueue`) | **34.53 ms** |
| GPU busy (kernels + memcpy) | 24.26 ms |
| **GPU idle (bubbles)** | **10.27 ms = 29.7%** |
| All 6 `trainStation` segments | 3.11 ms/inf |
| `trainStation2` (largest) | 1.77 ms/inf |
| `GetIndicePairsImplicitGemm` (rulebook build) | 6.32 ms/inf |
| `DeviceToShapeHostCopy` sync points | **exactly 4** |

### 2.1 The four DDS sync points

`DeviceToShapeHostCopy` appears at exactly the **four stride-2 downsampling layers** (and nowhere
else):

| Layer (downsample) | Shape-copy duration | GPU idle immediately after |
|--------------------|--------------------:|---------------------------:|
| `encoder_layer1.2` → stage2 | 0.248 ms | 0.102 ms |
| `encoder_layer2.2` → stage3 | 0.301 ms | 0.082 ms |
| `encoder_layer3.2` → stage4 | 0.280 ms | 0.077 ms |
| `conv_out` | 0.329 ms | 0.073 ms |

**Submanifold convolutions** (`conv1`/`conv2`, which preserve the active-site set) produce **no**
shape copy — confirming the DDS overhead is exclusively tied to the layers that change the active
voxel count.

### 2.2 What a trainStation actually contains

A `[trainStationN]` NVTX range is **not** a pure stall — it wraps a chunk of real engine work. Inside
one `trainStation2` window (1.977 ms): 7 GPU kernels, ~1.54 ms GPU-busy, ~0.44 ms GPU-idle. The
trainStation is the **segment of the graph between two DDS boundaries**; the cost is the loss of
cross-segment pipelining plus the host syncs at the boundaries, not the work inside.

## 3. Root Cause and Analogy to PTv3

| | PTv3 (already optimized) | BEVFusion spconv (this report) |
|---|--------------------------|--------------------------------|
| DDS source | `Unique` (pooling grouping) | `GetIndicePairsImplicitGemm` (rulebook + output coord count) |
| Why shape is dynamic | pooled voxel count is data-dependent | active-site count after downsample is data-dependent |
| In-graph symptom | CPU/GPU sync barrier | `DeviceToShapeHostCopy` + trainStation segmentation |
| Fix | precompute pooling metadata in CUDA preprocess, feed as static inputs | precompute rulebooks / output coords in CUDA preprocess, feed as static inputs |

**Key fact that makes the fix possible:** the spconv **rulebook (index pairs) and per-stage output
coordinates depend only on the input voxel geometry (which cells are occupied), not on feature
values.** This is precisely why spconv separates `GetIndicePairs` (geometry) from the GEMM
(features). Voxel coordinates are known right after voxelization in preprocessing — so the entire
cascade of active coordinates and rulebooks for every layer can be computed up front and passed in
as inputs with resolvable shapes, removing the in-graph DDS.

## 4. Proposed Optimization (two-part, mirrors the PTv3 PRs)

### 4.1 Runtime / preprocessing side (analogous to autoware_universe#12727)

1. After voxelization, run a **coordinate-only forward pass** of the downsampling cascade on the GPU
   to derive, for every sparse-conv layer:
   - output active coordinates,
   - index pairs (rulebook),
   - per-stage active-site counts.
2. Perform **one** `cudaMemcpyAsync` + single sync to bring back the per-stage counts (replaces the
   4 mid-graph syncs), set the engine's dynamic input shapes from them.
3. Bind the precomputed rulebooks/coordinates to engine input tensors.

### 4.2 ONNX export side (analogous to AWML#206)

- Replace `GetIndicePairsImplicitGemm` nodes with plugin nodes that **consume** the precomputed
  rulebook inputs instead of computing them in-graph.
- Add the rulebook/coordinate tensors as named graph inputs with dynamic axes; the active-site count
  becomes a symbolic dim resolved from preprocessing.

### 4.3 Expected benefit

- Removes the 4 `DeviceToShapeHostCopy` syncs and collapses the 6 trainStation segments.
- Enables capturing the whole sparse encoder as a **single CUDA Graph**, eliminating per-kernel
  launch overhead in addition to the sync bubbles.
- Not all 10.27 ms of idle is recoverable (some is launch latency / reformatting), but the DDS- and
  segmentation-attributable portion is significant. PTv3's analogous change yielded a 34% end-to-end
  latency reduction (29 ms → 19 ms) as a reference magnitude.

## 5. Lighter-Weight Alternative (no ONNX export change)

Modify the spconv plugin so it **never copies the count to host**: declare a static **upper-bound**
output shape (max active sites) and use masking/padding so downstream layers always run at the bound.
This removes the `DeviceToShapeHostCopy` and merges the trainStations without touching the exported
graph's input signature. Cost: some layers compute over padding (wasted work at the max size). Easier
to land than the full precompute, but recovers less cleanly.

## 6. Change-Point Evaluation (spconv_cpp + plugin)

### 6.1 Where the DDS / D2H actually lives

| Concern | File | Location |
|---------|------|----------|
| TRT plugin (IPluginV3) | `autoware.universe/.../autoware_tensorrt_plugins/src/get_indices_pairs_implicit_gemm_plugin.cpp` | class + `enqueue()` @288 |
| **DDS shape declaration** | same file | `getOutputShapes()` @186–244 → `declareSizeTensor(4, min, max)` for downsampling |
| **num_act_out → device write** (the H2D that follows the host read) | same file | @439–445 `cudaMemcpyAsync(..., HostToDevice)` |
| **D2H count read (thrust path)** | `spconv_cpp/.../SpconvOps_apply_thrust_unique_to_indice_pairs_uniq.cu` | @25–38 (`thrust::unique`, returns `int` to host) |
| **D2H count read (hash path)** | `spconv_cpp/.../SparseConvIndicesKernel_unique_hash.cu` | @14–36 (`uniq_cnt.cpu(tvctx)`) |
| Rulebook build entry | `spconv_cpp/.../SpconvOps.h` | `get_indice_pairs_implicit_gemm()` @544 |
| Submanifold (NO DDS) | `spconv_cpp/.../SparseConvIndicesKernel_generate_subm_conv_inds.cu` | output count == input count |
| Downsample stages (DDS) | `generate_conv_inds_stage1/1_5/stage2` | unique/sort step is the data-dependent point |

So `[trainStationN]` is created by exactly one mechanism: `getOutputShapes()` calling
`declareSizeTensor()` for the 4 downsampling layers, whose value is produced by the unique/sort D2H
read inside `enqueue()`.

### 6.2 Key enabler already present

`SpconvOps::get_indice_pairs_implicit_gemm()` **already accepts a `preallocated` map** and will reuse
caller-supplied rulebook tensors instead of recomputing them:
`"PairFwd"`, `"IndiceNumPerLoc"`, `"HashKOrKV"`, `"PairMask"`
(`SpconvOps_get_indice_pairs_implicit_gemm.cc` @63–76, 127–136). The plugin's `enqueue()` currently
never populates this map — so the precompute path is **half-built already** in the library; the
missing piece is wiring it through the plugin I/O and the export graph.

### 6.3 Candidate implementation routes

- **Route A — Full precompute (matches §4, PTv3-style).** Run a coordinate-only forward pass of the
  4 downsampling stages in preprocessing → precomputed rulebooks + per-stage counts; feed as engine
  inputs; plugin `getOutputShapes()` derives output dim from an input dim (no `declareSizeTensor`);
  `enqueue()` consumes `preallocated`. One preprocessing sync replaces 4 in-graph syncs. Cleanest
  result; largest change (plugin I/O + export graph + runtime preprocess). Sequential geometric
  cascade (stage N+1 needs stage N coords) makes the preprocess pass more involved than PTv3.
- **Route B — Static upper-bound shape (lighter).** `getOutputShapes()` returns a constant per-stage
  bound instead of `declareSizeTensor`; kernels pad/mask to the bound; drop the D2H/H2D. No export
  input-signature change. Removes trainStations but wastes compute on padding (active sites shrink
  through downsampling, so a flat bound is costly).
- **Route C — Lift the sparse encoder out of TensorRT.** Run the spconv backbone as native libspconv
  outside the engine (as NVIDIA CUDA-BEVFusion does), feed the dense output back into a TRT engine.
  No trainStations because there is no TRT graph for the sparse part. Biggest architectural change;
  decouples spconv from TRT entirely.

### 6.4 Recommendation

**Route A** is the faithful analogue of the PTv3 PRs and gives the cleanest, fully-static engine,
and the library already supports preallocated rulebooks — but it touches the plugin I/O, the ONNX
export, and the runtime preprocessing together (a breaking pair, like AWML#206 + autoware_universe
#12727). Route B is a good incremental first step to validate the trainStation removal in isolation
before committing to the full export/runtime contract change.

## 7. Route A — Detailed Implementation Plan (file-by-file)

### 7.0 Architecture decision (important simplification)

The DDS does **not** originate in the `ImplicitGemm` conv plugin — that plugin already derives its
output extent from an *input* dim:
`implicit_gemm_plugin.cpp:269–286` → `outputs[0].d[0] = inputs[3].d[0]` (pair_mask dim0),
`outputs[0].d[1] = inputs[1].d[0]` (C_out). The DDS is created **only** by
`GetIndicePairsImplicitGemm::getOutputShapes()` calling `declareSizeTensor(4, …)`
(`get_indices_pairs_implicit_gemm_plugin.cpp:217–238`), which then propagates through the pair
tensors into every downstream layer.

**Consequence:** if the rulebook (`pair_fwd`, `pair_mask_fwd`, `mask_argsort_fwd`, `out_indices`,
`num_act_out`) becomes a real **graph input** (shape resolved by `setInputShape` before `enqueueV3`),
the size tensor disappears and `ImplicitGemm` needs **no change** — its input-derived output shape is
already correct. So Route A = *remove the GetIndicePairs nodes from the graph, expose their outputs
as graph inputs, and precompute them in preprocessing.* This mirrors PTv3 exactly (precompute → graph
inputs → bind), and the `GetIndicePairs` plugin is simply no longer instantiated in the graph (kept
in the registry for backward compat).

### 7.1 Layer structure to precompute (from AWML BEVFusion config)

`pts_middle_encoder` (`BEVFusionSparseEncoder`), `sparse_shape=[1440,1440,41]`, kernel=3 unless noted:

| Stage | Layers | Type | Stride | Changes coords? |
|-------|--------|------|--------|-----------------|
| conv_input | 1 | SubMConv3d | 1 | no |
| encoder_layer1 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | downsample only |
| encoder_layer2 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | downsample only |
| encoder_layer3 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | downsample only |
| encoder_layer4 | subm,subm | SubM×2 | 1 | no |
| conv_out | 1 | SparseConv3d k=(1,1,3) | (1,1,2) | downsample only |

Only the 4 stride>1 layers carry DDS today (matches the 4 `DeviceToShapeHostCopy` in §2.1). Submanifold
layers reuse the prior stage's coordinates (their rulebook is geometry off the same coords).

### 7.2 Runtime side (autoware_bevfusion) — analogous to autoware_universe#12727

- **New module** `lib/preprocess/sparse_rulebook_precompute.{hpp,cu}`:
  - Inputs: voxel coords (`coors`, [A,4]) already produced by voxelization
    (`bevfusion_trt.cpp:initPtr/preProcess`).
  - For each sparse layer in declared order, call
    `SpconvOps::get_indice_pairs_implicit_gemm(...)` (same call the plugin used,
    `get_indices_pairs_implicit_gemm_plugin.cpp:392/432`) into stable preallocated device buffers,
    threading each downsample layer's `out_indices` as the next layer's input coords.
  - Collect every layer's `num_act_out`; do **one** `cudaMemcpyAsync`+`cudaStreamSynchronize` for all
    counts (replaces the 4 in-graph syncs).
- **`lib/bevfusion_trt.cpp`**:
  - `initTrt()` (~L141–187): register the new rulebook graph inputs in the optimization profile with
    `[min,opt,max]` (max = `out_indices_num_limit_`), like PTv3's per-stage inputs.
  - new `bindSerializedRulebookAddresses()`: `setTensorAddress()` for every rulebook buffer.
  - `preProcess()`: call the precompute module, then `setInputShape()` for each rulebook input using
    the synced counts, before `enqueueV3()`.
- **Config/schema** (`config/bevfusion_lidar.param.yaml`, `schema/*.json`): add the sparse encoder
  layer descriptor list (ksize/stride/padding/subm per layer) so preprocessing knows the cascade —
  analogous to PTv3's `pooling_strides`.

### 7.3 Export side (AWML) — analogous to AWML#206

- **`projects/SparseConvolution/sparse_functional.py`** (`GetIndicePairsImplicitGemm.symbolic`
  @243–292): add an `export_precomputed` path that, instead of emitting the
  `autoware::GetIndicePairsImplicitGemm` op, returns 5 tensors sourced from **named graph inputs**
  (`rulebook_{i}_out_indices`, `_pair_fwd`, `_pair_mask`, `_mask_argsort`, `_num_act_out`).
- **`projects/BEVFusion/bevfusion/sparse_encoder.py`** (`forward` @147+): in export mode, pull each
  layer's rulebook from the injected inputs rather than computing it; keep the `ImplicitGemm` calls
  unchanged (they already take pair tensors as args).
- **`projects/BEVFusion/deploy/exporter.py`** (`_export_main_body` @187+, `torch.onnx.export` @173,
  `_fix_onnx_graph`): declare the new inputs + dynamic axes; use a distinct symbolic dim for the
  `indptr`-like `[N+1]`/`num_act_out` tensors; ensure no `GetIndicePairs` node remains.
- Document the new input contract in the BEVFusion export README (mirrors AWML#206 README section).

### 7.4 Plugin side

- `ImplicitGemmPlugin`: **no change** (input-derived shapes already correct).
- `GetIndicesPairsImplicitGemmPlugin`: no change required for Route A (node removed from graph). The
  `out_indices_num_limit_ = 256000` upper bound becomes the profile max for the rulebook inputs.

### 7.5 Verification

- Port the spconv equivalence-test idea (PTv3's `serialized_pooling_metadata_test.cpp`): a gtest that
  runs the preprocessing rulebook cascade and checks it byte-matches the in-graph
  `SpconvOps::get_indice_pairs_implicit_gemm` output for a fixture point cloud.
- End-to-end: rebuild ONNX (new contract) → rebuild engine → confirm `[trainStation]` markers and the
  4 `DeviceToShapeHostCopy` are gone in a fresh nsys capture, and detection output is unchanged.

### 7.6 Build / test constraint

This is a breaking pair (export + runtime) requiring a CUDA build, TensorRT engine rebuild, ONNX
re-export, and a dataset to validate — none runnable in the analysis sandbox here. Implementation
must land slice-by-slice with the build/test loop on the target machine. Lead with the export
contract (like AWML#206), then runtime, then re-profile.

## 8. Implementation Status

### Slice 1 — Export graph surgery ✅ implemented & ONNX-validated

`AWML/deployment/projects/bevfusion/export/sparse_trainstation_transform.py`
(`remove_trainstation_dds`). Deletes the 4 down-sampling `GetIndicePairsImplicitGemm` nodes and
promotes their consumed outputs (`out[0..3]`; `out[4]` num_act_out has no consumer, dropped) to graph
inputs with a per-stage shared symbolic dim.

Verified in the `awml-bevfusion` container against the baseline
`bevfusion_sparse.onnx`:
- 21 → **17** `GetIndicePairsImplicitGemm` (all remaining `subm=1`, i.e. no `declareSizeTensor`, no DDS).
- `ImplicitGemm` unchanged (21); 12 of its input edges now sourced from the new graph inputs
  (4 stages × pair_fwd/pair_mask/mask_argsort).
- Graph inputs 3 → **19** (+16: 4 stages × out_indices/pair_fwd/pair_mask/mask_argsort).
- `onnx.checker` OK; `shape_inference(strict_mode=True)` OK → graph consistent end-to-end.

New input names per stage (`l1/l2/l3/out`), INT32:
`…GetIndicePairsImplicitGemm_output_{0,1,2,3}` with shapes
`[N,4] / [KV,N] / [N,1] / [N]` (KV=27 for l1–l3, 3 for conv_out).

### Build / test workflow (this machine)

- Container `awml-bevfusion` (`awml-bevfusion:full`); host `AWML` is mounted at `/workspace`, so AWML
  edits apply live. Plugin `.so` at `/opt/plugins/libautoware_tensorrt_plugins.so` (prebuilt from
  fork `vividf/autoware.universe@feat/implicit_gemm_int8`; rebuild via
  `projects/BEVFusion/plugins/build_plugin_inside_container.sh`).
- Export/build CLI:
  `python -m deployment.cli.main bevfusion <deploy_cfg> <model_cfg>`
  deploy cfg: `deployment/projects/bevfusion/config/deploy_config_split_fp16_opt_trainstation.py`.
- Sparse ONNX = `pts_middle_encoder` only; split from dense. Baseline graph: 21 GetIndicePairs +
  21 ImplicitGemm; inputs voxels/coors/num_points_per_voxel; output lidar_bev.

### Slice 1c — Engine build + trainStation removal ✅ PROVEN

Built FP16 sparse engines from baseline vs. surgically-modified ONNX in the container (plugin
`/opt/plugins/libautoware_tensorrt_plugins.so`), then dumped TensorRT **engine-inspector** layer info
(nsys-free structural proof — `trainStation` is TRT's internal Myelin region name and appears verbatim
in the engine layer list):

| Engine | total layers | `trainStation` layers |
|--------|-------------:|----------------------:|
| Baseline (`bevfusion_sparse.onnx`) | 135 | **6** (`[trainStation1]`…`[trainStation6]`) |
| Modified (`bevfusion_sparse_nots.onnx`) | 125 | **0** |

The 6 baseline trainStations match the 6 seen in the on-board nsys profile (§2). Removing the 4
down-sample `GetIndicePairsImplicitGemm` nodes eliminates **all** of them. Both engines build cleanly
(all 21+21 plugins instantiate; 19 inputs with a consistent optimization profile — note
voxels/coors/num_points_per_voxel share dim_param `voxels_num`, so their profiles must be identical).

> Throwaway harness: `AWML/_ts_tmp/{build_sparse_engine.py,inspect_engine.py}`.

### Slice 1d — Numerical equivalence ✅ PROVEN

`AWML/_ts_tmp/validate_equiv.py`: feeds BOTH engines the same synthetic sparse input (40k random
voxels). The modified engine additionally receives the 4 down-sample rulebooks, precomputed via
`sparse_functional.GetIndicePairsImplicitGemm.apply` cascaded over the 4 down-sample stages (the exact
reference logic for the Slice-2 CUDA runtime). Result:

```
baseline (1,256,180,180)  vs  modified (1,256,180,180)
max abs diff = 0.0088   mean = 0.00014   relative max = 0.0034   -> MATCH (fp16-level)
```

Confirms: (1) the Python precompute matches what the baseline computes in-graph; (2) the modified
engine correctly consumes external rulebooks; (3) the graph surgery preserves semantics. The
precompute cascade (feed conv coords → `get_indice_pairs_implicit_gemm` per down-sample stage,
threading `out_indices` forward; coords normalized `[z,y,x] → [batch,x,y,z]`; spatial_shape per stage
`1440→720→360→180`) is the reference for the Slice-2 C++/CUDA runtime.

**Route A is end-to-end validated at the export+engine level: trainStation removed AND output
numerically equivalent.**

### Slice 1b — Export-pipeline integration ✅ done & verified via official CLI

- `onnx_export_pipeline.py::_postprocess_sparse_onnx_fp`: runs `remove_trainstation_dds` on the sparse
  ONNX when `deploy_cfg.spconv_remove_trainstation` is set (independent of the ReLU-fuse flag;
  composes cleanly with it).
- `deploy_config_split_fp16_opt_trainstation.py`: `spconv_remove_trainstation = True` + programmatic
  injection of the 16 rulebook inputs into `components.bevfusion_sparse.tensorrt_profile`
  (N∈[1,256000]; KV=27 for l1–l3, 3 for conv_out).
- Ran the official CLI end-to-end
  (`python -m deployment.cli.main bevfusion <trainstation cfg> <model cfg>`): log shows
  "trainStation/DDS removal done (removed 4 … added 16 rulebook graph inputs)" then both engines
  build. **Engine-inspector on the CLI-produced `bevfusion_sparse.engine`: 127 layers, 0 trainStation
  layers.** (Co-exists with the ImplicitGemm ReLU fusion, 13 relus.)

**Export side complete (Slices 1/1b/1c/1d): the official pipeline now emits a trainStation-free,
numerically-equivalent sparse engine behind one deploy-cfg flag.** Remaining work is the runtime that
supplies the 16 rulebook inputs.

### Slice 2 (Python runtime) — rulebook precompute wired into the deploy eval pipeline

The deployment's own Python TensorRT pipeline (`pipelines/tensorrt.py`) also needs to supply the 16
rulebook inputs (eval failed with "Address is not set for input … GetIndicePairsImplicitGemm_output_0"
until wired). Added:
- `pipelines/sparse_rulebook_precompute.py`: `compute_rulebook_inputs(coors_zyx, input_names)` —
  cascades `GetIndicePairsImplicitGemm` over the 4 down-sample stages (the validated Slice-1d logic),
  returns `{input_name: int32 np.ndarray}`. `has_rulebook_inputs()` gates it (no-op for baseline).
- `pipelines/tensorrt.py::_trt_infer_voxel_inputs`: when the sparse engine exposes rulebook inputs,
  precompute from the same `coors` and add to the bind map before `enqueueV3`.

This lets mAP be validated in-container (no autoware build needed) and is the exact reference for the
C++/CUDA autoware_bevfusion port.

**Clean A/B (both `export.mode="none"`, engines prebuilt, GPU-timed, 5 samples):**

| stage | baseline (trainStation ON) | trainStation removed | Δ |
|-------|---------------------------:|---------------------:|---|
| mAP Center-BEV / Plane | 0.9066 / 0.9502 | 0.9068 / 0.9503 | identical (fp16 noise) |
| Sparse Encoder | 9.37 ± 0.33 ms | 8.00 ± 0.40 ms | −1.4 ms (~15%) |
| Dense Engine (unchanged — control) | 7.25 ms | 7.04 ms | ~equal ✓ |
| Model total | 16.63 ms | 15.03 ms | −1.6 ms |

**mAP is preserved; the Sparse Encoder is ~15% faster on this GPU.** The dense engine (byte-identical
between the two) matches within noise, confirming the comparison is clean (an earlier A/B was
confounded because the baseline ran `mode="both"` — a heavy engine build right before eval inflated
*all* stages incl. the unchanged dense 44→7 ms; that run's latency deltas are invalid, its mAP is not).

**Honest caveats on the latency number:**
- This is a strong dGPU. The on-board target (the original nsys profile, §2) showed ~30% GPU idle
  from the 6 trainStations, so the relative benefit there is expected to be larger than 15%.
- The Python prototype's rulebook precompute time is **not** counted in "Sparse Encoder" (that stage
  is only the TRT enqueue). It replaces work the baseline did *in-graph* (which WAS in the baseline's
  9.37 ms) and collapses 4 mid-graph syncs into 1 preprocessing sync — but a fully fair end-to-end
  number must include the precompute cost in preprocessing. The decisive, hardware-independent result
  is structural: trainStations 6→0 and mAP unchanged.

### Slice 2b — autoware_bevfusion C++/CUDA runtime ✅ implemented (see Slice 2c for build/verify)

Ported the validated Python precompute to the on-vehicle node. New + edited files in
`autoware.universe/perception/autoware_bevfusion/`:

- **`preprocess/sparse_rulebook_precompute.{hpp,cu}`** (new): `SparseRulebookPrecompute` —
  owns stable per-stage device buffers (out_indices/pair_fwd/pair_mask/mask_argsort, sized to the
  256000 upper bound) and a shared spconv workspace; a `buildBatchedCoordsKernel` converts the
  `coors` (`[z,y,x]` → `[batch,x,y,z]`); `compute()` cascades `SpconvOps::get_indice_pairs_implicit_gemm`
  over the 4 down-sample stages (mirrors the plugin's non-subm `enqueue` path), threading
  `out_indices` forward; exposes per-stage counts + device pointers. `default_bevfusion_downsample_stages()`
  encodes the 4 stages (ksize/stride/padding/spatial 1440→720→360→180).
- **`bevfusion_trt.{hpp,cpp}`**: `addSparseRulebookNetworkIO` / `addSparseRulebookProfileDims`
  (declare the 16 inputs + `[min,opt,max]` profiles, max = limit), `bindSparseRulebookAddresses`
  (`setTensorAddress` once to the stable buffers), `setSparseRulebookInputShapes` (`setInputShape`
  per-stage from the synced counts). `preProcess` calls `compute()` right after voxelization; all
  gated on `config_.sparse_remove_trainstation_` (no-op otherwise → baseline engine still works).
- **`bevfusion_config.hpp`**: plain members `sparse_remove_trainstation_`,
  `sparse_out_indices_num_limit_` (256000), `sparse_coors_is_zyx_`.
- **`bevfusion_node.cpp`** + **`config/ml_package_bevfusion_lidar.param.yaml`** +
  **`schema/ml_package_bevfusion.schema.json`**: `sparse_remove_trainstation` ROS param (default false).
- **`CMakeLists.txt`**: added the new `.cu` to `${PROJECT_NAME}_cuda_lib` (already links `spconv::spconv`).

Not built/verified here (autoware.universe is not mounted in the awml-bevfusion container and needs a
colcon/autoware + spconv build). The `.cu` faithfully mirrors the proven plugin `enqueue` and the
validated Python cascade; integration points to confirm on first build: exact `SpconvOps` API
signatures, the `coors` order (`sparse_coors_is_zyx_`), and the spconv workspace sizing.

### Slice 2c — first autoware-env build + end-to-end run ✅ PROVEN (pilot-auto.x2)

Built `autoware_bevfusion` in `pilot-auto.x2` against the merged single-file engine
(`bevfusion_lidar.onnx`, exported with `spconv_remove_trainstation=True`) and ran it end-to-end on a
real `concatenated/pointcloud` rosbag. Three issues surfaced — all three were exactly the
"confirm on first build" points flagged in Slice 2b — plus the config needed to enable the path.
Fixes (in `autoware.universe/perception/autoware_bevfusion/`):

1. **`SpconvOps` API signature — `std::string` vs `const char*` (compile error).**
   `network_trt_ptr_->setTensorAddress(...)` / `setInputShape(...)` in `bevfusion_trt.cpp`
   (`bindSparseRulebookAddresses` / `setSparseRulebookInputShapes`) were called with
   `s.onnx_base + "_output_N"` (a `std::string`), but the installed `autoware_tensorrt_common`
   only exposes `(const char*, ...)` / `(int32_t, ...)` overloads — no implicit `std::string`
   conversion. **Fix:** wrap each name in `(...).c_str()` (8 call sites).

2. **Merged-engine tensor-name prefix (engine builds, but profiles set on the wrong tensors).**
   `onnx.compose.merge_models` namespaces the sparse subgraph with `sparse/`, and the merge step only
   renamed the 3 *declared* `io.inputs` (`voxels`/`coors`/`num_points_per_voxel`) back — so the 16
   trainStation rulebook inputs (added later by `sparse_trainstation_transform`) kept the prefix and
   came out as `sparse//pts_middle_encoder/.../GetIndicePairsImplicitGemm_output_*` (double `//`). The
   runtime's hardcoded `default_bevfusion_downsample_stages()` base names have no prefix, so the
   optimization profiles were registered for non-existent tensors and the real inputs were left
   without a profile → `Error Code 4: ... is missing dimensions in profile 0`.
   **Fix (export side, keeps the runtime clean):** in `onnx_export_pipeline._merge_split_onnx`, after
   the declared-input/output rename, strip the `sparse/` namespace from every remaining graph input
   so the rulebook inputs keep their original un-prefixed `GetIndicePairsImplicitGemm` node names
   (`gs` renames by object identity, so consumers update too). The merged ONNX input names then match
   both the runtime's hardcoded stage names and the deploy-cfg `tensorrt_profile` names. No runtime
   prefix knob is needed — `autoware_bevfusion` binds the names as-is. (AWML eval is unaffected:
   `sparse_rulebook_precompute.has_rulebook_inputs` / `compute_rulebook_inputs` match by the
   `GetIndicePairsImplicitGemm_output_` marker + node `infix`, both prefix-agnostic.)
   *Requires re-exporting the ONNX with the fixed pipeline and rebuilding the engine.*

3. **spconv workspace under-sized for the down-sample stages (runtime abort on first frame).**
   `SparseRulebookPrecompute` passed `N` (= `out_indices_num_limit_`, 256000) as `max_act_out_in_theory`
   to `get_indice_gen_workspace_size` / `get_indice_gen_tensors_from_workspace`, so the internal
   `indice_pairs_uniq` buffer was carved at `N*1.1 = 281600`. The first stage actually needs
   `get_handcrafted_max_act_out(num_in, ...) ≈ 808121`, tripping the spconv `StaticAllocator`
   `res.nbytes() >= total ... assert faild. alloc failed, tensor size too small [2, 808121] [2, 281600]`.
   The plugin's `enqueue` sizes this from `SpconvOps::get_handcrafted_max_act_out(num_act_in, ...)`,
   not `N`. **Fix:** mirror the plugin — `computeStage()` derives
   `max_act_out_theory = get_handcrafted_max_act_out(num_in, ksize, stride, padding, dilation)` and
   feeds it to both workspace-size and tensor-carving calls; `allocateStageBuffers()` sizes the shared
   workspace for the worst case (max over stages of `get_handcrafted_max_act_out(N, ...)`), which bounds
   every per-stage carve since runtime `num_in ≤ N`.

**Config required to enable the path** (in the *loaded* ml-package param file — for the default launch
that is `~/autoware_data/bevfusion/ml_package_bevfusion_lidar.param.yaml`, resolved from
`model_path = $(data_path)/bevfusion`, **not** the package `config/` copy):

```yaml
sparse_remove_trainstation: true
```

**Verified:** engine builds with all 16 `sparse//...GetIndicePairsImplicitGemm_output_*` profiles set
(`Engine generation completed`), node loads it, and replaying the pointcloud rosbag drives inference
with **no crash** and populated `/objects` detections (the `compatibleCallback` PointCloud2 path —
the one that previously aborted on frame 0).

> Build note (this machine): the shell auto-activates conda base, whose `colcon` lacks `colcon_core`.
> Drop miniconda from `PATH` (and unset `PYTHONPATH`/`CONDA_PREFIX`) before `colcon build`, else the
> build silently no-ops before compiling.

### Next slices (pending)

- **1b** Wire `remove_trainstation_dds` into `onnx_export_pipeline.py` as a sparse-ONNX post-process
  gated by a deploy-cfg flag (e.g. `spconv_remove_trainstation=True`); add the 16 inputs to the
  TensorRT optimization profile (deploy cfg `tensorrt_profile`).
- **1c** Decisive proof: build engine from the surgically-modified ONNX and capture nsys → confirm
  `[trainStation]` / `DeviceToShapeHostCopy` gone (feed once-computed rulebooks).
- **2** Runtime (autoware_bevfusion): CUDA precompute of the 4 down-sample rulebooks
  (`SpconvOps::get_indice_pairs_implicit_gemm` cascade) + `setInputShape` + bind. One sync for counts.
- **3** Equivalence gtest + end-to-end mAP unchanged.
