# PTv3 Serialized Pooling Metadata Precomputation

> Related PRs:
> - [tier4/AWML#206](https://github.com/tier4/AWML/pull/206) — ONNX export side
> - [autowarefoundation/autoware_universe#12727](https://github.com/autowarefoundation/autoware_universe/pull/12727) — Runtime inference side

## Background

PTv3 (Point Transformer V3) is a LiDAR segmentation model that uses **Serialized Pooling** to aggregate point cloud features across resolution levels in its encoder. Each pooling stage groups voxels by a stride-shifted key derived from their serialized Morton code, then computes per-group statistics: gather indices, CSR row pointers, cluster labels, serialization orders, and their inverses.

### Original Problem: Data-Dependent Shapes Inside TensorRT

In the original implementation, this grouping was performed **inside the TensorRT graph** using `Unique` operations. Because `Unique` has data-dependent output shapes (its output size depends on the actual point cloud content, not just its dimensions), TensorRT could not statically infer downstream tensor shapes. This forced TensorRT to insert **CPU/GPU synchronization barriers** mid-graph to read back the dynamic sizes before continuing.

The result: every inference call stalled while the CPU waited for the GPU to finish the `Unique` stage, then resumed — a costly and avoidable overhead.

**Measured latency before optimization: 29.093 ms**

---

## Solution: Precompute Pooling Metadata Before TensorRT

The fix moves all pooling metadata discovery out of the TensorRT graph and into the **CUDA preprocessing stage** that already runs before each inference call. The computed tensors are passed into TensorRT as ordinary dynamic inputs. Because TensorRT receives them as externally-provided data (not as the result of an in-graph computation), all shapes are known up front and no mid-graph synchronization is needed.

**Measured latency after optimization: 19.138 ms — a 34% reduction**

---

## Changes Overview

This optimization is split across two companion PRs that must be deployed together.

### PR 1: AWML#206 — Restructure the ONNX export

**Goal:** Produce an ONNX model that accepts precomputed metadata as inputs instead of computing it internally.

The exported ONNX graph changes from:

```
Inputs: grid_coord, feat, serialized_code
Graph:  Unique → argsort → segment_csr → pooled features
Issue:  Unique output shape is data-dependent → TRT cannot infer static shapes
```

To:

```
Inputs: grid_coord, feat, serialized_code,
        serialized_pooling_0_{indices,indptr,cluster,...},
        serialized_pooling_1_{indices,indptr,cluster,...},
        ...  (one group per encoder stage)
Graph:  Gather + autoware::SegmentCSR plugin
Result: All shapes statically known → no CPU/GPU sync needed
```

Key code changes in `point_transformer_v3m1_base.py`:

- New **`SerializedPoolingMeta`** dataclass holding the 7 metadata tensors per stage
- New **`build_serialized_pooling_meta()`** function computing these tensors in Python at export time
- `SerializedPooling` gains an `export_mode` flag: in export mode it consumes a pre-built `SerializedPoolingMeta` instead of running `Unique`/`argsort`

Key changes in `tools/export.py`:

- Builds `SerializedPoolingMeta` for each encoder stage from a sample frame
- Registers metadata tensors as named ONNX inputs with proper dynamic axes
- Uses a distinct symbolic dimension `serialized_pooling_i_out_voxels_plus_one` for the `indptr` tensor (shape `[M+1]`) to avoid dim aliasing

### PR 2: autoware_universe#12727 — Precompute metadata in CUDA preprocessing

**Goal:** At runtime, compute the pooling metadata on the GPU before calling TensorRT, then bind it as engine inputs.

#### CUDA Kernel Pipeline (`preprocess_kernel.cu`)

The following runs on the GPU stream for each encoder pooling stage, before `enqueueV3`:

| Step | Kernel / API | Purpose |
|------|-------------|---------|
| 1 | `preparePoolingSortInputKernel` | Right-shift `serialized_code` by `pooling_depth × 3` bits to obtain group keys; fill out-of-range slots with `INT64_MAX` sentinel |
| 2 | `cub::DeviceRadixSort::SortPairs` | Sort voxels by group key entirely on the GPU |
| 3 | `markPoolingRunsKernel` | Flag the first voxel of each new group (run-length encoding) |
| 4 | `cub::DeviceScan::InclusiveSum` | Prefix-sum the flags to assign a contiguous group ID to each voxel |
| 5 | `fillPoolingStageKernel` | Populate `indices`, `indptr`, `head_indices`, `cluster`, `grid_coord` |
| 6 | `prepareOrderSortInputKernel` + `fillOrderAndInverseKernel` | Compute `serialized_order` and `serialized_inverse` for each serialization order (`z`, `z-trans`) |
| 7 | `cudaMemcpyAsync` + `cudaStreamSynchronize` | Copy per-stage output voxel counts (a few integers) back to CPU — **the only CPU/GPU sync** |
| 8 | `setSerializedPoolingInputShapes` | Set actual runtime shapes on the TRT engine using the synced counts |
| 9 | `enqueueV3` | TensorRT inference — all shapes known, no further sync |

The single sync in step 7 transfers only scalar integers (one per stage), making it negligible compared to the original mid-graph syncs.

---

## ONNX Input Contract

After precomputation the TensorRT engine receives 7 extra inputs per pooling stage `i`:

| Tensor name | Shape | Description |
|-------------|-------|-------------|
| `serialized_pooling_{i}_indices` | `[N_in]` | Per-voxel parent group index |
| `serialized_pooling_{i}_indptr` | `[N_out+1]` | CSR row pointer (one entry per output voxel, plus one) |
| `serialized_pooling_{i}_cluster` | `[N_in]` | Per-voxel cluster label |
| `serialized_pooling_{i}_head_indices` | `[N_out]` | Representative (head) voxel per group |
| `serialized_pooling_{i}_grid_coord` | `[N_out, 4]` | Grid coordinates of pooled voxels |
| `serialized_pooling_{i}_serialized_order` | `[N_in, 2]` | Serialization permutation (one column per order) |
| `serialized_pooling_{i}_serialized_inverse` | `[N_in, 2]` | Inverse of the serialization permutation |

`N_in` = voxel count entering stage `i`; `N_out = N_in / pooling_stride`.

`serialized_depth` is folded as a compile-time constant and is **not** a runtime input.

---

## Required Config Parameters

Two new parameters must be set in `config/ml_package_ptv3.param.yaml` to match the model training configuration:

```yaml
serialization_orders: ["z", "z-trans"]   # must match training config exactly
pooling_strides: [2, 2, 2, 2]            # one entry per encoder pooling stage; must be positive powers of two
```

These are validated at startup: `serialization_orders` must be exactly `["z", "z-trans"]` and every stride must be a positive power of two.

---

## Before vs After

```
Before
──────
Point cloud
  └─ CUDA preprocess
       └─ TensorRT graph
            ├─ voxelize
            ├─ Unique  ← data-dependent shape
            │   └─ CPU/GPU sync  ← stall
            ├─ argsort / segment_csr
            └─ PTv3 attention blocks
Latency: 29.093 ms


After
─────
Point cloud
  └─ CUDA preprocess
       ├─ voxelize
       ├─ RadixSort + InclusiveSum (GPU)
       ├─ fillPoolingStageKernel (GPU)
       ├─ fillOrderAndInverseKernel (GPU)
       └─ cudaMemcpyAsync (sync: copy ~4 ints)  ← only sync
            └─ TensorRT graph
                 ├─ Gather + SegmentCSR  ← shapes fully static
                 └─ PTv3 attention blocks
Latency: 19.138 ms  (↓ 34%)
```

---

## Deployment Notes

- The two PRs are a **breaking pair**: AWML#206 changes the ONNX input signature, and autoware_universe#12727 provides those inputs at runtime. Neither works without the other.
- The TensorRT engine **must be rebuilt** from an ONNX model generated with AWML#206.
- An equivalence test in `autoware_ptv3` (`serialized_pooling_metadata_test.cpp`) validates all 8 output tensors against a CPU reference implementation across both pooling stages.
