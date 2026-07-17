# StreamPETR → `deployment/` Migration Spec

Status: **ALL PHASES IMPLEMENTED (2026-07-17, Docker-verified)** — `deployment/projects/streampetr/`
exports the 3 ONNX components with exact I/O and op-count parity against the deployed reference
(`work_dirs/streampetr/simplify_*.onnx`); TensorRT FP16 engines build and run; stateful
inference pipelines pass 3-backend evaluation (mAP 0.544–0.549 BEV-center, TRT 65.5 ms vs
PyTorch 237 ms); PyTorch-vs-ONNX FP32 verification passes at max_diff 3e-4. Known limitation:
FP16 element-wise verification fails by design (discrete top-k reordering) — judge FP16 by mAP
(documented in the deploy config). Remaining follow-up: switch the model config to T4MetricV2
for training-comparable absolute mAP. It follows the framework's project-layout contract
(`deployment/docs/architecture.md#project-layout-contract`) and the design philosophy in
`CLAUDE.md` (deep modules, small interfaces, information hiding, evolutionary change).

Author aid: line references were taken from the repo at the time of writing; verify against
source before implementing (Graphify/report may be stale).

---

## 0. TL;DR

- Old StreamPETR deployment is **export-only**: 389 lines (`projects/StreamPETR/deploy/torch2onnx.py`
  + `containers.py`) that emit **3 independent ONNX files** from **random dummy inputs**, with no
  data loader, no inference pipeline, no verification, and no evaluation.
- The new framework expects a full project bundle (config, io, export, inference, evaluation,
  runner, entrypoint) wired to shared orchestrators. CenterPoint is the multi-component reference.
- StreamPETR is a **camera-based, temporally-stateful 3D detector**. Two facts drive every design
  decision:
  1. The shared 3D entrypoint `run_detection3d_deployment` **hard-wires `PointCloudDataLoader`**
     (`deployment/runtime/detection3d_entrypoint.py:81`). A camera model must self-wire its
     entrypoint (the YOLOX pattern) while still reusing the modality-agnostic `Detection3DEvaluator`.
  2. StreamPETR carries a **temporal memory queue** across frames. There is no dedicated framework
     seam for state — it must live inside the project's inference pipelines, and it interacts
     awkwardly with the stateless, warmup-replaying evaluator loop (`base_evaluator.py:167,197`).
- Recommended sequencing: **export parity first** (Phase 1–3), then **inference + verification**
  (Phase 4–5), then **evaluation** (Phase 6). Evaluation is the highest-risk, lowest-parity-value
  piece and should not block ONNX/TRT export.

---

## 1. Understand — current architecture (old deploy)

### 1.1 What exists today

| File | LOC | Role |
| --- | --- | --- |
| `projects/StreamPETR/deploy/torch2onnx.py` | 189 | CLI: build model from config, load ckpt, export **one** of 3 "sections" to ONNX + `onnxsim` |
| `projects/StreamPETR/deploy/containers.py` | 200 | 3 `torch.nn.Module` proxy wrappers that reconstruct sub-graphs of the model for tracing |

### 1.2 The three export "sections"

The model is deliberately **split into 3 ONNX graphs** because the runtime (Autoware / the
DL4AGX TensorRT reference) runs them as a pipeline with host-side glue between them:

1. **`extract_img_feat`** — `TrtEncoderContainer`: image backbone + neck.
   Input: `img [1, N_cam, 3, H, W]` → Output: `img_feats`.
2. **`position_embedding`** — `TrtPositionEmbeddingContainer`: 3D position encoding from camera
   geometry. Requires `onnxruntime.tools.pytorch_export_contrib_ops.register()` before export
   (custom op). Inputs: `img_metas_pad, img_feats, intrinsics, img2lidar` → Outputs: `pos_embed, cone`.
3. **`pts_head_memory`** — `TrtPtsHeadContainer`: transformer decoder head **plus explicit
   temporal-memory I/O**. The memory queue (`memory_embedding / reference_point / timestamp /
   egopose / velo`) is passed **in** as `pre_memory_*` tensors and returned **out** as
   `post_memory_*`. The container manually reimplements `_pts_head` and `_post_update_memory`
   (mutating `head.memory_*` in place) and sets `head.with_dn = False`.

### 1.3 Data flow (old)

```
img ─▶ [extract_img_feat] ─▶ img_feats ─┐
                                          ├─▶ [position_embedding] ─▶ pos_embed, cone ─┐
intrinsics, img2lidar, img_metas ────────┘                                             │
                                                                                       ▼
pre_memory_* (host state) ─────────────────────────────────────────▶ [pts_head_memory] ─▶ preds + post_memory_*
```

The 3 graphs are chained **outside** the model, by the C++/host runtime. The temporal state lives
on the host and is threaded through `pts_head_memory` each frame.

### 1.4 Load-time model surgery

`torch2onnx.py:63-64` swaps flash-attention for `PETRMultiheadAttention` in the decoder config
**before** building the model. This is mandatory for exportability and must be preserved.

### 1.5 What is NOT present today

- No `BaseDataLoader` — inputs are `np.random.uniform(...)`, so the ONNX graph is **never validated
  against a real sample**.
- No inference pipeline, no cross-backend verification, no accuracy evaluation in the deploy path.
- `work_dir` and shapes are read ad-hoc from the training config (`cfg.ida_aug_conf.final_dim`,
  `cfg.num_cameras`, `cfg.stride`, `cfg.model.pts_bbox_head.in_channels`).

---

## 2. Preserve — what is good and must survive the migration

These are load-bearing design decisions in the old code. The migration must **keep their behavior
byte-compatible with the deployed runtime**, because downstream Autoware C++ consumes the exact
graph I/O names and the 3-file split.

1. **The 3-file split and the exact input/output tensor names.** Autoware and the DL4AGX reference
   depend on them. This is a stable, externally-observed contract — treat it as frozen. It maps
   cleanly onto the framework's multi-component export (`components` config + `ModelComponentBuilder`),
   so we *keep the contract and change only the machinery*.
2. **Flash-attention → `PETRMultiheadAttention` surgery.** Correct and necessary; relocate verbatim
   into `io/model_loader.py`.
3. **Host-side temporal state threading.** Making the memory queue explicit graph I/O (rather than
   hidden module state) is the right call for a stateless TRT engine — it keeps the engine pure and
   puts state ownership on the caller. Preserve this shape; do not try to bake the queue into the
   engine.
4. **`with_dn = False` at export.** Denoising is train-only; disabling it for export is correct.

**Design note:** the old `containers.py` reimplements the head forward by hand. That is the one part
that is *fragile* (it silently drifts from `streampetr_head.py`), but its **externally observable
result** — the graph I/O — is the contract we preserve. We keep the I/O, and we should try to reduce
the hand-reimplementation (see §4.2 open question).

---

## 3. Target — the framework seam checklist

From `deployment/docs/architecture.md#project-layout-contract`, a project bundle at
`deployment/projects/streampetr/` must satisfy:

| Seam | Base / contract | StreamPETR obligation | Difficulty |
| --- | --- | --- | --- |
| `__init__.py` | `ProjectAdapter(name, run)` → `project_registry.register` | Register `"streampetr"` | trivial |
| `entrypoint.py` | `run(args) -> int` | **Self-wired** (camera loader) — cannot use `run_detection3d_deployment` | medium |
| `config/` | `BaseDeploymentConfig` subclass | Project keys + `_REQUIRED_COMPONENTS` = the 3 sections + `_validate_components` | low |
| `io/` (data) | `BaseDataLoader` | **New multi-camera + temporal loader** returning `input`/`metadata`/`ground_truth` | **high** |
| `io/` (model) | wrap `build_mmdet3d_model` | Register StreamPETR modules + apply attention surgery | low |
| `export/` sample extractor | `SampleExtractor` | Extract a **real** sample; run encoder→pos-embed to chain component inputs | **high** |
| `export/` component builder | `ModelComponentBuilder` | Emit the 3 `ExportableComponent`s | medium |
| `export/onnx_models/` | export-time graph modules | Port the 3 containers (as ONNX-friendly modules) | medium |
| `export/` contrib-op hook | per-component `post_transforms` or pipeline `finalize` | Register contrib ops for `position_embedding` | medium |
| `inference/` | `BaseInferencePipeline` ×3 backends (+`GPUResourceMixin` for TRT) | **Stateful 3-stage pipeline** threading the memory queue | **high** |
| `evaluation/executor.py` | `BackendExecutor` (not `PointCloudBackendExecutor`) | `create_pipeline` + camera `prepare_input` | medium |
| `evaluation/` evaluator | `Detection3DEvaluator` (reuse) | Reuse directly (modality-agnostic; `min_num_points` no-ops without points) | low |
| metrics | `extract_t4metric_v2_config` + `Detection3DMetricsInterface` | Model config already uses `T4MetricV2` + `class_names` — reuse | none |
| `runner.py` | `BaseDeploymentRunner` subclass | Override `load_pytorch_model`; wire custom `onnx_pipeline` | low |

**Reused for free** (framework already handles): `OnnxExportPipeline`, `TensorRTExportPipeline`,
exporters, `ArtifactManager`, all three orchestrators, `BackendVerifier`, `OutputComparator`,
`Detection3DEvaluator`, `Detection3DMetricsInterface`, config schema, CLI dispatch.

---

## 4. Review — the hard problems and design decisions

Each subsection states the problem, root cause, alternatives with trade-offs, and a
**recommendation**. These are the decisions worth debating before writing code.

### 4.1 Camera data loader + self-wired entrypoint

**Problem.** `run_detection3d_deployment` builds `PointCloudDataLoader` unconditionally
(`detection3d_entrypoint.py:81`). StreamPETR needs multi-view images + camera intrinsics/extrinsics
+ ego-pose + timestamps, and temporally-ordered frames.

**Root cause.** The shared 3D entrypoint conflated "3D detection" with "point-cloud input." The
*evaluator* is already modality-agnostic (`detection_3d_evaluator.py`, documented as such); only the
*loader construction* is LiDAR-specific.

**Alternatives.**

- **(A) Self-wire the entrypoint (YOLOX pattern).** StreamPETR's `entrypoint.py` writes its own
  `run(args)` that builds a `StreamPETRDataLoader`, derives metrics via `extract_t4metric_v2_config`,
  constructs the shared `Detection3DEvaluator`, and builds the runner.
  - *Pro:* zero change to shared code; localized; matches the existing camera precedent (YOLOX).
  - *Con:* duplicates ~30 lines of the shared wiring (logging, config parse, metrics extraction).
- **(B) Generalize the shared entrypoint** to inject the data loader (`data_loader_factory`).
  - *Pro:* removes the LiDAR assumption; the next camera-3D model reuses it.
  - *Con:* speculative generality for N=1 today; changes shared code that CenterPoint/BEVFusion
    depend on; adds a factory seam before a second consumer exists. Violates "don't abstract until
    the second case."

**Recommendation: (A) now, (B) later.** Follow YOLOX. If/when a *second* camera-3D detector lands,
extract the shared camera wiring then — that is the evolutionary-architecture move. Record a TODO in
`detection3d_entrypoint.py` noting the LiDAR coupling.

### 4.2 Multi-component export with real, chained inputs

**Problem.** The 3 components are **interdependent** (encoder output feeds position-embedding;
both feed the head), but the old exporter fed each one **random** tensors. The framework's
`SampleExtractor` is expected to produce a *real* payload, and CenterPoint's builder shows the
precedent of running early stages to produce later-stage inputs
(`CenterPointComponentBuilder._prepare_backbone_input` runs the voxel + middle encoder).

**Root cause.** In the old code, correctness of the *graph* was assumed, not verified; random inputs
are fine for shape-tracing but useless for numerical verification/evaluation.

**Design.** `StreamPETRSampleExtractor.extract_sample` loads one real frame and runs
`extract_img_feat` → produces `img_feats`; runs `position_embeding` → produces `pos_embed, cone`;
seeds a **zeroed memory queue** (`memory_len` from `pts_bbox_head`). It returns a typed
`StreamPETRExportSample` holding every tensor the 3 components need. `StreamPETRComponentBuilder`
then slices that sample into 3 `ExportableComponent`s whose `name` matches the deploy-config keys.

**Open question — reduce hand-reimplementation.** `containers.py` manually rewrites the head forward.
Two options for the `onnx_models/`:
- **(i) Port the containers verbatim** (fast, low-risk parity, keeps the fragility).
- **(ii) Refactor so the export modules call the real `streampetr_head` methods** (`position_embeding`,
  `temporal_alignment`, `transformer`, …) instead of re-implementing them.
  - *Pro:* export graph can't silently drift from the model; deep-module win.
  - *Con:* the head methods may contain non-traceable control flow the containers were written to
    avoid; higher migration risk.

**Recommendation: (i) for Phase 1 parity, then evaluate (ii) as a follow-up** once the ported graph
is byte-verified against the current deployed ONNX. Do not couple parity to a refactor.

### 4.3 Stateful temporal memory vs. the stateless evaluator loop

**Problem.** The framework's inference contract is stateless per-sample
(`BaseInferencePipeline.preprocess → run_model → postprocess`), and `BaseEvaluator.evaluate` iterates
`for idx in range(actual_samples)` **and runs warmup that replays samples**
(`base_evaluator.py:167,197-218`). StreamPETR's accuracy depends on a memory queue carried **in frame
order**, reset at **sequence/clip boundaries**. Warmup replay and any reordering corrupt the queue.

**Root cause.** The framework assumes IID samples; StreamPETR samples are a time series.

**Where state can live (framework facts).**
- The eval loop **does** iterate in index order — the correct order for a temporal model — *if* the
  dataset is arranged as contiguous ordered clips (the training `GroupStreamingSampler` does this;
  the deploy loader must reproduce that ordering).
- There is **no** state seam; state must live inside the project pipeline instance (it persists
  across `infer()` calls because the executor creates the pipeline once per backend).

**Alternatives for state ownership.**

- **(A) Pipeline-owned queue.** The pipeline holds `self._memory_*`, updates it in `run_model`, and
  resets on a frame-metadata flag (`prev_exists == False` / new `scene_token`). `preprocess` reads
  the reset flag from sample metadata.
  - *Pro:* localized; matches how the PyTorch model already threads state; no shared-code change.
  - *Con:* must guarantee ordering + disable warmup for StreamPETR; a subtle coupling to eval-loop
    internals.
- **(B) Loader-owned queue threaded through metadata.** The queue lives outside the pipeline and is
  passed in/out via `InferenceInput.metadata`.
  - *Pro:* pipeline stays pure/stateless — closer to framework spirit.
  - *Con:* `infer()` returns `InferenceResult`, not an updated queue; would require reaching around
    the return contract. More friction than (A).

**Interactions with the shared loop that must be handled:**
- **Warmup must be 0** for StreamPETR (`evaluation.num_warmup = 0` in deploy config), else replay
  corrupts the queue. Document this as a hard config constraint.
- **Sequence-boundary reset** must be driven by sample metadata, surfaced by the data loader.
- **Verification** (`BackendVerifier`) compares a few samples; with state, ref and test must both
  start from the same (zeroed) queue and step the same frames — verify **from the clip start**, not
  arbitrary indices.

**Recommendation: (A) pipeline-owned queue**, with the loader guaranteeing clip-ordered samples and
exposing a per-frame `is_sequence_start`/`prev_exists` flag; `num_warmup=0` enforced (ideally
validated in the StreamPETR config subclass). This is the smallest, most local change and mirrors the
model's own design. Flag the deeper question — *should the framework grow a first-class "sequential
evaluation" mode?* — as future work, not part of this migration.

### 4.4 The `position_embedding` contrib-op registration

**Problem.** Exporting `position_embedding` needs
`onnxruntime.tools.pytorch_export_contrib_ops.register()` first (a global side-effect before
`torch.onnx.export`).

**Alternatives.**
- **(A) Register inside the export module / component builder** just before building that component.
- **(B) A per-component `post_transforms`** — *wrong tool*: `post_transforms` runs on the already-
  exported `ModelProto`; registration must happen **before** tracing.
- **(C) Pipeline `finalize` hook** — also post-export; wrong phase.

**Recommendation: (A).** Register the contrib ops in `StreamPETRComponentBuilder.build_components`
(or in the `position_embedding` `onnx_models` module import) guarded so it runs once. Keep it out of
the shared exporter — it is StreamPETR-specific.

### 4.5 `onnxsim` simplification

Old code runs `onnxsim.simplify` per section. The shared `ONNXExporter` already supports optional
`onnxsim` via `OnnxConfig.simplify` (`config/schema.py:110`). **Recommendation:** set `simplify:
true` in the deploy config and delete the manual simplify call — reuse the framework capability.

### 4.6 Evaluation & metrics — the easy win

`Detection3DEvaluator` is explicitly modality-agnostic and its `_filter_by_min_num_points` no-ops
when `points is None` (`detection_3d_evaluator.py:106-119`). The T4 model configs already declare
`val_evaluator = T4MetricV2` with `class_names`, so `extract_t4metric_v2_config` works unchanged.
**No new evaluator or metrics code is needed** — only correct GT parsing in the camera data loader
(`ground_truth` = 3D boxes + labels in the T4 frame).

---

## 5. Discussion — questions to resolve before coding

1. **Scope of v1.** Is the goal (a) *export parity* with the current deployed 3-file ONNX, or
   (b) a *full* pipeline incl. camera evaluation? These have very different risk profiles.
   Recommendation: land (a) first as a self-contained PR, then (b).
2. **Numerical parity oracle.** Do we have the current deployed `simplify_*.onnx` (v2.5 model-zoo
   links exist in `docs/t4dataset/v2/base.md`) to diff the migrated export against? Verification is
   only meaningful with a reference.
3. **Dataset ordering.** Can the T4 test info be loaded as contiguous ordered clips for temporal
   inference, and does the info carry `prev_exists`/scene boundaries? This gates §4.3.
4. **Head reimplementation.** Accept the ported containers verbatim for v1 (§4.2 option i), or invest
   in calling real head methods now?
5. **TensorRT plugins.** Does StreamPETR TRT need custom plugins (the DL4AGX reference uses some)? If
   so, they flow through `tensorrt_config.plugin_libraries` and the executor factory, like BEVFusion.

---

## 6. Proposed phased plan

Each phase is independently reviewable and leaves the tree working. Verify on host with
ast/pyflakes; run e2e in Docker (host python lacks torch — see project memory).

### Phase 0 — Scaffolding (no behavior)
- Create `deployment/projects/streampetr/` with the directory skeleton mirroring CenterPoint.
- `__init__.py` registers `ProjectAdapter(name="streampetr", run=run)`.
- `config/streampetr_deployment_config.py`: `StreamPETRDeploymentConfig(BaseDeploymentConfig)` with
  `_REQUIRED_COMPONENTS = ("extract_img_feat", "position_embedding", "pts_head_memory")` +
  `_validate_components`. Enforce `num_warmup == 0` here.
- `config/deploy_config.py`: components (3), `export`, `evaluation`, `verification`, `devices`,
  `checkpoint_path`, top-level `model_cfg`. Mirror CenterPoint's config shape.
- **Exit:** CLI lists `streampetr`; config parses; `--help` works.

### Phase 1 — Model loading + camera data loader
- `io/model_loader.py`: `build_streampetr_model` wrapping `build_mmdet3d_model`, importing the
  StreamPETR module variants and applying the flash-attn → `PETRMultiheadAttention` surgery.
- `io/data_loader.py`: `StreamPETRDataLoader(BaseDataLoader)` — multi-cam images + intrinsics +
  img2lidar + ego-pose + timestamps; `ground_truth` = T4 3D boxes/labels; expose
  `is_sequence_start`; guarantee clip-ordered samples.
- `io/sample_types.py`: typed `StreamPETRExportSample` + inference sample dataclasses.
- **Exit:** load a checkpoint; iterate N ordered samples with correct GT and metadata.

### Phase 2 — Export components (ONNX)
- `export/onnx_models/`: port the 3 containers as ONNX-friendly modules (v1: verbatim, §4.2 option i).
- `export/sample_extractor.py`: `StreamPETRSampleExtractor` — real sample, chained component inputs,
  zeroed memory seed.
- `export/component_builder.py`: `StreamPETRComponentBuilder` → 3 `ExportableComponent`s; register
  contrib ops for `position_embedding` (§4.4).
- `runner.py`: `StreamPETRDeploymentRunner(BaseDeploymentRunner)` overriding `load_pytorch_model`,
  wiring `OnnxExportPipeline(sample_extractor=…, component_builder=…)`.
- `entrypoint.py`: self-wired `run(args)` (§4.1 option A).
- Deploy config `export.mode = ONNX`, `onnx.simplify = true`.
- **Exit:** `python -m deployment.cli.main streampetr <deploy_cfg> <model_cfg>` emits the 3 ONNX with
  the exact deployed I/O names; diff against reference ONNX if available (§5.2).

### Phase 3 — TensorRT export
- Component `tensorrt_profile` + `tensorrt_config` (plugins if needed, §5.5).
- **Exit:** 3 engines build; `export.mode = BOTH` works.

### Phase 4 — Inference pipelines (stateful)
- `inference/streampetr_inference_pipeline.py`: shared base implementing `preprocess/run_model/
  postprocess` with the pipeline-owned memory queue (§4.3 option A) + per-stage abstract hooks
  (`run_encoder`, `run_position_embedding`, `run_head`).
- Three thin backend subclasses (pytorch/onnx/tensorrt); TRT mixes in `GPUResourceMixin`.
- `evaluation/executor.py`: `StreamPETRExecutor(BackendExecutor)` — `create_pipeline` (3-way) +
  camera `prepare_input`; `get_output_names` for verification.
- **Exit:** run a single frame end-to-end on each backend.

### Phase 5 — Cross-backend verification
- `verification` scenarios in the deploy config (ONNX-vs-PyTorch, TRT-vs-ONNX). Verify **from clip
  start** with identical zeroed queues (§4.3).
- **Exit:** verification passes within tolerance for the first K frames of a clip.

### Phase 6 — Evaluation (camera 3D mAP)
- Wire the shared `Detection3DEvaluator` in the entrypoint; `num_warmup = 0`.
- **Exit:** mAP within tolerance of the training-time T4Metric on the eval set; matches the numbers
  in `docs/t4dataset/v2/base.md` (v2.5 total mAP ≈ 0.45).

### Phase 7 — Docs + cleanup
- `deployment/projects/streampetr/README.md` (CenterPoint README as template).
- Update `projects/StreamPETR/README.md` §5 to point at the new CLI; decide whether to delete the old
  `deploy/` or leave a deprecation shim.

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Ported head graph drifts from `streampetr_head.py` | med | med | v1 verbatim + byte-diff vs reference ONNX; follow-up §4.2(ii) |
| Temporal state corrupted by warmup/reordering | high | high | `num_warmup=0` enforced in config; clip-ordered loader; verify from clip start |
| No reference ONNX to verify against | med | high | Pull v2.5 model-zoo ONNX (§5.2); else gate verification on availability |
| Camera GT parsing wrong (frame/coords) | med | high | Cross-check against training eval numbers in docs |
| TRT plugins required but missing | low | med | Mirror BEVFusion `plugin_libraries` path |
| Shared entrypoint LiDAR coupling tempts a premature refactor | med | low | Self-wire now (§4.1 A); refactor only at 2nd camera-3D model |

---

## 8. Explicitly out of scope (for this migration)

- Quantization (PTQ/QAT) of StreamPETR — the framework supports it, but parity + FP path come first.
- A first-class framework "sequential/temporal evaluation" mode — flagged as future work (§4.3).
- Generalizing `run_detection3d_deployment` to camera loaders — deferred to the 2nd consumer (§4.1).
- Changing the deployed 3-file ONNX contract or its tensor names — frozen (§2).
