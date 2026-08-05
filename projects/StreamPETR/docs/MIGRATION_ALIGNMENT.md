# StreamPETR: AWML ↔ autoware-ml Alignment Status

This document answers one question: **after the migration (2026-07-22), what
is aligned between the two frameworks and what still differs?** It is the
companion to [TRAINING_AWML.md](TRAINING_AWML.md) and
[TRAINING_AUTOWARE_ML.md](TRAINING_AUTOWARE_ML.md); the underlying analysis
and implementation log live in
[MIGRATION_AUTOWARE_ML_SPEC.md](MIGRATION_AUTOWARE_ML_SPEC.md).

Legend: ✅ aligned (verified) · ⚠️ intentionally different (understood, should
not block result parity) · ❌ not aligned yet (open work).

---

## 1. Input pipeline

| Item | AWML | autoware-ml | Status |
|---|---|---|---|
| Pixel value scale | [0,255] | [0,255] (`normalize_to_unit: false` — was a /255 **bug**, fixed) | ✅ |
| Normalization stats | ImageNet mean/std, 0-255 scale | same | ✅ |
| Color order | **BGR** end-to-end | **RGB** end-to-end; converter flips the stem conv (`--bgr-to-rgb`), which is mathematically equivalent | ⚠️ equivalent by construction |
| Resize/crop/flip aug | jitter ±0.02, rand_flip on | same (`_awml_parity.yaml`) | ✅ |
| BEV rot/scale aug | rot ±0.3925 rad, scale 0.95–1.05 | same | ✅ |
| Pad to /32 | yes | yes | ✅ |
| 2D aux annotations | `StreamPETRLoadAnnotations2D` (project augmented 3D boxes per camera) | `LoadAnnotations2DFromBoxes3D`, same math, same pipeline position (after aug, before range filter) | ✅ |
| 3D range filter | ±51.2 m | ±51.2 m | ✅ |
| Camera-order shuffle (train) | every frame (`shuffle_cameras=True` dataset default) | **ported**: `LoadMultiViewImagesFromFiles(shuffle_order: true)` in the train pipeline only; all per-camera arrays follow the shuffled order (unit-tested) | ✅ |
| Sampler random frame drop | `random_drop_probability` **defaults to 0.0 and the target configs never set it** — inactive in the AWML recipe | not ported (nothing to port) | ✅ inactive |
| Ego poses | re-centered per sequence (`reset_origin=True`); **`ego_pose = ego2global @ lidar2ego`** (lidar→global) | global (km-scale) poses + fp32 islands; **`ego_pose = ego2global` only** — differs by `lidar2ego`. Single-frame math is unaffected (memory starts empty), but multi-frame memory alignment sees the composed pose. If T4's `lidar2ego` is not identity this is a real gap — **verify** (open work §6) | ⚠️ verify |
| Timestamps | relative to sequence start (float64) | epoch seconds (float64); only Δt is consumed | ⚠️ |

## 2. Model

| Item | AWML | autoware-ml | Status |
|---|---|---|---|
| Backbone | VoVNet V-99-eSE, norm_eval | `VoVNet99MultiScale`, identical parameter names | ✅ |
| Neck | CPFPN | **CPFPN (ported natively)** — the pre-migration `GeneralizedLSSFPN` is *not* weight-compatible and is no longer used by the parity configs | ✅ |
| 3D head structure | StreamPETRHead: 644 queries, memory 1024, propagate/topk 256, DN, shared cls/reg branches | same | ✅ |
| 2D aux head | FocalHead (5 losses: QFL×2, L1 bbox×5, GIoU×2, L1 centers2d×10, GaussianFocal centerness×1) | **FocalHead2D (ported)**, same tower layout, same parameter names, same loss weights | ✅ |
| Focal token sampling | topk over all tokens at `train_ratio=1.0` | not ported — at ratio 1.0 it is a pure permutation, attention-invariant | ⚠️ no-op by construction |
| Partial-ignore | zero cone/barrier cls columns for negative queries when `traffic_cone_barrier_status=False`; also in DN losses and FocalHead | same semantics, unit-tested against AWML behavior (`test_streampetr_partial_ignore.py`) | ✅ |
| Cross-attention kernel | flash-attention 2.7.3 (fp16) | `nn.MultiheadAttention` (SDPA); **same packed weight layout**, no repacking needed | ⚠️ numerically different, weight-compatible |
| Precision | fp16 AMP (dynamic loss scale, autocast cache off) | bf16-mixed + TF32, fp32 islands for geometry/heads | ⚠️ |
| GridMask | in-model, prob 0.7 | same | ✅ |
| Assigner / coder / costs | HungarianAssigner3D, NMSFreeCoder | native ports, same weights | ✅ |

## 3. Training recipe

| Item | AWML fine-tune | autoware-ml `..._finetune_cone_barrier` | Status |
|---|---|---|---|
| Epochs | 40 | 40 | ✅ |
| Batch size | 1 | 1 | ✅ |
| Effective LR | 5e-5 × total_bs/8 (auto_scale_lr) = 6.25e-6 @ 1 GPU | 6.25e-6 hard-coded (comment explains scaling; **rescale manually for multi-GPU**) | ⚠️ manual |
| Backbone LR | ×0.1 | ×0.1 | ✅ |
| Weight decay / grad clip | 0.01 / norm 1.0 | same | ✅ |
| LR schedule | LinearLR 500 iters ×1/3 → per-epoch cosine to ×1e-4 | `IterWarmupEpochCosineLR`, same shape, stepped per iteration | ✅ |
| Seed | 0 (deterministic) | 0 (`seed: 0`) | ✅ |
| Best-checkpoint criterion | T4Metric mAP | `val/det3d/mAP` (max) | ✅ criterion, ⚠️ metric impl (§4) |
| EarlyStopping | none | disabled | ✅ |
| Validation cadence | every 5 epochs, every epoch for the last 5 | **every epoch** (Lightning has no dynamic intervals; strictly more checkpoints, no recipe impact) | ⚠️ |
| Init checkpoint | `epoch_20.pth` (mm auto-load) | converted `streampetr_2_7_epoch_20_converted.pth` via `--weights` — **conversion verified: full parameter coverage** | ✅ |

The base configs (`vov_flash_480x640_baseline.py` ↔
`vov_480x640_t4dataset_j6gen2_base.yaml`) align the same way with
35 epochs / bs 4 / LR 5e-5 (=1e-4×4/8). **TODO:** the nuScenes-pretrain
download + conversion for base training is documented but not yet executed
on this machine (see TRAINING_AUTOWARE_ML.md §4.2).

## 4. Evaluation & infrastructure (framework-level, not recipe)

| Item | AWML | autoware-ml | Status |
|---|---|---|---|
| Metric implementation | `T4Metric` (NuScenes protocol) | native `Detection3DMetricSuite` | ⚠️ **different code** — same checkpoint scores slightly differently; final acceptance must compare both stacks under ONE implementation (spec §4) |
| Experiment tracking | text logs + work_dirs | MLflow | ⚠️ cosmetic |
| Checkpoint format | mm dict | Lightning ckpt | ⚠️ converter bridges one direction (mm→native); native→mm not written |
| Distributed | mm DistributedDataParallel + auto_scale_lr | Lightning DDP, sampler shards scenes by rank; LR scaling manual | ⚠️ |
| Determinism | `deterministic=True` cudnn | seeded but non-deterministic kernels allowed | ⚠️ |

## 5. Why bit-identical results are impossible (and what to expect)

fp16 flash-attention vs bf16 SDPA, TF32 matmuls, different op schedules and
RNG consumers make step-by-step identity unattainable. The realistic ladder
(spec §4): forward parity (fp32, atol ≤1e-4) → loss parity (≤1e-3 rel) →
recipe parity (§3 above — done) → **result parity**: fine-tune both stacks
from the same converted `epoch_20`, evaluate both under AWML's T4Metric,
accept within ±0.5 mAP overall / ±1.0 per-class on cone/barrier.

## 6. Open work (ordered)

1. **Verify the `lidar2ego` ego-pose gap** (§1 table) — if T4's
   `lidar2ego` is not identity, the native datamodule must compose it into
   `ego_pose` like AWML does; a 2-frame parity extension would prove it.
2. **Full P4 acceptance** — full 40-epoch fine-tunes in both stacks on the
   complete dataset (this dev machine's `info2/` train pkl holds only ~19
   frames), evaluated under ONE metric implementation (AWML T4Metric),
   acceptance ±0.5 mAP overall / ±1.0 per-class on cone/barrier.
3. Decide validation cadence / multi-GPU LR policy if the runs move off a
   single GPU.
4. DN-loss parity is unit-tested structurally but excluded from the harness
   (denoising draws random noise; the RNG streams cannot be aligned across
   the two stacks without injecting recorded noise).

## 7. Parity harness results (2026-07-22) — **PARITY: PASS**

One real j6gen2 val frame (with `traffic_cone_barrier_status=False`, so the
partial-ignore path was **active**), AWML `epoch_20.pth` vs the converted
checkpoint, both stacks fp32/eval/no-DN/no-dropout:

| Comparison | Result |
|---|---|
| `pos_embed`, `cone` (geometry path) | **bit-exact (0.0)** |
| 2D aux head on identical features (isolated) | **bit-exact (0.0)** — head + converted weights fully equivalent |
| `img_feats` (VoVNet+CPFPN end-to-end) | max rel 3.2e-4 (cross-container cuDNN algorithm noise) |
| `all_cls_scores` / `all_bbox_preds` (6 decoder layers) | max rel 3.9e-4 / 2.7e-4 |
| All 12 3D loss terms (incl. partial-ignore cls) | max rel 2.8e-4 |
| All 5 auxiliary 2D loss terms | max rel 2.3e-4 |
| Decoded 2D outputs end-to-end | ≤4.1e-3 (conv noise amplified at sigmoid saturation; isolated check above is the equivalence proof) |

Lessons encoded in the harness: mm's `clip_sigmoid` mutates the centerness
logits **in place** and the mm head pixel-scales `location` in place — the
dump snapshots both before loss computation.

Smoke fine-tune (P4 item 11, dev-subset scale): 2 epochs through the real
`autoware-ml train` entry with the converted `epoch_20` — train/loss
32.2→30.8, `val/det3d/mAP` 0.767→0.834, best-checkpoint selection on mAP
working, all 30 loss terms logged to MLflow.

## 8. Verification already done (2026-07-22, autoware-ml docker)

- 188 unit tests pass, incl. 10 new partial-ignore/converter/head tests and
  a converter round-trip against a real synthetic mm state dict.
- Both parity configs compose and instantiate end-to-end (model, transforms,
  scheduler, callbacks).
- Real `epoch_20.pth` converted: 894/899 tensors mapped, every native model
  parameter initialized (only alias keys unloaded).
- One real j6gen2 training batch through the full stack with converted
  weights: all 30 loss terms finite (incl. 5 aux-2D and DN losses,
  partial-ignore active on a status-False frame), gradients flow to every
  trainable parameter.
- Camera-order shuffle: unit test asserts images/intrinsics stay consistent
  under shuffling and that the default (val/test) path never shuffles.
- Box-encoding channel order audited: this AWML fork's `normalize_bbox` is
  `(cx,cy,cz,logw,logl,logh,sin,cos,vx,vy)` with centers at channels 0–2 in
  the reg branch (`tmp[..., 0:3] += reference`) — identical to the native
  encoding, so the converter correctly copies reg branches without any
  channel permutation.
