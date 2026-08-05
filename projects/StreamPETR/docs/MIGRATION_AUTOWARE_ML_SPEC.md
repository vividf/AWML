# StreamPETR Migration Spec: AWML → autoware-ml

**Goal:** Reproduce the AWML fine-tuning run

```bash
python tools/detection3d/train.py \
  projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py
```

(j6gen2 fine-tune with `traffic_cone` + `barrier`, partial-ignore) inside the new
**autoware-ml** framework (`~/ml_workspace/autoware-ml`), with **equal or near-equal
training results**.

This document is written so that a future agent can (a) understand both architectures,
(b) know exactly where they differ, and (c) execute the migration plan in §6.

Facts marked **[VERIFIED]** were confirmed by reading source in both repos on 2026-07-22.

---

## 1. The two architectures at a glance

| Dimension | AWML (`~/ml_workspace/AWML`) | autoware-ml (`~/ml_workspace/autoware-ml`) |
|---|---|---|
| Framework | OpenMMLab: mmengine 0.10.7 / mmdet3d 1.4.0 / mmcv 2.1.0 / mmdet 3.3.0 | PyTorch Lightning 2.6.1 + Hydra 1.3.2 + Pixi + MLflow |
| PyTorch | 2.8.0 + cu12.9 (Docker) | 2.9.1 + cu12.8 (Pixi) |
| Config | Python configs, `_base_` inheritance, mmengine `Registry` + `custom_imports` | YAML under `autoware_ml/configs/`, Hydra `defaults:` composition, `hydra.utils.instantiate` (`_target_`) |
| Train entry | `tools/detection3d/train.py` → `Runner.from_cfg(cfg)` → `runner.train()` | `autoware-ml train --config-name …` → `cli/cli.py` → `cli/runtime.py` → `scripts/train.py` (`@hydra.main`) → `lightning.Trainer.fit` |
| Session | foreground process | `autoware-ml session start --name X -- train …` = detached tmux-managed session |
| Model code location | per-project plugin: `projects/StreamPETR/stream_petr/**` (registered via registry) | first-class package: `autoware_ml/models/**` (plain classes, no registry) |
| Training loop | mmengine `EpochBasedTrainLoop`, hooks (Logger/Checkpoint/ParamScheduler) | LightningModule (`autoware_ml/models/base.py`), callbacks (ModelCheckpoint, EarlyStopping, LRMonitor) |
| Precision | custom `NoCacheAmpOptimWrapper` (fp16 AMP, dynamic loss scale, autocast cache off) | Lightning `precision: bf16-mixed`; TF32 matmul enabled globally (`utils/runtime.py`) |
| Checkpoints | mm format (`state_dict` + meta), `work_dirs/<config>/epoch_N.pth`, `save_best=mAP` | Lightning ckpt into MLflow run dir; monitor `val/loss`, `save_top_k=1` + `last`; **EarlyStopping(val/loss, patience=20)** |
| Weights init | `load_from=<path>` in config (mm loader, handles any matching keys) | `--weights <path>`: `utils/checkpoints.py:apply_matching_weights` — requires top-level `"state_dict"`, **exact key-name + shape match, no remapping** [VERIFIED] |
| Eval metric | `T4Metric` (NuScenesMetric subclass; mAP/NDS via autoware_perception_evaluation-style ranges) | native `Detection3DMetricSuite` (`autoware_ml/metrics/detection3d/`): MeanAP at val; NDS/TpErrors/HeadingAP at test only |
| Dataset infos | mmdet3d-v2 pkl `{data_list, metainfo{classes,version}}` from `tools/detection3d/create_data_t4dataset.py` | same pkl schema consumed by `datamodule/common/detection3d.py:load_detection_data_infos` [VERIFIED — **format compatible**] |
| Sampler (temporal) | `GroupStreamingSampler` (projects/StreamPETR/…/samplers) — seq lanes per batch column, **default `seed=10`**, `trim_sequences`, `random_drop_probability`, camera shuffle in dataset | `GroupStreamingSampler` (`autoware_ml/datamodule/samplers.py`) — same lane concept, rank-sharded, epoch shuffle via `set_epoch`; **no camera shuffle, no random drop** [VERIFIED] |
| Determinism | `randomness=dict(seed=0, deterministic=True)` | `seed` **not set** in StreamPETR configs → unseeded unless `+seed=N` passed [VERIFIED via `scripts/train.py:set_seed`] |

### Data path conventions
- AWML: `data/t4dataset/` + `info_directory_path="info2/"` → `data/t4dataset/info2/t4dataset_j6gen2_base_infos_{train,val,test}.pkl`.
- autoware-ml: `data_root: ${AUTOWARE_ML_DATA_PATH}/t4dataset`, ann files default to
  `info/detection3d/…`; overridable on CLI: `datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl`
  (paths are joined to `data_root` by `resolve_data_path`).
- **This machine uses the original `info2/` infos, NOT the `kokseang_2_8_1` ones** from the
  reference command.

---

## 2. StreamPETR component mapping

| Component | AWML | autoware-ml | Status |
|---|---|---|---|
| Detector | `stream_petr/models/detectors/petr3d.py:Petr3D` | `models/detection3d/streampetr.py:StreamPETRDetectionModel` (LightningModule) | ported |
| 3D head | `dense_heads/streampetr_head.py:StreamPETRHead` | `models/detection3d/heads/streampetr.py:StreamPETRHead` | ported (native rewrite) |
| **2D aux head** | `dense_heads/focal_head.py:FocalHead` (`img_roi_head`, 5 losses: QFL cls2d 2.0, GaussianFocal centerness 1.0, L1 bbox2d 5.0, GIoU 2.0, L1 centers2d 10.0) + `StreamPETRLoadAnnotations2D` | — | **MISSING** [VERIFIED] |
| Backbone | `backbones/vovnet.py:VoVNet` V-99-eSE | `models/common/backbones/vovnet.py:VoVNet99MultiScale` | ported (different param names) |
| Neck | `necks/cp_fpn.py:CPFPN` | `models/common/necks/lss_fpn.py:GeneralizedLSSFPN` | **different neck class** — output-parity must be checked |
| Transformer | `PETRTemporalTransformer` + `PETRMultiheadFlashAttention` (**flash-attn 2.7.3, fp16 kernels**, cross-attn) | `StreamPETRDecoderLayer` with **`torch.nn.MultiheadAttention`**, no flash-attn | ported, numerically different |
| Pos. encoding | `utils/positional_encoding.py` (`pos2posemb3d/1d`, nerf) | `task_modules/streaming.py` (same functions; pos2posemb3d in (y,x,z) order for pretrained-weight compat) | ported |
| Memory bank | in head (`memory_len=1024`, propagate 256, topk 256) | same, with explicit fp32 islands + float64 timestamps (bf16-safety) | ported |
| GridMask | in-model, `(True,True,rotate=1,offset=False,ratio=0.5,mode=1,prob=0.7)` | `models/common/grid_mask.py`, same params, `use_grid_mask: true` | ported |
| Assigner/coder | `HungarianAssigner3D`, `NMSFreeCoder`, mm match costs | `task_modules/{assigners,match_costs,bbox_coders}.py` | ported |
| Focal loss | `mmdet.FocalLoss` (supports 2-D per-class `label_weights` — used by partial-ignore) | `losses/detection3d/focal.py:SigmoidFocalLoss` | ported; **per-class label-weight support needed for partial-ignore — verify** |
| Dataset | `datasets/pipelines/dataset.py:StreamPETRDataset` (`reset_origin`, camera shuffle, seq grouping, `traffic_cone_barrier_status` → img_metas) | `datamodule/common/multiview_detection3d.py` + `datamodule/t4dataset/…` (`prev_exists`, ego_pose from `ego2global`; **global poses, no reset_origin**) | ported, feature gaps |
| **Partial-ignore** | `StreamPETRHead._get_target_single` + `dn_loss_single` + `FocalHead`: zero cls label-weights on cone/barrier columns for **background** queries when `traffic_cone_barrier_status=False` | — (no `partial_ignore` / `traffic_cone_barrier_status` anywhere) | **MISSING** [VERIFIED] |
| Image transforms | `ResizeCropFlipRotImage`, `GlobalRotScaleTransImage`, `PadMultiViewImage`, `NormalizeMultiviewImage` | `transforms/camera/{resize,geometry,normalize,loading}.py` equivalents | ported, config gaps (§4) |
| AMP wrapper | `NoCacheAmpOptimWrapper` (fp16 + cache off, pytorch#142234 workaround) | not needed (bf16 + fp32 islands) | by design |

---

## 3. CRITICAL confirmed defects / blockers in autoware-ml

### 3.1 [P0, VERIFIED] Image value-scale mismatch (255×)
`transforms/camera/loading.py:LoadMultiViewImagesFromFiles` defaults
`normalize_to_unit=True` → pixels divided by 255 → RGB in [0,1].
Then `NormalizeMultiviewImage` applies `mean=[123.675,116.280,103.530]`,
`std=[58.395,57.120,57.375]` — **[0,255]-scale ImageNet stats**. Net effect: every input
pixel ≈ `(0..1 − 123.7)/58.4 ≈ −2.1` (near-constant, information crushed to ~0.004 dynamic
range vs mean shift). No compensating rescale exists downstream (`data_preprocessing` is
empty for StreamPETR).
**Fix:** set `normalize_to_unit: false` on the loader in the StreamPETR configs (keep RGB
[0,255] + ImageNet RGB stats), or divide mean/std by 255. Any previously trained
autoware-ml StreamPETR checkpoints were trained on the broken scale and are not
comparable.

### 3.2 [P0, VERIFIED] BGR vs RGB — deliberate divergence, breaks weight reuse
- AWML: images stay **BGR**, `mean=[103.530,116.280,123.675]`, `std=[57.375,57.120,58.395]`, `to_rgb=False`.
- autoware-ml: loader converts to **RGB**, ImageNet stats in RGB order.
Colleague's note confirms this was an intentional change ("in inference we use bgr…
I converted it to rgb"). Channel order is *internally consistent* in each repo, but any
checkpoint transfer (nuScenes pretrain, streampetr_2_7 epoch_20) crosses the BGR↔RGB
boundary. First-conv-weight channel flip is **not** exactly equivalent because mean/std
per-channel values also swap (they do swap consistently here, so flipping conv1 input
channels IS a valid conversion for VoVNet stem — implement it in the converter, §6 P3).
Recommendation: make color order a config knob (`color_type: bgr` exists on
`LoadImageFromFile` but **not** on `LoadMultiViewImagesFromFiles` — add it), and for
strict parity runs use BGR + AWML stats.

### 3.3 [P0, VERIFIED] Missing 2D auxiliary FocalHead
AWML training uses `img_roi_head=mmdet.FocalHead` with 5 auxiliary 2D losses fed by
`StreamPETRLoadAnnotations2D` (3D boxes projected to per-camera 2D boxes/centers/depths).
This shapes the image features during training (Focal-PETR-style spatial alignment) and
materially affects final mAP. autoware-ml has no 2D aux head, no 2D annotation transform,
no QualityFocal/GaussianFocal/GIoU-2D losses in the StreamPETR path. Must be ported for
result parity.

### 3.4 [P0, VERIFIED] Missing partial-ignore (`traffic_cone_barrier_status`)
The whole point of the target config: scenes without cone/barrier annotation must not
punish cone/barrier predictions as false positives. Mechanism in AWML:
- info pkl per-frame bool `traffic_cone_barrier_status` (already present in the existing
  `info2/` j6gen2 pkls) → `img_metas`;
- in `_get_target_single`: `label_weights` becomes `(num_queries, num_classes)` and
  columns [5,6] (traffic_cone, barrier) are zeroed for **negative** queries when the
  status is False; same in `dn_loss_single` for DN negatives; mirrored in `FocalHead`.
autoware-ml drops the field at load time and its `StreamPETRTargets`/loss path has no
per-class label-weight concept. Must be ported (dataset plumb-through + head + focal loss
2-D weight support).

### 3.5 [P0, VERIFIED] No mm→Lightning checkpoint converter
`apply_matching_weights` does exact key-name matching with **no remapping**. The AWML
fine-tune starts from `work_dirs/streampetr_2_7/epoch_20.pth` (mm format, mm module
names: `img_backbone.*` VoVNet mm names, `pts_bbox_head.*`, packed flash-attn
`in_proj_weight`, CPFPN names). autoware-ml native names differ (`bbox_head.*`,
`nn.MultiheadAttention` params, `GeneralizedLSSFPN` lateral/output convs, VoVNet
`_OSAModule` naming). A bespoke key-mapping + structural conversion script (incl.
attention weight repacking, conv1 BGR→RGB flip, cls/reg branch sharing) **does not exist
anywhere in the repo** and must be written. (The colleague's
`streampetr_vov99_nuscenes_pretrain_converted.pth` implies a private script exists
somewhere — worth asking, but plan to write one.)

### 3.6 [P1] Recipe divergences vs. the target AWML config

| Hyperparameter | AWML target config | autoware-ml `vov_480x640_t4dataset_j6gen2` |
|---|---|---|
| point_cloud_range | `[-51.2,-51.2,-5.0, 51.2,51.2,3.0]` | `[-54,-54,-5, 54,54,3]` |
| eval_class_range | 51.2 m all classes | 54 m all classes |
| Epochs | 40 (`val_interval=5`, dynamic `[(35,1)]`) | 35, val every epoch, **EarlyStopping patience 20 on val/loss** |
| Batch / workers | bs=1, workers=32, `trim_sequences=True` | bs=8, workers=16 |
| LR | AdamW 5e-5, backbone ×0.1, wd 0.01, **`auto_scale_lr(base_batch_size=8, enable=True)`** → effective LR scales with total batch | AdamW 5e-5, backbone 5e-6, wd 0.01, no auto-scaling |
| LR schedule | LinearLR warmup 500 **iters** (start 1/3) → CosineAnnealing per-epoch, `eta_min=lr·1e-4` | `CyclicCosineAnnealingLR` warmup 1 **epoch** (`max_lr_factor=1.0` → flat), decay 34 ep, min 1e-4× — stepped per epoch |
| Train aug | resize jitter 0.02, **rand_flip=True**, GlobalRotScaleTrans rot ±0.3925 rad, scale [0.95,1.05]; **camera-order shuffle each frame**; sampler `random_drop_probability` | resize jitter 0.02, flip **off**, rot/scale **identity**; no camera shuffle; no random drop [VERIFIED] |
| Precision | fp16 AMP (dynamic scale) + flash-attn fp16 cross-attn; self-attn/cls/reg fp32 | bf16-mixed + TF32; same fp32 islands |
| Seed | seed=0 deterministic (sampler seed=10) | unseeded by default |
| load_from | `work_dirs/streampetr_2_7/epoch_20.pth` | `--weights` (converted ckpt required) |
| Best-model criterion | save_best = T4Metric mAP | monitor = `val/loss` |
| test_ann_file | `…_infos_test.pkl` | **points to `…_infos_val.pkl`** (likely unintended) [VERIFIED] |

### 3.7 [P2] Other differences to be aware of
- **reset_origin:** AWML re-centers ego poses per sequence (`reset_origin=True`);
  autoware-ml keeps global (kilometer-scale) poses but protects memory alignment with
  fp32 islands (regression-tested). Functionally OK, numerically different.
- **Metrics are not the same code:** T4Metric (mm/NuScenes protocol + T4 ranges) vs
  native `Detection3DMetricSuite`. Even a perfect model will score slightly differently.
  For acceptance, evaluate the *same* checkpoint under **AWML's T4Metric** (export
  predictions or convert the ckpt back) — see §7.
- **Timestamps:** AWML makes timestamps relative to sequence start (float64); autoware-ml
  keeps epoch-seconds float64. Both feed `Δt`; equivalent if diffs are used — verify once
  in the parity harness.
- flash-attn fp16 vs `nn.MultiheadAttention` bf16: bit-identical training is impossible;
  target statistical parity (§7).

---

## 4. What "same results" can realistically mean

Bit-identical training across the two stacks is **not achievable** (fp16 flash kernels vs
bf16 SDPA, TF32, different op schedules, different RNG consumers). The achievable and
recommended equivalence ladder:

1. **Forward parity (must-pass):** identical weights + identical preprocessed input →
   per-layer outputs match within fp tolerance (fp32 mode, atol ~1e-4).
2. **Loss parity (must-pass):** identical batch + weights → each loss term matches within
   tolerance (incl. DN terms and partial-ignore masking).
3. **Recipe parity:** every row of the §3.6 table aligned in a new autoware-ml config
   variant.
4. **Result parity (acceptance):** fine-tune both stacks from the same converted
   `epoch_20` init on the same `info2/` splits; final mAP (evaluated under ONE metric
   implementation) within noise band, suggested ±0.5 mAP overall and per-class cone/
   barrier AP within ±1.0.

---

## 5. Reference commands

AWML (ground truth):
```bash
# inside AWML docker
python tools/detection3d/train.py \
  projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py
```

autoware-ml (after migration; use the machine's original infos, not kokseang_2_8_1):
```bash
autoware-ml session start --name streampetr -- train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier \
  datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info2/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info2/t4dataset_j6gen2_base_infos_test.pkl \
  --weights <converted_streampetr_2_7_epoch_20.pth>
```

---

## 6. Migration plan

### P0 — Input correctness (blocking everything)
1. Fix the 255× scale bug: `normalize_to_unit: false` in both StreamPETR yaml configs
   (train/val/test/predict pipelines) **or** rescale mean/std; add a unit test asserting
   post-normalize pixel statistics on a real image.
2. Add `color_type: {rgb,bgr}` to `LoadMultiViewImagesFromFiles`; decide policy: keep RGB
   as repo default, use BGR+AWML stats for the parity config (weights convert either way
   via conv1 channel flip — do it once in the converter and keep RGB if preferred).
3. Fix `test_ann_file` pointing at val pkl.

### P1 — Feature ports for the target task
4. Plumb `traffic_cone_barrier_status` from info pkl → dataset sample → batch.
5. Port partial-ignore into `StreamPETRHead` target/DN-loss paths: 2-D `label_weights`
   with cone/barrier columns zeroed for negative queries when status is False;
   extend `SigmoidFocalLoss` to accept per-class weights. Unit-test against the AWML
   logic with a synthetic batch.
6. Port the 2D auxiliary `FocalHead` + `StreamPETRLoadAnnotations2D` (3D→2D projection)
   + the 4 missing loss types, wired as an optional `img_roi_head` on
   `StreamPETRDetectionModel`. (If a first milestone without aux head is acceptable,
   quantify its impact by an AWML ablation run instead — but for "same results" it must
   exist.)

### P2 — Recipe-parity config
7. New Hydra variant `vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml`:
   pc_range ±51.2, eval range 51.2, epochs 40, warmup-by-iteration LinearLR(1/3, 500
   iters)+cosine (add an iteration-based scheduler or configure Lightning
   `interval: step`), batch size + LR scaling decision (either bs=1 lr matching AWML's
   auto-scaled effective LR, or document the intended difference), train aug enabled
   (flip, rot ±0.3925, scale 0.95–1.05) to match AWML, `+seed=0`, disable/loosen
   EarlyStopping, checkpoint-monitor on mAP.
8. Add camera-order shuffle (train only) and optional sampler random-drop to match AWML's
   regularization, or run an AWML ablation to prove they don't matter for this fine-tune.

### P3 — Checkpoint conversion + parity harness
9. Write `tools/convert_mm_streampetr_checkpoint.py`: mm `epoch_20.pth` →
   `{"state_dict": …}` with native names. Handles: VoVNet name map, CPFPN→GeneralizedLSSFPN
   (verify structural equivalence first — if not equivalent, port CPFPN), flash-attn
   packed `in_proj_weight` → `nn.MultiheadAttention` params, shared cls/reg branches,
   BGR→RGB conv-stem flip (if RGB policy), embeddings/memory buffers.
10. Forward/loss parity harness: one real sample through AWML (docker) dumped to disk →
    same tensors through autoware-ml model with converted weights → compare head inputs,
    pos_embed, per-layer cls/bbox outputs, each loss term. Gate: fp32 atol ≤1e-4
    (forward), losses ≤1e-3 relative.

### P4 — Training runs & acceptance
11. Short smoke fine-tune (1–2 epochs) both stacks from converted epoch_20; compare loss
    curves step-aligned (expect overlap within AMP noise).
12. Full 40-epoch fine-tune; evaluate BOTH resulting checkpoints under AWML T4Metric
    (convert autoware-ml ckpt back, or add a prediction-dump + offline T4Metric path).
    Acceptance: §4 item 4.

---

## 7. Implementation status (2026-07-22)

P0–P3 items 1–9 are **IMPLEMENTED** in `~/ml_workspace/autoware-ml` (branch
`feat/add-streampetr`, uncommitted) and verified in the autoware-ml docker
(unit tests + real-data smoke). Summary:

- **P0.1** `normalize_to_unit: false` set in all StreamPETR pipelines (both yamls).
- **P0.2** `color_type: {rgb,bgr}` added to `LoadMultiViewImagesFromFiles`;
  repo policy stays RGB, the converter flips the stem conv instead.
- **P0.3** `test_ann_file` now points at the test pkl.
- **P1.4** `traffic_cone_barrier_status` plumbed pkl → `get_data_info` →
  collation (list) → `compute_metrics` → head losses.
- **P1.5** Partial-ignore ported: `StreamPETRHead` (`class_names` +
  `partial_ignore_classes`), 2-D `label_weights` on negative queries in
  `_get_targets`, DN background masking (`_dn_label_weights`),
  `SigmoidFocalLoss` accepts per-class weights. Unit-tested against the AWML
  semantics (`tests/models/test_streampetr_partial_ignore.py`).
- **P1.6** Auxiliary 2D head ported natively: `FocalHead2D` (5 losses:
  QualityFocal 2.0 / L1 bbox 5.0 / GIoU 2.0 / L1 centers2d 10.0 / Gaussian
  centerness 1.0), `HungarianAssigner2D` + 2D costs, 2D box utils,
  `LoadAnnotations2DFromBoxes3D` transform (3D→2D projection after aug,
  before range filter). Wired as optional `img_roi_head`; training-only.
  Focal token sampling not ported: `train_ratio=1.0` makes it a permutation
  (attention-invariant).
- **P2.7** `vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml`:
  pc_range ±51.2, eval ranges 51.2, 40 epochs, bs=1, lr 6.25e-6 (AWML
  auto-scaled 5e-5·bs/8, backbone ×0.1), `IterWarmupEpochCosineLR`
  (500-iter warmup ×1/3 → per-epoch cosine, `interval: step`), rand_flip +
  rot ±0.3925 + scale 0.95–1.05, seed 0, EarlyStopping disabled,
  checkpoint monitor `val/det3d/mAP` (max).
  **Camera-order shuffle and sampler random-drop are NOT ported** (P2 item 8
  second half — ablate or port later).
- **P3.9** `autoware_ml/tools/convert_streampetr_checkpoint.py` — needs no mm
  install (tolerant unpickler). Verified against the real `epoch_20.pth`:
  894/899 tensors mapped (5 skipped mm buffers), **every** native model
  parameter initialized (the only unloaded state-dict keys are
  `image_feature_extractor.*` aliases of the same tensors). Output at
  `autoware-ml/work_dirs/streampetr_2_7_epoch_20_converted.pth`.
- **CPFPN verdict (was open question 2):** CPFPN and GeneralizedLSSFPN are
  **not** weight-compatible (concat+BN+ReLU vs add, no norm/act). CPFPN was
  ported natively (`models/common/necks/cp_fpn.py`, mm-compatible parameter
  names) and the finetune config uses it.
- **Color-order verdict (was open question 3):** RGB kept as repo policy;
  BGR→RGB conv-stem flip done in the converter (`--bgr-to-rgb`).

**Added 2026-07-22 (second pass):**
- Shared parity settings extracted to `_awml_parity.yaml`; the fine-tune
  config is now a thin variant on top of it.
- **Base-training config** `vov_480x640_t4dataset_j6gen2_base.yaml`
  (35 epochs / bs 4 / LR 5e-5 = 1e-4·4/8), mirroring
  `vov_flash_480x640_baseline.py`. Init from the nuScenes model-zoo
  pretrain via the converter (`--drop-pattern` strips the 10-vs-7-class
  layers — autoware-ml's loader refuses shape mismatches that mm's
  strict=False silently skips). **TODO: zoo checkpoint download +
  conversion not yet executed on this machine.**
- Converter gained `--drop-pattern REGEX` (repeatable) + unit test.
- Beginner docs added alongside this spec: `TRAINING_AWML.md`,
  `TRAINING_AUTOWARE_ML.md`, `MIGRATION_ALIGNMENT.md`.

**Added 2026-07-22 (third pass):**
- **Camera-order shuffle ported** (`LoadMultiViewImagesFromFiles`
  `shuffle_order: true`, train pipeline only, unit-tested). **Sampler
  random-drop needs no port**: `random_drop_probability` defaults to 0.0
  and no target config sets it — inactive in the AWML recipe.
- **Parity harness written (P3.10)**: `projects/StreamPETR/tools/parity_dump.py`
  (AWML side, fp32/eval/no-DN, FlashMHA→fp32 SDPA monkeypatch) +
  `autoware_ml/tools/streampetr_parity_check.py` (native side, PASS/FAIL
  table; forward atol 1e-4, loss rel 1e-3). Not yet executed.
- **Box-encoding audit**: this fork's `normalize_bbox` = native encoding
  (centers at reg channels 0–2) — converter needs no channel permutation.
- **New finding**: AWML `ego_pose = ego2global @ lidar2ego`; native uses
  `ego2global` only. Irrelevant for single frames, affects multi-frame
  memory alignment if `lidar2ego` ≠ identity — must verify.
- nuScenes zoo checkpoint downloaded to
  `autoware-ml/work_dirs/nuscenes_vov99_baseline_320x800.pth`.

**Added 2026-07-22 (fourth pass — all remaining verifications executed):**
- **nuScenes pretrain conversion VERIFIED**: 880/899 tensors mapped, 0
  unexpected; only the 14 class-count layers stay at their focal-prior
  init (equivalent to mm strict=False). Ready at
  `autoware-ml/work_dirs/nuscenes_vov99_baseline_320x800_converted.pth`.
- **P3.10 parity harness EXECUTED — PARITY: PASS** (see
  MIGRATION_ALIGNMENT.md §7): geometry path and isolated 2D head
  **bit-exact**; all 17 loss terms (partial-ignore active) within rel
  3e-4; forward within rel 4e-4 (conv-noise scale). Harness lessons: mm
  `clip_sigmoid` and `position_embeding` mutate tensors in place — the
  dump snapshots before loss; the AWML docker image lacks
  `fvcore`/`flash_attn`, so the dump pip-installs fvcore and stubs
  flash_attn (safe: FlashMHA.forward is replaced by fp32 SDPA).
- **P4 smoke fine-tune DONE** (dev-subset): 2 epochs via the real
  `autoware-ml train` entry, converted `epoch_20` init — train/loss
  32.2→30.8, val/det3d/mAP 0.767→0.834, mAP-based best-checkpoint works.

**Still pending:** the `lidar2ego` ego-pose composition check (multi-frame
memory alignment), and full-dataset P4 acceptance runs under one metric
implementation (this machine only has a ~19-frame dev subset).

Launch command (verified to compose; datamodule + one real training batch +
converted weights produce finite losses and full gradient coverage):

```bash
cd ~/ml_workspace/autoware-ml
./docker/container.sh --run --data-path ~/ml_workspace/AWML/data --cmd \
  "autoware-ml train \
    --config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier \
    datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl \
    datamodule.val_ann_file=info2/t4dataset_j6gen2_base_infos_val.pkl \
    datamodule.test_ann_file=info2/t4dataset_j6gen2_base_infos_test.pkl \
    --weights /workspace/work_dirs/streampetr_2_7_epoch_20_converted.pth"
```

---

## 8. Open questions for the team
- Does the private converter behind `streampetr_vov99_nuscenes_pretrain_converted.pth`
  exist and can it be upstreamed? (Would shortcut P3.)
- Is CPFPN vs GeneralizedLSSFPN weight-compatible/output-equivalent? If not, which is
  canonical going forward?
- Is the intended production color order RGB (autoware-ml) — i.e., should AWML parity be
  achieved in BGR then converted, or should the whole pipeline be re-standardized on RGB?
- Was any autoware-ml StreamPETR result obtained *with* the 255× scale bug? If yes, those
  baselines need to be re-run after the P0 fix.
