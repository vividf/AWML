# StreamPETR Training in autoware-ml — Beginner's Guide

This guide explains how StreamPETR training works in **autoware-ml**
(`~/ml_workspace/autoware-ml`, Lightning/Hydra-based): the commands, what
happens internally, and how it maps onto the AWML workflow you may already
know. Companion docs:

- [TRAINING_AWML.md](TRAINING_AWML.md) — the same walkthrough for AWML
- [MIGRATION_ALIGNMENT.md](MIGRATION_ALIGNMENT.md) — what is aligned / still different after the migration
- [MIGRATION_AUTOWARE_ML_SPEC.md](MIGRATION_AUTOWARE_ML_SPEC.md) — the migration spec and status

---

## 1. The stack in one paragraph

autoware-ml replaces the whole OpenMMLab stack with plain, first-class code:
**PyTorch Lightning** (training loop, checkpointing, DDP), **Hydra** (YAML
config composition, `_target_` instantiation), **Pixi** (environment), and
**MLflow** (experiment tracking). There are **no registries** — a config's
`_target_: autoware_ml.models...ClassName` is just an import path, and
`hydra.utils.instantiate` builds the object tree recursively. Model code
lives in `autoware_ml/models/**` as ordinary classes.

## 2. Environment

Everything runs inside the autoware-ml docker container:

```bash
cd ~/ml_workspace/autoware-ml
./docker/container.sh --run --data-path /path/to/your/datasets
```

On this machine the datasets live in the AWML repo, so:

```bash
./docker/container.sh --run --data-path ~/ml_workspace/AWML/data
```

What the script mounts (see `docker/container.sh`):

| Host | Container | Meaning |
|---|---|---|
| the autoware-ml repo | `/workspace` | code, configs, work_dirs |
| `--data-path` dir | `/workspace/data` | datasets + info pkls |

The image presets `AUTOWARE_ML_DATA_PATH=/workspace/data`, and every dataset
config resolves `data_root: ${oc.env:AUTOWARE_ML_DATA_PATH}/t4dataset` —
so with the mount above, `data/t4dataset/info2/*.pkl` from AWML is visible
unchanged. Useful variants:

```bash
./docker/container.sh --exec                      # enter the running container
./docker/container.sh --stop                      # stop it
./docker/container.sh --run --headless --cmd "pytest autoware_ml/tests -q"   # one-shot command
```

## 3. Data

autoware-ml consumes the **same mmdet3d-v2 info pkls** as AWML (verified in
the migration): `{data_list: [...], metainfo: {classes, version}}` per split.
No re-generation needed — point the config at AWML's `info2/` files via CLI
overrides (see §4). The per-frame `traffic_cone_barrier_status` flag in those
pkls is read by the datamodule since the migration.

## 4. The two training commands

Both run **inside the container**. Configs are addressed by their path under
`autoware_ml/configs/tasks/` (the `tasks/` prefix is added automatically).

### 4.1 Fine-tune (cone/barrier partial-ignore — mirrors the AWML fine-tune)

```bash
autoware-ml train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier \
  datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info2/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info2/t4dataset_j6gen2_base_infos_test.pkl \
  --weights /workspace/work_dirs/streampetr_2_7_epoch_20_converted.pth
```

The `--weights` file is AWML's `work_dirs/streampetr_2_7/epoch_20.pth`
converted with the tool in §6 (already generated on this machine).

### 4.2 Base training (from the nuScenes pretrain) — complete walkthrough

Unlike AWML (which auto-downloads `load_from` URLs and partial-loads with
`strict=False`), autoware-ml takes only **local** `--weights` paths and
refuses shape mismatches — so the pretrain is downloaded and converted
**once**, with the class-count-dependent layers dropped at conversion time.
End to end:

```bash
# 0. On the host: start the container (repo -> /workspace, data -> /workspace/data)
cd ~/ml_workspace/autoware-ml
./docker/container.sh --run --data-path ~/ml_workspace/AWML/data

# 1. (once, inside the container) Download + convert the nuScenes pretrain.
wget -c -O work_dirs/nuscenes_vov99_baseline_320x800.pth \
  'https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth'
python -m autoware_ml.tools.convert_streampetr_checkpoint \
  --input  work_dirs/nuscenes_vov99_baseline_320x800.pth \
  --output work_dirs/nuscenes_vov99_baseline_320x800_converted.pth \
  --bgr-to-rgb \
  --drop-pattern 'cls_branches\.\d+\.6\.' \
  --drop-pattern 'img_roi_head\.cls\.'

# 2. Train. On this machine the j6gen2 infos live in info2/:
autoware-ml train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base \
  datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info2/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info2/t4dataset_j6gen2_base_infos_test.pkl \
  --weights /workspace/work_dirs/nuscenes_vov99_baseline_320x800_converted.pth

#    To train on the full t4dataset *base* split instead (like AWML's
#    2_8 base run), point the ann files at the base pkls, e.g.:
#      datamodule.train_ann_file=info2/t4dataset_base_infos_train.pkl ...

# 3. Long runs: detached managed session instead of a foreground process
autoware-ml session start --name streampetr-base -- train \
  --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base \
  datamodule.train_ann_file=info2/t4dataset_j6gen2_base_infos_train.pkl \
  datamodule.val_ann_file=info2/t4dataset_j6gen2_base_infos_val.pkl \
  datamodule.test_ann_file=info2/t4dataset_j6gen2_base_infos_test.pkl \
  --weights /workspace/work_dirs/nuscenes_vov99_baseline_320x800_converted.pth
```

> **Step 1 is done and verified (2026-07-22):** the converted file exists at
> `work_dirs/nuscenes_vov99_baseline_320x800_converted.pth` — 880/899
> tensors mapped, 0 unexpected keys, and the only non-alias unloaded model
> keys are exactly the 14 class-count-dependent layers
> (`bbox_head.cls_branches.*.6.*`, `img_roi_head.cls.*`), which keep their
> focal-prior initialization just like AWML's `strict=False` load.

**Batch-size / LR parity note.** AWML has two base recipes that land on the
same effective LR of 5e-5: the default baseline (bs=4/GPU, lr=1e-4,
auto-scaled ×4/8) and the production 2_8 partial-ignore base (bs=8/GPU,
lr=5e-5, auto-scaled ×8/8). The autoware-ml config defaults to **bs=4 with
lr=5e-5** (matches the default baseline exactly). For strict parity with the
2_8 recipe add `batch_size=8` (memory permitting) — the LR is already 5e-5;
for any other total batch size scale `model.optimizer.lr` and
`model.optimizer_group_overrides.img_backbone.lr` by total_batch/8 yourself
(autoware-ml has no auto_scale_lr).

Long runs are best started as **managed sessions** (detached tmux, survives
your terminal):

```bash
autoware-ml session start --name streampetr-ft -- train --config-name ... --weights ...
autoware-ml session list / logs / stop
```

## 5. Config anatomy (Hydra composition)

```
vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml   <- run config: bs=1, 40 epochs, LRs
vov_480x640_t4dataset_j6gen2_base.yaml                    <- run config: bs=4, 35 epochs, LRs
  └── defaults:
       - vov_480x640_t4dataset_j6gen2      <- colleague's j6gen2 config (pipelines, model dims)
       - _reset_scheduler                  <- clears the inherited scheduler node (Hydra merge trick)
       - _awml_parity                      <- everything AWML-recipe-shaped (shared by both runs)
       - _self_                            <- this file's own overrides win last
```

`_awml_parity.yaml` is the heart of the migration; it holds the shared
AWML-parity settings: pc_range ±51.2, eval ranges 51.2, full train
augmentation (flip / rot ±0.3925 / scale 0.95–1.05), the 2D annotation
projection transform, the CPFPN neck, the auxiliary 2D `FocalHead2D`,
partial-ignore wiring, the iteration-warmup + epoch-cosine scheduler,
`seed: 0`, checkpoint selection by `val/det3d/mAP`, and EarlyStopping
disabled.

Hydra notes for AWML users:

- `defaults:` list ≈ `_base_` inheritance; later entries override earlier,
  `_self_` positions this file's own keys.
- Merging is per-key like mm; to **replace** a whole node you must first
  null it (that is what `_reset_scheduler.yaml` does).
- Any key can be overridden from the CLI: `batch_size=2`,
  `model.optimizer.lr=1e-5`, `+seed=1` (leading `+` adds a new key).

## 6. Checkpoint conversion (mm → Lightning)

`autoware_ml/tools/convert_streampetr_checkpoint.py` converts AWML/mm
StreamPETR checkpoints to the native naming; it needs **no mm packages**
(tolerant unpickler skips mmengine metadata). What it does:

| Source (mm) | Target (native) | Transform |
|---|---|---|
| `img_backbone.*` | `img_backbone.*` | name identical (VoVNet) |
| `img_neck.*` | `img_neck.*` | name identical (CPFPN was ported natively) |
| `pts_bbox_head.*` | `bbox_head.*` | prefix rename; attention layouts already match |
| `img_roi_head.*` | `img_roi_head.*` | name identical (FocalHead2D mirrors FocalHead) |
| stem conv (with `--bgr-to-rgb`) | | flips input channels: AWML trains on BGR, autoware-ml loads RGB |
| `--drop-pattern REGEX` | | drops class-count-dependent layers when source/target class counts differ |

Verified on `epoch_20.pth`: 894/899 tensors mapped, **full coverage** of the
native model (the 5 skipped keys are non-persistent mm buffers, recomputed
from the config).

`--weights` then loads by **exact key name + shape**; shape mismatches are a
hard error (that's why `--drop-pattern` exists), and unexpected/unloaded
keys are printed in a load report — read it once per new checkpoint.

## 7. What happens when you press Enter

1. **CLI → Hydra**: `autoware-ml train` resolves the config, applies your
   CLI overrides, and calls the training entrypoint
   (`autoware_ml/scripts/train.py`).
2. **Seeding**: `seed: 0` (set in `_awml_parity`) seeds python/numpy/torch.
3. **Instantiate**: `hydra.utils.instantiate` builds the datamodule and the
   model (`StreamPETRDetectionModel`, a `LightningModule`, with backbone /
   CPFPN / `bbox_head` / `img_roi_head` children).
4. **Weights**: `--weights` triggers `apply_matching_weights` (§6).
5. **Dataloaders**: the datamodule builds streaming loaders — its
   `GroupStreamingSampler` assigns whole scenes to batch lanes, shards
   scenes by DDP rank, reshuffles per epoch; `prev_exists` marks scene
   starts so the head resets its memory. Same idea as AWML's sampler
   (camera-order shuffle and random frame drop are **not** ported — see
   MIGRATION_ALIGNMENT.md).
6. **Transforms per frame** (train):
   load 5 cams (RGB, [0,255] since the `normalize_to_unit` fix) →
   ResizeCropFlipRot (flip on) → GlobalRotScaleTrans (±0.3925 / 0.95–1.05) →
   pad /32 → normalize → **project 3D boxes to per-camera 2D annotations** →
   range filter ±51.2.
7. **Training step** (`bf16-mixed` autocast, fp32 islands for geometry):
   images → backbone+CPFPN → `bbox_head` (memory propagation, 6 decoder
   layers, per-layer cls/bbox losses + DN losses) and, training only,
   `img_roi_head` (5 auxiliary 2D losses). **Partial-ignore** zeroes the
   cone/barrier classification columns for negative queries on frames whose
   `traffic_cone_barrier_status` is False — in the main losses, the DN
   losses, and the 2D head.
8. **Scheduler**: `IterWarmupEpochCosineLR` steps **per iteration** — linear
   warmup ×1/3 → 1 over 500 iters, multiplied by a per-epoch cosine decay to
   ×1e-4 (mirrors AWML's LinearLR + CosineAnnealingLR pair).
9. **Validation** (every epoch): decoded predictions go to the native
   `Detection3DMetricSuite`; validation logs `val/det3d/mAP` and per-class
   APs. `ModelCheckpoint` keeps `best.ckpt` by **max val/det3d/mAP** plus
   `last.ckpt`. EarlyStopping is disabled (AWML recipes always run all
   epochs).
10. **Tracking**: metrics, hyperparameters, and checkpoints are logged to
    MLflow (`sqlite:///mlruns/mlflow.db`); inspect with
    `autoware-ml mlflow-ui`.

## 8. Cheat sheet

| I want to… | Command / place |
|---|---|
| Enter the container | `./docker/container.sh --run --data-path ~/ml_workspace/AWML/data` |
| Fine-tune cone/barrier | §4.1 command |
| Train from base | §4.2 command (convert the pretrain first — see TODO) |
| Long run in background | `autoware-ml session start --name X -- train ...` |
| Evaluate a checkpoint | `autoware-ml test --config-name <same config> --checkpoint <ckpt>` |
| See metrics dashboards | `autoware-ml mlflow-ui` |
| Convert an AWML checkpoint | `python -m autoware_ml.tools.convert_streampetr_checkpoint ...` (§6) |
| Check parity vs AWML | dump with `AWML/projects/StreamPETR/tools/parity_dump.py`, then `python -m autoware_ml.tools.streampetr_parity_check --reference ... --checkpoint ...` |
| Change the recipe | edit the run config or override on the CLI |
| Find the model code | `autoware_ml/models/detection3d/{streampetr.py, heads/streampetr.py, heads/focal2d.py}` |
| Find data code | `autoware_ml/datamodule/common/multiview_detection3d.py`, `autoware_ml/transforms/camera/` |
| Run the StreamPETR tests | `pytest autoware_ml/tests/models/test_streampetr*.py -q` |
