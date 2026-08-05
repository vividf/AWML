# StreamPETR Training in AWML — Beginner's Guide

This guide explains how StreamPETR training works in **AWML** (this repo,
OpenMMLab-based): the commands, what happens internally at each step, and
where every moving part lives. Read this together with:

- [TRAINING_AUTOWARE_ML.md](TRAINING_AUTOWARE_ML.md) — the same walkthrough for the new **autoware-ml** framework
- [MIGRATION_ALIGNMENT.md](MIGRATION_ALIGNMENT.md) — what is aligned between the two after the migration, and what still differs
- [MIGRATION_AUTOWARE_ML_SPEC.md](MIGRATION_AUTOWARE_ML_SPEC.md) — the full migration spec and implementation status

---

## 1. The stack in one paragraph

AWML is built on the **OpenMMLab** ecosystem: `mmengine` (training runner,
config system, hooks), `mmdet3d`/`mmdet` (detection models, transforms,
losses), and `mmcv` (ops). Models and datasets are **registered by name** in
global registries and **instantiated from Python config files**. Everything
about a run — model architecture, data pipeline, optimizer, schedule, hooks —
is described in one config that inherits from `_base_` configs.

## 2. Environment

Training runs inside the AWML docker image. From the repo root:

```bash
docker run -it --rm --gpus all --shm-size=64g --name awml \
  -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml

# one-time inside the container
cd projects/StreamPETR && pip install -e . && cd /workspace
```

Key point: the repo is mounted at `/workspace` and the dataset at
`/workspace/data`. The host `data/t4dataset/` directory holds both the raw
T4dataset and the **info pkl files** (see §3).

## 3. Data: info files

Training never reads the raw dataset directly — it reads **info pkls**
(one dict per frame: image paths, calibration, ego pose, boxes, and for our
task the per-frame `traffic_cone_barrier_status` flag). They are created once:

```bash
python tools/detection3d/create_data_t4dataset.py \
  --root_path ./data \
  --config /workspace/autoware_ml/configs/detection3d/dataset/t4dataset/base.py \
  --version base --max_sweeps 1 --out_dir ./data/info/cameraonly/baseline
```

On this machine the StreamPETR configs point at `data/t4dataset/info2/`
(`info_directory_path = "info2/"`), e.g.
`info2/t4dataset_j6gen2_base_infos_{train,val,test}.pkl`.

## 4. The two training commands

### 4.1 Base training (from the nuScenes pretrain) — complete walkthrough

The production base recipe for the cone/barrier task is
`t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base_partialignore.py`
(35 epochs, **batch_size=8 per GPU**, lr=5e-5 with `auto_scale_lr(base=8)`,
partial-ignore + 2D aux head on, trained on the t4dataset **base** split).
End to end:

```bash
# 0. On the host: enter the AWML container (repo -> /workspace, data -> /workspace/data)
cd ~/ml_workspace/AWML
docker run -it --rm --gpus all --shm-size=64g --name awml \
  -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml

# 1. (once) Create the t4dataset *base* info pkls if they don't exist yet.
#    The base split needs the full T4 base DB downloaded under data/t4dataset/.
python tools/detection3d/create_data_t4dataset.py \
  --root_path ./data \
  --config autoware_ml/configs/detection3d/dataset/t4dataset/base.py \
  --version base --max_sweeps 1 --out_dir ./data/t4dataset/info2

# 2. (once) Prefetch the nuScenes pretrain. The 2_8 config's load_from
#    points at this local file (a URL there also works — mmengine downloads
#    automatically — but shows 0% GPU for a long time first):
mkdir -p pretrained
wget -c -O pretrained/nuscenes_vov99_baseline_320x800.pth \
  'https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth'

# 3. Train (single GPU)
python tools/detection3d/train.py \
  projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base_partialignore.py

#    Multi-GPU (auto_scale_lr rescales the LR by total batch / 8)
bash tools/detection3d/dist_script.sh \
  projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base_partialignore.py \
  2 train
```

Two machine-specific lines near the top of that config may need editing
before step 3 (`--cfg-options` cannot fix them, because the config bakes
them into the dataloader dicts at parse time):

```python
info_directory_path = "info/kokseang_2_8/"   # -> where YOUR base info pkls live, e.g. "info2/"
data_root = "data/"                          # -> e.g. "data/t4dataset/"
```

The pretrain loads with `strict=False`: tensors whose name+shape match are
loaded; everything else (the 10-class nuScenes classification layers vs our
7 classes) is skipped with a warning and stays randomly initialized. This
"download + partial load" is the part autoware-ml does **not** do
automatically — see the other guide.

Outputs land in `work_dirs/<config_name>/`: `epoch_N.pth` checkpoints, the
best-mAP checkpoint (`save_best="NuScenes metric/T4Metric/mAP"`), logs, and
a dump of the resolved config.

### 4.2 Fine-tune (cone/barrier partial-ignore, the migration target)

```bash
python tools/detection3d/train.py \
  projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py
```

This config inherits everything from the baseline and overrides:
`load_from = work_dirs/streampetr_2_7/epoch_20.pth` (a local checkpoint from
a previous base run), `batch_size=1`, `num_epochs=40`, `lr=5e-5`, and the
j6gen2 `info2/` annotation files.

Multi-GPU for either command:

```bash
bash tools/detection3d/dist_script.sh <config> <num_gpus> train
```

## 5. Config anatomy (what inherits what)

```
t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py   <- the run config (fine-tune deltas)
  └── _base_: ../default/vov_flash_480x640_baseline.py                      <- model + pipeline + recipe
        ├── _base_: autoware_ml/configs/detection3d/default_runtime.py      <- hooks, logging
        └── _base_: autoware_ml/configs/detection3d/dataset/t4dataset/base.py <- class names, name_mapping
```

Inheritance is **dict merge**: the child overrides keys of the parent
(`_delete_=True` replaces a whole node). The important blocks in
`vov_flash_480x640_baseline.py`:

| Block | What it configures |
|---|---|
| `model = dict(type="Petr3D", ...)` | detector: VoVNet-99 backbone → CPFPN neck → `img_roi_head` (FocalHead, 2D aux) + `pts_bbox_head` (StreamPETRHead, 3D) |
| `train_pipeline` / `test_pipeline` | per-frame transform list (load images → aug → normalize → 2D annotation projection → range filter → bundle) |
| `train_dataloader` | `StreamPETRDataset` + **`GroupStreamingSampler`** (see §6.2) |
| `optim_wrapper` | `NoCacheAmpOptimWrapper` = **fp16 AMP** with dynamic loss scale; backbone `lr_mult=0.1`; grad-clip 1.0 |
| `param_scheduler` | LinearLR warmup (500 **iterations**, start ×1/3) + CosineAnnealingLR (per **epoch**, `eta_min = lr*1e-4`) |
| `val_evaluator` / `test_evaluator` | `T4Metric` (NuScenes-protocol mAP/NDS with T4 class ranges) |
| `auto_scale_lr = dict(base_batch_size=8, enable=True)` | effective LR = configured LR × (total batch / 8) |
| `randomness = dict(seed=0, deterministic=True)` | reproducibility |

Registration: `custom_imports = dict(imports=["projects.StreamPETR.stream_petr"])`
makes mmengine import the plugin package, which registers `Petr3D`,
`StreamPETRHead`, `FocalHead`, `StreamPETRDataset`, `GroupStreamingSampler`,
etc. into the mm registries so the strings in the config resolve to classes.

## 6. What happens when you press Enter

### 6.1 Boot

`tools/detection3d/train.py` reads the config, applies `custom_imports`,
builds a `mmengine.Runner` from it (`Runner.from_cfg`), and calls
`runner.train()`. The Runner instantiates the model, dataloaders, optimizer
wrapper, param schedulers, and hooks from the config dicts.

### 6.2 Streaming batches (the StreamPETR-specific part)

StreamPETR is a **temporal** model: its head keeps a memory bank of the top
256 queries from the previous frame. Batches must therefore be
**scene-contiguous**: each batch lane walks one driving scene frame by frame.
`GroupStreamingSampler` does this — it groups dataset indices by scene,
assigns whole scenes to batch lanes, shuffles scene order per epoch
(`seed=10` internally), and optionally trims lanes (`trim_sequences=True`) so
they stay aligned. The dataset marks the first frame of each scene with
`prev_exists=False`, which tells the head to reset its memory.

The train pipeline per frame:

1. `LoadMultiViewImageFromFiles` — 5 cameras, **BGR**, float32, [0,255]
2. `LoadAnnotations3D` — 3D boxes + labels
3. `ResizeCropFlipRotImage` — resize jitter ±0.02, random flip
4. `GlobalRotScaleTransImage` — BEV rot ±0.3925 rad, scale 0.95–1.05 (camera matrices updated inversely)
5. `PadMultiViewImage` — pad to /32
6. `NormalizeMultiviewImage` — mean/std in **BGR order**, `to_rgb=False`
7. `StreamPETRLoadAnnotations2D` — project the (augmented) 3D boxes into each camera → 2D boxes / centers / depths for the aux head
8. `ObjectRangeFilter` — drop 3D boxes outside ±51.2 m
9. `PETRFormatBundle3D` — pack tensors + `img_metas`

The dataset additionally **shuffles camera order** every frame (train only)
and forwards `traffic_cone_barrier_status` into `img_metas`.

### 6.3 Forward and losses

`Petr3D.forward_train` per frame:

1. backbone+neck → stride-16 multiview features
2. `img_roi_head` (**FocalHead**) → dense per-token 2D predictions → 5 aux
   losses (QualityFocal cls ×2, L1 bbox ×5, GIoU ×2, L1 centers2d ×10,
   Gaussian centerness ×1) — training only, shapes the image features
3. `pts_bbox_head` (**StreamPETRHead**) → propagates memory, runs the
   6-layer decoder (self-attn fp32, **flash-attention** cross-attn fp16),
   emits per-layer class scores + boxes
4. losses: focal cls ×2 + L1 bbox ×0.25 per decoder layer, plus denoising
   (DN) query losses

**Partial-ignore** (the whole point of the fine-tune config): when a frame's
`traffic_cone_barrier_status` is `False` (its scene has no cone/barrier
annotation), the classification `label_weights` become per-class and the
cone/barrier columns are zeroed for **negative** (unmatched) queries — the
model is not punished for predicting cones/barriers there. Applied in
`StreamPETRHead._get_target_single`, `dn_loss_single`, and `FocalHead.loss`.

### 6.4 Loop, validation, checkpoints

`EpochBasedTrainLoop` runs 40 epochs (fine-tune config), validating every 5
epochs and **every epoch for the last 5** (`dynamic_intervals=[(35, 1)]`).
Validation decodes predictions and feeds them to **T4Metric** which reports
`NuScenes metric/T4Metric/mAP`, NDS, and per-class APs.

`CheckpointHook` writes `work_dirs/<config_name>/epoch_N.pth` every 2 epochs
and keeps the best by `save_best="NuScenes metric/T4Metric/mAP"`. Logs and a
dump of the resolved config land in the same `work_dirs/<config_name>/`
directory.

## 7. Checkpoint format (matters for migration)

An mm checkpoint is `{"state_dict": {...}, "meta": {...}, "optimizer": ...,
"param_schedulers": ..., "message_hub": ...}`. Keys are mm module paths:
`img_backbone.*` (VoVNet), `img_neck.*` (CPFPN), `img_roi_head.*`
(FocalHead), `pts_bbox_head.*` (StreamPETRHead, with flash-attn packed
`in_proj_weight`). The `meta`/`message_hub` entries pickle mmengine objects —
this is why loading an mm checkpoint normally requires mmengine installed
(the autoware-ml converter works around it; see the other guide).

## 8. Cheat sheet

| I want to… | Command / place |
|---|---|
| Train base model | `python tools/detection3d/train.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py` |
| Fine-tune cone/barrier | `python tools/detection3d/train.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py` |
| Evaluate a checkpoint | `python tools/detection3d/test.py <config> <ckpt>` |
| See training outputs | `work_dirs/<config_name>/` |
| Change the recipe | edit the run config (child) — never the `_base_` files |
| Find the model code | `projects/StreamPETR/stream_petr/models/` |
| Find dataset/sampler code | `projects/StreamPETR/stream_petr/datasets/` |
