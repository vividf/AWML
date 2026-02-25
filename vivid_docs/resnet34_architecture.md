# BEVResNet34 + SECONDFPN Architecture

## 1. Overview

This document describes the architecture of the CenterPoint backbone variant using **BEVResNet34** as `pts_backbone` and **SECONDFPN** as `pts_neck`.

- **Config**: `resnet34_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Base**: `second_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Input BEV feature**: `(B, 32, 1020, 1020)`

### Key Config Values

```python
pts_backbone = dict(
    type="BEVResNet",
    depth=34,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=(0, 1, 2),
    deep_stem=True,       # Three 3x3 convs instead of one 7x7
    conv1_stride=1,       # No downsampling in stem
    with_pool=False,       # No maxpool after stem
    frozen_stages=-1,
    base_channels=64,
    norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
    norm_eval=False,
    style="pytorch",
    in_channels=32,
    with_cp=True,
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[64, 128, 256],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
)
```

---

## 2. Backbone: BEVResNet34

ResNet34 uses BasicBlock (expansion=1), so `base_channels=64` gives output channels `[64, 128, 256]` for the 3 stages.

### 2.1 Stem (No Downsampling)

The BEV-friendly stem configuration avoids spatial downsampling:

- `deep_stem=True`: Uses three 3x3 convs instead of a single 7x7 conv.
- `conv1_stride=1`: First conv uses stride=1 (no downsampling).
- `with_pool=False`: MaxPool is disabled.

```text
Input: (B, 32, 1020, 1020)
  → Conv2d(32→32, k=3, s=1, p=1) + BN + ReLU
  → Conv2d(32→32, k=3, s=1, p=1) + BN + ReLU
  → Conv2d(32→64, k=3, s=1, p=1) + BN + ReLU
  → (No MaxPool)
Output: (B, 64, 1020, 1020)
```

### 2.2 Layer1 / Stage0 (stride=1)

- **3 BasicBlocks** (ResNet34 layer configuration: [3, 4, 6, 6], using first 3 stages)
- First block: stride=1, identity shortcut
- Output: `(B, 64, 1020, 1020)` — no spatial change

```text
BasicBlock × 3:
  Conv2d(64→64, k=3, s=1, p=1) + BN + ReLU
  Conv2d(64→64, k=3, s=1, p=1) + BN
  + shortcut (identity)
  ReLU
```

### 2.3 Layer2 / Stage1 (stride=2)

- **4 BasicBlocks**
- First block: stride=2 with downsample shortcut (1x1 conv + BN)
- Remaining blocks: stride=1, identity shortcut
- Input: `(B, 64, 1020, 1020)` → Output: `(B, 128, 510, 510)`

```text
BasicBlock 1 (downsample):
  Conv2d(64→128, k=3, s=2, p=1) + BN + ReLU
  Conv2d(128→128, k=3, s=1, p=1) + BN
  + shortcut: Conv2d(64→128, k=1, s=2) + BN
  ReLU

BasicBlock 2-4:
  Conv2d(128→128, k=3, s=1, p=1) + BN + ReLU
  Conv2d(128→128, k=3, s=1, p=1) + BN
  + shortcut (identity)
  ReLU
```

### 2.4 Layer3 / Stage2 (stride=2)

- **6 BasicBlocks**
- First block: stride=2 with downsample shortcut
- Remaining blocks: stride=1, identity shortcut
- Input: `(B, 128, 510, 510)` → Output: `(B, 256, 255, 255)`

### 2.5 Backbone Summary

```text
Input: (B, 32, 1020, 1020)

BEVResNet34:
    Stem (deep_stem, s=1, no pool) → (B, 64, 1020, 1020)
    Layer1 (3 BasicBlock, s=1)     → (B, 64, 1020, 1020)   ← out_indices[0]
    Layer2 (4 BasicBlock, s=2)     → (B, 128, 510, 510)    ← out_indices[1]
    Layer3 (6 BasicBlock, s=2)     → (B, 256, 255, 255)    ← out_indices[2]
```

---

## 3. Neck: SECONDFPN

### Configuration

```python
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[64, 128, 256],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
    norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)
```

### Branch Operations

| Branch | Input | upsample_stride | Operation | Output |
|--------|-------|-----------------|-----------|--------|
| 0 (layer1) | `64 x 1020 x 1020` | 0.5 | Conv2d(64→128, k=2, s=2) | `128 x 510 x 510` |
| 1 (layer2) | `128 x 510 x 510` | 1 | Conv2d(128→128, k=1, s=1) | `128 x 510 x 510` |
| 2 (layer3) | `256 x 255 x 255` | 2 | ConvTranspose2d(256→128, k=2, s=2) | `128 x 510 x 510` |

### Concatenation

```text
up0 = (B, 128, 510, 510)   ← downsampled from 1020
up1 = (B, 128, 510, 510)   ← identity scale
up2 = (B, 128, 510, 510)   ← upsampled from 255

out = torch.cat([up0, up1, up2], dim=1) → (B, 384, 510, 510)
```

---

## 4. Full Architecture Diagram

```text
Point Cloud
  → Voxelization (max_num_points=32)
  → PillarFeatureNet (in=5, out=32)
  → PointPillarsScatter
  → BEV Feature: 32 x 1020 x 1020

  → BEVResNet34 Stem (deep_stem, conv1_stride=1, no maxpool)
     → 64 x 1020 x 1020
  → Layer1 (BasicBlock × 3, stride=1)
     → 64 x 1020 x 1020
  → Layer2 (BasicBlock × 4, stride=2)
     → 128 x 510 x 510
  → Layer3 (BasicBlock × 6, stride=2)
     → 256 x 255 x 255

  → SECONDFPN
     branch 0: 64 x 1020  → Conv(s=2)   → 128 x 510
     branch 1: 128 x 510  → Conv(s=1)   → 128 x 510
     branch 2: 256 x 255  → Deconv(s=2) → 128 x 510
     concat → 384 x 510 x 510

  → CenterHead (in_channels=384, out_size_factor=2)
```

---

## 5. Key Design Decisions

### Why deep_stem with no downsampling?

Standard ResNet uses a 7x7 conv (stride=2) + MaxPool (stride=2) for 4x downsampling at the stem. For BEV features from PointPillarsScatter, the spatial resolution is already determined by the voxel grid (1020x1020), and aggressive early downsampling would lose spatial precision. The BEV-friendly configuration:
- `deep_stem=True`: Replaces 7x7 conv with three 3x3 convs (more efficient, better boundary handling)
- `conv1_stride=1`: Preserves resolution through stem
- `with_pool=False`: No MaxPool

### Why eps=1e-5 in BatchNorm?

Changed from the typical `eps=1e-3` to `eps=1e-5` for better numerical stability when training with mixed precision (AMP with float16). Smaller epsilon prevents division-by-near-zero in BN's normalization.

### Why activation_checkpointing?

BEVResNet34 operates on high-resolution BEV features (1020x1020 in layer1), consuming significantly more GPU memory than SECOND backbone. Activation checkpointing on `pts_backbone` trades recomputation for memory, enabling training with reasonable batch sizes.

---

## 6. Comparison with SECOND and BEVVoVNet

| | SECOND | BEVResNet34 | BEVVoVNet V-99-eSE |
|---|---|---|---|
| **Stage outputs (ch)** | 64 / 128 / 256 | 64 / 128 / 256 | 256 / 512 / 768 |
| **Stage outputs (spatial)** | 510 / 255 / 128 | 1020 / 510 / 255 | 1020 / 510 / 255 |
| **Downsampling strategy** | Conv stride=2 at every stage | No-ds stem, stride (1,2,2) | No-ds stem, MaxPool at stage3/4 |
| **Block type** | Plain Conv+BN+ReLU | BasicBlock (residual) | OSA (dense aggregation) + eSE |
| **Neck in_channels** | [64, 128, 256] | [64, 128, 256] | [256, 512, 768] |
| **Capacity** | Low | Medium | High |
| **Key advantage** | Fast, simple | Residual learning, pretrained weights | High capacity, multi-scale aggregation |
