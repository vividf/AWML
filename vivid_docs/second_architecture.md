# SECOND + SECONDFPN Architecture

## 1. Overview

This document describes the architecture of the default CenterPoint backbone using **SECOND** as `pts_backbone` and **SECONDFPN** as `pts_neck`.

- **Config**: `second_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Input BEV feature**: `(B, 32, 1020, 1020)`

### Key Config Values

```python
pts_backbone = dict(
    type="SECOND",
    in_channels=32,
    out_channels=[64, 128, 256],
    layer_nums=[3, 5, 5],
    layer_strides=[1, 2, 2],
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    conv_cfg=dict(type="Conv2d", bias=False),
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[64, 128, 256],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
    norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)
```

Each SECOND stage consists of:
- One 3x3 Conv (stride from `layer_strides`) for channel expansion / downsampling
- Multiple 3x3 Conv (stride=1) for feature extraction
- BN + ReLU after every conv

---

## 2. Backbone: SECOND

### 2.1 Stage 0 (64ch, stride=2)

- **layer_nums[0] = 3** total conv layers (1 strided + 2 identity)
- First Conv: `Conv2d(32 → 64, k=3, s=2, p=1)` — spatial: `1020 → 510`
- Remaining: `Conv2d(64 → 64, k=3, s=1, p=1)` × 2

| Step | Operation | Output |
|------|-----------|--------|
| Conv (strided) | `Conv2d(32→64, k=3, s=2, p=1) + BN + ReLU` | `(B, 64, 510, 510)` |
| Conv × 2 | `Conv2d(64→64, k=3, s=1, p=1) + BN + ReLU` | `(B, 64, 510, 510)` |

**Output**: `x0 = (B, 64, 510, 510)`

### 2.2 Stage 1 (128ch, stride=2)

- **layer_nums[1] = 5** total conv layers (1 strided + 4 identity)
- First Conv: `Conv2d(64 → 128, k=3, s=2, p=1)` — spatial: `510 → 255`
- Remaining: `Conv2d(128 → 128, k=3, s=1, p=1)` × 4

| Step | Operation | Output |
|------|-----------|--------|
| Conv (strided) | `Conv2d(64→128, k=3, s=2, p=1) + BN + ReLU` | `(B, 128, 255, 255)` |
| Conv × 4 | `Conv2d(128→128, k=3, s=1, p=1) + BN + ReLU` | `(B, 128, 255, 255)` |

**Output**: `x1 = (B, 128, 255, 255)`

### 2.3 Stage 2 (256ch, stride=2)

- **layer_nums[2] = 5** total conv layers (1 strided + 4 identity)
- First Conv: `Conv2d(128 → 256, k=3, s=2, p=1)` — spatial: `255 → 128`
- Remaining: `Conv2d(256 → 256, k=3, s=1, p=1)` × 4

| Step | Operation | Output |
|------|-----------|--------|
| Conv (strided) | `Conv2d(128→256, k=3, s=2, p=1) + BN + ReLU` | `(B, 256, 128, 128)` |
| Conv × 4 | `Conv2d(256→256, k=3, s=1, p=1) + BN + ReLU` | `(B, 256, 128, 128)` |

**Output**: `x2 = (B, 256, 128, 128)`

### 2.4 Backbone Summary

```text
Input: (B, 32, 1020, 1020)

SECOND Backbone:
    Stage0 (3 conv, s=2) → (B, 64, 510, 510)
    Stage1 (5 conv, s=2) → (B, 128, 255, 255)
    Stage2 (5 conv, s=2) → (B, 256, 128, 128)
```

---

## 3. Neck: SECONDFPN

SECONDFPN aligns all multi-scale features to a **common resolution** (the middle scale, `x1` at 255x255), then concatenates them.

### 3.1 Branch Operations

| Branch | Input | upsample_stride | Operation | Output |
|--------|-------|-----------------|-----------|--------|
| 0 (stage0) | `64 x 510 x 510` | 0.5 | Conv2d(64→128, k=2, s=2) | `128 x 255 x 255` |
| 1 (stage1) | `128 x 255 x 255` | 1 | Conv2d(128→128, k=1, s=1) | `128 x 255 x 255` |
| 2 (stage2) | `256 x 128 x 128` | 2 | ConvTranspose2d(256→128, k=2, s=2) | `128 x 255 x 255` |

### 3.2 How stride=0.5 Works (Branch 0)

When `upsample_stride < 1`, SECONDFPN converts it to a **downsample** convolution:

```python
stride = round(1 / 0.5) = 2
# Uses Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
```

This downsamples `x0` from 510x510 to 255x255 to match the target resolution.

### 3.3 How stride=2 Works (Branch 2)

Uses `ConvTranspose2d` to upsample:

```python
# ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# H_out = (H_in - 1) * stride - 2*padding + kernel + output_padding
# H_out = (128 - 1) * 2 - 0 + 2 + 1 = 255
```

This upsamples `x2` from 128x128 to 255x255.

### 3.4 Concatenation

All branches are now aligned to 255x255:

```text
up0 = (B, 128, 255, 255)   ← downsampled from 510
up1 = (B, 128, 255, 255)   ← identity scale
up2 = (B, 128, 255, 255)   ← upsampled from 128

out = torch.cat([up0, up1, up2], dim=1) → (B, 384, 255, 255)
```

---

## 4. Full Architecture Diagram

```text
Point Cloud
  → Voxelization (max_num_points=32)
  → PillarFeatureNet (in=5, out=32)
  → PointPillarsScatter
  → BEV Feature: 32 x 1020 x 1020

  → SECOND Backbone
     Stage0 (3 conv, stride=2) → 64 x 510 x 510
     Stage1 (5 conv, stride=2) → 128 x 255 x 255
     Stage2 (5 conv, stride=2) → 256 x 128 x 128

  → SECONDFPN
     branch 0: 64 x 510  → Conv(s=2)    → 128 x 255   (downsample)
     branch 1: 128 x 255 → Conv(s=1)    → 128 x 255   (keep)
     branch 2: 256 x 128 → Deconv(s=2)  → 128 x 255   (upsample)
     concat → 384 x 255 x 255

  → CenterHead (in_channels=384, out_size_factor=2)
```

---

## 5. Key Insight: Alignment to Middle Resolution

SECONDFPN does **not** upsample everything to the largest feature map. Instead, it aligns all branches to the **middle resolution** (`x1`):

- `stride=0.5` → downsample the high-res map (`x0`)
- `stride=1` → keep the mid-res map (`x1`)
- `stride=2` → upsample the low-res map (`x2`)

This is why `torch.cat()` works without shape errors — all three branches produce `(B, 128, 255, 255)` before concatenation.

---

## 6. Comparison with BEVResNet34 and BEVVoVNet

| | SECOND | BEVResNet34 | BEVVoVNet V-99-eSE |
|---|---|---|---|
| **Stage outputs (ch)** | 64 / 128 / 256 | 64 / 128 / 256 | 256 / 512 / 768 |
| **Stage outputs (spatial)** | 510 / 255 / 128 | 1020 / 510 / 255 | 1020 / 510 / 255 |
| **Downsampling** | Conv stride=2 at every stage | No-ds stem, stride (1,2,2) | No-ds stem, MaxPool at stage3/4 |
| **FPN alignment target** | 255 x 255 | 510 x 510 | 510 x 510 |
| **Final neck output** | 384 x 255 x 255 | 384 x 510 x 510 | 384 x 510 x 510 |
| **Block type** | Plain Conv+BN+ReLU | BasicBlock (residual) | OSA (dense) + eSE |
| **Capacity** | Low | Medium | High |

Note that SECOND aligns to a smaller resolution (255x255) because all three backbone stages use stride=2, producing spatial sizes at 510/255/128. The BEV-friendly variants (ResNet34, VoVNet) preserve full resolution at the first stage (1020), resulting in a higher-resolution neck output (510x510).
