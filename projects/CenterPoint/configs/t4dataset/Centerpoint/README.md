# CenterPoint Backbone Variants

This directory contains CenterPoint configs with different backbone architectures, all sharing the same base pipeline (`second_secfpn_4xb16_121m_j6gen2_base_amp.py`).

## Common Parameters

| Parameter | Value |
|-----------|-------|
| `point_cloud_range` | `[-122.40, -122.40, -3.0, 122.40, 122.40, 5.0]` |
| `voxel_size` | `[0.24, 0.24, 8.0]` |
| `grid_size` | `[1020, 1020, 1]` |
| `out_size_factor` | `2` |
| `pts_neck.type` | `SECONDFPN` |
| `pts_neck.upsample_strides` | `[0.5, 1, 2]` |
| `final_output` | `384 x 510 x 510` |

## Backbone Comparison

| | SECOND (base) | BEVResNet34 | BEVVoVNet V-99-eSE |
|---|---|---|---|
| **Config** | `second_secfpn_*.py` | `resnet34_secfpn_*.py` | `vov99_secfpn_*.py` |
| **Batch Size** | 16 | 8 | 8 |
| **Backbone Type** | `SECOND` | `BEVResNet` | `BEVVoVNet` |
| **Stage Outputs (ch)** | 64 / 128 / 256 | 64 / 128 / 256 | 256 / 512 / 768 |
| **Stage Outputs (spatial)** | 510 / 255 / 128 | 1020 / 510 / 255 | 1020 / 510 / 255 |
| **Downsampling** | Conv stride=2 each stage | Stem no-ds, stride (1,2,2) | Stem no-ds, MaxPool at stage3/4 |
| **Neck in_channels** | [64, 128, 256] | [64, 128, 256] | [256, 512, 768] |
| **Activation Checkpointing** | No | Yes | Yes |
| **Param Count** | Smallest | Medium | Largest |

## Architecture Docs

Detailed architecture documentation for each backbone variant:

- [SECOND + SECONDFPN](../../../../../vivid_docs/second_architecture.md)
- [BEVResNet34 + SECONDFPN](../../../../../vivid_docs/resnet34_architecture.md)
- [BEVVoVNet V-99-eSE + SECONDFPN](../../../../../vivid_docs/vov99_architecture.md)

## Common Data Flow

All backbone variants follow the same pipeline:

```text
Point Cloud
  → Voxelization (max_num_points=32)
  → PillarFeatureNet (in=5, out=32)
  → PointPillarsScatter → BEV: 32 x 1020 x 1020
  → Backbone (variant-specific) → multi-scale features
  → SECONDFPN (upsample_strides=[0.5, 1, 2]) → 384 x 510 x 510
  → CenterHead (in_channels=384, out_size_factor=2)
```
