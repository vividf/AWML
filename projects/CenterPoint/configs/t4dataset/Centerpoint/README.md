# CenterPoint ResNet34 + SECFPN Architecture Notes

This README explains the model structure used by:

- `resnet34_secfpn_4xb16_121m_base_amp.py`

It focuses on:

- End-to-end feature flow
- Where downsampling happens
- Feature map sizes and channels

## Key Config Values

- `point_cloud_range = [-122.40, -122.40, -3.0, 122.40, 122.40, 5.0]`
- `voxel_size = [0.24, 0.24, 8.0]`
- `grid_size = [1020, 1020, 1]`
- `out_size_factor = 2`
- `pts_backbone.type = BEVResNet`
- `pts_backbone.strides = (1, 2, 2)`
- `pts_backbone.deep_stem = True`
- `pts_backbone.conv1_stride = 1`
- `pts_backbone.with_pool = False`
- `pts_neck.type = SECONDFPN`
- `pts_neck.upsample_strides = [0.5, 1, 2]`

## Architecture Diagram (README text format)

```text
Point Cloud
  -> Voxelization (max_num_points=32)
  -> PillarFeatureNet (in=4, out=32)
  -> PointPillarsScatter
  -> BEV Feature: 32 x 1020 x 1020

  -> BEVResNet Stem (deep_stem, conv1_stride=1, no maxpool)
     -> Stage0 / layer1 (BasicBlock x3, first stride=1)
        -> 64 x 1020 x 1020
     -> Stage1 / layer2 (BasicBlock x4, first stride=2)
        -> 128 x 510 x 510
     -> Stage2 / layer3 (BasicBlock x6, first stride=2)
        -> 256 x 255 x 255

  -> SECONDFPN
     branch from layer1: 64 x 1020 x 1020 --upsample_stride=0.5--> 128 x 510 x 510
     branch from layer2: 128 x 510 x 510 --upsample_stride=1----> 128 x 510 x 510
     branch from layer3: 256 x 255 x 255 --upsample_stride=2----> 128 x 510 x 510
     concat -> 384 x 510 x 510

  -> CenterHead (in_channels=384, out_size_factor=2)
```

## ResNet Block-Level View (README text format)

```text
Input BEV: 32 x 1020 x 1020
  -> Stem: no downsample

layer1 (3 blocks, stride setting = 1)
  B1: stride=1, shortcut=identity
  B2: stride=1
  B3: stride=1
  Output: 64 x 1020 x 1020

layer2 (4 blocks, stride setting = 2)
  B1: stride=2, shortcut downsample: 1x1 conv(s=2) + BN
  B2: stride=1
  B3: stride=1
  B4: stride=1
  Output: 128 x 510 x 510

layer3 (6 blocks, stride setting = 2)
  B1: stride=2, shortcut downsample: 1x1 conv(s=2) + BN
  B2: stride=1
  B3: stride=1
  B4: stride=1
  B5: stride=1
  B6: stride=1
  Output: 256 x 255 x 255
```

## Where Downsampling Happens

### 1) Stem: no downsampling

In this config, BEVResNet stem is intentionally BEV-friendly:

- `deep_stem=True`
- `conv1_stride=1`
- `with_pool=False` (maxpool disabled)

So the stem keeps the input BEV resolution (`1020 x 1020`).

### 2) Backbone stages: downsampling only at stage boundaries

`strides=(1, 2, 2)` means:

- `layer1` first block uses stride 1 -> no spatial downsampling
- `layer2` first block uses stride 2 -> downsample by 2
- `layer3` first block uses stride 2 -> downsample by 2 again

Inside the same stage, remaining blocks use `stride=1`.

### 3) SECONDFPN: resize to a unified BEV scale

SECONDFPN receives:

- `64 x 1020 x 1020`
- `128 x 510 x 510`
- `256 x 255 x 255`

Then converts each to `128 x 510 x 510` using:

- stride `0.5` (downsample)
- stride `1` (identity scale)
- stride `2` (upsample)

Finally concatenates them to:

- `384 x 510 x 510`

which feeds `CenterHead`.

## Quick Summary

- Stem does not downsample.
- Downsampling is performed by the first block of `layer2` and `layer3`.
- SECONDFPN aligns multi-scale features to `510 x 510` and concatenates them.
