# BEVVoVNet V-99-eSE + SECONDFPN Architecture

## 1. Overview

This document describes the architecture of the CenterPoint backbone variants using **BEVVoVNet V-99-eSE** or **V-57-eSE** as `pts_backbone` and **SECONDFPN** as `pts_neck`.

- **Config**: `vov99_secfpn_4xb16_121m_j6gen2_base_amp.py`（V-57：`vov57_secfpn_4xb16_121m_j6gen2_base_amp.py`）
- **Base**: `second_secfpn_4xb16_121m_j6gen2_base_amp.py`
- **Input BEV feature**: `(B, 32, 1020, 1020)`

### What does "99" mean?

The number **99** in VoVNet (V-99-eSE) comes from the **original VoVNet/OSANet naming**: it indicates the **network depth** (number of layers). VoVNet variants are named by approximate layer count:

- **V-19**: ~19 layers (lightweight)
- **V-39**: ~39 layers
- **V-57**: ~57 layers
- **V-99**: ~99 layers (deepest standard variant)

So "VoV99" = VoVNet with depth ~99. The same spec uses **OSA (One-Shot Aggregation)** blocks: each block has 5 conv layers, and the count 1+3+9+3 blocks × 5 layers plus stem gives a deep network suited for higher accuracy. The "eSE" suffix means **eSE (effective Squeeze-Excitation)** attention is used in the OSA modules.

### Key Config Values

```python
pts_backbone = dict(
    type="BEVVoVNet",
    spec_name="V-99-eSE",
    input_ch=32,
    stem_strides=(1, 1, 1),       # No downsampling in stem
    out_features=("stage3", "stage4", "stage5"),   # last 3 stages → SECONDFPN (stage5 is used)
    frozen_stages=-1,
    norm_eval=False,
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[512, 768, 1024],
    out_channels=[128, 128, 128],
    upsample_strides=[1, 2, 4],   # 510→510, 255→510, 128→510
)
```

### V-99-eSE Specification

| Parameter | Value |
|-----------|-------|
| Stem channels | [64, 64, 128] |
| Stage conv channels | [128, 160, 192, 224] |
| Stage output channels | [256, 512, 768, 1024] |
| Layers per OSA block | 5 |
| Blocks per stage | [1, 3, 9, 3] |
| eSE (Squeeze-Excitation) | Yes |
| Depthwise separable | No |

### Why are the OSA block counts 1, 3, 9, 3?

The pattern **1, 3, 9, 3** (for stages 2, 3, 4, 5) comes from the **original VoVNet paper** (CVPRW 2019, Lee et al.). The design follows a common backbone principle: **spend more depth where spatial size is smaller** (so compute per layer is lower) and **keep the first stage light** (highest resolution is the most expensive).

| Stage | Blocks | Spatial (BEV) | Rationale |
|-------|--------|----------------|-----------|
| **Stage 2** | **1** | 1020×1020 | Highest resolution; one OSA block is enough to build low-level features without blowing up compute. |
| **Stage 3** | **3** | 510×510 | After 2× downsampling; a few more blocks add capacity at moderate cost. |
| **Stage 4** | **9** | 255×255 | Heaviest stage. Resolution is ¼ of stage 2, so each conv is much cheaper; more blocks here give strong semantic features for detection. |
| **Stage 5** | **3** | 128×128 | Used: output to SECONDFPN (last 3 stages = stage3, 4, 5). |

So: **light at high resolution (1) → ramp up in the middle (3, 9) → taper at the end (3)**. The exact numbers 1-3-9-3 were chosen empirically in the paper to get a good accuracy/speed trade-off and the desired depth; lighter variants (e.g. V-39, V-57) use fewer blocks per stage (e.g. [1,1,2,2] or [1,1,4,3]).

**來源說明**  
- **1, 3, 9, 3 對應 V-99**：出自原始 VoVNet 論文 *An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection* (CVPRW 2019, Lee et al., [arxiv 1904.09730](https://arxiv.org/abs/1904.09730)) 中的設計；本 repo 實作見 `projects/CenterPoint/models/backbones/vovnet.py` 的 `VoVNet99_eSE["block_per_stage"]`（改寫自 StreamPETR 的 VoVNet）。  
- **V-39 用 [1,1,2,2]、V-57 用 [1,1,4,3]**：同一論文的輕量變體配置，在本 repo 的 **StreamPETR** 中有完整定義：`projects/StreamPETR/stream_petr/models/backbones/vovnet.py` 裡的 `VoVNet39_eSE`（`block_per_stage`: [1, 1, 2, 2]）與 `VoVNet57_eSE`（`block_per_stage`: [1, 1, 4, 3]）。CenterPoint 的 backbone 已實作 **V-99-eSE** 與 **V-57-eSE**（`projects/CenterPoint/models/backbones/vovnet.py` 的 `_STAGE_SPECS`）；**預設 V-99 使用 stage3 / stage4 / stage5 接 SECONDFPN**（不忽略 stage5）；V-57 同樣使用後三階，見 §1.1。

> **Note**: Default V-99 uses `out_features=("stage3", "stage4", "stage5")` so stage5 (1024ch, 3 blocks) is used. Stage2 (1020×1020) is not fed to the neck, reducing high-res compute.

### 1.1 BEVVoVNet V-57-eSE: Last Three Stages (Lighter Config)

A lighter variant uses **V-57-eSE** with **only the last three stages** (stage3, stage4, stage5) instead of the first three (stage2, stage3, stage4). This drops the highest-resolution stage2 (1020×1020), reducing compute and memory while keeping the same neck output layout (384ch @ 510×510).

| Item | V-99 (last 3 stages) | V-57 (last 3 stages) |
|------|------------------------|------------------------|
| **Backbone spec** | V-99-eSE | V-57-eSE |
| **block_per_stage** | [1, 3, 9, 3] | [1, 1, 4, 3] |
| **out_features** | `("stage3", "stage4", "stage5")` | `("stage3", "stage4", "stage5")` |
| **Neck in_channels** | [512, 768, 1024] | [512, 768, 1024] |
| **Neck upsample_strides** | [1, 2, 4] | [1, 2, 4] |
| **Spatial sizes** | 510, 255, 128 | 510, 255, 128 |

**Config files** (under `projects/CenterPoint/configs/t4dataset/Centerpoint/`):

- **Base (AMP)**: `vov57_secfpn_4xb16_121m_j6gen2_base_amp.py` — base training config with V-57-eSE and last 3 stages.
- **T4Metric V2**: `vov57_secfpn_4xb16_121m_j6gen2_base_amp_t4metric_v2.py` — inherits base, uses T4MetricV2 evaluator.
- **RFS + BF16**: `vov57_secfpn_4xb16_121m_j6gen2_base_amp_rfs_bf16.py` — inherits base, adds repeat-frame sampling and BF16.

**When to use**: Prefer V-57 last-3-stages when you want lower latency and can trade some accuracy; the backbone has fewer OSA blocks (1+1+4+3) and no 1020×1020 branch.

### 1.2 若改用 StreamPETR 的 stem downsampling（4×）

目前 BEVVoVNet 使用 `stem_strides=(1, 1, 1)`，不做下採樣。若改成和 StreamPETR 一樣的 **`stem_strides=(2, 1, 2)`**（4× 下採樣），整條 backbone 的空間尺寸會變成：

| 位置 | 現在 (1,1,1) | 改用 (2,1,2) 後 |
|------|----------------|-------------------|
| **Stem 輸入** | 1020×1020 | 1020×1020 |
| **Stem 輸出** | 1020×1020 | **255×255**（1020÷2÷2） |
| **Stage2** | 1020×1020 | **255×255**（無 MaxPool） |
| **Stage3** | 510×510 | **128×128**（255÷2） |
| **Stage4** | 255×255 | **64×64**（128÷2） |
| **Stage5** | 128×128 | **32×32**（64÷2） |

也就是：**一開始就 4× 縮小，後面每個 stage 再各 2×，整體更小、更快，但高解析度細節會少**。

若仍取 **out_features=("stage2", "stage3", "stage4")**，則給 neck 的是：

- stage2: **256ch @ 255×255**
- stage3: **512ch @ 128×128**
- stage4: **768ch @ 64×64**

SECONDFPN 若要把三支對齊到同一個 grid（例如 510×510 以配合現有 head），可設：

- **in_channels** = [256, 512, 768]（不變）
- **upsample_strides** = **[2, 4, 8]**（255→510、128→510、64→510）

```python
# 概念配置（尚未實作）
pts_backbone = dict(
    type="BEVVoVNet",
    spec_name="V-99-eSE",
    input_ch=32,
    stem_strides=(2, 1, 2),   # 與 StreamPETR 相同，4× downsampling
    out_features=("stage2", "stage3", "stage4"),
    ...
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 768],
    out_channels=[128, 128, 128],
    upsample_strides=[2, 4, 8],  # 對應 255→510, 128→510, 64→510
    ...
)
```

**小結**：加上 StreamPETR 的 stem downsampling 後，stem 出 255×255，stage2/3/4 分別是 255、128、64，neck 用 [2, 4, 8] 上採樣到 510×510；計算量與顯存會下降，但 BEV 高解析度資訊在 stem 就丟失，適合更在意速度的設定。

### 1.3 折衷：stem_strides=(1, 1, 2) — 對齊 SECOND 空間 510 / 255 / 128

若只做 **2× 下採樣**（最後一層 stem 用 stride=2），即 **`stem_strides=(1, 1, 2)`**：

| 位置 | 空間尺寸 |
|------|----------|
| Stem 輸出 | **510×510**（1020÷2） |
| Stage2 | **510×510**（無 MaxPool） |
| Stage3 | **255×255**（510÷2） |
| Stage4 | **128×128**（255÷2） |

這樣 **stage2 / stage3 / stage4 的空間正好是 510、255、128**，與 SECOND 的 multi-scale 金字塔一致。Neck 只需用 **`upsample_strides=[1, 2, 4]`**（510→510、255→510、128→510），不需 0.5 這種「在 neck 裡再下採樣」的設定，邏輯更直觀，也和常見 510/255/128 的設計對齊。

| 比較 | 目前 V-99 (1,1,1) | stem (1,1,2) |
|------|-------------------|--------------|
| Stem 後 | 1020×1020 | **510×510** |
| Stage2/3/4 空間 | 1020, 510, 255 | **510, 255, 128** |
| Neck upsample_strides | [0.5, 1, 2] | **[1, 2, 4]** |
| 與 SECOND 空間對齊 | 否 | **是** |

**概念配置**（僅說明，未實作）：

```python
pts_backbone = dict(
    ...
    stem_strides=(1, 1, 2),   # 僅 2×，對齊 SECOND 510/255/128
    out_features=("stage2", "stage3", "stage4"),
    ...
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 768],
    out_channels=[128, 128, 128],
    upsample_strides=[1, 2, 4],   # 510→510, 255→510, 128→510
    ...
)
```

**小結**：`(1, 1, 2)` 比 `(1,1,1)` 省約一半 stage2 計算、又比 `(2,1,2)` 保留更多解析度；且 510/255/128 與 SECOND 一致，neck 用 [1, 2, 4] 即可，是較折衷的選項。

---

## 2. Backbone: BEVVoVNet

### 2.1 Stem (No Downsampling)

Unlike the original VoVNet which uses `stem_strides=(2, 1, 2)` for 4x downsampling, BEVVoVNet uses `stem_strides=(1, 1, 1)` to preserve full BEV spatial resolution.

| Layer | Operation | Output |
|-------|-----------|--------|
| stem_1 | Conv2d(32 → 64, k=3, s=1, p=1) + BN + ReLU | `(B, 64, 1020, 1020)` |
| stem_2 | Conv2d(64 → 64, k=3, s=1, p=1) + BN + ReLU | `(B, 64, 1020, 1020)` |
| stem_3 | Conv2d(64 → 128, k=3, s=1, p=1) + BN + ReLU | `(B, 128, 1020, 1020)` |

### 2.2 Stage2 (256ch, no MaxPool)

- **No MaxPool** because `stage_num == 2` (first stage after stem).
- **1 OSA block**: 5 conv layers + concat + 1x1 conv + eSE attention.
- Input: `(B, 128, 1020, 1020)` → Output: `(B, 256, 1020, 1020)`

**OSA Block Structure**:
```text
input (128ch)
  ├─ Conv3x3 → 128ch  (layer 0)
  ├─ Conv3x3 → 128ch  (layer 1)
  ├─ Conv3x3 → 128ch  (layer 2)
  ├─ Conv3x3 → 128ch  (layer 3)
  └─ Conv3x3 → 128ch  (layer 4)
Concat([input, layer0..4]) → (128 + 5×128 = 768ch)
Conv1x1(768 → 256) + eSE → 256ch
```

### 2.3 Stage3 (512ch, MaxPool ↓2)

- **MaxPool2d(k=3, s=2, ceil_mode=True)**: `1020 → 510`
- **3 OSA blocks** (block_per_stage[1] = 3):
  - First block: stage_conv_ch=160, no identity shortcut, no eSE
  - Middle block: identity shortcut, no eSE
  - Last block: identity shortcut + eSE attention
- Input: `(B, 256, 1020, 1020)` → After MaxPool: `(B, 256, 510, 510)` → Output: `(B, 512, 510, 510)`

### 2.4 Stage4 (768ch, MaxPool ↓2)

- **MaxPool2d(k=3, s=2, ceil_mode=True)**: `510 → 255`
- **9 OSA blocks** (block_per_stage[2] = 9) — the heaviest stage.
  - stage_conv_ch=192, output=768ch
  - Last block has eSE; all blocks after the first have identity shortcut.
- Input: `(B, 512, 510, 510)` → After MaxPool: `(B, 512, 255, 255)` → Output: `(B, 768, 255, 255)`

### 2.5 Backbone Summary

```text
Input: (B, 32, 1020, 1020)

BEVVoVNet V-99-eSE:
    Stem (s=1,1,1) → (B, 128, 1020, 1020)
    Stage2 (1 OSA)  → (B, 256, 1020, 1020)   [not used by neck]
    Stage3 (3 OSA)  → (B, 512, 510, 510)     ← out_features
    Stage4 (9 OSA)  → (B, 768, 255, 255)     ← out_features
    Stage5 (3 OSA)  → (B, 1024, 128, 128)    ← out_features
```

---

## 3. Neck: SECONDFPN

### Configuration

Default V-99 feeds **stage3, stage4, stage5** to SECONDFPN (spatial 510, 255, 128):

```python
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[512, 768, 1024],
    out_channels=[128, 128, 128],
    upsample_strides=[1, 2, 4],
    norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)
```

### Branch Operations

| Branch | Input | upsample_stride | Operation | Output |
|--------|-------|-----------------|-----------|--------|
| 0 (stage3) | `512 x 510 x 510` | 1 | Conv2d(512→128, k=1, s=1) | `128 x 510 x 510` |
| 1 (stage4) | `768 x 255 x 255` | 2 | ConvTranspose2d(768→128, k=2, s=2) | `128 x 510 x 510` |
| 2 (stage5) | `1024 x 128 x 128` | 4 | ConvTranspose2d(1024→128, k=2, s=4) | `128 x 512 x 512` → 510 |

### Concatenation

```text
up0 = (B, 128, 510, 510)
up1 = (B, 128, 510, 510)
up2 = (B, 128, 510, 510)

out = torch.cat([up0, up1, up2], dim=1) → (B, 384, 510, 510)
```

### 3.1 upsample_strides: [0.5, 1, 2] vs [1, 2, 4] — 哪個比較好？

兩種設定對應**不同的 backbone 輸出**接到 neck，沒有絕對「哪個比較好」，要看你要的是**精度優先**還是**速度／對齊 SECOND 金字塔**。

| 項目 | **[0.5, 1, 2]**（stage2,3,4） | **[1, 2, 4]**（stage3,4,5，目前預設） |
|------|-------------------------------|----------------------------------------|
| **Backbone 給 neck 的空間** | 1020, 510, 255 | 510, 255, 128 |
| **Neck 各 branch 做什麼** | 1020**↓2**→510、510 不變、255**↑2**→510 | 510 不變、255**↑2**→510、128**↑4**→510 |
| **有沒有 1020 高解析度** | ✅ 有（stage2） | ❌ 沒有 |
| **Neck 裡有沒有下採樣** | ✅ 有（branch0 從 1020 壓到 510） | ❌ 沒有（只有不變或上採樣） |
| **與 SECOND 空間金字塔** | 不一致（1020/510/255） | **一致（510/255/128）** |
| **計算／顯存** | 較重（要算 stage2 @ 1020×1020） | 較輕 |
| **適合** | 要最高空間細節、小目標、可接受較慢 | 要速度、或希望與 510/255/128 設計一致 |

**簡要結論：**

- **選 [1, 2, 4]**（現在預設）：  
  - 用 stage3、4、5，**不忽略 stage5**，且 510/255/128 與 SECOND 一致。  
  - Neck 只做「保持 510」或「上採樣到 510」，語意單純；且不做 1020 的 branch，**較快、較省顯存**。  
  - 代價是**沒有 1020 那一層**，極高解析度的細節會少一點。

- **選 [0.5, 1, 2]**：  
  - 用 stage2、3、4，**帶 1020 高解析度**，對小目標、邊界定位可能較有利。  
  - Neck 要在 branch0 做一次**下採樣**（1020→510），實作上沒問題，但計算與顯存較大。  
  - 若你更在意精度、且能接受較重 backbone，可以改回 `out_features=("stage2","stage3","stage4")` 並配 `in_channels=[256,512,768]`、`upsample_strides=[0.5,1,2]`。

所以：**沒有單一「比較好」**——[1, 2, 4] 較適合速度與 510/255/128 對齊；[0.5, 1, 2] 較適合極致利用 1020 的精度。目前 **V-99 與 V-57** 預設皆用 [1, 2, 4]（stage3,4,5），以與 SECOND 空間一致並控制計算量。

---

## 4. Full Architecture Diagram

```text
Point Cloud
  → Voxelization (max_num_points=32)
  → PillarFeatureNet (in=5, out=32)
  → PointPillarsScatter
  → BEV Feature: 32 x 1020 x 1020

  → BEVVoVNet V-99-eSE Stem (stride=1,1,1, no downsampling)
     → 128 x 1020 x 1020
  → Stage2 (1 OSA)  → 256 x 1020 x 1020  [not used]
  → Stage3 (MaxPool ↓2, 3 OSA blocks)
     → 512 x 510 x 510   ← to SECONDFPN
  → Stage4 (MaxPool ↓2, 9 OSA blocks)
     → 768 x 255 x 255   ← to SECONDFPN
  → Stage5 (MaxPool ↓2, 3 OSA blocks)
     → 1024 x 128 x 128  ← to SECONDFPN

  → SECONDFPN
     branch 0: 512 x 510  → stride 1 → 128 x 510
     branch 1: 768 x 255  → stride 2 → 128 x 510
     branch 2: 1024 x 128 → stride 4 → 128 x 510
     concat → 384 x 510 x 510

  → CenterHead (in_channels=384, out_size_factor=2)
```

---

## 5. Comparison with SECOND and BEVResNet34

| | SECOND | BEVResNet34 | BEVVoVNet V-99-eSE | BEVVoVNet V-57-eSE |
|---|---|---|---|---|
| **Stage outputs (ch)** | 64 / 128 / 256 | 64 / 128 / 256 | 512 / 768 / 1024 (stage3,4,5) | 512 / 768 / 1024 (stage3,4,5) |
| **Stage outputs (spatial)** | 510 / 255 / 128 | 1020 / 510 / 255 | 510 / 255 / 128 | 510 / 255 / 128 |
| **Blocks** | 3+5+5 Conv | 3+4+6 BasicBlock | 3+9+3 OSA (5 layers each) | 3+4+3 OSA (5 layers each) |
| **Neck in_channels** | [64, 128, 256] | [64, 128, 256] | [512, 768, 1024] | [512, 768, 1024] |
| **Key feature** | Simple stacked Conv | Residual connections | Dense aggregation + eSE | Dense aggregation + eSE (lighter) |
| **Batch size** | 16 | 8 | 8 | 8 |
| **Activation Checkpoint** | No | Yes (`pts_backbone`) | Yes (`pts_backbone`) | Yes (`pts_backbone`) |
| **Capacity** | Low | Medium | High | Medium–High |

BEVVoVNet (V-99 and V-57) has more parameters and FLOPs than SECOND/ResNet34 due to:
1. Higher channel counts (512/768/1024 from stage3,4,5 vs 64/128/256)
2. Dense OSA aggregation (each OSA block concatenates all intermediate features)
3. V-99: 9 OSA blocks in stage4 (45 conv layers in stage4 alone). V-57: 4 blocks in stage4 → lighter than V-99.

---

## 7. BEVVoVNet (AWML) vs StreamPETR VoVNet

AWML supports **V-99-eSE** and **V-57-eSE**; both use the same stage_out_ch [256, 512, 768, 1024] and feed **stage3, stage4, stage5** to SECONDFPN (in_channels=[512, 768, 1024]). V-99 has block_per_stage [1, 3, 9, 3]; V-57 has [1, 1, 4, 3]. StreamPETR uses the same VoVNet spec family; differences below are between StreamPETR (image) and AWML (BEV).

| Aspect | StreamPETR VoVNet | AWML BEVVoVNet (V-99 / V-57) |
|--------|-------------------|-------------------------------|
| **Class** | `VoVNet` | `BEVVoVNet` (subclass of VoVNet) |
| **Variants** | V-19/39/57/99-eSE | **V-99-eSE**, **V-57-eSE** (stage3,4,5) |
| **Input** | Camera/image features (e.g. 3ch or feature dim) | BEV pillar features (32ch, 1020×1020) |
| **Stem strides** | Default `(2, 1, 2)` → **4× spatial downsampling** | **`(1, 1, 1)`** → no downsampling |
| **Spatial size after stem** | 4× smaller (e.g. 256×256 for 1024 input) | Same as input (1020×1020) |
| **Purpose** | Image backbone for multi-view 3D (camera) | BEV backbone for LiDAR/point-cloud BEV |
| **Neck** | Projection/FPN for camera stream | SECONDFPN with in_channels=[512, 768, 1024] (stage3,4,5) |

**Summary**: StreamPETR uses VoVNet as designed for **image** backbones (stem downsamples 4×). AWML adds **BEVVoVNet** with configurable stem strides; both **V-99-eSE** and **V-57-eSE** use stage3,4,5 → SECONDFPN. V-57 is lighter (fewer OSA blocks) for speed/accuracy trade-off.

---

## 8. Why Stage Output Channels Differ from SECOND / BEVResNet34

| Backbone | Stage output channels | Neck `in_channels` |
|----------|------------------------|--------------------|
| SECOND | 64 / 128 / 256 | [64, 128, 256] |
| BEVResNet34 | 64 / 128 / 256 | [64, 128, 256] |
| BEVVoVNet V-99-eSE | **512 / 768 / 1024** (stage3,4,5) | **[512, 768, 1024]** |
| BEVVoVNet V-57-eSE | **512 / 768 / 1024** (stage3,4,5) | **[512, 768, 1024]** |

**Reasons:**

1. **Different backbone families**  
   SECOND and BEVResNet34 use a **lightweight** channel schedule: base 64, then 64→128→256 per stage. VoVNet comes from the **original paper** (CVPRW 2019, ETRI/Megvii) and uses a **heavier** schedule: stage_out_ch = [256, 512, 768, 1024]. So 256/512/768 are the **native** VoVNet design for higher capacity, not chosen to match SECOND.

2. **SECONDFPN is backbone-agnostic**  
   The neck only requires that `in_channels` matches the backbone’s multi-scale output channels. It then projects each branch to 128 and concatenates → 384 for the head. So we set `pts_neck.in_channels=[512, 768, 1024]` when using VoV99 or VoV57 (stage3,4,5); no need to change SECOND or ResNet.

3. **Accuracy vs cost**  
   VoV99’s larger channels and more OSA blocks aim for **higher accuracy** at the cost of more parameters and compute; VoV57 is a lighter variant (fewer blocks) for a speed/accuracy trade-off. SECOND/ResNet34’s 64/128/256 are a **smaller, faster** design. The stage output channel difference is therefore an intentional capacity choice per backbone family, not a bug or inconsistency.

---

## 9. Latency Reduction Strategies

If you need to reduce the overall network latency of the VoVNet-based CenterPoint pipeline, consider the following directions:

### 9.1 Backbone Architecture Changes

1. **Use a lighter VoVNet variant**: Use **V-57-eSE** (config: `vov57_secfpn_4xb16_121m_j6gen2_base_amp.py`) or V-39-eSE instead of V-99-eSE. V-57 has fewer OSA blocks (1+1+4+3) and no stage2 in the neck path; stage4 has 4 blocks instead of 9, yielding the largest speedup.

2. **Reduce stage4 depth**: Stage4 dominates compute with 9 OSA blocks at 255x255 resolution. Reducing `block_per_stage[2]` from 9 to 3-5 blocks can cut backbone latency by ~30-40% with moderate accuracy impact.

3. **Add stage2 (high-resolution branch)** (optional): The default V-99 uses only stage3,4,5. If you need more spatial detail, you can set `out_features=("stage2", "stage3", "stage4")` and neck `in_channels=[256, 512, 768]`, `upsample_strides=[0.5, 1, 2]`. The trade-off is higher compute (stage2 at 1020×1020 and neck branch 0).

4. **Enable depthwise separable convolutions**: Set `dw=True` in the VoVNet spec to use depthwise separable conv3x3 in OSA blocks. This reduces parameter count and FLOPs at each layer.

5. **Reduce OSA layers per block**: The `layer_per_block=5` setting means each OSA block has 5 intermediate conv layers. Reducing to 3 decreases both compute and the concat dimension.

### 9.2 Input Resolution / Voxelization

6. **Increase voxel size**: Changing `voxel_size` from `[0.24, 0.24, 8.0]` to `[0.32, 0.32, 8.0]` reduces the BEV grid from 1020x1020 to 765x765. This quadratically reduces compute in the backbone (especially stage2 at full resolution). This is the single most impactful change for latency.

7. **Reduce point cloud range**: Narrowing the detection range (e.g., from 122.4m to 100m) shrinks the grid proportionally.

### 9.3 Deployment Optimizations

8. **INT8 quantization (PTQ/QAT)**: Apply post-training quantization or quantization-aware training for TensorRT deployment. INT8 can provide ~2x speedup over FP16 for conv-heavy backbones. See `deployment/projects/centerpoint/config/` for reference INT8 configs.

9. **TensorRT FP16**: Ensure the model is exported with FP16 precision in TensorRT. The config already uses `AmpOptimWrapper` with `dtype="float16"`, which helps maintain numerical compatibility.

10. **Operator fusion**: TensorRT automatically fuses Conv+BN+ReLU sequences. Ensure BN is fused before export (`fuse_bn=True` in deploy config) for maximum throughput.

### 9.4 Recommended Priority

For the best latency-accuracy trade-off, the recommended approach (from most impactful to least):

| Priority | Strategy | Expected Speedup | Accuracy Impact |
|----------|----------|------------------|-----------------|
| 1 | Increase voxel_size (0.24→0.32) | ~40-50% | Moderate (loss of fine detail) |
| 2 | Reduce stage4 blocks (9→3-5) | ~20-30% | Small to moderate |
| 3 | INT8 quantization | ~2x over FP16 | Small (with proper calibration) |
| 4 | Add stage2 (use high-res branch) | -15~20% slower | Better detail |
| 5 | Use lighter VoVNet variant (e.g. **V-57-eSE** instead of V-99) | ~30-50% | Depends on variant |

### 9.5 Notes

- When changing voxel_size or point_cloud_range, remember to update `grid_size`, `out_size_factor`, and all dependent configs (neck upsample_strides, head, train_cfg, test_cfg).
- Reducing backbone capacity may require tuning the learning rate schedule, batch size, and training epochs to maintain convergence.
- For deployment, always profile on the target hardware (e.g., NVIDIA Orin) since inference characteristics differ from training GPUs.
