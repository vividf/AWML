# CenterPoint pts_backbone 從 BEVResNet 遷移至 VoVNet V-99-eSE 技術文件

## 1. 目標

將 `vov99_secfpn_4xb16_121m_j6gen2_base_amp.py` 中的 `pts_backbone` 從 `BEVResNet`（ResNet34）替換為 `VoVNet V-99-eSE`，用於 BEV（鳥瞰圖）點雲特徵提取。

---

## 2. 現有架構分析

### 2.1 資料流

```
LiDAR 點雲
  → PillarFeatureNet (pts_voxel_encoder)
    → 32 通道的 pillar 特徵
  → PointPillarsScatter (pts_middle_encoder)
    → 32ch @ 1020×1020 的 BEV 特徵圖
  → pts_backbone (BEVResNet / SECOND)
    → 多尺度特徵 [64ch@1020, 128ch@510, 256ch@255]
  → pts_neck (SECONDFPN)
    → 384ch @ 510×510（3 組 128ch 特徵串接）
  → CenterHead (pts_bbox_head)
    → 偵測結果
```

### 2.2 BEVResNet（現有 backbone）輸出

BEVResNet 的關鍵設計：stem 不做降採樣（`conv1_stride=1`, `with_pool=False`），
保留 BEV 特徵圖的空間解析度。

| 層級    | 輸出通道 | 空間大小    | stride |
|---------|----------|-------------|--------|
| Stage 0 | 64       | 1020×1020   | 1      |
| Stage 1 | 128      | 510×510     | 2      |
| Stage 2 | 256      | 255×255     | 2      |

### 2.3 SECONDFPN（現有 neck）處理方式

```
Stage 0: 64ch  @ 1020×1020 → upsample_stride=0.5 (Conv stride=2 降採樣) → 128ch @ 510×510
Stage 1: 128ch @ 510×510   → upsample_stride=1   (Conv stride=1 保持)   → 128ch @ 510×510
Stage 2: 256ch @ 255×255   → upsample_stride=2   (Deconv stride=2 上採樣) → 128ch @ 510×510
                                                                    串接 → 384ch @ 510×510
```

最終輸出 510×510 = grid_size(1020) / out_size_factor(2)，供 CenterHead 使用。

---

## 3. VoVNet V-99-eSE 架構特性

### 3.1 網路規格

```python
VoVNet99_eSE = {
    "stem": [64, 64, 128],           # stem 三層的輸出通道
    "stage_conv_ch": [128, 160, 192, 224],  # 每個 stage 的中間通道
    "stage_out_ch": [256, 512, 768, 1024],  # 每個 stage 的輸出通道
    "layer_per_block": 5,             # 每個 OSA block 內的卷積層數
    "block_per_stage": [1, 3, 9, 3],  # 每個 stage 內的 OSA block 數量
    "eSE": True,                      # 使用 eSE (effective Squeeze-Excitation) 模組
}
```

### 3.2 原始 VoVNet 的降採樣行為

**Stem（共 4× 降採樣）：**
- Conv1: stride=2（1020 → 510）
- Conv2: stride=1（510 → 510）
- Conv3: stride=2（510 → 255）

**Stages（每個 stage 除了 stage2 外，都有 MaxPool stride=2）：**

| 層級    | 輸出通道 | 空間大小    | 降採樣方式             |
|---------|----------|-------------|----------------------|
| Stem    | 128      | 255×255     | Conv stride 2,1,2    |
| Stage 2 | 256      | 255×255     | 無（不加 MaxPool）    |
| Stage 3 | 512      | 127×127     | MaxPool(3, stride=2, ceil_mode=True) |
| Stage 4 | 768      | 63×63       | MaxPool(3, stride=2, ceil_mode=True) |
| Stage 5 | 1024     | 31×31       | MaxPool(3, stride=2, ceil_mode=True) |

---

## 4. 直接使用 VoVNet 的問題分析

### 4.1 問題一：Stem 降採樣過度

VoVNet 的 stem 進行 4× 降採樣（1020 → 255），而原本的 BEVResNet stem 不做降採樣（1020 → 1020）。
BEV 特徵圖不像 RGB 影像，不需要在 stem 階段就大幅降低解析度。

### 4.2 問題二：空間維度不對齊（最嚴重）

MaxPool(kernel=3, stride=2, ceil_mode=True) 在奇數輸入時產生的輸出：
- 255 → 127（奇數）
- 127 → 63（奇數）
- 63 → 31（奇數）

SECONDFPN 使用 ConvTranspose2d 上採樣，輸出大小為 `H_in × stride`：
- 255 × 2 = 510 ✓
- 127 × 4 = 508 ≠ 510 ✗
- 63 × 8 = 504 ≠ 510 ✗

**不同大小的特徵圖無法串接（concatenate），SECONDFPN 會報錯。**

### 4.3 問題三：通道數不匹配

| 來源       | 輸出通道          |
|-----------|-------------------|
| BEVResNet | [64, 128, 256]    |
| VoVNet    | [256, 512, 768, 1024] |

SECONDFPN 的 `in_channels` 需要對應修改。

### 4.4 問題四：模組匯入

VoVNet 註冊在 StreamPETR 專案下（`projects/StreamPETR/stream_petr/models/backbones/vovnet.py`），
CenterPoint 的 config 需要額外匯入才能使用。

---

## 5. 解決方案：建立 BEVVoVNet

### 5.1 核心思路

仿照 `BEVResNet` 對 ResNet 的改造方式，建立 `BEVVoVNet` 類別：
**將 stem 的降採樣從 4× 改為 1×（不降採樣）**，讓 MaxPool 在 stages 中自然產生整數倍的空間大小。

### 5.2 修改後的空間維度

Stem 的三個 conv 全部使用 stride=1（不降採樣）：

| 層級    | 輸出通道 | 空間大小    | 降採樣方式             |
|---------|----------|-------------|----------------------|
| Stem    | 128      | **1020×1020** | Conv stride 1,1,1（不降採樣） |
| Stage 2 | 256      | **1020×1020** | 無                    |
| Stage 3 | 512      | **510×510**   | MaxPool(3, stride=2, ceil_mode=True) |
| Stage 4 | 768      | **255×255**   | MaxPool(3, stride=2, ceil_mode=True) |
| Stage 5 | 1024     | 127×127     | MaxPool(3, stride=2, ceil_mode=True) |

**驗算 MaxPool 輸出大小：**
```
MaxPool(kernel=3, stride=2, ceil_mode=True):
  H_out = ceil((H_in - 3) / 2 + 1) = ceil((H_in - 1) / 2)

1020 → ceil((1020-1)/2) = ceil(509.5) = 510 ✓（偶數）
510  → ceil((510-1)/2)  = ceil(254.5) = 255 ✓
255  → ceil((255-1)/2)  = ceil(127)   = 127  （奇數，但不使用此層）
```

### 5.3 選取輸出特徵層

使用 `out_features=("stage2", "stage3", "stage4")`：

| 特徵層  | 通道 | 空間大小  | 對應原本 BEVResNet |
|---------|------|-----------|-------------------|
| Stage 2 | 256  | 1020×1020 | Stage 0 (64ch @ 1020) |
| Stage 3 | 512  | 510×510   | Stage 1 (128ch @ 510) |
| Stage 4 | 768  | 255×255   | Stage 2 (256ch @ 255) |

空間維度 [1020, 510, 255] **完全一致**！

### 5.4 SECONDFPN 配置更新

```python
pts_neck=dict(
    type="SECONDFPN",
    in_channels=[256, 512, 768],       # ← 更新：對應 VoVNet stage2/3/4 的輸出通道
    out_channels=[128, 128, 128],      # 不變
    upsample_strides=[0.5, 1, 2],      # 不變：與原本的空間比例完全一致
    ...
)
```

```
Stage 2: 256ch @ 1020 → upsample_stride=0.5 (Conv stride=2) → 128ch @ 510 ✓
Stage 3: 512ch @ 510  → upsample_stride=1   (Conv stride=1) → 128ch @ 510 ✓
Stage 4: 768ch @ 255  → upsample_stride=2   (Deconv stride=2) → 128ch @ 510 ✓ (255×2=510)
                                                         串接 → 384ch @ 510×510 ✓
```

**CenterHead 的 `in_channels=384` 不需要改變。**

---

## 6. 需要進行的程式碼變更

### 6.1 新增模型檔案

**檔案：`projects/CenterPoint/models/backbones/vovnet.py`**

建立 `BEVVoVNet` 類別，繼承 StreamPETR 的 `VoVNet`，覆寫 stem 建構方法：
- 將 stem 的 3 個 conv 的 stride 改為可配置參數（預設全部為 1）
- 保留 VoVNet 其餘架構不變（OSA modules、eSE、MaxPool 等）

### 6.2 更新模組匯入

**檔案：`projects/CenterPoint/models/backbones/__init__.py`**
- 新增 `BEVVoVNet` 匯入

**檔案：`projects/CenterPoint/models/__init__.py`**
- 新增 `BEVVoVNet` 匯出

### 6.3 更新設定檔

**檔案：`projects/CenterPoint/configs/t4dataset/Centerpoint/vov99_secfpn_4xb16_121m_j6gen2_base_amp.py`**

更新 `pts_backbone` 和 `pts_neck` 配置。

---

## 7. 效能考量與替代方案

### 7.1 計算成本警告

使用 3 層輸出 (stage2, stage3, stage4) 時，**stage2 的 OSA 模組在 1020×1020 的特徵圖上運算**，
計算量和記憶體消耗會顯著高於原本的 BEVResNet。

VoVNet99 的 stage2 有 1 個 OSA block，內含 5 層卷積（128ch），在 1020×1020 上運算。

### 7.2 替代方案：2 層輸出（更省資源）

只使用 `out_features=("stage3", "stage4")` → [512, 768] at [510, 255]：

```python
pts_neck=dict(
    type="SECONDFPN",
    in_channels=[512, 768],
    out_channels=[128, 128],
    upsample_strides=[1, 2],
)
# CenterHead in_channels 需改為 256 (128+128)
```

優點：跳過 stage2 在大尺寸特徵圖上的運算，顯著降低計算量。
缺點：少一個尺度的特徵，CenterHead 輸入通道從 384 減少至 256。

### 7.3 建議

先以 **3 層輸出方案（stage2, stage3, stage4）** 實作，確保功能正確後，
若遇到記憶體或速度瓶頸，再切換至 2 層輸出方案。
