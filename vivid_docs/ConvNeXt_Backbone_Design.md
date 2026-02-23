# ConvNeXt Backbone 設計詳解

## 概述

ConvNeXt 是一個純卷積神經網絡架構，設計靈感來自 Vision Transformer (ViT) 和 Swin Transformer，但完全使用卷積操作實現。在 CenterPoint 中，`ConvNeXt_PC` 是專門為點雲（Point Cloud）BEV（Bird-Eye-View）特徵提取設計的版本。

## 整體架構

### 1. 架構層次結構

```
Input (BEV Feature Map)
    ↓
Stage 0: [32 channels] × 3 blocks (depth=3)
    ↓ (downsample)
Stage 1: [64 channels] × 2 blocks (depth=2)
    ↓ (downsample)
Stage 2: [128 channels] × 1 block (depth=1)  ← out_indices[0]
    ↓ (downsample)
Stage 3: [128 channels] × 1 block (depth=1)  ← out_indices[1]
    ↓ (downsample)
Stage 4: [128 channels] × 1 block (depth=1)  ← out_indices[2]
    ↓
Output Features (multi-scale)
```

### 2. 配置參數解析

以 `pillar_020_convnext_small` 為例：

```python
pts_backbone=dict(
    type="ConvNeXt_PC",
    in_channels=32,                    # 輸入通道數（來自 middle encoder）
    out_channels=[32, 64, 128, 128, 128], # 每個 stage 的輸出通道數
    depths=[3, 2, 1, 1, 1],            # 每個 stage 的 block 數量
    out_indices=[2, 3, 4],             # 輸出哪些 stage 的特徵
    drop_path_rate=0.0,                 # Stochastic depth 比率
    layer_scale_init_value=1.0,        # Layer scale 初始值
    gap_before_final_norm=False,        # 是否在最後的 norm 前做 GAP
    with_cp=True,                      # 使用 checkpoint 節省內存
    first_downsample=1,                 # 第一個下採樣層的位置
)
```

### 3. 關鍵組件

#### 3.1 Downsample Layers（下採樣層）

每個 stage 之間的下採樣層結構：

```python
downsample_layer = nn.Sequential(
    LayerNorm2d(prev_channels),        # 先做 LayerNorm
    nn.Conv2d(prev_channels, new_channels, kernel_size=2, stride=2)  # 2x2 下採樣
)
```

**特點：**
- 使用 LayerNorm 而不是 BatchNorm
- 使用 2×2 卷積進行下採樣（stride=2）
- 每次下採樣空間分辨率減半，通道數可能增加

#### 3.2 ConvNeXtBlock（核心構建塊）

標準 ConvNeXtBlock 的結構（來自 mmpretrain）：

```
Input
  ↓
Depthwise Conv (7×7)          # 深度可分離卷積，大感受野
  ↓
LayerNorm
  ↓
Pointwise Conv (1×1)          # 擴展通道 (mlp_ratio=4)
  ↓
GELU Activation
  ↓
Pointwise Conv (1×1)          # 壓縮回原通道
  ↓
Layer Scale (可選)            # 可學習的縮放因子
  ↓
Drop Path (可選)              # Stochastic depth
  ↓
Residual Connection          # 殘差連接
  ↓
Output
```

**關鍵設計特點：**

1. **Depthwise Separable Convolution（深度可分離卷積）**
   - 標準版本：7×7 depthwise conv
   - Large 版本：可配置更大的 kernel（如 9×9）

2. **LayerNorm 替代 BatchNorm**
   - 更適合小 batch size
   - 對 batch 統計不敏感

3. **Layer Scale**
   - 可學習的縮放因子，初始值通常很小（1e-6）
   - 幫助訓練穩定性

4. **GELU 激活函數**
   - 相比 ReLU 更平滑

#### 3.3 ConvNeXtBlockLarge（大卷積核版本）

```python
class ConvNeXtBlockLarge(ConvNeXtBlock):
    def __init__(self, kernel_size=9, padding=4, ...):
        # 使用更大的卷積核（如 9×9）替代標準的 7×7
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels  # depthwise
        )
```

**用途：** 在特定 stage 使用更大的感受野，捕捉更大範圍的空間特徵。

### 4. 前向傳播流程

```python
def forward(self, x):
    outs = []
    for i, stage in enumerate(self.stages):
        # 1. 下採樣（如果需要）
        if i >= self.first_downsample:
            x = self.downsample_layers[i](x)

        # 2. 通過 stage 中的所有 blocks
        x = stage(x)

        # 3. 如果是指定的輸出 stage，進行歸一化並保存
        if i in self.out_indices:
            norm_layer = getattr(self, f"norm{i}")
            if self.gap_before_final_norm:
                gap = x.mean([-2, -1], keepdim=True)  # Global Average Pooling
                outs.append(norm_layer(gap).flatten(1))
            else:
                outs.append(norm_layer(x).contiguous())

    return tuple(outs)  # 返回多尺度特徵
```

### 5. 在 CenterPoint 中的應用

#### 5.1 輸入特徵

- **來源：** `pts_middle_encoder` (PointPillarsScatter)
- **形狀：** `[B, 32, H, W]` (例如 `[B, 32, 1216, 1216]`)
- **含義：** BEV 平面上的 pillar 特徵圖

#### 5.2 多尺度特徵提取

根據 `out_indices=[2, 3, 4]`，backbone 輸出三個不同分辨率的特徵：

```
Stage 2: [B, 128, H/4, W/4]   # 1/4 分辨率
Stage 3: [B, 128, H/8, W/8]   # 1/8 分辨率  
Stage 4: [B, 128, H/16, W/16]  # 1/16 分辨率
```

#### 5.3 與 Neck 的連接

這些多尺度特徵被傳遞給 `SECONDFPN`：

```python
pts_neck=dict(
    type="SECONDFPN",
    in_channels=[128, 128, 128],  # 對應 stage 2, 3, 4 的輸出
    out_channels=[128, 128, 128],
    upsample_strides=[1, 2, 4],   # 上採樣到相同分辨率
)
```

FPN 會將這些特徵上採樣並融合，生成最終的檢測特徵。

### 6. 設計優勢

#### 6.1 相比 ResNet 的改進

1. **LayerNorm vs BatchNorm**
   - 不依賴 batch 統計，更適合小 batch 訓練
   - 推理時行為一致

2. **大卷積核**
   - 7×7 depthwise conv 提供大感受野
   - 參數量遠少於標準 7×7 conv

3. **簡化的激活函數位置**
   - 只在 MLP 中使用激活，卷積後直接 LayerNorm

#### 6.2 相比 Transformer 的優勢

1. **純卷積實現**
   - 無需複雜的 attention 機制
   - 計算效率高，易於優化

2. **局部歸納偏置**
   - 卷積的局部性適合圖像/BEV 特徵
   - 不需要大量數據預訓練

### 7. 關鍵參數說明

| 參數 | 說明 | 典型值 |
|------|------|--------|
| `in_channels` | 輸入通道數 | 32 (來自 middle encoder) |
| `out_channels` | 每個 stage 的通道數 | `[32, 64, 128, 128, 128]` |
| `depths` | 每個 stage 的 block 數 | `[3, 2, 1, 1, 1]` |
| `out_indices` | 輸出哪些 stage | `[2, 3, 4]` |
| `drop_path_rate` | Stochastic depth 比率 | 0.0 (small) 或 0.4 (standard) |
| `layer_scale_init_value` | Layer scale 初始值 | 1.0 |
| `first_downsample` | 第一個下採樣位置 | 1 (stage 1 開始下採樣) |
| `with_cp` | 使用 checkpoint | True (節省內存) |

### 8. 計算複雜度分析

對於輸入 `[B, 32, 1216, 1216]`：

- **Stage 0:** `[B, 32, 1216, 1216]` → 3 blocks
- **Stage 1:** `[B, 64, 608, 608]` → 2 blocks (下採樣 2×)
- **Stage 2:** `[B, 128, 304, 304]` → 1 block (下採樣 2×)
- **Stage 3:** `[B, 128, 152, 152]` → 1 block (下採樣 2×)
- **Stage 4:** `[B, 128, 76, 76]` → 1 block (下採樣 2×)

**總計算量：** 主要在前幾個 stage，後續 stage 由於分辨率降低，計算量較小。

### 9. 與標準 ConvNeXt 的差異

1. **無 Stem Layer**
   - 標準 ConvNeXt 有 patch embedding stem
   - `ConvNeXt_PC` 直接從 BEV 特徵開始，無需 stem

2. **可配置的 first_downsample**
   - 允許第一個 stage 不下採樣
   - 保持輸入分辨率

3. **Large Kernel 支持**
   - 可選的大卷積核版本
   - 用於特定 stage 以增加感受野

### 10. 訓練技巧

1. **Checkpoint (`with_cp=True`)**
   - 節省 GPU 內存
   - 以計算時間換取內存

2. **Stochastic Depth (`drop_path_rate`)**
   - 正則化技術
   - 訓練時隨機跳過某些 block

3. **Layer Scale**
   - 幫助訓練穩定性
   - 初始值通常很小，逐漸學習

## 總結

ConvNeXt_PC backbone 是一個高效的多尺度特徵提取器，通過：
- 分層的下採樣結構提取多尺度特徵
- 深度可分離卷積減少參數量
- LayerNorm 提供穩定的歸一化
- 殘差連接保證梯度流動

為 CenterPoint 的 3D 檢測任務提供了強有力的 BEV 特徵表示。
