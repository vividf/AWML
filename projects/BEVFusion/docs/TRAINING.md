# BEVFusion 訓練流程說明

本文件說明如何使用 AWML 訓練 BEVFusion（以 LiDAR-only nuScenes 為例），並重點說明 **spconv**、**cumm** 在訓練中扮演的角色、網路架構與資料流。

---

## 1. 訓練指令與流程概覽

### 1.1 單 GPU 訓練指令（nuScenes）

```bash
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py
```

### 1.2 訓練流程簡圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  tools/detection3d/train.py                                                  │
│    → 載入 config (bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e)   │
│    → 建立 Runner / 資料載入 / 模型 / 優化器 / scheduler                        │
│    → 每個 epoch: batch → model(batch) → loss → backward → optimizer.step()   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataLoader                                                                  │
│    → NuScenesDataset + train_pipeline (LoadPoints, MultiSweeps, Augmentation) │
│    → Pack3DDetInputs → batch["inputs"]["points"] (list of LiDAR point clouds) │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BEVFusion.extract_pts_feat() → pts_middle_encoder()  ← 這裡使用 spconv/cumm │
│    → 見下方「網路架構與 spconv/cumm 角色」                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Config** 決定：voxel 大小、sparse shape、encoder 通道數、backbone/neck/head、dataset、scheduler 等。
- **訓練迴圈** 由 mmengine Runner 驅動；每個 step 會執行一次 `model(batch)`，內部會呼叫 `extract_pts_feat`，也就是 **sparse encoder（spconv）** 的入口。

---

## 2. 網路架構與資料流（LiDAR-only）

整體 LiDAR 分支可簡化為：

```
Points (N, 5)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 1. Voxelization (data_preprocessor / pts_voxel_layer)                      │
│    - point_cloud_range, voxel_size, max_num_points, max_voxels             │
│    - 輸出: voxel_features (M, C_in), coords (M, 4), [num_points_per_voxel] │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 2. Voxel Feature Encoder (pts_voxel_encoder)                                │
│    - HardSimpleVFE: 每個 voxel 內特徵聚合 → 仍為 per-voxel 特徵             │
└───────────────────────────────────────────────────────────────────────────┘
    │  feats (M, 5), coords (M, 4), batch_size
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 3. pts_middle_encoder = BEVFusionSparseEncoder  ★ 使用 spconv（與 cumm）   │
│    - 輸入: SparseConvTensor(feats, coords, sparse_shape, batch_size)       │
│    - SubMConv3d / SparseConv3d 多層 3D 稀疏卷積                             │
│    - 最後 conv_out 做 (1,1,3) stride (1,1,2) 再 .dense() → 4D 再 view       │
│    - 輸出: spatial_features (N, C*D, H, W) 供後續 2D backbone               │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 4. pts_backbone (SECOND) + pts_neck (SECONDFPN)                            │
│    - 2D 卷積處理 BEV 特徵 → 多尺度特徵                                       │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 5. bbox_head (BEVFusionHead)                                               │
│    - TransFusion 風格：heatmap + decoder → 3D bbox + 類別                   │
└───────────────────────────────────────────────────────────────────────────┘
```

- **Sparse 部分僅出現在步驟 3**：從 voxel 的 (feats, coords) 建 `SparseConvTensor`，經過多層 **SubMConv3d / SparseConv3d**，最後 `conv_out` 再轉成 dense 特徵給 SECOND。
- **spconv** 提供這些 3D 稀疏卷積的 **Python API 與 C++/CUDA 介面**；**cumm** 是 spconv 2.x 底層用來產生與執行 **稀疏卷積 CUDA kernel** 的庫（含 Tensor Core 等優化）。

---

## 3. BEVFusionSparseEncoder 與 SparseEncoder 的差異

BEVFusion 的 `BEVFusionSparseEncoder`（`bevfusion/sparse_encoder.py`）繼承自 mmdet3d 的 `SparseEncoder`（`mmdet3d.models.middle_encoders.sparse_encoder`），但只呼叫 `super(SparseEncoder, self).__init__()`（即 `nn.Module`），**不呼叫** `SparseEncoder.__init__`，並自行建構 `conv_input`、`encoder_layers`、`conv_out`，且改寫 `forward`。兩者差異如下。

### 3.1 用途與所屬框架

| 項目 | SparseEncoder | BEVFusionSparseEncoder |
|------|----------------|-------------------------|
| 所屬 | mmdet3d 內建 | BEVFusion 專案（AWML） |
| 設計對象 | SECOND、Part-A2 等 | BEVFusion（與其 voxelization 一致） |
| 繼承 | `nn.Module` | `SparseEncoder`（僅用 `make_encoder_layers` 等邏輯，init/forward 自實作） |

### 3.2 3D 體素座標與形狀順序（最重要）

- **SparseEncoder** 假設體素網格的 **維度順序為 (D, H, W)**，與 mmdet3d 常見的 voxelization 一致：
  - `coors` 為 `(batch_idx, z_idx, y_idx, x_idx)`，對應 3D 張量為 **(Z, Y, X)** 即 **(D, H, W)**。
  - 最後一層 `conv_out` 的 downsampling 做在 **第一個空間維**：`kernel_size=(3, 1, 1)`、`stride=(2, 1, 1)`。
  - `out.dense()` 得到 **(N, C, D, H, W)**，直接 `view(N, C*D, H, W)` 得到 BEV。

- **BEVFusionSparseEncoder** 假設體素網格的 **維度順序為 (H, W, D)**，與 BEVFusion 的 voxelization 輸出一致：
  - 同一份 `coors` 在 spconv 內部被解讀為 **(batch_idx, x_idx, y_idx, z_idx)** 或對應 **(H, W, D)** 的 layout（依 BEVFusion 的 voxel 實作）。
  - 最後一層 `conv_out` 的 downsampling 做在 **第三個空間維（對應高度 D）**：`kernel_size=(1, 1, 3)`、`stride=(1, 1, 2)`。
  - `out.dense()` 得到 **(N, C, H, W, D)**，需先 `permute(0, 1, 4, 2, 3)` 變成 **(N, C, D, H, W)**，再 `view(N, C*D, H, W)` 與 SparseEncoder 輸出格式對齊，供後續 SECOND 使用。

因此：**若在 BEVFusion 裡改用原生 `SparseEncoder` 而不改 voxelization，3D 維度順序會不一致，結果會錯誤**；必須使用 `BEVFusionSparseEncoder` 才能與 BEVFusion 的體素化座標與形狀順序對齊。

### 3.3 conv_out 的 kernel / stride

| 項目 | SparseEncoder | BEVFusionSparseEncoder |
|------|----------------|-------------------------|
| conv_out kernel_size | (3, 1, 1) | (1, 1, 3) |
| conv_out stride | (2, 1, 1) | (1, 1, 2) |
| 作用 | 在第一維（D）做下採樣 | 在第三維（D）做下採樣，對應 (H,W,D) layout |

兩者都是把 3D 稀疏特徵壓成 2D BEV（把 D 維與 channel 合併成 C*D），只是對應的空間軸不同。

### 3.4 可選的額外特徵（aug features）

- **SparseEncoder**：無額外輸入維度或編碼，僅 `in_channels`。
- **BEVFusionSparseEncoder**：支援 `aug_features_min_values`、`aug_features_max_values`、`num_aug_features`。
  - 若 `num_aug_features > 0`，會在 forward 裡對部分維度做歸一化後，用 cos/sin 編碼擴成 `in_channels * num_aug_features * 2`，再送入 `conv_input`。
  - 用於將某些連續特徵（例如高度、強度）編成週期性編碼，提升表達能力。

### 3.5 小結

- **BEVFusionSparseEncoder** 與 **SparseEncoder** 的 **encoder 主體**（conv_input、encoder_layers 的建構方式）概念相同，都透過 `make_sparse_convmodule` / `make_encoder_layers` 建 SubMConv3d、SparseConv3d。
- 差異在：**(1) 3D 形狀順序 (D,H,W) vs (H,W,D)**、**(2) conv_out 的 kernel/stride 與對應軸**、**(3) dense 後的 permute 有無**、**(4) BEVFusion 多了可選的 aug_features 編碼**。  
訓練 BEVFusion 時必須使用 **BEVFusionSparseEncoder**，才能與專案內的 voxelization 與後續 SECOND 輸入格式一致。

---

## 4. Sparse Encoder Forward 實作細節（Code Trace）

以下依呼叫順序追蹤 **BEVFusionSparseEncoder.forward** 的實作，對應程式碼位置與資料形狀。Config 以 nuScenes 為例：`sparse_shape=[1440,1440,41]`、`encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128))`、`block_type="basicblock"`。

### 4.1 入口與輸入

- **呼叫處**：`BEVFusion.extract_pts_feat()` → `self.pts_middle_encoder(feats, coords, batch_size)`  
  （`bevfusion/bevfusion.py` 約 222 行）
- **forward 簽名**（`bevfusion/sparse_encoder.py` 124 行）：
  ```python
  def forward(self, voxel_features, coors, batch_size):
  ```
- **輸入**：
  - `voxel_features`: `(M, C_in)`，M 為非空 voxel 數，C_in=5（HardSimpleVFE 輸出）。
  - `coors`: `(M, 4)`，每行 `(batch_idx, z_idx, y_idx, x_idx)`（或與 voxelization 一致的座標順序），dtype 會轉成 `int`。
  - `batch_size`: int。

### 4.2 [可選] Aug features 編碼

- **程式碼**：`bevfusion/sparse_encoder.py` 145–154 行。
- 若 config 中 `num_aug_features > 0`：
  - 對 `voxel_features` 做 min-max 歸一化，再對部分維度做 cos/sin 編碼，並 concat 成 `(M, in_channels*num_aug_features*2)`，取代原本的 `voxel_features`。
- 本專案 nuScenes config 未使用 aug_features，此段不執行。

### 4.3 建構 SparseConvTensor

- **程式碼**：`bevfusion/sparse_encoder.py` 156–157 行。
  ```python
  coors = coors.int()
  input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
  ```
- **SparseConvTensor**（spconv 或 mmcv.ops）內含：
  - `features`: `(M, C_in)`，每個非空 voxel 的特徵。
  - `indices`: 由 `coors` 轉成，表示每個 voxel 在 3D 網格中的位置。
  - `spatial_shape`: `sparse_shape`，即 `[1440, 1440, 41]`（BEVFusion 為 H, W, D 順序）。
  - `batch_size`。
- 之後所有層的輸入/輸出都是 **SparseConvTensor**，僅在最後一層之後才轉成 dense。

### 4.4 conv_input（第一層稀疏卷積）

- **程式碼**：`bevfusion/sparse_encoder.py` 158 行，`x = self.conv_input(input_sp_tensor)`。
- **建構處**：同檔案 99–107 行，`make_sparse_convmodule(..., conv_type="SubMConv3d", indice_key="subm1", ...)`。
- **make_sparse_convmodule**（`mmdet3d/models/layers/sparse_block.py` 157–224 行）會依 `order`（預設 `("conv","norm","act")`）組出一個 **SparseSequential**：
  - **Conv**：`SubMConv3d`，kernel_size=3，padding=1，stride=1，in_channels→base_channels（16）。SubM 表示 submanifold：只在不改變稀疏模式的條件下做 3×3×3 卷積。
  - **Norm**：BN1d(base_channels)。
  - **Act**：ReLU(inplace=True)。
- **indice_key="subm1"**：spconv 用來快取該層的 rule book（哪些輸出 voxel 對應哪些輸入 voxel），同一 key 可複用，加速後續 forward。
- **輸出**：仍是 SparseConvTensor，`features` shape 為 `(M, 16)`，稀疏索引不變。

### 4.5 encoder_layers（多階段 3D 稀疏卷積）

- **程式碼**：`bevfusion/sparse_encoder.py` 160–163 行。
  ```python
  encode_features = []
  for encoder_layer in self.encoder_layers:
      x = encoder_layer(x)
      encode_features.append(x)
  ```
- **encoder_layers** 由 **SparseEncoder.make_encoder_layers** 建立（`mmdet3d/models/middle_encoders/sparse_encoder.py` 164–241 行），BEVFusion 使用 `block_type="basicblock"`，`encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128))`，`encoder_paddings=((0,0,1),(0,0,1),(0,0,(1,1,0)),(0,0))`。

**make_encoder_layers 邏輯摘要**（basicblock）：
- 每個 stage 有多個 block；若「當前是 stage 的最後一個 block」且「不是最後一個 stage」，則插入 **SparseConv3d stride=2** 做下採樣並增加 channel；否則插入 **SparseBasicBlock**（兩個 SubMConv3d + residual）。
- 對應到 nuScenes config，各 stage 結構如下（通道數 in→out，indice_key 用於 spconv 快取）：

| Stage | 結構 | 說明 |
|-------|------|------|
| encoder_layer1 | SparseBasicBlock(16,16) → SparseBasicBlock(16,16) → **SparseConv3d**(16→32, stride=2, indice_key=`spconv1`) | 兩次 SubM 再下採樣 |
| encoder_layer2 | SparseBasicBlock(32,32) → SparseBasicBlock(32,32) → **SparseConv3d**(32→64, stride=2, indice_key=`spconv2`) | 同上 |
| encoder_layer3 | SparseBasicBlock(64,64) → SparseBasicBlock(64,64) → **SparseConv3d**(64→128, stride=2, indice_key=`spconv3`) | 同上 |
| encoder_layer4 | SparseBasicBlock(128,128) → SparseBasicBlock(128,128) | 最後 stage 不再下採樣，輸出 128 通道 |

- **SparseBasicBlock**（`mmdet3d/models/layers/sparse_block.py` 92–154 行）：兩個 SubMConv3d(3×3×3) + BN + ReLU + residual，輸入輸出 sparse 結構一致（除最後一層接 SparseConv3d 時會下採樣並改變 voxel 數）。
- 每經過一層 **SparseConv3d stride=2**，空間維度約減半，voxel 數會減少；經過四段後，最後一層的 `encode_features[-1]` 的 `features` shape 為 `(M', 128)`，M' 為最後階段的非空 voxel 數。

### 4.6 conv_out（最後一層稀疏卷積 → 壓成 2D）

- **程式碼**：`bevfusion/sparse_encoder.py` 166–167 行。
  ```python
  out = self.conv_out(encode_features[-1])
  spatial_features = out.dense()
  ```
- **建構處**：同檔案 114–123 行。
  - `make_sparse_convmodule(encoder_out_channels, output_channels, kernel_size=(1,1,3), stride=(1,1,2), conv_type="SparseConv3d", indice_key="spconv_down2")`  
  即 128→128，在 **第三個空間維（D）** 上做 stride 2 下採樣（對應 BEVFusion 的 (H,W,D) 順序）。
- **輸出**：仍為 SparseConvTensor；接著呼叫 **.dense()** 把稀疏 3D 張量填成 dense 的 5D 張量。

### 4.7 dense() 與形狀轉換為 BEV

- **程式碼**：`bevfusion/sparse_encoder.py` 167–172 行。
  ```python
  spatial_features = out.dense()   # (N, C, H, W, D)
  N, C, H, W, D = spatial_features.shape
  spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, C, D, H, W)
  spatial_features = spatial_features.view(N, C * D, H, W)               # (N, C*D, H, W)
  ```
- **.dense()**（spconv）：根據目前 sparse 的 indices 與 spatial_shape，將 `(M', C)` 的 features 還原成一個 dense 的 5D 張量。在 BEVFusion 的座標順序下，該張量為 **(N, C, H, W, D)**（N=batch_size, C=128, H/W 為 BEV 網格兩軸，D 為高度維）。
- **permute(0,1,4,2,3)**：把 D 維移到第 3 維，得到 **(N, C, D, H, W)**，方便與 channel 合併。
- **view(N, C*D, H, W)**：把 C 與 D 合併成一個維度，得到 **BEV 特徵圖 (N, 128*D, H, W)**，供後續 **SECOND backbone** 當 2D 輸入。D 的具體值由 sparse_shape 與前面各層 stride 決定（最後一層在 D 上 stride 2，故 D 約為原始 41 的一半再取整）。

### 4.8 回傳

- **程式碼**：`bevfusion/sparse_encoder.py` 174–176 行。
  - 若 `return_middle_feats=False`（預設）：回傳 `spatial_features`，shape **(N, C*D, H, W)**。
  - 若 `return_middle_feats=True`：回傳 `(spatial_features, encode_features)`，後者為各 stage 輸出的 SparseConvTensor 列表，可供其他模組使用。

### 4.9 小結：Forward 資料流一覽

```
voxel_features (M, 5), coors (M, 4), batch_size
    → [可選] aug_features 編碼
    → SparseConvTensor(feats, coors, sparse_shape, batch_size)
    → conv_input: SubMConv3d 5→16, indice_key=subm1     → (M, 16) sparse
    → encoder_layer1: BasicBlock×2 + SparseConv3d 16→32 stride2  → 下採樣
    → encoder_layer2: BasicBlock×2 + SparseConv3d 32→64 stride2
    → encoder_layer3: BasicBlock×2 + SparseConv3d 64→128 stride2
    → encoder_layer4: BasicBlock×2                        → (M', 128) sparse
    → conv_out: SparseConv3d (1,1,3) stride(1,1,2) 128→128
    → .dense() → (N, C, H, W, D)
    → permute → view → (N, C*D, H, W)  → 輸出
```

- **關鍵檔案**：`projects/BEVFusion/bevfusion/sparse_encoder.py`（forward、conv 建構）、`mmdet3d/models/middle_encoders/sparse_encoder.py`（make_encoder_layers）、`mmdet3d/models/layers/sparse_block.py`（make_sparse_convmodule、SparseBasicBlock）。

---

## 5. spconv 在訓練中扮演的角色

### 5.1 什麼是 spconv？

- **spconv**（traveller59/spconv）是「**Spatially Sparse Convolution**」的實作庫，專為 3D 點雲/體素設計，只對有資料的 voxel 做卷積，避免在空體素上浪費計算。
- BEVFusion 的 LiDAR 分支中，**所有 3D 稀疏卷積** 都依賴 spconv 提供的型別與算子。

### 5.2 BEVFusion 裡如何用到 spconv？

- **後端選擇**（`mmdet3d.models.layers.spconv`）：
  - 若已安裝 **spconv 2.x**（例如 `pip install spconv-cu120`），則 `IS_SPCONV2_AVAILABLE = True`，使用 **traveller59 的 spconv**（建議，效能與部署較佳）。
  - 若未安裝或版本 < 2.0，則使用 **mmcv 的 SparseConvTensor** 等實作作為 fallback。
- **BEVFusionSparseEncoder**（`bevfusion/sparse_encoder.py`）：
  - 使用 `make_sparse_convmodule()` 建立多個 **SubMConv3d** 與最後一層 **SparseConv3d**（`conv_out`）。
  - 輸入：`voxel_features`、`coors`、`batch_size`；建構 `SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)` 作為稀疏卷積的輸入。
  - Forward 時依序通過 `conv_input` → 多個 `encoder_layers` → `conv_out`，最後 `out.dense()` 再 permute/view 成 `(N, C*D, H, W)`。
- **Config 對應**（nuScenes 範例）：
  - `sparse_shape=[1440, 1440, 41]`：由 `point_cloud_range` 與 `voxel_size=[0.075, 0.075, 0.2]` 推得體素網格大小。
  - `encoder_channels=((16,16,32), (32,32,64), (64,64,128), (128,128))`、`block_type="basicblock"` 等，決定 sparse encoder 的通道與 block 結構，底層均由 **spconv** 的 SubMConv3d / SparseConv3d 實作。

因此，**訓練時每一 forward 的 3D 稀疏卷積計算**（含 backward）都是由 **spconv** 提供的算子完成；若安裝的是 spconv-cu120，底層會進一步用到 **cumm** 產生的 CUDA kernel。

---

## 6. cumm 在訓練中扮演的角色

### 6.1 什麼是 cumm？

- **cumm**（FindDefinition/cumm）是「**CUda Matrix Multiply**」相關的庫，用於產生與優化 CUDA 程式碼（例如 GEMM、sparse 相關運算）。
- 與 **pccm**（Python 作為 meta-programming）一起，成為 spconv 2.x 的 **底層程式碼生成與執行框架**。

### 6.2 與 spconv 的關係

- **spconv 2.x** 的 C++/CUDA 實作依賴 **cumm**：
  - 稀疏卷積的實際 **CUDA kernel**（含 rule-based 的 im2col、GEMM、Tensor Core 等）是由 cumm 相關的程式碼生成或呼叫。
  - 安裝 `spconv-cu120` 時，通常會一併依賴對應的 **cumm**（或由 spconv 的 build 過程引入）；訓練時 **不需在 Python 裡直接 import cumm**，而是透過 spconv 間接使用。
- 因此：
  - **Python 層**：只看到 **spconv**（SparseConvTensor、SubMConv3d、SparseConv3d 等）。
  - **執行層**：spconv 的 forward/backward 會呼叫到 **cumm** 產生的 CUDA 程式碼，在訓練中負責 **稀疏卷積的實際 GPU 計算**。

總結：**cumm** 在訓練中扮演的是 **底層 CUDA 稀疏卷積算子的實現與加速**，不直接出現在 BEVFusion 的 Python 程式碼中，而是透過 spconv 使用。

---

## 7. 如何利用這些 library（實作面）

### 7.1 訓練前依賴

- 建議安裝 traveller59 的 spconv（會自動帶入對應 cumm 或由 spconv 編譯使用）：
  ```bash
  pip install spconv-cu120
  ```
- 安裝後，AWML 會透過 `IS_SPCONV2_AVAILABLE` 自動選用此後端；無需改 BEVFusion 程式碼。

### 7.2 程式碼中的使用點

| 位置 | 用途 |
|------|------|
| `bevfusion/sparse_encoder.py` | 使用 `SparseConvTensor`、`make_sparse_convmodule` 建構 SubMConv3d/SparseConv3d，forward 時建 `SparseConvTensor` 並呼叫 `.dense()` 轉成 dense 特徵。 |
| `mmdet3d.models.layers.spconv` | 判斷是否使用 spconv 2.x（`IS_SPCONV2_AVAILABLE`），決定 `SparseConvTensor` 從 `spconv.pytorch` 還是 `mmcv.ops` 匯入。 |
| Config `pts_middle_encoder` | 設定 `sparse_shape`、`encoder_channels`、`encoder_paddings`、`block_type` 等，全部由 spconv 的 3D 稀疏卷積實作。 |

### 7.3 與 config 的對應關係

- `voxel_size=[0.075, 0.075, 0.2]`、`point_cloud_range=[-54,-54,-5,54,54,3]`  
  → 推得 `sparse_shape=[1440, 1440, 41]`（各軸格數）。
- `pts_middle_encoder.type="BEVFusionSparseEncoder"`、`in_channels=5`、`encoder_channels=((16,16,32), ...)`  
  → 決定 sparse encoder 的層數與通道，全部由 spconv（與底層 cumm）執行。

---

## 8. 總結

- **訓練指令**：使用  
  `python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py`  
  即可啟動 LiDAR-only BEVFusion 訓練。
- **spconv**：在訓練中負責 **3D 稀疏卷積的 API 與執行**（SparseConvTensor、SubMConv3d、SparseConv3d），集中在 **BEVFusionSparseEncoder**；若安裝 spconv-cu120，會自動被選用。
- **cumm**：作為 spconv 2.x 的 **底層 CUDA 實作與 kernel 生成/優化**，在訓練時透過 spconv 間接參與所有 3D 稀疏卷積的 forward/backward，無需在 Python 中直接呼叫。
- **網路架構**：Points → Voxelization → VFE → **Sparse Encoder (spconv)** → Dense BEV → SECOND backbone → SECONDFPN → BEVFusionHead；spconv/cumm 的影響範圍僅在 Sparse Encoder 這一段。

更多環境與資料準備請參考主 [README](../../README.md) 與 [安裝教學](/docs/tutorial/tutorial_detection_3d.md)。
