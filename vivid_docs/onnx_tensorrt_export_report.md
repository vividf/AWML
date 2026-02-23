# CenterPoint ONNX/TensorRT 導出問題解決報告

## 概述

本文檔記錄了將 CenterPoint (ConvNeXt-PC) 模型成功導出為 ONNX 和 TensorRT 引擎過程中遇到的問題及其解決方案。

**模型配置**：
- Backbone: ConvNeXt_PC (small)
- Neck: SECONDFPN
- Voxel Encoder: BackwardPillarFeatureNet
- Grid Size: 1216 x 1216

---

## 問題 1: TensorRT Concatenation 維度不匹配

### 錯誤訊息

```
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error
(IConcatenationLayer /model/neck/Concat: /model/neck/Concat: axis 2 dimensions
must be equal for concatenation on axis 1. Dimensions are 255 and 254.)
```

### 問題分析

1. **根本原因**：
   - SECONDFPN 使用 `ConvTranspose2d` (deconv) 進行上採樣
   - 不同 stride 的上採樣層（strides=[1, 2, 4]）產生的空間維度略有差異
   - 例如：一個特徵圖是 255x255，另一個是 254x254
   - ONNX 允許這種差異，但 TensorRT 在 concatenation 時要求嚴格匹配

2. **觸發條件**：
   - 當輸入空間尺寸不能被所有 stride 的乘積整除時更容易出現
   - 例如：1020 / 8 = 127.5（不能被整除）

### 解決方案

#### 修法 1: Crop 對齊（已實施，最穩定）

在 `SECONDFPN.forward()` 中添加 crop 對齊邏輯：

```python
# projects/CenterPoint/models/necks/second_fpn.py

def forward(self, x):
    assert len(x) == len(self.in_channels)
    ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

    if len(ups) > 1:
        # 修法1: 使用 crop 對齊到最小尺寸（最穩定）
        min_h = min(up.shape[2] for up in ups)
        min_w = min(up.shape[3] for up in ups)
        target_size = (min_h, min_w)

        aligned_ups = []
        for up in ups:
            h, w = up.shape[2], up.shape[3]
            if (h, w) != target_size:
                # 從左上角 crop（對 detection 最穩定）
                up = up[..., :min_h, :min_w]
            aligned_ups.append(up)
        out = torch.cat(aligned_ups, dim=1)
    else:
        out = ups[0]
    return [out]
```

**優點**：
- 最穩定，適合跨框架部署
- 不改變數值，只裁剪邊緣
- 生產環境常用的做法

**缺點**：
- 會損失邊緣 1 像素的信息（通常影響很小）

---

## 問題 2: Spatial Features 尺寸配置錯誤

### 錯誤訊息

雖然沒有直接錯誤，但會導致維度不匹配問題。

### 問題分析

1. **配置不一致**：
   - 訓練配置：`grid_size = [1216, 1216, 1]`
   - 部署配置：`spatial_features = [1, 32, 1020, 1020]` ❌

2. **整除性問題**：
   - 1020 / 8 = 127.5（不能被最大 stride 乘積整除）
   - 這會加劇 deconv 輸出尺寸不一致的問題

### 解決方案

修正 deploy config 中的 `spatial_features` 尺寸：

```python
# projects/CenterPoint/deploy/configs/deploy_config_fp16_convnext_small.py

spatial_features=dict(
    # 必須匹配訓練配置的 grid_size
    # For pillar_020_convnext_small: grid_size = [1216, 1216, 1]
    # 1216 能被 8 整除 (1216 / 8 = 152)，有助於避免維度不匹配
    min_shape=[1, 32, 1216, 1216],
    opt_shape=[1, 32, 1216, 1216],
    max_shape=[1, 32, 1216, 1216],
),
```

**驗證整除性**：
- 1216 / 1 = 1216 ✓
- 1216 / 2 = 608 ✓
- 1216 / 4 = 304 ✓
- 1216 / 8 = 152 ✓

---

## 問題 3: Input Features Channel 數配置錯誤

### 錯誤訊息

```
[TRT] [E] Dimension mismatch for tensor input_features and profile 0.
At dimension axis 2, profile has min=11, opt=11, max=11 but tensor has 9.
```

### 問題分析

1. **實際 Channel 計算**：

   實際的 channel 數取決於兩個因素：
   - **數據加載的 `use_dim`**：決定原始點雲數據有多少 channels
   - **Voxel Encoder 的 forward 邏輯**：會拼接原始 features + 計算出的 features

   **BackwardPillarFeatureNet** (ConvNeXt 使用)：
   - 數據：`point_use_dim = 3` (x, y, z) 或 `point_load_dim = 4` (x, y, z, intensity)
   - Forward 拼接：`features` (4) + `f_cluster` (3) + `f_center` (2, 只用 x, y)
   - 總計：4 + 3 + 2 = **9 channels** ✓

   **PillarFeatureNet** (SECOND/ResNet 使用)：
   - 數據：`point_load_dim = 5` (x, y, z, intensity, ring_id)，`use_dim=point_load_dim=5`
   - Forward 拼接：`features` (5) + `f_cluster` (3) + `f_center` (3, 使用 x, y, z)
   - 總計：5 + 3 + 3 = **11 channels** ✓

2. **配置錯誤**：
   - ConvNeXt 部署配置設定為 11 channels（錯誤）
   - 實際 ONNX 模型期望 9 channels（因為使用 BackwardPillarFeatureNet）

### 解決方案

修正 deploy config 中的 `input_features` channel 數：

```python
# projects/CenterPoint/deploy/configs/deploy_config_fp16_convnext_small.py

input_features=dict(
    # BackwardPillarFeatureNet channel 計算：
    # base (4) + cluster_center (3) + voxel_center (2) = 9 channels
    # Shape: (num_voxels, max_points_per_voxel, channels)
    min_shape=[1000, 32, 9],
    opt_shape=[20000, 32, 9],
    max_shape=[64000, 32, 9],
),
```

### Voxel Encoder 對照表

| Backbone | Voxel Encoder Type | Point Data Channels | Forward 拼接結果 | 總 Channels |
|----------|-------------------|---------------------|-----------------|-------------|
| ConvNeXt | `BackwardPillarFeatureNet` | 4 (x, y, z, intensity) | 4 + 3 + 2 | **9** |
| SECOND | `PillarFeatureNet` | 5 (x, y, z, intensity, ring_id) | 5 + 3 + 3 | **11** |
| ResNet | `PillarFeatureNet` | 5 (x, y, z, intensity, ring_id) | 5 + 3 + 3 | **11** |

**說明**：
- **Point Data Channels**：由數據加載配置的 `use_dim` 決定
- **Forward 拼接**：`原始features` + `cluster_center` (3) + `voxel_center` (2 或 3)
- **BackwardPillarFeatureNet** 的 `voxel_center` 只用 x, y（+2 channels）
- **PillarFeatureNet** 的 `voxel_center` 使用 x, y, z（+3 channels）

### mmdetection3d v0 vs v1 主要差異

#### 1. **PillarFeatureNet 的 voxel_center 計算差異**

**v0 (舊版本)**：
- `with_voxel_center=True` 時，只計算 **x, y** 兩個維度的距離
- 增加 **2 channels** (x, y)
- 不包含 z 維度的 voxel center 距離
- 代碼：`f_center = torch.zeros_like(features[:, :, :2])` （只取前 2 個 channel）

**v1 (新版本)**：
- `with_voxel_center=True` 時，計算 **x, y, z** 三個維度的距離
- 增加 **3 channels** (x, y, z)
- 包含完整的 3D voxel center 距離
- 代碼：`f_center = torch.zeros_like(features[:, :, :3])` （取前 3 個 channel）

**影響**：
- v0 模型：總 channels = base (4) + cluster_center (3) + voxel_center (2) = **9 channels**
- v1 模型：總 channels = base (4) + cluster_center (3) + voxel_center (3) = **10 channels**
- 如果直接用 v1 的 `PillarFeatureNet` 加載 v0 訓練的模型，會因為 channel 數不匹配而失敗

#### 2. **座標系統差異**

**v0 (舊版本)**：
- 旋轉系統：**clockwise y-axis**（順時針 y 軸）
- Bounding box 格式：`[x, y, z, w, l, h]`（寬度在前，長度在後）

**v1 (新版本)**：
- 旋轉系統：**right-handed x-axis reference**（右手 x 軸參考系）
- Bounding box 格式：`[x, y, z, l, w, h]`（長度在前，寬度在後）
- 參考：[mmdetection3d 座標系統文檔](https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html)

#### 3. **架構和功能差異**

**v0 (舊版本)**：
- 基於 mmdetection3d v0.x
- 不支持 Batch Inference（單張推理）
- 配置系統較簡單

**v1 (新版本)**：
- 基於 mmdetection3d v1.x
- **支持 Batch Inference**（批量推理，性能更好）
- 新的配置系統（純 Python 配置）
- 更完善的數據集支持（Waymo, nuScenes 等）

#### 4. **為什麼需要 BackwardPillarFeatureNet？**

為了**向後兼容**（Backward Compatibility）：

1. **舊模型保護**：v0 訓練的模型權重是 9 channels，需要匹配的實現才能加載
2. **平滑遷移**：不需要重新訓練所有模型，可以直接使用舊模型
3. **行為一致**：`BackwardPillarFeatureNet` 的行為與 v0 完全一致（不計算 z 維度）

**實際例子**：
```python
# v0 訓練的模型（9 channels）
old_checkpoint = "model_v0.pth"

# ✅ 使用 BackwardPillarFeatureNet（9 channels）- 可以加載
model = BackwardPillarFeatureNet(...)
model.load_state_dict(torch.load(old_checkpoint))  # 成功！

# ❌ 使用 v1 的 PillarFeatureNet（10 channels）- 會失敗
model = PillarFeatureNet(...)
model.load_state_dict(torch.load(old_checkpoint))  # RuntimeError: shape mismatch
```

#### 5. **代碼對比**

**v1 PillarFeatureNet (mmdet3d v1)**：
```python
# 計算 x, y, z 三個維度
if with_voxel_center:
    in_channels += 3  # +3 channels
    f_center = torch.zeros_like(features[:, :, :3])  # 3 channels
    f_center[:, :, 0] = ...  # x
    f_center[:, :, 1] = ...  # y
    f_center[:, :, 2] = ...  # z (包含 z 維度！)
```

**BackwardPillarFeatureNet (v0 兼容)**：
```python
# 只計算 x, y 兩個維度
if with_voxel_center:
    in_channels += 2  # +2 channels (關鍵差異！)
    f_center = torch.zeros_like(features[:, :, :2])  # 2 channels
    f_center[:, :, 0] = ...  # x
    f_center[:, :, 1] = ...  # y
    # 不計算 z 維度！
```

---

## 其他考慮的解決方案（未實施）

### 修法 2: 調整 ConvTranspose 的 output_padding

**原理**：通過調整 `output_padding` 和 `padding` 讓輸出尺寸一致

**未採用原因**：
- 需要修改 mmcv 的 `build_upsample_layer`
- 可能影響模型精度
- 修法 1 (crop) 已經足夠穩定

### 修法 3: 使用 align_corners=False

**狀態**：已確認代碼中未使用 `align_corners=True`

### 修法 4: 調整 Grid Size 為可整除值

**建議**：
- 如果仍有問題，可考慮使用 1024 或 1280（都是 8 的倍數）
- 但 1216 已經可以整除，且與訓練配置一致

---

## 最終配置總結

### 1. SECONDFPN 修復

**文件**：`projects/CenterPoint/models/necks/second_fpn.py`

```python
def forward(self, x):
    # ... 添加 crop 對齊邏輯
    min_h = min(up.shape[2] for up in ups)
    min_w = min(up.shape[3] for up in ups)
    # Crop 所有特徵圖到最小尺寸
```

### 2. Deploy Config 修正

**文件**：`projects/CenterPoint/deploy/configs/deploy_config_fp16_convnext_small.py`

```python
model_inputs=[
    dict(
        input_shapes=dict(
            input_features=dict(
                min_shape=[1000, 32, 9],   # ✅ 修正：9 channels
                opt_shape=[20000, 32, 9],
                max_shape=[64000, 32, 9],
            ),
            spatial_features=dict(
                min_shape=[1, 32, 1216, 1216],  # ✅ 修正：匹配 grid_size
                opt_shape=[1, 32, 1216, 1216],
                max_shape=[1, 32, 1216, 1216],
            ),
        )
    ),
],
```

---

## 驗證步驟

1. **ONNX 導出**：
   ```bash
   python projects/CenterPoint/deploy/main.py \
       projects/CenterPoint/deploy/configs/deploy_config_fp16_convnext_small.py \
       projects/CenterPoint/configs/t4dataset/CenterPoint-ConvNeXtPC/pillar_020_convnext_small_secfpn_4xb8_121m_base_t4metric_v2.py
   ```

2. **檢查 ONNX 模型**：
   - 確認 `pts_voxel_encoder.onnx` 輸入為 9 channels
   - 確認 `pts_backbone_neck_head.onnx` 輸入為 1216x1216

3. **TensorRT 構建**：
   - 確認沒有 concatenation 維度錯誤
   - 確認 engine 文件成功生成

---

## 經驗總結

### 關鍵要點

1. **配置一致性**：
   - 部署配置必須與訓練配置完全一致
   - 特別注意 `grid_size` 和 `spatial_features` 的對應關係

2. **Voxel Encoder 差異**：
   - 不同 backbone 可能使用不同的 voxel encoder
   - 必須根據實際使用的 encoder 設定正確的 channel 數

3. **維度對齊策略**：
   - Crop 對齊是最穩定且常用的方法
   - 適合跨框架部署場景

4. **整除性檢查**：
   - 確保輸入尺寸能被所有 stride 的乘積整除
   - 可以減少維度不匹配的發生

### 最佳實踐

1. **導出前檢查**：
   - 確認訓練配置中的 `grid_size`
   - 確認 `voxel_encoder` 類型
   - 計算正確的 channel 數

2. **逐步驗證**：
   - 先導出 ONNX，檢查輸入輸出尺寸
   - 再構建 TensorRT，檢查是否有錯誤
   - 最後進行推理驗證

3. **文檔記錄**：
   - 記錄每個配置的 channel 數和尺寸
   - 建立配置對照表，避免混淆

---

## 參考資料

- [SECONDFPN 實現](../../models/necks/second_fpn.py)
- [BackwardPillarFeatureNet 實現](../../models/voxel_encoders/pillar_encoder.py)
- [部署配置](../../deploy/configs/deploy_config_fp16_convnext_small.py)
- [訓練配置](../../configs/t4dataset/CenterPoint-ConvNeXtPC/pillar_020_convnext_small_secfpn_4xb8_121m_base.py)

---

**報告日期**：2026-02-03  
**狀態**：✅ 已成功導出 ONNX 和 TensorRT 引擎
