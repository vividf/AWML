# CenterPoint Verification 修復總結

## 問題描述

用戶報告 CenterPoint 的 PyTorch 和 ONNX 驗證顯示 **difference 完全為 0**，這不太可能。需要檢查是否兩個 ONNX 模型都有被正確使用。

## 根本原因分析

經過詳細分析，發現了 **三個關鍵問題**：

### 問題 1: `_get_input_features` 中的重複和錯誤邏輯

**位置**: `autoware_ml/deployment/backends/centerpoint_onnx_helper.py:145-221`

**原始代碼問題**:
```python
# 第一次調用 - 獲取原始特徵然後處理
input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(...)
input_features = self.pytorch_model.pts_voxel_encoder(input_features)  # ❌ 處理特徵

# 第二次調用 - 又重新獲取原始特徵
raw_features = self.pytorch_model.pts_voxel_encoder.get_input_features(...)
input_features = raw_features  # 覆蓋前面的結果
```

**問題**:
- 重複調用 `get_input_features`（浪費計算）
- 第一次調用 `forward` 會輸出處理過的特徵（錯誤的輸入給 ONNX）
- 邏輯混亂，不清楚最終返回的是什麼

**修復**: 簡化邏輯，只獲取原始特徵一次
```python
# 直接獲取原始特徵（不要調用 forward）
if hasattr(self.pytorch_model.pts_voxel_encoder, 'get_input_features'):
    input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(
        voxels_tensor, 
        num_points_tensor, 
        coors_tensor
    )
    logger.info(f"Got raw input features from PyTorch model: shape {input_features.shape}")
else:
    raise NotImplementedError("Standard voxel encoder not supported for ONNX inference")
```

### 問題 2: `postprocess_from_onnx` 返回錯誤的數據類型

**位置**: `autoware_ml/deployment/backends/centerpoint_onnx_helper.py:250-279`

**原始代碼問題**:
```python
for output in onnx_outputs:
    if isinstance(output, np.ndarray):
        output_tensor = torch.from_numpy(output)  # ❌ 轉換為 torch.Tensor
        processed_outputs.append(output_tensor)
```

**問題**:
- ONNX backend 返回 **torch.Tensor**
- 但驗證代碼期望 **numpy.ndarray**
- 導致驗證代碼中的類型檢查失敗：
  ```python
  if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
      # ❌ 這個條件不成立，因為 out 是 torch.Tensor
  ```
- **結果**: 沒有進行任何比較，max_diff 和 mean_diff 保持初始值 0.0

**修復**: 保持輸出為 numpy arrays
```python
for output in onnx_outputs:
    if isinstance(output, np.ndarray):
        # 保持為 numpy array 用於驗證
        processed_outputs.append(output)
    else:
        # 如果需要，轉換為 numpy
        if hasattr(output, 'numpy'):
            processed_outputs.append(output.numpy())
        else:
            processed_outputs.append(np.array(output))
```

### 問題 3: 驗證代碼缺少詳細日誌和錯誤檢測

**位置**: `autoware_ml/deployment/core/verification.py:191-261`

**原始代碼問題**:
- 沒有檢測空列表輸出
- 沒有記錄詳細的數值範圍和統計信息
- 無法診斷為什麼 difference 為 0

**修復**: 添加詳細日誌和錯誤檢測
```python
# 1. 檢測空列表
if isinstance(output, list) and len(output) == 0:
    logger.error(f"  {backend_name} verification FAILED: Empty output list")
    return False, None, 0.0

# 2. 記錄詳細的輸出信息
logger.info(f"  Reference output details:")
if isinstance(reference_output, list):
    for i, out in enumerate(reference_output):
        if isinstance(out, np.ndarray):
            logger.info(f"    Output[{i}] shape: {out.shape}, dtype: {out.dtype}")
            logger.info(f"    Output[{i}] range: [{out.min():.6f}, {out.max():.6f}]")
            logger.info(f"    Output[{i}] mean: {out.mean():.6f}, std: {out.std():.6f}")

# 3. 記錄每個輸出的差異
for i, (ref_out, out) in enumerate(zip(reference_output, output)):
    if isinstance(ref_out, np.ndarray) and isinstance(out, np.ndarray):
        diff = np.abs(ref_out - out)
        output_max_diff = diff.max()
        output_mean_diff = diff.mean()
        logger.info(f"    Output[{i}] - max_diff: {output_max_diff:.6f}, mean_diff: {output_mean_diff:.6f}")
```

## 兩個 ONNX 模型的使用確認

### ONNX Backend 流程

✅ **確認使用了兩個 ONNX 模型**:

1. **第一個 ONNX 模型**: `pts_voxel_encoder.onnx`
   - 位置: `centerpoint_onnx_helper.py:preprocess_for_onnx()` 第 234-238 行
   ```python
   voxel_features = self.voxel_encoder_session.run(
       ["pillar_features"], 
       voxel_encoder_inputs
   )[0]
   ```

2. **第二個 ONNX 模型**: `pts_backbone_neck_head.onnx`
   - 位置: `onnx_backend.py:_infer_centerpoint()` 第 169 行
   ```python
   head_outputs = self._session.run(None, backbone_head_inputs)
   ```

### PyTorch Backend 流程

✅ **使用完整的 PyTorch 管道**:

```python
# 1. Voxel Encoder (對應第一個 ONNX)
input_features = self._model.pts_voxel_encoder.get_input_features(...)
voxel_features = self._model.pts_voxel_encoder(input_features)

# 2. Middle Encoder
spatial_features = self._model.pts_middle_encoder(...)

# 3. Backbone, Neck, Head (對應第二個 ONNX)
backbone_features = self._model.pts_backbone(spatial_features)
neck_features = self._model.pts_neck(backbone_features)
head_outputs = self._model.pts_bbox_head(neck_features)
```

## 修復的文件列表

1. ✅ `autoware_ml/deployment/backends/centerpoint_onnx_helper.py`
   - 修復 `_get_input_features` 方法
   - 修復 `postprocess_from_onnx` 方法

2. ✅ `autoware_ml/deployment/core/verification.py`
   - 添加空輸出檢測
   - 添加詳細的輸出日誌
   - 添加每個輸出的差異記錄

## 測試驗證

### 測試腳本

創建了專門的測試腳本: `projects/CenterPoint/deploy/test_verification_fix.py`

### 運行測試

```bash
cd /home/yihsiangfang/ml_workspace/AWML

python projects/CenterPoint/deploy/test_verification_fix.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --onnx-dir work_dirs/centerpoint_deployment \
    --info-file data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl \
    --device cpu \
    --num-samples 1 \
    --tolerance 0.1 \
    --log-level INFO
```

### 期望結果

修復後，應該看到：

1. ✅ **詳細的輸出信息**:
   ```
   Reference output details:
     Output[0] shape: (1, 3, 200, 200), dtype: float32
     Output[0] range: [-5.123456, 3.456789]
     Output[0] mean: 0.012345, std: 1.234567
   ```

2. ✅ **非零的差異值**:
   ```
   Computing differences for 6 outputs...
     Output[0] - max_diff: 0.001234, mean_diff: 0.000123
     Output[1] - max_diff: 0.002345, mean_diff: 0.000234
     ...
   Overall Max difference: 0.002345
   Overall Mean difference: 0.000234
   ```

3. ✅ **正確的驗證結果**:
   ```
   sample_0_onnx: ✅ PASSED  (如果 max_diff < tolerance)
   或
   sample_0_onnx: ❌ FAILED  (如果 max_diff >= tolerance)
   ```

## 預期影響

### 修復前
- ❌ Difference 總是 0（不進行比較）
- ❌ 無法檢測 ONNX 導出的數值問題
- ❌ 給人虛假的信心（看起來完美匹配）

### 修復後
- ✅ 正確計算 PyTorch vs ONNX 的差異
- ✅ 能夠檢測到數值不一致
- ✅ 提供詳細的診斷信息
- ✅ 真實反映模型轉換的質量

## 後續建議

1. **運行完整驗證**:
   ```bash
   python projects/CenterPoint/deploy/main.py \
       --model-cfg <config> \
       --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
       --checkpoint <checkpoint> \
       --replace-onnx-models \
       --device cpu
   ```

2. **檢查 tolerance 設定**:
   - 當前設定: `1e-1` (0.1)
   - 如果差異大於預期，可能需要：
     - 檢查 ONNX 導出設定 (constant_folding, opset_version)
     - 檢查數值精度問題 (float32 vs float16)
     - 檢查運算順序差異

3. **監控差異趨勢**:
   - 記錄每次驗證的 max_diff 和 mean_diff
   - 確保差異在可接受範圍內
   - 如果差異突然增大，需要調查原因

## 總結

這次修復解決了三個關鍵問題：
1. ✅ 修復了 `_get_input_features` 的邏輯錯誤
2. ✅ 修復了輸出類型不匹配問題（最關鍵！）
3. ✅ 增強了驗證日誌和錯誤檢測

**根本原因**: `postprocess_from_onnx` 返回了 torch.Tensor 而不是 numpy.ndarray，導致驗證代碼的類型檢查失敗，沒有進行任何比較，所以 difference 一直是初始值 0.0。

**現在**: 修復後應該能夠正確比較 PyTorch 和 ONNX 的輸出，並得到真實的差異值。

