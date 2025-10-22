# CenterPoint Evaluation 快速修復指南

## TL;DR

**問題**: PyTorch 成功，ONNX 失敗，TensorRT 結果為 0

**原因**:
1. ONNX backend 使用了錯誤的模型配置（沒有 `get_input_features` 方法）
2. TensorRT middle encoder 使用了隨機數據而不是真實的 voxel features

**解決**: 已修復 ✅

## 快速測試

### 重新運行 Evaluation

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 確保使用 --replace-onnx-models flag!
python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu
```

### 期望看到

```
✅ PyTorch Results:
  mAP: 0.4400
  Predictions: 64

✅ ONNX Results:                   ← 現在應該成功！
  mAP: ~0.43-0.45                  ← 接近 PyTorch
  Predictions: ~60-68

✅ TensorRT Results:               ← 現在應該有正確結果！
  mAP: ~0.42-0.45                  ← 不再是 0！
  Predictions: ~60-68
```

## 如果 ONNX 還是失敗

### 檢查錯誤信息

如果看到:
```
NotImplementedError: Standard voxel encoder not supported
```

**原因**: 沒有使用 `--replace-onnx-models` flag

**解決**: 重新運行命令並添加 `--replace-onnx-models`

### 檢查模型配置

```bash
# 在日誌中搜索
grep "CenterPointONNX" <log_file>

# 應該看到:
# "Using ONNX-compatible model config for ONNX backend"
```

## 如果 TensorRT 結果還是 0

### 檢查 PyTorch 模型加載

```bash
# 在日誌中搜索
grep "Loading PyTorch model for TensorRT" <log_file>

# 應該看到:
# "Successfully loaded PyTorch model on cuda:0"
```

### 檢查 Heatmap 值

修復前（錯誤）:
```
Raw heatmap - min: -456.848145, max: -11.921435
Sigmoid heatmap - min: 0.000000, max: 0.000007  ← 全部接近0！
```

修復後（正確）:
```
Raw heatmap - min: -20.xxx, max: 5.xxx
Sigmoid heatmap - min: 0.xxx, max: 0.8xxx      ← 應該有正常範圍！
```

## 修復的文件

1. ✅ `projects/CenterPoint/deploy/evaluator.py`
2. ✅ `projects/CenterPoint/deploy/centerpoint_tensorrt_backend.py`
3. ✅ `autoware_ml/deployment/core/verification.py`

## 技術細節

### ONNX 修復
- 使用 `CenterPointONNX` 配置（有 `get_input_features` 方法）
- 不再使用 standard 模型配置

### TensorRT 修復
- Middle encoder 現在使用 PyTorch（不再用隨機數據）
- TensorRT 只運行 voxel encoder 和 backbone/neck/head
- 混合執行：TensorRT + PyTorch

## 為什麼這樣修復？

### ONNX 需要特殊模型
```python
# Standard 模型
pts_voxel_encoder.forward(voxels, num_points, coors)  # ❌ ONNX 不支持

# ONNX 模型
features = pts_voxel_encoder.get_input_features(...)   # ✅ 獲取原始特徵
output = pts_voxel_encoder.forward(features)           # ✅ ONNX 可以處理
```

### TensorRT Middle Encoder 使用 PyTorch
```python
# 修復前 - 使用隨機數據
dummy_features = torch.randn(...)  # ❌ 完全錯誤！

# 修復後 - 使用真實計算
spatial_features = pytorch_model.pts_middle_encoder(
    voxel_features, coors, batch_size
)  # ✅ 正確的 sparse convolution
```

Middle encoder 使用 sparse convolution（稀疏卷積），TensorRT 支持有限，所以保持 PyTorch 執行。

## 完整文檔

詳細的技術分析請查看：
- `CENTERPOINT_EVALUATION_FIXES_SUMMARY.md` - 完整修復總結
- `CENTERPOINT_VERIFICATION_FIXES_SUMMARY.md` - Verification 修復
- `CENTERPOINT_VERIFICATION_USAGE_GUIDE.md` - 使用指南

## 常見問題

### Q1: 為什麼一定要用 `--replace-onnx-models`？

A: 因為 ONNX/TensorRT 需要特殊的模型配置：
- `CenterPointONNX` 而不是 `CenterPoint`
- `PillarFeatureNetONNX` 而不是 `PillarFeatureNet`
- 這些 ONNX 版本有特殊的方法（如 `get_input_features`）

### Q2: TensorRT 為什麼比 ONNX 快但還用 PyTorch？

A: 
- TensorRT 執行: Voxel Encoder + Backbone/Neck/Head（~95% 計算量）
- PyTorch 執行: Middle Encoder（~5% 計算量，sparse convolution）
- 總體還是比純 PyTorch 快很多

### Q3: ONNX 和 TensorRT 的精度會完全一樣嗎？

A: 不會完全一樣，但應該很接近：
- PyTorch: mAP 0.4400
- ONNX: mAP ~0.43-0.45（±0.01 可接受）
- TensorRT: mAP ~0.42-0.45（±0.02 可接受，可能有更多數值誤差）

### Q4: 我該用哪個 backend？

| Backend | 速度 | 精度 | 部署複雜度 | 推薦場景 |
|---------|------|------|-----------|---------|
| PyTorch | 慢 | 最高 | 簡單 | 開發/調試 |
| ONNX | 中 | 高 | 中等 | 跨平台部署 |
| TensorRT | 快 | 高 | 複雜 | NVIDIA GPU 生產環境 |

## 聯繫支持

如果問題仍未解決：
1. 收集完整的日誌輸出
2. 確認使用了 `--replace-onnx-models`
3. 檢查 ONNX/TensorRT 文件是否存在
4. 查看詳細文檔

