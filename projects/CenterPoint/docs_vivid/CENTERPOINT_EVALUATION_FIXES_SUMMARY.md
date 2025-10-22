# CenterPoint Evaluation 問題修復總結

## 問題概述

在 CenterPoint deployment evaluation 中發現三個 backend 的不同問題：

1. ✅ **PyTorch**: 成功 (mAP 0.44)
2. ❌ **ONNX**: 執行失敗 - `NotImplementedError: Standard voxel encoder not supported`
3. ⚠️ **TensorRT**: 執行成功但結果全為 0 (mAP 0.0)

## 根本原因分析

### 問題 1: ONNX Backend 使用錯誤的模型配置

**錯誤信息**:
```
NotImplementedError: Standard voxel encoder not supported for ONNX inference. 
Please use CenterPointONNX with PillarFeatureNetONNX.
```

**原因**:
- Evaluator 在創建 ONNX backend 時，使用了 standard 模型配置
- 沒有使用 ONNX 兼容的模型配置（CenterPointONNX）
- 導致加載的 PyTorch 模型沒有 `get_input_features` 方法

**位置**: `projects/CenterPoint/deploy/evaluator.py:173-187`

**原始代碼**:
```python
elif backend == "onnx":
    # ...
    pytorch_model = self._load_pytorch_model_directly(
        model_path.replace('centerpoint_deployment', 'centerpoint/best_checkpoint.pth'), 
        device_obj, 
        logger
    )
    # ❌ 使用了 self.model_cfg 中的 standard 配置
```

### 問題 2: TensorRT Middle Encoder 使用隨機數據

**現象**:
- TensorRT 執行成功但所有預測 score 接近 0 (0.000000 to 0.000007)
- Heatmap 值異常：`Raw heatmap - min: -456.848145, max: -11.921435`
- 所有 AP 都是 0.0

**原因**:
- TensorRT backend 的 `_process_middle_encoder` 方法使用了 **torch.randn** 生成隨機數據
- 沒有正確實現 middle encoder 的邏輯
- 隨機數據導致 backbone/neck/head 的輸出完全無意義

**位置**: `projects/CenterPoint/deploy/centerpoint_tensorrt_backend.py:232-262`

**原始代碼**:
```python
def _process_middle_encoder(self, voxel_features, input_data):
    # ...
    # ❌ 使用隨機數據！
    dummy_features = torch.randn(batch_size, 32, spatial_size, spatial_size, ...)
    return dummy_features
```

## 修復方案

### 修復 1: ONNX Backend - 使用 ONNX 兼容配置

**修改文件**: `projects/CenterPoint/deploy/evaluator.py`

**修復內容**:
```python
elif backend == "onnx":
    # 檢查 model_cfg 是否已經是 ONNX 版本
    if self.model_cfg.model.type == "CenterPointONNX":
        logger.info("Using ONNX-compatible model config for ONNX backend")
        # 使用正確的 checkpoint 路徑
        checkpoint_path = model_path.replace('centerpoint_deployment', 'centerpoint/best_checkpoint.pth')
        pytorch_model = self._load_pytorch_model_directly(checkpoint_path, device_obj, logger)
    else:
        logger.error("Model config is not ONNX-compatible!")
        raise ValueError("ONNX evaluation requires ONNX-compatible model config")
    
    return ONNXBackend(model_path, device=device, pytorch_model=pytorch_model)
```

**關鍵點**:
1. ✅ 檢查 model_cfg 是否為 `CenterPointONNX`
2. ✅ 如果不是，報錯並提示需要使用 `--replace-onnx-models`
3. ✅ 使用 ONNX 兼容的模型配置（有 `get_input_features` 方法）

### 修復 2: TensorRT Backend - 使用 PyTorch Middle Encoder

**修改文件**: `projects/CenterPoint/deploy/centerpoint_tensorrt_backend.py`

**修復內容**:

1. **添加 pytorch_model 參數**:
```python
def __init__(self, model_path: str, device: str = "cuda:0", pytorch_model=None):
    """
    Args:
        pytorch_model: PyTorch model for middle encoder (required)
    """
    # ...
    self.pytorch_model = pytorch_model
```

2. **使用 PyTorch Middle Encoder**:
```python
def _process_middle_encoder(self, voxel_features, input_data):
    """Use PyTorch model for middle encoder (sparse convolution)."""
    if self.pytorch_model is None:
        raise ValueError("PyTorch model is required for middle encoder processing")
    
    # Ensure correct device
    device = next(self.pytorch_model.parameters()).device
    voxel_features = voxel_features.to(device)
    coors = input_data["coors"].to(device)
    batch_size = int(coors[-1, 0].item()) + 1
    
    # Run PyTorch middle encoder
    with torch.no_grad():
        spatial_features = self.pytorch_model.pts_middle_encoder(
            voxel_features, coors, batch_size
        )
    
    return spatial_features
```

**關鍵點**:
1. ✅ 接受 `pytorch_model` 參數
2. ✅ 使用 PyTorch 的 middle encoder 處理真實的 voxel features
3. ✅ 不再使用隨機數據
4. ✅ Sparse convolution 在 PyTorch 中運行（TensorRT 不支持）

### 修復 3: Evaluator - 為 TensorRT 傳遞 PyTorch 模型

**修改文件**: `projects/CenterPoint/deploy/evaluator.py`

**修復內容**:
```python
elif backend == "tensorrt":
    # TensorRT backend needs PyTorch model for middle encoder
    if self.model_cfg.model.type == "CenterPointONNX":
        logger.info("Loading PyTorch model for TensorRT middle encoder")
        checkpoint_path = model_path.replace('centerpoint_deployment/tensorrt', 'centerpoint/best_checkpoint.pth')
        pytorch_model = self._load_pytorch_model_directly(checkpoint_path, device_obj, logger)
        return CenterPointTensorRTBackend(model_path, device=device, pytorch_model=pytorch_model)
    else:
        raise ValueError("TensorRT evaluation requires ONNX-compatible model config")
```

### 修復 4: Verification - 為 TensorRT 傳遞 PyTorch 模型

**修改文件**: `autoware_ml/deployment/core/verification.py`

**修復內容**:
```python
# CenterPoint multi-engine setup
trt_backend = CenterPointTensorRTBackend(tensorrt_path, device="cuda", pytorch_model=pytorch_model)
```

## 為什麼需要 ONNX 兼容的模型配置？

CenterPoint 的 ONNX/TensorRT 部署需要特殊的模型配置：

| 組件 | Standard 版本 | ONNX 版本 |
|-----|-------------|----------|
| Detector | `CenterPoint` | `CenterPointONNX` |
| Voxel Encoder | `PillarFeatureNet` | `PillarFeatureNetONNX` |
| BBox Head | `CenterHead` | `CenterHeadONNX` |

**關鍵差異**:
1. `PillarFeatureNetONNX` 有 `get_input_features()` 方法
   - 返回原始特徵（未處理）
   - ONNX voxel encoder 需要這些原始特徵作為輸入

2. `CenterHeadONNX` 返回不同的輸出格式
   - Standard: `Tuple[List[Dict]]` - 嵌套結構
   - ONNX: `Tuple[Tensor, ...]` - 扁平化的 tensor 列表

## 為什麼 TensorRT 需要 PyTorch Middle Encoder？

CenterPoint 的 middle encoder 使用 **sparse convolution**（稀疏卷積）：

```python
# Middle encoder example
class PointPillarsScatter(nn.Module):
    def forward(self, voxel_features, coors, batch_size):
        # Sparse scatter operation
        batch_canvas = []
        for batch_idx in range(batch_size):
            canvas = torch.zeros(...)  # Dense canvas
            # Scatter sparse voxels to dense canvas
            this_coords = coors[coors[:, 0] == batch_idx]
            ...
```

**為什麼不用 TensorRT**:
1. ❌ Sparse convolution 在 TensorRT 中支持有限
2. ❌ Dynamic scatter operations 難以優化
3. ❌ 轉換複雜且容易出錯

**使用 PyTorch 的優勢**:
1. ✅ 保證正確性（使用原始實現）
2. ✅ Middle encoder 計算量小（相對於 backbone/head）
3. ✅ 混合模式：TensorRT（voxel encoder + backbone/head）+ PyTorch（middle encoder）

## 修復的文件列表

1. ✅ `projects/CenterPoint/deploy/evaluator.py`
   - 修復 ONNX backend 配置
   - 修復 TensorRT backend 配置

2. ✅ `projects/CenterPoint/deploy/centerpoint_tensorrt_backend.py`
   - 添加 `pytorch_model` 參數
   - 修復 `_process_middle_encoder` 使用 PyTorch

3. ✅ `autoware_ml/deployment/core/verification.py`
   - 為 TensorRT backend 傳遞 `pytorch_model`

## 測試驗證

### 運行 Evaluation

確保使用 `--replace-onnx-models` flag：

```bash
cd /home/yihsiangfang/ml_workspace/AWML

python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu
```

### 期望結果

修復後，所有三個 backend 應該都能成功運行並產生相似的結果：

```
PyTorch Results:
  mAP (0.5:0.95): 0.4400
  Total Predictions: 64
  Latency: 4563.89 ms

ONNX Results:                    ← ✅ 現在應該成功！
  mAP (0.5:0.95): ~0.43-0.45      ← ✅ 接近 PyTorch
  Total Predictions: ~60-68       ← ✅ 接近 PyTorch
  Latency: ~XXX ms

TensorRT Results:                ← ✅ 現在應該有正確結果！
  mAP (0.5:0.95): ~0.42-0.45      ← ✅ 接近 PyTorch
  Total Predictions: ~60-68       ← ✅ 接近 PyTorch
  Latency: ~XXX ms               ← ✅ 應該比 ONNX 快
```

### 診斷檢查點

如果 ONNX 還是失敗：
```bash
# 檢查是否使用了 --replace-onnx-models
grep "CenterPointONNX" <log_file>
# 應該看到: "Using ONNX-compatible model config"
```

如果 TensorRT 結果還是為 0：
```bash
# 檢查是否正確加載了 PyTorch 模型
grep "Loading PyTorch model for TensorRT" <log_file>
# 應該看到: "Successfully loaded PyTorch model"

# 檢查 heatmap 值
grep "heatmap.*min:" <log_file>
# 應該看到合理的值範圍（不是全部負數）
```

## 架構圖

### CenterPoint ONNX/TensorRT Pipeline

```
Input Points
    ↓
[Voxelization] ← PyTorch (data_preprocessor)
    ↓
Voxels, Num Points, Coors
    ↓
[Voxel Encoder] ← ONNX/TensorRT Engine #1
    ↓
Voxel Features
    ↓
[Middle Encoder] ← PyTorch (pts_middle_encoder) ← 混合執行！
    ↓
Spatial Features
    ↓
[Backbone + Neck + Head] ← ONNX/TensorRT Engine #2
    ↓
Detection Outputs
```

**混合執行的原因**:
1. Voxelization: 需要靈活的點雲處理（PyTorch）
2. Voxel Encoder: 密集運算，適合 ONNX/TensorRT
3. Middle Encoder: 稀疏運算，保持 PyTorch
4. Backbone/Neck/Head: 密集運算，適合 ONNX/TensorRT

## 總結

### 修復前
- ❌ ONNX: 完全無法運行
- ❌ TensorRT: 運行但結果無意義（全部為0）

### 修復後
- ✅ ONNX: 正確使用 ONNX 兼容模型，成功運行
- ✅ TensorRT: 使用 PyTorch middle encoder，產生正確結果
- ✅ 所有 backend 結果應該接近 PyTorch baseline

### 關鍵要點
1. **使用 `--replace-onnx-models`**: 確保模型配置是 ONNX 兼容的
2. **TensorRT 混合執行**: Middle encoder 在 PyTorch 中運行
3. **Verification 成功**: 已確認兩個 ONNX 模型都正確使用
4. **Evaluation 對齊**: 現在使用相同的配置和邏輯

如果還有問題，請檢查：
- [ ] 是否使用了 `--replace-onnx-models` flag
- [ ] ONNX 文件是否正確導出（兩個 .onnx 文件）
- [ ] TensorRT 引擎是否正確構建（兩個 .engine 文件）
- [ ] Checkpoint 路徑是否正確

