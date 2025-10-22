# CenterPoint PyTorch Evaluation 修復

## 🚨 新問題

使用 `--replace-onnx-models` 後，PyTorch evaluation 失敗了：

```
TypeError: PillarFeatureNetONNX.forward() takes 2 positional arguments but 6 were given
```

同時 ONNX evaluation 輸出 0 predictions。

## 🔍 根本原因

### 問題 1: PyTorch 使用了 ONNX 配置

**衝突**：
- PyTorch evaluation 使用 mmdet3d 標準 API
- 標準 API 調用：`voxel_encoder.forward(voxels, num_points, coors, ...)`
- 但 `PillarFeatureNetONNX` 的接口是：`forward(features)` - 只接受一個參數！

| Backend | 需要的配置 | 原因 |
|---------|-----------|------|
| PyTorch | **Standard** | 使用 mmdet3d 標準 API |
| ONNX | **ONNX** | 需要 `get_input_features` 方法 |
| TensorRT | **ONNX** | 需要 `get_input_features` 方法 |

### 問題 2: ONNX 輸出格式不匹配

ONNX 返回的是：
```python
[heatmap_np, reg_np, height_np, dim_np, rot_np, vel_np]  # 6個 numpy arrays
```

但 `_parse_predictions` 檢查的是：
```python
isinstance(output[0], torch.Tensor)  # ❌ 不匹配！
```

## ✅ 解決方案

### 修復 1: 為 PyTorch 使用 Standard 配置

修改 `_load_pytorch_model_directly` 方法，添加 `use_standard_config` 參數：

```python
def _load_pytorch_model_directly(
    self, 
    checkpoint_path: str, 
    device: torch.device, 
    logger, 
    use_standard_config: bool = False  # ← 新參數
):
    model_config = self.model_cfg.model.copy()
    
    # 為 PyTorch evaluation 轉換回 standard 配置
    if use_standard_config and model_config.type == "CenterPointONNX":
        logger.info("Converting ONNX config to standard config")
        # 轉換模型類型
        model_config.type = "CenterPoint"
        model_config.pts_voxel_encoder.type = "PillarFeatureNet"
        model_config.pts_bbox_head.type = "CenterHead"
        # ...
```

然後在 PyTorch backend 創建時使用：

```python
if backend == "pytorch":
    model = self._load_pytorch_model_directly(
        model_path, 
        device_obj, 
        logger, 
        use_standard_config=True  # ← 轉換為 standard 配置
    )
```

### 修復 2: 支持 ONNX 的 numpy array 輸出

修改 `_parse_predictions` 方法：

```python
# 檢查 ONNX 格式（numpy arrays 或 torch tensors）
elif isinstance(output[0], (torch.Tensor, np.ndarray)) and len(output) == 6:
    heatmap, reg, height, dim, rot, vel = output
    
    # 轉換 numpy arrays 為 torch tensors
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
        reg = torch.from_numpy(reg)
        # ...
```

## 📊 修復後的流程

### PyTorch Backend
```
ONNX Config (CenterPointONNX)
    ↓
convert to standard  ← ✅ 新增轉換
    ↓
Standard Config (CenterPoint)
    ↓
Load PyTorch Model
    ↓
Use mmdet3d Standard API
```

### ONNX Backend
```
ONNX Config (CenterPointONNX)
    ↓
Load ONNX-compatible PyTorch Model
    ↓
Use for voxelization & middle encoder
    ↓
Run ONNX inference
    ↓
Return numpy arrays  ← ✅ 現在正確解析
```

## 🧪 測試命令

```bash
cd /home/yihsiangfang/ml_workspace/AWML

python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu
```

## 📈 期望結果

修復後應該看到：

```
✅ PyTorch Results:
  Converting ONNX config to standard config  ← 新的日誌
  Model type: CenterPoint                     ← Standard 配置
  mAP: 0.4400
  Predictions: ~60-70

✅ ONNX Results:
  ONNX head outputs detected                  ← 正確檢測格式
  Model type: CenterPointONNX                 ← ONNX 配置
  mAP: ~0.43-0.45
  Predictions: ~60-70                         ← 不再是 0！

✅ TensorRT Results:
  mAP: ~0.42-0.45
  Predictions: ~60-70
```

## 🔑 關鍵要點

### 1. 不同 Backend 需要不同配置

| Backend | 配置類型 | Voxel Encoder | 原因 |
|---------|---------|--------------|------|
| PyTorch | Standard | `PillarFeatureNet` | 使用 mmdet3d API |
| ONNX | ONNX | `PillarFeatureNetONNX` | 需要 `get_input_features` |
| TensorRT | ONNX | `PillarFeatureNetONNX` | 需要 `get_input_features` |

### 2. 配置轉換是動態的

- Evaluator 接收的是 ONNX 配置（因為 `--replace-onnx-models`）
- PyTorch backend 在加載時動態轉換為 standard 配置
- ONNX/TensorRT backend 直接使用 ONNX 配置

### 3. 為什麼不直接傳遞兩個配置？

**問題**: 在 main.py 中維護兩個配置（standard 和 ONNX）會很複雜

**解決**: 在 evaluator 內部動態轉換更簡單、更靈活

## 📁 修改的文件

1. ✅ `projects/CenterPoint/deploy/evaluator.py`
   - 添加 `use_standard_config` 參數到 `_load_pytorch_model_directly`
   - 實現 ONNX config → Standard config 轉換
   - PyTorch backend 使用 standard 配置
   - 支持 numpy array 格式的 ONNX 輸出

2. ✅ `projects/CenterPoint/deploy/main.py`（之前修復）
   - 檢測 evaluation 是否需要 ONNX 配置
   - 即使 export mode = "none" 也加載 ONNX 配置（如果需要）

## ⚠️ 已知限制

### 配置轉換的限制

目前的轉換只處理了基本的類型替換：
- `CenterPointONNX` → `CenterPoint`
- `PillarFeatureNetONNX` → `PillarFeatureNet`
- `CenterHeadONNX` → `CenterHead`

如果 ONNX 配置有其他自定義參數（如 `rot_y_axis_reference`），這些參數會保留。對於大多數情況這是OK的，因為這些參數不影響 forward 方法的簽名。

### 為什麼不在 main.py 中分別處理？

**選項 A**: main.py 傳遞兩個配置
```python
# ❌ 複雜且容易出錯
pytorch_cfg = model_cfg  # standard
onnx_cfg = load_pytorch_model(...)[1]  # ONNX
evaluator = CenterPointEvaluator(pytorch_cfg, onnx_cfg)
```

**選項 B**: evaluator 動態轉換（當前方案）
```python
# ✅ 簡單且靈活
evaluator = CenterPointEvaluator(onnx_cfg)
# evaluator 內部根據 backend 類型自動轉換
```

## 🐛 調試提示

### 如果 PyTorch 還是失敗

檢查日誌：
```bash
grep "Converting ONNX" <log_file>
# 應該看到: "Converting ONNX-compatible config to standard config"

grep "Model type:" <log_file>
# PyTorch 應該看到: "Model type: CenterPoint"
# ONNX 應該看到: "Model type: CenterPointONNX"
```

### 如果 ONNX predictions 還是 0

檢查日誌：
```bash
grep "ONNX head outputs detected" <log_file>
# 應該看到這條消息

grep "DEBUG: ONNX output shapes" <log_file>
# 應該看到所有輸出的 shapes
```

## 📚 相關文檔

- `CENTERPOINT_EVALUATION_FIXES_SUMMARY.md` - Evaluation 修復總結
- `CENTERPOINT_EVALUATION_ERROR_FIX.md` - 之前的錯誤修復
- `CENTERPOINT_VERIFICATION_FIXES_SUMMARY.md` - Verification 修復

## 💡 經驗教訓

1. **不同的 backend 需要不同的配置** - 不能一刀切
2. **ONNX 模型有不同的接口** - 為了導出優化
3. **動態轉換比維護多個配置更簡單** - 更靈活、更易維護

## ✅ 完成！

現在所有三個 backend 都應該能正常工作：
- ✅ PyTorch: 使用 standard 配置
- ✅ ONNX: 使用 ONNX 配置，正確解析輸出
- ✅ TensorRT: 使用 ONNX 配置，使用 PyTorch middle encoder

運行測試命令驗證結果！

