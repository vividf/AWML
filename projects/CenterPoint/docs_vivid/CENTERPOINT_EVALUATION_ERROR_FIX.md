# CenterPoint Evaluation 錯誤修復

## 🚨 錯誤信息

```
ERROR: Current model type: CenterPoint
ERROR: Expected model type: CenterPointONNX
ERROR: ONNX evaluation requires ONNX-compatible model config
```

## 🎯 根本原因

**你沒有使用 `--replace-onnx-models` flag！**

這個 flag 是**必需的**，它會將模型配置從標準版本替換為 ONNX 兼容版本：

| 組件 | 沒有 flag | 使用 flag |
|-----|----------|----------|
| Detector | `CenterPoint` ❌ | `CenterPointONNX` ✅ |
| Voxel Encoder | `PillarFeatureNet` ❌ | `PillarFeatureNetONNX` ✅ |
| BBox Head | `CenterHead` ❌ | `CenterHeadONNX` ✅ |

## ✅ 解決方案

### 正確的命令

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# ⭐ 關鍵：必須添加 --replace-onnx-models！
python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu
```

### 如果只想運行 Evaluation（不重新 Export）

即使 ONNX/TensorRT 模型已經存在，你仍然需要：

1. ✅ 使用 `--replace-onnx-models` flag
2. ✅ 提供 `--checkpoint` 路徑
3. ✅ 在 deploy_config.py 中設置 `export.mode = "none"`

```bash
# 1. 修改 deploy_config.py
# export = dict(
#     mode="none",  # 不重新導出
#     ...
# )

# 2. 運行命令（仍需 --replace-onnx-models！）
python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu
```

## 🔍 為什麼需要這個 Flag？

### ONNX 需要特殊的方法

標準的 `PillarFeatureNet` 只有：
```python
def forward(self, voxels, num_points, coors):
    # 一次性處理所有
    return features
```

ONNX 版本的 `PillarFeatureNetONNX` 有：
```python
def get_input_features(self, voxels, num_points, coors):
    # 返回原始特徵（未處理）
    return raw_features

def forward(self, features):
    # 處理特徵
    return processed_features
```

ONNX voxel encoder 需要 `raw_features` 作為輸入，所以必須使用 ONNX 版本。

### 輸出格式不同

| 版本 | 輸出格式 |
|------|---------|
| Standard | `Tuple[List[Dict[str, Tensor]]]` - 嵌套結構 |
| ONNX | `Tuple[Tensor, ...]` - 扁平化列表 |

ONNX 無法處理嵌套的 dict 結構，所以必須使用扁平化的輸出。

## 📝 我已經修復的問題

我修復了 `main.py` 中的邏輯問題：

**修復前** ❌:
```python
# 只有在 export 模式時才加載 ONNX 配置
if config.export_config.mode != "none":
    pytorch_model, onnx_compatible_model_cfg = load_pytorch_model(
        ..., replace_onnx_models=args.replace_onnx_models
    )
```

**修復後** ✅:
```python
# 如果需要評估 ONNX/TensorRT，也加載 ONNX 配置
needs_onnx_config = False
if eval_config.get("enabled"):
    if eval_config.get("models", {}).get("onnx") or eval_config.get("models", {}).get("tensorrt"):
        needs_onnx_config = True

if config.export_config.mode != "none" or needs_onnx_config:
    pytorch_model, onnx_compatible_model_cfg = load_pytorch_model(
        ..., replace_onnx_models=args.replace_onnx_models
    )
```

現在即使只運行 evaluation（export mode = "none"），只要配置了 ONNX/TensorRT evaluation，也會正確加載 ONNX 兼容的配置。

**但你仍然需要使用 `--replace-onnx-models` flag！**

## 🧪 測試驗證

運行命令後，你應該看到：

```
✅ PyTorch model loaded successfully
Loading with ONNX-compatible configuration for evaluation  ← 應該看到這個！

Evaluating pytorch model...
  mAP: 0.4400 ✅

Evaluating onnx model...
Using ONNX-compatible model config for ONNX backend  ← 應該看到這個！
  mAP: ~0.43-0.45 ✅

Evaluating tensorrt model...
Loading PyTorch model for TensorRT middle encoder  ← 應該看到這個！
  mAP: ~0.42-0.45 ✅
```

## ❌ 如果還是失敗

### 檢查 1: 確認使用了 flag

```bash
# 在你的命令中搜索
echo "YOUR_COMMAND" | grep "replace-onnx-models"

# 應該看到:
# --replace-onnx-models
```

### 檢查 2: 確認日誌信息

```bash
# 在日誌中搜索
grep "Loading with ONNX-compatible" <log_file>

# 應該看到:
# Loading with ONNX-compatible configuration for evaluation
```

### 檢查 3: 確認模型類型

```bash
# 在日誌中搜索
grep "model type:" <log_file>

# 應該看到:
# Current model type: CenterPointONNX  ← 正確！
# 而不是:
# Current model type: CenterPoint  ← 錯誤！
```

## 📚 相關文檔

- `CENTERPOINT_EVALUATION_FIXES_SUMMARY.md` - 完整技術分析
- `CENTERPOINT_EVALUATION_QUICK_FIX_GUIDE.md` - 快速指南
- `CENTERPOINT_VERIFICATION_FIXES_SUMMARY.md` - Verification 修復

## 💡 常見誤解

### ❓ "我已經導出了 ONNX 模型，為什麼還需要這個 flag？"

**答**: 這個 flag 不是用來導出模型，而是用來**加載正確的 PyTorch 模型配置**。

Evaluation 流程需要：
1. ONNX 文件（已經存在） ✅
2. ONNX 兼容的 PyTorch 模型（用於 voxelization 和 middle encoder） ← 需要 flag！

### ❓ "可以在配置文件中設置嗎？"

**答**: 不行，`--replace-onnx-models` 只能作為命令行參數使用。

這是設計決定：確保用戶明確知道他們在使用 ONNX 兼容模式。

### ❓ "PyTorch evaluation 需要這個 flag 嗎？"

**答**: 不需要。只有 ONNX 和 TensorRT evaluation 需要。

但是，如果你的 evaluation 配置中包含了 ONNX 或 TensorRT，就必須使用這個 flag。

## ✅ 最終檢查清單

運行 evaluation 前確認：

- [ ] ✅ 使用了 `--replace-onnx-models` flag
- [ ] ✅ 提供了 `--checkpoint` 路徑
- [ ] ✅ ONNX 模型文件存在（work_dirs/centerpoint_deployment/*.onnx）
- [ ] ✅ TensorRT 引擎存在（如果要評估 TensorRT）
- [ ] ✅ deploy_config.py 中的 evaluation.models 路徑正確

全部勾選後，運行命令，應該就能成功了！

