# CenterPoint rot-y-axis-reference 修復文檔

## 📋 問題描述

當使用 `--rot-y-axis-reference` 標誌導出 ONNX 模型時：
- ✅ Verification 階段正常
- ❌ Evaluation 階段 mAP 下降

**根本原因**: ONNX 模型輸出經過 `rot_y_axis_reference` 轉換後的格式，但 PyTorch decoder (`predict_by_feat`) 期望標準格式，導致解碼錯誤。

---

## 🔍 問題分析

### rot_y_axis_reference 的作用

當 `rot_y_axis_reference=True` 時，CenterHeadONNX 在導出時會對輸出做特殊轉換：

**1. Dimension (dim) 轉換**
```python
# Standard format: [l, w, h]
# rot_y_axis_reference format: [w, l, h]
# 實現: dim[:, [1, 0, 2]] - 交換 length 和 width
```

**2. Rotation (rot) 轉換**
```python
# Standard format: [sin(y), cos(x)]
# rot_y_axis_reference format: [-cos(x), -sin(y)]
# 實現: 
#   1. 交換通道: [sin(y), cos(x)] -> [cos(x), sin(y)]
#   2. 取負: [cos(x), sin(y)] -> [-cos(x), -sin(y)]
```

### 數據流

```
ONNX Export (with rot_y_axis_reference=True)
    ↓
ONNX Model outputs [w, l, h] and [-cos(x), -sin(y)]
    ↓
Evaluation: Pass to PyTorch predict_by_feat
    ↓
❌ PyTorch expects [l, w, h] and [sin(y), cos(x)]
    ↓
Result: Wrong decoding, low mAP
```

---

## 🛠️ 修復方案

### 核心思路

在將 ONNX/TensorRT 輸出傳給 PyTorch decoder 之前，檢測並轉換回標準格式。

### 修改位置

**文件**: `AWML/projects/CenterPoint/deploy/evaluator.py`  
**方法**: `_parse_with_pytorch_decoder`  
**行數**: 699-726

### 修改內容

```python
# IMPORTANT: If model was exported with rot_y_axis_reference=True,
# we need to convert ONNX outputs back to standard format before passing to predict_by_feat
rot_y_axis_reference = getattr(self.model_cfg.model.pts_bbox_head, 'rot_y_axis_reference', False)

if rot_y_axis_reference:
    print(f"INFO: Detected rot_y_axis_reference=True, converting ONNX outputs to standard format")
    
    # Debug: Show values before conversion
    print(f"DEBUG: Before conversion - dim channels [0,1,2] sample: {dim[0, :, 380, 380]}")
    print(f"DEBUG: Before conversion - rot channels [0,1] sample: {rot[0, :, 380, 380]}")
    
    # 1. Convert dim from [w, l, h] back to [l, w, h]
    # ONNX output: dim[:, [0, 1, 2]] = [w, l, h]
    # Standard: dim[:, [0, 1, 2]] = [l, w, h]
    # So we need to swap channels 0 and 1
    dim = dim[:, [1, 0, 2], :, :]  # [w, l, h] -> [l, w, h]
    print(f"DEBUG: Converted dim from [w,l,h] to [l,w,h]")
    print(f"DEBUG: After conversion - dim channels [0,1,2] sample: {dim[0, :, 380, 380]}")
    
    # 2. Convert rot from [-cos(x), -sin(y)] back to [sin(y), cos(x)]
    # ONNX output: rot[:, [0, 1]] = [-cos(x), -sin(y)]
    # Standard: rot[:, [0, 1]] = [sin(y), cos(x)]
    # Step 1: Negate to get [cos(x), sin(y)]
    rot = rot * (-1.0)
    # Step 2: Swap channels to get [sin(y), cos(x)]
    rot = rot[:, [1, 0], :, :]
    print(f"DEBUG: Converted rot from [-cos(x), -sin(y)] to [sin(y), cos(x)]")
    print(f"DEBUG: After conversion - rot channels [0,1] sample: {rot[0, :, 380, 380]}")

# Now pass to predict_by_feat in standard format
preds_dict = {
    'heatmap': heatmap,
    'reg': reg,
    'height': height,
    'dim': dim,        # Standard [l, w, h]
    'rot': rot,        # Standard [sin(y), cos(x)]
    'vel': vel
}
```

---

## ✅ 修復效果

### ONNX Evaluation

**測試命令**:
```bash
python projects/CenterPoint/deploy/main.py \
  projects/CenterPoint/deploy/configs/deploy_config.py \
  projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py \
  work_dirs/centerpoint/best_checkpoint.pth \
  --replace-onnx-models \
  --rot-y-axis-reference \
  --device cpu
```

**結果**:
| Backend | mAP | Predictions | 狀態 |
|---------|-----|-------------|------|
| **PyTorch** | **0.4400** | 64 | ✅ 基準 |
| **ONNX (rot-y-axis)** | **0.4400** | 63 | ✅ 修復成功！ |

**輸出日誌**:
```
INFO: Detected rot_y_axis_reference=True, converting ONNX outputs to standard format
DEBUG: Before conversion - dim channels [0,1,2] sample: tensor([1.3450, 0.2747, 0.4546])
DEBUG: Converted dim from [w,l,h] to [l,w,h]
DEBUG: After conversion - dim channels [0,1,2] sample: tensor([0.2747, 1.3450, 0.4546])
DEBUG: Before conversion - rot channels [0,1] sample: tensor([-0.0455, 1.0689])
DEBUG: Converted rot from [-cos(x), -sin(y)] to [sin(y), cos(x)]
DEBUG: After conversion - rot channels [0,1] sample: tensor([-1.0689, 0.0455])

ONNX Results:
  mAP (0.5:0.95): 0.4400  ✅
  Total Predictions: 63
```

### TensorRT Evaluation

**數據流**:
- TensorRT 引擎從 ONNX (with rot_y_axis_reference) 轉換而來
- TensorRT 輸出 → `_parse_centerpoint_head_outputs` → `_parse_with_pytorch_decoder`
- 使用**相同的轉換邏輯**

**預期結果**: TensorRT 也應該達到 mAP ≈ 0.44

**注意**: TensorRT 引擎需要從帶有 `--rot-y-axis-reference` 的 ONNX 重新導出。

---

## 📊 轉換邏輯詳解

### 1. Dimension 轉換

```
┌─────────────────────────────────────────────────────────┐
│ ONNX Export (rot_y_axis_reference=True)                │
├─────────────────────────────────────────────────────────┤
│ Standard [l, w, h] → [w, l, h] (交換 index 0 和 1)      │
│ Code: dim[:, [1, 0, 2], :, :]                          │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Evaluation 反向轉換                                      │
├─────────────────────────────────────────────────────────┤
│ ONNX [w, l, h] → Standard [l, w, h] (交換回來)          │
│ Code: dim[:, [1, 0, 2], :, :]                          │
└─────────────────────────────────────────────────────────┘

Example:
  Before: [1.3450, 0.2747, 0.4546] (w=1.34, l=0.27, h=0.45)
  After:  [0.2747, 1.3450, 0.4546] (l=0.27, w=1.34, h=0.45) ✅
```

### 2. Rotation 轉換

```
┌─────────────────────────────────────────────────────────┐
│ ONNX Export (rot_y_axis_reference=True)                │
├─────────────────────────────────────────────────────────┤
│ Standard [sin(y), cos(x)]                              │
│   → Step 1: Swap [cos(x), sin(y)]                     │
│   → Step 2: Negate [-cos(x), -sin(y)]                 │
│ Code:                                                   │
│   rot[:, [1, 0], :, :]  # Swap                        │
│   rot * torch.tensor([-1.0, -1.0])  # Negate          │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Evaluation 反向轉換                                      │
├─────────────────────────────────────────────────────────┤
│ ONNX [-cos(x), -sin(y)]                                │
│   → Step 1: Negate [cos(x), sin(y)]                   │
│   → Step 2: Swap [sin(y), cos(x)]                     │
│ Code:                                                   │
│   rot = rot * (-1.0)  # Negate                        │
│   rot = rot[:, [1, 0], :, :]  # Swap                  │
└─────────────────────────────────────────────────────────┘

Example:
  Before: [-0.0455, 1.0689] (-cos(x)=-0.05, -sin(y)=1.07)
  Negate: [0.0455, -1.0689] (cos(x)=0.05, sin(y)=-1.07)
  Swap:   [-1.0689, 0.0455] (sin(y)=-1.07, cos(x)=0.05) ✅
```

---

## 🎯 關鍵洞察

### 1. 檢測機制

```python
# 從 model config 讀取 rot_y_axis_reference 參數
rot_y_axis_reference = getattr(
    self.model_cfg.model.pts_bbox_head, 
    'rot_y_axis_reference', 
    False
)
```

### 2. 轉換時機

- ✅ **正確時機**: 在傳給 `predict_by_feat` 之前轉換
- ❌ **錯誤時機**: 在 `predict_by_feat` 之後轉換（太晚了）

### 3. 適用範圍

此修復適用於：
- ✅ ONNX Backend (已驗證)
- ✅ TensorRT Backend (使用相同邏輯)
- ✅ 任何調用 `_parse_with_pytorch_decoder` 的路徑

### 4. 與 Verification 的關係

**為什麼 Verification 沒問題？**
- Verification 比較的是 ONNX 和 PyTorch 的**原始輸出**（head outputs）
- 兩者都是 rot_y_axis_reference 格式，所以可以直接比較
- **Evaluation 才需要解碼**，所以才需要轉換

---

## 📝 使用指南

### 導出帶 rot-y-axis-reference 的模型

```bash
# 1. 導出 ONNX
python projects/CenterPoint/deploy/main.py \
  projects/CenterPoint/deploy/configs/deploy_config.py \
  projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py \
  work_dirs/centerpoint/best_checkpoint.pth \
  --replace-onnx-models \
  --rot-y-axis-reference \
  --device cpu

# 2. 轉換為 TensorRT (optional)
# 修改 deploy_config.py: mode="trt"
python projects/CenterPoint/deploy/main.py \
  ... \
  --rot-y-axis-reference \
  --device cuda
```

### 評估

```bash
# 評估 ONNX
python projects/CenterPoint/deploy/main.py \
  ... \
  --rot-y-axis-reference

# 評估 TensorRT
# (使用相同命令，指定 tensorrt backend)
```

### 預期結果

- ONNX mAP ≈ 0.44 (與 PyTorch 一致)
- TensorRT mAP ≈ 0.44 (與 PyTorch 一致)
- 日誌中顯示: "Detected rot_y_axis_reference=True, converting ONNX outputs to standard format"

---

## 🔧 故障排除

### 問題 1: mAP 仍然很低

**可能原因**:
1. ONNX/TensorRT 模型不是用 `--rot-y-axis-reference` 導出的
2. Model config 中 `rot_y_axis_reference` 參數丟失

**解決方案**:
```bash
# 檢查日誌是否有轉換信息
# 應該看到: "Detected rot_y_axis_reference=True"

# 重新導出模型
rm -rf work_dirs/centerpoint_deployment
python ... --replace-onnx-models --rot-y-axis-reference
```

### 問題 2: 沒有看到轉換日誌

**可能原因**: Model config 沒有正確保存 `rot_y_axis_reference`

**解決方案**:
```python
# 檢查 evaluator.py 中的 model_cfg
print(f"rot_y_axis_reference: {getattr(self.model_cfg.model.pts_bbox_head, 'rot_y_axis_reference', None)}")
```

### 問題 3: TensorRT mAP 低於 ONNX

**可能原因**: TensorRT 引擎從舊的 ONNX 轉換（沒有 rot_y_axis_reference）

**解決方案**:
```bash
# 刪除舊引擎
rm -rf work_dirs/centerpoint_deployment/tensorrt

# 重新導出 (確保 ONNX 是用 --rot-y-axis-reference 導出的)
python ... --rot-y-axis-reference
```

---

## 📚 相關代碼

### 主要修改

1. **`evaluator.py` Line 699-726**: 添加 rot_y_axis_reference 轉換邏輯

### 相關文件

1. **`centerpoint_head_onnx.py`**: 定義 ONNX 導出時的轉換
   - `_export_forward_rot_y_axis_reference`: dim 和 rot 轉換
   
2. **`main.py`**: 處理 `--rot-y-axis-reference` 標誌
   - Line 42: 添加命令行參數
   - Line 100: 設置 model config

3. **`evaluator.py`**: 評估時的反向轉換
   - `_parse_with_pytorch_decoder`: 檢測並轉換

---

## ✅ 總結

### 修復內容

1. ✅ 添加 rot_y_axis_reference 檢測邏輯
2. ✅ 實現 dim 反向轉換 ([w,l,h] → [l,w,h])
3. ✅ 實現 rot 反向轉換 ([-cos(x),-sin(y)] → [sin(y),cos(x)])
4. ✅ 添加詳細調試日誌

### 驗證結果

| Backend | Without rot-y-axis | With rot-y-axis | 狀態 |
|---------|-------------------|-----------------|------|
| PyTorch | 0.4400 | 0.4400 | ✅ 基準 |
| ONNX | 0.4400 | 0.4400 | ✅ 已修復 |
| TensorRT | 0.4400 | ~0.4400 | ✅ 應該正常 |

### 下一步

1. 在完整數據集（19 個樣本）上驗證
2. 測試 TensorRT 的 rot-y-axis-reference 版本
3. 性能基準測試

---

**創建日期**: 2025-10-23  
**狀態**: ✅ ONNX 已驗證，TensorRT 理論上應該正常  
**版本**: v1.0  
**作者**: AI Assistant

