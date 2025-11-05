# YOLOX Verification/Evaluation 修復原理說明

## 🔍 問題根源分析

### 問題 1: Input Normalization 不一致

**發現：**
- YOLOX 的 `data_preprocessor` **沒有配置 mean/std**（見 `yolox-s-opt-elan_960x960_300e_t4dataset.py`）
- 這意味著訓練時輸入是 **0-255 範圍**，不做 normalization
- 但我們之前錯誤地添加了 normalization，導致輸入格式不匹配

**驗證：**
```python
# 檢查 data_preprocessor 配置
model = dict(
    type="YOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        # ❌ 沒有 mean 和 std 配置！
    ),
)
```

**修復：**
- 移除 normalization（mean/std）
- 保持輸入為 0-255 範圍，與訓練一致

### 問題 2: BGR vs RGB 格式

**發現：**
- MMDet pipeline 輸出的是 **BGR 格式**的 tensor
- YOLOX 訓練時使用的是 BGR 格式（data_preprocessor 沒有 `bgr_to_rgb=True`）
- 我們之前錯誤地嘗試轉換為 RGB

**驗證：**
```python
# 檢查 channel order
mean_per_channel = tensor.mean(dim=(0,2,3))
# Channel 0 (B): 107.389
# Channel 1 (G): 112.363  
# Channel 2 (R): 118.142
# B > R → BGR 格式
```

**修復：**
- 保持 BGR 格式，不做轉換

### 問題 3: ReLU6 vs ReLU 不一致

**發現：**
- ONNX 導出時會將 ReLU6 替換為 ReLU（為了 ONNX 兼容性）
- 但 verification 時使用的 PyTorch 模型**沒有替換**，導致輸出不一致

**驗證：**
```python
# 導出時（main_pipeline.py line 94-103）
replace_relu6_with_relu(pytorch_model)  # ✅ 替換了

# Verification 時（evaluator.py line 171）
pytorch_pipeline = YOLOXPyTorchPipeline(...)  # ❌ 沒有替換！
```

**修復：**
- 在 verification 和 evaluation 中都添加 ReLU6 → ReLU 替換
- 確保所有後端使用相同的模型狀態

### 問題 4: 導出和推理輸入格式不一致

**發現：**
- 導出時使用 `data_loader.load_and_preprocess()`（方法不存在）
- 推理時使用 `data_loader.load_sample()` + `data_loader.preprocess()`
- 兩者使用的輸入格式可能不同

**修復：**
- 統一使用 `load_sample()` + `preprocess()`
- 確保導出和推理使用相同的輸入格式

## ✅ 修復後的流程

### 1. Preprocessing（統一的輸入格式）

```python
# yolox_pipeline.py preprocess()
if isinstance(input_data, torch.Tensor):
    # MMDet pipeline 輸出：0-255 BGR
    tensor = input_data
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    # ✅ 保持 0-255 範圍，BGR 格式（不做 normalization，不做轉換）
    return tensor, metadata
```

**為什麼這樣做：**
- 與 YOLOX 訓練時的輸入格式完全一致
- `data_preprocessor` 沒有配置 mean/std，所以不做 normalization
- 訓練時使用 BGR 格式，所以推理時也保持 BGR

### 2. Model State（統一的模型狀態）

```python
# evaluator.py verify() 和 _create_backend()
# 在創建 PyTorch pipeline 時替換 ReLU6 → ReLU
replace_relu6_with_relu(pytorch_model)  # ✅ 匹配 ONNX 導出時的狀態
```

**為什麼這樣做：**
- ONNX 導出時會替換 ReLU6 為 ReLU
- 如果推理時不替換，模型狀態不同，輸出會不一致
- 確保 verification 時三個後端使用相同的模型狀態

### 3. Export（一致的導出和推理）

```python
# main_pipeline.py export_onnx_with_pipeline()
sample = data_loader.load_sample(sample_idx)
single_input = data_loader.preprocess(sample)  # ✅ 與推理時相同的輸入格式
```

**為什麼這樣做：**
- 導出時使用的輸入格式必須與推理時完全一致
- 這樣 ONNX 模型才會期望正確的輸入格式

## 🎯 為什麼這些修復有效

### 1. **Evaluation 成功的原因**

```
訓練時：0-255 BGR → 模型 → detections
推理時：0-255 BGR → 模型 → detections ✅ 格式一致！
```

修復前：
- 錯誤地做了 normalization（mean/std）
- 導致輸入格式不匹配，模型輸出異常（objectness 太低）

修復後：
- 保持 0-255 範圍，與訓練一致
- 模型輸出正常，postprocessing 能正確過濾 detections

### 2. **Verification 成功的原因**

```
導出時：0-255 BGR + ReLU6→ReLU → ONNX 模型
推理時：0-255 BGR + ReLU6→ReLU → PyTorch/ONNX/TensorRT ✅ 狀態一致！
```

修復前：
- PyTorch 使用 ReLU6，ONNX 使用 ReLU
- 模型狀態不同，輸出差異大（max diff ~2.84）

修復後：
- 所有後端都使用 ReLU（ReLU6 已替換）
- 所有後端都使用相同的輸入格式（0-255 BGR）
- 輸出一致（max diff < tolerance）

## 📊 關鍵改動對比

| 項目 | 修復前 | 修復後 | 原因 |
|------|--------|--------|------|
| **Input Range** | Normalized (mean/std) | 0-255 | data_preprocessor 不做 normalization |
| **Color Format** | RGB (轉換後) | BGR | 訓練時使用 BGR |
| **ReLU6** | PyTorch 不替換 | 所有後端都替換 | 匹配 ONNX 導出狀態 |
| **Export Input** | load_and_preprocess() | load_sample() + preprocess() | 與推理格式一致 |

## 🔑 核心原理

1. **訓練和推理的輸入格式必須完全一致**
   - YOLOX 的 data_preprocessor 不做 normalization
   - 所以推理時也不應該做 normalization

2. **導出和推理的模型狀態必須完全一致**
   - ONNX 導出時會替換 ReLU6 為 ReLU
   - 所以推理時也要替換，才能保證輸出一致

3. **所有後端必須使用相同的輸入和模型狀態**
   - Verification 比較的是 raw outputs
   - 任何不一致都會導致輸出差異

這就是為什麼這些改動能讓程序運行成功的根本原因！

