# YOLOX_opt_elan ONNX Export with Decoding

本指南說明如何使用新方法導出包含解碼層的 ONNX 模型，使其輸出格式與 Tier4 YOLOX 部署方法一致。

## 修改內容

### 1. 新增文件

#### `onnx_wrapper.py`
- 實現 `YOLOXONNXWrapper` 類，將 MMDetection YOLOX 模型包裝起來
- 在模型前向傳播中添加解碼邏輯
- 輸出格式：`[batch_size, num_predictions, 7]`
  - 其中 7 = `[x1, y1, x2, y2, obj_conf, cls_conf, cls_id]`

### 2. 修改文件

#### `deploy/main.py`
- 導入 `YOLOXONNXWrapper`
- 修改 `export_onnx()` 函數，支持選擇性使用解碼層
- 新增 `model_cfg` 參數以獲取類別數量

#### `deploy_config.py`
- 新增 `decode_in_inference` 配置選項（默認為 `True`）
- 新增解碼相關參數：`score_thr`, `nms_thr`, `max_per_img`

## 使用方法

### 導出包含解碼層的 ONNX（新方法，Tier4兼容）

```bash
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_opt_elan_deployment
```

配置文件中確保：
```python
# deploy_config.py
onnx_config = dict(
    ...
    decode_in_inference=True,  # 啟用解碼層
    score_thr=0.01,
    nms_thr=0.65,
    max_per_img=300,
)
```

### 導出原始輸出的 ONNX（無解碼層）

如果需要導出原始輸出格式（與之前一樣），修改配置：

```python
# deploy_config.py
onnx_config = dict(
    ...
    decode_in_inference=False,  # 禁用解碼層
)
```

## 輸出格式對比

### 新方法（decode_in_inference=True）
```
Output Shape: [batch_size, num_predictions, 7]
Format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]

示例：
- batch_size: 1
- num_predictions: 約 8400（960x960 輸入，三個檢測層）
- Shape: [1, 8400, 7]
```

### 舊方法（Tier4 YOLOX）
```
Output Shape: [batch_size, num_predictions, 7]
Format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]

完全相同的格式！
```

### 原始輸出（decode_in_inference=False）
```
Output: Tuple of (cls_scores, bbox_preds, objectnesses)
- cls_scores: List[Tensor], 每層 shape [B, num_classes, H, W]
- bbox_preds: List[Tensor], 每層 shape [B, 4, H, W]
- objectnesses: List[Tensor], 每層 shape [B, 1, H, W]
```

## 驗證一致性

### 使用 Netron 檢查 ONNX 模型

```bash
# 安裝 netron
pip install netron

# 查看新方法生成的 ONNX
netron work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx

# 查看舊方法生成的 ONNX（用於對比）
netron work_dirs/old_yolox_elan/yolox_s_opt_elan_batch_6.onnx
```

檢查項目：
1. **輸入節點**：`images`, shape `[batch_size, 3, 960, 960]`
2. **輸出節點**：`outputs`, shape `[batch_size, num_predictions, 7]`
3. **層數**：新方法應該包含額外的 sigmoid、reshape、concat 等解碼層

### 使用 Python 驗證輸出

```python
import onnx
import numpy as np

# 加載新方法的 ONNX
model_new = onnx.load("work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx")

# 檢查輸入輸出
print("Inputs:")
for input in model_new.graph.input:
    print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

print("\nOutputs:")
for output in model_new.graph.output:
    print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
```

預期輸出：
```
Inputs:
  images: [batch_size, 3, 960, 960]

Outputs:
  outputs: [batch_size, num_predictions, 7]
```

### 數值驗證

運行推理並比較結果：

```python
import onnxruntime as ort
import numpy as np
import cv2

# 準備輸入
img = cv2.imread("test_image.jpg")
img_resized = cv2.resize(img, (960, 960))
img_normalized = img_resized.astype(np.float32) / 255.0
img_transposed = img_normalized.transpose(2, 0, 1)[np.newaxis, ...]

# 新方法推理
sess_new = ort.InferenceSession("work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx")
outputs_new = sess_new.run(None, {"images": img_transposed})[0]

print(f"Output shape: {outputs_new.shape}")
print(f"Sample detection: {outputs_new[0, 0]}")
# 預期輸出: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
```

## 解碼邏輯說明

新方法實現的解碼流程：

1. **提取特徵**：通過 backbone 和 neck 提取多尺度特徵
2. **頭部輸出**：獲取分類分數、bbox 預測、objectness
3. **生成先驗框**：為每個特徵層生成網格先驗框
4. **展平和拼接**：將多層輸出展平並拼接
5. **應用 sigmoid**：對分類分數和 objectness 應用 sigmoid
6. **解碼 bbox**：
   ```python
   center = prior_center + bbox_pred[:2] * prior_stride
   size = exp(bbox_pred[2:]) * prior_stride
   bbox = [center - size/2, center + size/2]  # 轉換為 [x1, y1, x2, y2]
   ```
7. **組合輸出**：
   ```python
   outputs[:, :, 0:4] = bboxes         # bbox 坐標
   outputs[:, :, 4] = objectness       # 目標置信度
   outputs[:, :, 5] = max_cls_scores   # 類別置信度
   outputs[:, :, 6] = labels           # 類別 ID
   ```

## 配置參數說明

### `decode_in_inference`
- **類型**：bool
- **默認值**：True
- **說明**：是否在 ONNX 模型中包含解碼層
- **建議**：True（與 Tier4 YOLOX 一致）

### `score_thr`
- **類型**：float
- **默認值**：0.01
- **說明**：檢測分數閾值，用於後處理時過濾低分檢測
- **注意**：在 ONNX 導出時，這個值主要用於文檔記錄，實際過濾在推理時進行

### `nms_thr`
- **類型**：float
- **默認值**：0.65
- **說明**：NMS IoU 閾值
- **注意**：ONNX 模型不包含 NMS，需要在後處理中實現

### `max_per_img`
- **類型**：int
- **默認值**：300
- **說明**：每張圖片的最大檢測數量

## 與舊方法的差異

| 項目 | 舊方法（Tier4 YOLOX） | 新方法（decode_in_inference=True） |
|------|---------------------|--------------------------------|
| 模型來源 | Tier4 YOLOX | MMDetection YOLOX |
| 權重轉換 | 需要格式轉換 | 直接使用 |
| 解碼位置 | ONNX 內部 | ONNX 內部（通過 wrapper） |
| 輸出格式 | [B, N, 7] | [B, N, 7] ✅ 相同 |
| Batch Size | Static | Dynamic ✅ 更靈活 |
| 依賴 | 需要安裝 Tier4 YOLOX | 只需 MMDetection |

## 故障排除

### 問題：ONNX 導出失敗

**解決方法**：
1. 確認 PyTorch 和 ONNX 版本兼容
2. 檢查輸入張量形狀是否正確
3. 嘗試降低 opset_version（如 11 或 13）

### 問題：輸出形狀不正確

**解決方法**：
1. 檢查 `decode_in_inference` 設置
2. 驗證模型配置中的類別數量
3. 使用 Netron 檢查 ONNX 圖結構

### 問題：推理結果與舊方法不同

**可能原因**：
1. **Batch size 不同**：舊方法可能固定為 6，新方法默認為 1
2. **權重不完全相同**：檢查是否使用相同的 checkpoint
3. **預處理差異**：確認輸入圖像預處理一致

## 測試命令

完整測試流程：

```bash
# 1. 導出 ONNX（新方法）
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_opt_elan_new

# 2. 驗證 ONNX 結構
python -c "
import onnx
model = onnx.load('work_dirs/yolox_opt_elan_new/yolox_opt_elan.onnx')
print('Input:', model.graph.input[0].name, [d.dim_value or d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim])
print('Output:', model.graph.output[0].name, [d.dim_value or d.dim_param for d in model.graph.output[0].type.tensor_type.shape.dim])
"

# 3. 測試推理
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('work_dirs/yolox_opt_elan_new/yolox_opt_elan.onnx')
dummy_input = np.random.randn(1, 3, 960, 960).astype(np.float32)
output = sess.run(None, {'images': dummy_input})[0]
print(f'Output shape: {output.shape}')
print(f'Expected: [1, ~8400, 7]')
print(f'Sample detection: {output[0, 0]}')
"
```

## 總結

新方法通過 `YOLOXONNXWrapper` 實現了與 Tier4 YOLOX 完全兼容的 ONNX 導出格式，同時保持了以下優勢：

✅ 無需格式轉換，直接使用 MMDetection checkpoint  
✅ 支持動態 batch size  
✅ 輸出格式與舊方法一致  
✅ 更易於維護和擴展  
✅ 統一的部署框架  

建議使用新方法進行 ONNX 導出！
