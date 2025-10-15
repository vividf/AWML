# 快速開始：導出 Tier4 兼容格式的 ONNX

## TL;DR (太長不看版)

```bash
# 1. 使用新方法導出 ONNX（與舊方法格式相同）
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_opt_elan_deployment

# 2. 驗證 ONNX 模型
python projects/YOLOX_opt_elan/deploy/verify_onnx.py \
    work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx

# 3. (可選) 與舊方法生成的 ONNX 對比
python projects/YOLOX_opt_elan/deploy/verify_onnx.py \
    work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx \
    --reference work_dirs/old_yolox_elan/yolox_s_opt_elan_batch_6.onnx
```

## 解決方案說明

### 問題
你原先使用兩種方法導出 ONNX：
1. **新方法** (`deploy/main.py`): 輸出原始 head 結果（未解碼）
2. **舊方法** (`deploy_yolox_s_opt.py`): 輸出解碼後的檢測結果

這導致兩個 ONNX 模型的輸出格式完全不同。

### 解決方案
我已經修改新方法，添加了一個 `YOLOXONNXWrapper`，讓新方法也能導出與舊方法相同格式的 ONNX。

### 核心修改
1. **新增 `onnx_wrapper.py`**: 包裝器模型，添加解碼層
2. **修改 `deploy/main.py`**: 支持選擇性使用解碼層
3. **更新 `deploy_config.py`**: 添加 `decode_in_inference` 選項

## 使用示例

### 1. 導出 ONNX（Tier4 兼容格式）

確保 `deploy_config.py` 中設置：

```python
# ONNX configuration
onnx_config = dict(
    ...
    decode_in_inference=True,  # 重要！啟用解碼層
    ...
)
```

然後運行：

```bash
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_opt_elan_deployment
```

### 2. 驗證輸出格式

```bash
python projects/YOLOX_opt_elan/deploy/verify_onnx.py \
    work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx
```

預期輸出：
```
📤 Outputs:
  - outputs: ['batch_size', 'num_predictions', 7] (dtype=1)

🎯 Detected format: [batch, num_predictions, 7]
   Format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
```

### 3. Python 推理示例

```python
import onnxruntime as ort
import numpy as np

# 加載模型
sess = ort.InferenceSession("work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx")

# 準備輸入（假設已經預處理）
# input_image: [1, 3, 960, 960], dtype=float32
input_image = np.random.randn(1, 3, 960, 960).astype(np.float32)

# 推理
outputs = sess.run(None, {"images": input_image})[0]

# 輸出格式: [batch_size, num_predictions, 7]
# 7 = [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
print(f"Output shape: {outputs.shape}")  # [1, ~8400, 7]

# 過濾高置信度檢測
batch_idx = 0
detections = outputs[batch_idx]  # [num_predictions, 7]

# 計算最終分數: obj_conf * cls_conf
final_scores = detections[:, 4] * detections[:, 5]
high_conf_mask = final_scores > 0.5

# 獲取高置信度檢測
high_conf_dets = detections[high_conf_mask]

print(f"High confidence detections: {len(high_conf_dets)}")
for det in high_conf_dets:
    x1, y1, x2, y2 = det[:4]
    obj_conf = det[4]
    cls_conf = det[5]
    cls_id = int(det[6])
    final_score = obj_conf * cls_conf

    print(f"Class {cls_id}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], score={final_score:.3f}")
```

## 輸出格式說明

### 新方法（decode_in_inference=True）
```
Shape: [batch_size, num_predictions, 7]

每個檢測框包含 7 個值：
- [0:4]: bbox 坐標 [x1, y1, x2, y2]
- [4]:   objectness confidence (目標置信度)
- [5]:   class confidence (類別置信度)
- [6]:   class id (類別 ID, 0-7)

最終分數 = objectness × class_confidence
```

### 與舊方法的對比

| 項目 | 舊方法 | 新方法 | 一致性 |
|------|--------|--------|--------|
| 輸出形狀 | [B, N, 7] | [B, N, 7] | ✅ 相同 |
| 輸出格式 | [x1,y1,x2,y2,obj,cls,id] | [x1,y1,x2,y2,obj,cls,id] | ✅ 相同 |
| Batch Size | 固定（如 6） | 動態 | ✅ 更好 |
| 需要轉換 checkpoint | 是 | 否 | ✅ 更好 |
| 需要安裝 Tier4 YOLOX | 是 | 否 | ✅ 更好 |

## 配置選項

在 `deploy_config.py` 中：

```python
onnx_config = dict(
    # ... 其他配置 ...

    # 解碼選項
    decode_in_inference=True,  # True: Tier4 格式, False: 原始輸出
    score_thr=0.01,           # 分數閾值
    nms_thr=0.65,             # NMS IoU 閾值
    max_per_img=300,          # 每張圖片最大檢測數
)
```

## 故障排除

### Q: 輸出形狀不是 [B, N, 7]
**A:** 檢查 `deploy_config.py` 中 `decode_in_inference=True`

### Q: 推理速度很慢
**A:**
1. 檢查是否使用 GPU: `ort.InferenceSession(..., providers=['CUDAExecutionProvider'])`
2. 考慮轉換為 TensorRT 以獲得更好性能

### Q: 檢測結果與舊方法不完全一致
**A:**
1. 確認使用相同的 checkpoint
2. 檢查預處理是否一致
3. Batch size 可能影響結果（建議都用 batch_size=1 對比）

## 完整工作流程

```bash
# 步驟 1: 導出 ONNX
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_new_deploy

# 步驟 2: 驗證 ONNX
python projects/YOLOX_opt_elan/deploy/verify_onnx.py \
    work_dirs/yolox_new_deploy/yolox_opt_elan.onnx

# 步驟 3: (可選) 轉換為 TensorRT
# 修改 deploy_config.py: mode='trt' 或 'both'
python projects/YOLOX_opt_elan/deploy/main.py \
    ... (same as step 1) ...

# 步驟 4: (可選) 運行評估
# 修改 deploy_config.py: evaluation.enabled=True
python projects/YOLOX_opt_elan/deploy/main.py \
    ... (same as step 1) ...
```

## 總結

✅ **優點**：
- 輸出格式與舊方法完全一致
- 無需轉換 checkpoint 格式
- 支持動態 batch size
- 無需安裝額外依賴（Tier4 YOLOX）
- 統一的部署框架

🎯 **建議**：使用新方法（`decode_in_inference=True`）替代舊方法！

## 相關文檔

- 詳細說明：`DECODING_GUIDE.md`
- 原始 README：`README.md`
