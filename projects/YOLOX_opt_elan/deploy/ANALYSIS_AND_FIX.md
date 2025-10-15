# YOLOX_opt_elan ONNX 導出問題分析與修正

## 📊 問題分析步驟

### 步驟 1: 分析舊方法 ONNX (Tier4)

**文件**: `work_dirs/old_yolox_elan/yolox_s_opt_elan_batch_6.onnx`

**關鍵發現**:
```
輸入:  images, shape [6, 3, 960, 960]
輸出:  output, shape [6, 18900, 13]
節點數: 245
主要操作: Conv(106), Relu(97), Sigmoid(6), Concat(16), Reshape(3), Transpose(1)
```

**輸出結構**:
- 18900 = 30×30 + 60×60 + 120×120 (三個檢測層的總錨點數)
- 13 = bbox_reg(4) + objectness(1) + class_scores(8)

**最後幾個節點**:
1. 每個檢測層: `Concat([reg_preds, Sigmoid(obj_preds), Sigmoid(cls_preds)])`
2. Reshape 每層到 `[B, -1, 13]`
3. Concat 所有層 → `[B, 18900, 13]`
4. Transpose → 最終輸出

**關鍵點**:
- ✅ bbox_reg 是**原始回歸輸出**，未解碼
- ✅ objectness 和 class_scores 經過 **sigmoid**
- ✅ 簡單的 concat 操作，無複雜解碼

---

### 步驟 2: 分析原始新方法 ONNX

**文件**: `work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx` (第一版)

**關鍵發現**:
```
輸入:  images, shape ['batch_size', 3, 960, 960]
輸出:  outputs, shape ['batch_size', 'ScatterNDoutputs_dim_1', 'ScatterNDoutputs_dim_2']
節點數: 1016 (!!!)
主要操作: Constant(403), Conv(106), Clip(97), Concat(53), Shape(53), Unsqueeze(45), ...
```

**問題**:
- ❌ 輸出形狀未定義（動態形狀處理失敗）
- ❌ 節點數過多 (1016 vs 245)
- ❌ 包含大量動態形狀處理操作 (Shape, Reshape, ScatterND, etc.)
- ❌ 複雜的 bbox 解碼邏輯導致 ONNX 圖過於複雜

**根本原因**:
第一版 wrapper 試圖在 ONNX 中實現完整的 bbox 解碼，包括：
- 生成 priors/anchors
- 解碼 bbox (center + exp(size))
- 計算最終坐標
這導致 PyTorch 的動態操作在導出時生成了大量冗餘節點。

---

### 步驟 3: 理解 Tier4 YOLOX 的實現

通過分析舊 ONNX 的節點結構，發現：

**Tier4 的實現邏輯**:
```python
# 對每個檢測層 (3層)
for level in [0, 1, 2]:
    reg_pred = Conv(features)        # [B, 4, H, W]
    obj_pred = Conv(features)        # [B, 1, H, W]
    cls_pred = Conv(features)        # [B, 8, H, W]

    # 只對 obj 和 cls 做 sigmoid，reg 保持原始
    obj_pred = Sigmoid(obj_pred)
    cls_pred = Sigmoid(cls_pred)

    # Concat: [reg(4), obj(1), cls(8)]
    level_output = Concat([reg_pred, obj_pred, cls_pred])  # [B, 13, H, W]

    # Reshape
    level_output = Reshape(level_output, [B, 13, H*W])

# Concat 所有層
all_outputs = Concat([level0, level1, level2])  # [B, 13, 18900]

# Transpose
output = Transpose(all_outputs)  # [B, 18900, 13]
```

**關鍵洞察**:
- 不做 bbox 解碼！
- 只做簡單的 sigmoid 和 concat
- 輸出的是原始預測值，解碼留給後處理

---

## 🔧 修正方案

### 修正後的 `onnx_wrapper.py`

```python
class YOLOXONNXWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int = 8):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.bbox_head = model.bbox_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 取得特徵
        feat = self.model.backbone(x)
        if hasattr(self.model, 'neck') and self.model.neck is not None:
            feat = self.model.neck(feat)

        # 2. 取得 head 輸出
        cls_scores, bbox_preds, objectnesses = self.bbox_head(feat)

        # 3. 處理每個檢測層
        mlvl_outputs = []
        for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
            batch_size = cls_score.shape[0]

            # Reshape: [B, C, H, W] → [B, C, H*W]
            cls_score = cls_score.reshape(batch_size, self.num_classes, -1)
            bbox_pred = bbox_pred.reshape(batch_size, 4, -1)
            objectness = objectness.reshape(batch_size, 1, -1)

            # Sigmoid (只對 objectness 和 cls_score)
            cls_score = cls_score.sigmoid()
            objectness = objectness.sigmoid()

            # Concat: [bbox(4), objectness(1), cls(num_classes)]
            level_output = torch.cat([bbox_pred, objectness, cls_score], dim=1)
            mlvl_outputs.append(level_output)

        # 4. Concat 所有層: [B, 13, 18900]
        all_outputs = torch.cat(mlvl_outputs, dim=2)

        # 5. Transpose: [B, 18900, 13]
        all_outputs = all_outputs.permute(0, 2, 1).contiguous()

        return all_outputs
```

###修正要點:
1. ✅ **不做 bbox 解碼**: bbox_pred 保持原始值
2. ✅ **只做 sigmoid**: 對 objectness 和 class_scores
3. ✅ **簡單操作**: Reshape → Sigmoid → Concat → Transpose
4. ✅ **避免動態操作**: 不使用 grid_priors, max, argmax 等

---

## ✅ 驗證結果

### 邏輯測試

運行 `simple_wrapper_test.py`:

```
Parameters:
  Batch size: 6
  Num classes: 8
  Output channels: 4 (bbox) + 1 (obj) + 8 (cls) = 13

1. Simulating YOLOX outputs...
   Level 1: 30x30 = 900 anchors
   Level 2: 60x60 = 3600 anchors
   Level 3: 120x120 = 14400 anchors
   Total anchors: 18900

2. Processing outputs...
   Output shape: torch.Size([6, 18900, 13])
   Expected: [6, 18900, 13]
   ✅ Shape matches!

3. Comparison with old ONNX:
   Old ONNX output: [6, 18900, 13]
   New output:      [6, 18900, 13]
   ✅ Matches old ONNX format!

4. Output content verification:
   Objectness in [0,1]? True
   Class scores in [0,1]? True

✅ Logic Test Complete!
```

---

## 📝 配置更新

### `deploy_config.py` 更新:

```python
onnx_config = dict(
    opset_version=11,  # 改為 11 (匹配 Tier4)
    input_names=["images"],
    output_names=["output"],  # 改為 "output" (匹配 Tier4)
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"},  # 更新名稱
    },
    decode_in_inference=True,  # 使用新的簡化 wrapper
)
```

### `deploy/main.py` 更新:

- 簡化 wrapper 調用，移除不需要的參數
- 輸出名稱改為 "output" (匹配 Tier4)
- 更新日誌信息說明輸出格式

---

## 📊 新舊方法對比

| 項目 | 舊方法 (Tier4) | 原始新方法 | 修正後新方法 |
|------|---------------|-----------|-------------|
| 輸入形狀 | `[6, 3, 960, 960]` | `['batch', 3, 960, 960]` | `['batch', 3, 960, 960]` |
| 輸出形狀 | `[6, 18900, 13]` | 動態（未定義） | `['batch', 18900, 13]` ✅ |
| 輸出內容 | `[reg(4), obj(1), cls(8)]` | `[x1,y1,x2,y2, obj,cls_max,id]` | `[reg(4), obj(1), cls(8)]` ✅ |
| 節點數 | 245 | 1016 | ~250 (預估) ✅ |
| Batch Size | 固定 (6) | 動態 | 動態 ✅ |
| Opset 版本 | 11 | 16 | 11 ✅ |
| Bbox 解碼 | 否 | 是 | 否 ✅ |
| 複雜度 | 簡單 | 複雜 | 簡單 ✅ |

---

## 🚀 使用新方法

### 導出 ONNX:

```bash
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    work_dirs/old_yolox_elan/yolox_epoch24.pth \
    --work-dir work_dirs/yolox_opt_elan_new
```

### 預期輸出:

```
ONNX Model Info:
  Input: images, shape ['batch_size', 3, 960, 960]
  Output: output, shape ['batch_size', 18900, 13]
  Total nodes: ~245

Output format: [batch_size, 18900, 13]
  where 13 = [bbox_reg(4), objectness(1), class_scores(8)]
  Note: bbox_reg are raw outputs (NOT decoded coordinates)
```

---

## 🎯 關鍵結論

### 問題根源:
原始 wrapper 試圖在 ONNX 中實現完整的目標檢測後處理（bbox 解碼、argmax、過濾等），導致：
1. 生成過多動態操作節點
2. 輸出形狀無法確定
3. ONNX 圖過於複雜

### 解決方案:
模仿 Tier4 YOLOX 的簡單設計：
1. **只輸出原始預測**：不做 bbox 解碼
2. **只做基本操作**：Sigmoid + Concat + Transpose
3. **後處理分離**：將 bbox 解碼、NMS 等留給推理端

### 優勢:
- ✅ 輸出格式與 Tier4 完全一致
- ✅ ONNX 圖簡單高效
- ✅ 支持動態 batch size
- ✅ 無需轉換 checkpoint 格式
- ✅ 易於維護和部署

---

## 📚 相關文件

- `onnx_wrapper.py`: 修正後的 wrapper 實現
- `deploy_config.py`: 更新後的配置
- `analyze_onnx.py`: ONNX 分析工具
- `simple_wrapper_test.py`: 邏輯驗證測試
- `DECODING_GUIDE.md`: 詳細技術文檔
- `QUICK_START_DECODED.md`: 快速開始指南

---

**修正完成日期**: 2025-10-10  
**狀態**: ✅ 邏輯驗證通過，等待完整 ONNX 導出測試
