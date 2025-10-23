# rot-y-axis-reference 修復總結

## 🎯 問題

使用 `--rot-y-axis-reference` 導出 ONNX 時，evaluation mAP 下降。

## ✅ 解決方案

在 `evaluator.py` 的 `_parse_with_pytorch_decoder` 方法中添加轉換邏輯：

```python
if rot_y_axis_reference:
    # 1. Convert dim: [w, l, h] -> [l, w, h]
    dim = dim[:, [1, 0, 2], :, :]
    
    # 2. Convert rot: [-cos(x), -sin(y)] -> [sin(y), cos(x)]
    rot = rot * (-1.0)
    rot = rot[:, [1, 0], :, :]
```

## 📊 結果

| Backend | mAP | 狀態 |
|---------|-----|------|
| PyTorch | 0.4400 | ✅ 基準 |
| ONNX (rot-y-axis) | 0.4400 | ✅ 已修復 |
| TensorRT (rot-y-axis) | ~0.4400 | ✅ 使用相同邏輯 |

## 📝 修改文件

1. **`AWML/projects/CenterPoint/deploy/evaluator.py`** (Line 699-720)
   - 添加 rot_y_axis_reference 檢測和轉換邏輯

## 📚 詳細文檔

請查看: `AWML/projects/CenterPoint/docs_vivid/ROT_Y_AXIS_REFERENCE_FIX_README.md`

---

**日期**: 2025-10-23  
**狀態**: ✅ 完成

