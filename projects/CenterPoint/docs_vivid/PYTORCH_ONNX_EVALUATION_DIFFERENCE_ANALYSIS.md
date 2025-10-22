# PyTorch vs ONNX 評估差異分析報告

## 📊 問題概述

PyTorch 和 ONNX 的評估結果存在顯著差異：

| 指標 | PyTorch | ONNX (手動解碼) | 差異 |
|------|---------|-----------------|------|
| mAP | 0.4400 | 0.5082 | +15.5% |
| 預測數量 | 64 | 100 | +56% |
| car AP | 0.5091 | 0.2794 | -45.1% |
| truck AP | 0.5455 | 0.2616 | -52.0% |
| bus AP | 1.0000 | 1.0000 | 0% |
| bicycle AP | 0.0000 | 1.0000 | +∞ |

## 🔍 根本原因分析

### 1. 後處理邏輯不同

**PyTorch 評估**：
```python
# 使用模型內置的完整後處理流程
output = model(input_data)  # 返回 Det3DDataSample
# 內部已經完成：
# - CenterPointBBoxCoder 解碼
# - NMS（非極大值抑制）
# - 分數閾值過濾
# 最終只保留高質量預測（64個）
```

**ONNX 評估**：
```python
# 使用自定義的簡化解碼
head_outputs = onnx_model(input_data)  # 返回原始 head outputs
# 手動處理：
# - Top-K 選擇（100個）
# - 手動坐標/維度解碼
# - 沒有 NMS
# - 沒有分數閾值過濾
# 保留所有 top-100 預測
```

### 2. 第一個預測對比（Truck）

| 屬性 | PyTorch | ONNX | 差異 | 說明 |
|------|---------|------|------|------|
| x | 21.88 | 21.87 | ~0m | ✅ 一致 |
| y | -61.58 | -61.59 | ~0m | ✅ 一致 |
| **z** | **-1.41** | **0.56** | **~2m** | ❌ 嚴重不一致 |
| w | 12.22 | 12.21 | ~0m | ✅ 一致 |
| l | 2.59 | 2.59 | ~0m | ✅ 一致 |
| h | 3.90 | 3.90 | ~0m | ✅ 一致 |
| **yaw** | **1.50** | **0.07** | **1.43 rad (82°)** | ❌ 嚴重不一致 |

### 3. 關鍵差異來源

#### 3.1 z 坐標差異（~2米）

**可能原因**：
1. PyTorch 的 z 坐標可能包含額外的偏移或基準調整
2. ONNX 直接從 `height` 輸出讀取，沒有應用相同的變換
3. PyTorch 模型在後處理中可能應用了地面高度校正

**位置**：
- ONNX: `evaluator.py` Line 813
  ```python
  z = height[b, 0, y_idx, x_idx].item()
  ```
- PyTorch: 內置於 `CenterPointBBoxCoder.decode()`

####3.2 yaw 角度差異（82度）

**可能原因**：
1. ONNX 和 PyTorch 使用不同的旋轉參考系
2. `rot_y_axis_reference=False` 的解碼可能不完整
3. PyTorch 的後處理可能應用了額外的角度變換

**位置**：
- ONNX: `evaluator.py` Line 821-827
  ```python
  rot_sin = rot[b, 1, y_idx, x_idx].item()
  rot_cos = rot[b, 0, y_idx, x_idx].item()
  yaw = np.arctan2(rot_sin, rot_cos)
  # 條件性轉換（目前 rot_y_axis_reference=False，不轉換）
  ```
- PyTorch: 內置於 `CenterPointBBoxCoder.decode()`

#### 3.3 NMS 缺失

**影響**：
- ONNX 保留 100 個預測（包含重複和低質量）
- PyTorch 只保留 64 個高質量預測
- 導致 mAP 計算基準不同

---

## 🛠️ 解決方案

### 方案 1: 使用 PyTorch 模型進行 ONNX 後處理 ⭐⭐⭐⭐⭐

**描述**: 讓 ONNX backend 使用 PyTorch 模型的完整後處理流程

**實施**:
1. ONNX 只負責推理到 head outputs
2. 將 ONNX head outputs 轉換為 torch tensors
3. 調用 PyTorch 模型的後處理（包括 bbox_coder, NMS等）

**優點**:
- ✅ 100% 與 PyTorch 一致
- ✅ 自動處理所有變換和過濾
- ✅ 易於維護

**缺點**:
- ❌ ONNX 輸出格式需要與 PyTorch 完全兼容
- ❌ 當前發現 z 和 yaw 不兼容

**狀態**: ✅ **已成功實施！** （使用 `predict_by_feat` 替代 `bbox_coder.decode`）

**更新**: 2025-10-22
- 使用 `predict_by_feat` 方法成功統一 PyTorch 和 ONNX 評估
- mAP 完全一致：0.4400 vs 0.4400
- 所有類別 AP 完全一致
- 詳見：`PYTORCH_ONNX_UNIFIED_SOLUTION.md`

### 方案 2: 修正手動解碼邏輯 ⭐⭐⭐

**描述**: 改進 ONNX 的手動解碼，使其與 PyTorch 一致

**需要修正**:
1. **z 坐標計算**：
   - 研究 PyTorch 的 z 坐標計算邏輯
   - 應用相同的偏移和變換

2. **yaw 角度計算**：
   - 研究 PyTorch 的角度解碼邏輯
   - 確認 `rot` 輸出的實際格式（sin/cos 順序）

3. **添加 NMS**：
   - 實施與 PyTorch 相同的 NMS 算法
   - 應用相同的分數閾值

**優點**:
- ✅ 完全控制解碼邏輯
- ✅ 可以逐步調試和優化

**缺點**:
- ❌ 需要深入研究 PyTorch 內部實現
- ❌ 維護成本高
- ❌ 容易出錯

**狀態**: ⚠️ 部分完成（x, y, w, l, h 已一致）

### 方案 3: 接受差異，統一 metric 計算 ⭐⭐

**描述**: 承認 PyTorch 和 ONNX 的後處理不同，但確保 metric 計算一致

**實施**:
1. 為 PyTorch 和 ONNX 使用相同的 metric 計算代碼
2. 標準化預測格式
3. 接受預測數量不同，但期望 mAP 相近（±5%）

**優點**:
- ✅ 現實可行
- ✅ 關注最終性能指標

**缺點**:
- ❌ 不能保證完全一致
- ❌ 仍需修正 z 和 yaw

**狀態**: ⚠️ 可行，但需先修正 z 和 yaw

---

## 📋 當前狀態

### 已完成 ✅
1. 深拷貝配置，避免污染
2. 正確的坐標轉換（x, y）
3. 正確的維度解碼（w, l, h + exp()）
4. 條件性 w/l 交換

### 待解決 ❌
1. **z 坐標計算不一致**（差 2米）
2. **yaw 角度計算不一致**（差 82度）
3. **缺少 NMS**（導致預測數量不同）

### 測試結果

**手動解碼（當前）**:
- ONNX mAP: 0.5082 vs PyTorch: 0.4400
- 部分類別 AP 差異大（car, truck）
- 部分類別完全一致（bus）

**PyTorch decoder（已禁用）**:
- 所有 AP = 0（因為 z 和 yaw 不一致）

---

## 🎯 建議行動

### 短期（立即）

1. **調試 z 坐標**：
   ```python
   # 檢查 PyTorch 的 z 計算
   - 閱讀 CenterPointBBoxCoder 源碼
   - 對比 ONNX 的 height 輸出
   - 找出差異來源
   ```

2. **調試 yaw 角度**：
   ```python
   # 檢查 PyTorch 的 yaw 計算
   - 確認 rot 輸出格式（sin/cos 順序）
   - 檢查是否需要額外變換
   - 驗證 rot_y_axis_reference 的影響
   ```

3. **添加調試日誌**：
   ```python
   # 在 PyTorch 評估中添加日誌
   print(f"PyTorch z: {z}, yaw: {yaw}")
   print(f"ONNX height raw: {height_raw}")
   print(f"ONNX rot raw: sin={sin}, cos={cos}")
   ```

### 中期（1-2天）

1. **實施 NMS**：
   - 使用 `torchvision.ops.nms` 或自定義實現
   - 應用與 PyTorch 相同的參數

2. **完整測試**：
   - 在完整數據集（19個樣本）上測試
   - 驗證 mAP 差異 < 5%

### 長期（重構）

1. **統一後處理**：
   - 創建共享的後處理模塊
   - PyTorch 和 ONNX 都使用相同邏輯

2. **改進 ONNX 導出**：
   - 確保 ONNX 輸出格式與 PyTorch 完全兼容
   - 包含必要的變換和偏移

---

## 🔬 調試建議

### 1. 對比單個預測

```python
# 在 PyTorch 評估中
print(f"PyTorch prediction 0:")
print(f"  bbox: {bbox}")
print(f"  z_before_transform: {z_raw}")
print(f"  yaw_before_transform: {yaw_raw}")

# 在 ONNX 評估中  
print(f"ONNX prediction 0:")
print(f"  bbox: {bbox}")
print(f"  height_raw: {height[y_idx, x_idx]}")
print(f"  rot_raw: sin={rot[1, y_idx, x_idx]}, cos={rot[0, y_idx, x_idx]}")
```

### 2. 檢查中間值

```python
# 添加到 _parse_centerpoint_head_outputs
print(f"Input height range: {height.min():.3f} to {height.max():.3f}")
print(f"Input rot range: {rot.min():.3f} to {rot.max():.3f}")
print(f"Selected height at ({y_idx}, {x_idx}): {height[0, 0, y_idx, x_idx]}")
print(f"Selected rot at ({y_idx}, {x_idx}): sin={rot[0, 1, y_idx, x_idx]}, cos={rot[0, 0, y_idx, x_idx]}")
```

### 3. 單元測試

```python
# 創建測試用例
def test_z_calculation():
    height_onnx = 0.56
    z_pytorch = -1.41
    # 找出變換
    offset = z_pytorch - height_onnx  # -1.97
    print(f"z offset: {offset}")
```

---

## 📝 結論

1. **主要問題**：z 坐標和 yaw 角度的計算邏輯不一致
2. **次要問題**：缺少 NMS 導致預測數量不同
3. **建議方案**：修正手動解碼邏輯（方案 2）
4. **預期效果**：修正後 ONNX mAP 應與 PyTorch 在 ±5% 範圍內

---

**創建日期**: 2025-10-22  
**狀態**: 📝 待修正  
**優先級**: 🔴 高（影響部署質量）

