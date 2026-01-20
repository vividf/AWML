# Calibrate Batches vs Batch Size 關係說明

## 核心概念

### 兩個參數的定義

1. **`--calibrate-batches`** (或 `num_batches`)
   - **含義**: 用於校準的 **batch 數量**
   - **單位**: 個數（batches）
   - **作用**: 決定從 dataloader 中取多少個 batch 進行校準

2. **`--batch-size`**
   - **含義**: 每個 batch 包含的 **樣本數量**
   - **單位**: 個數（samples per batch）
   - **作用**: 決定每個 batch 包含多少個樣本

### 關鍵關係

```
總校準樣本數 = calibrate-batches × batch-size
```

## 實際示例

### 示例 1: batch-size=1, calibrate-batches=938

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --calibrate-batches 938 \
    --batch-size 1 \
    ...
```

**結果**:
- 每個 batch 包含 **1 個樣本**
- 使用 **938 個 batches**
- **總校準樣本數** = 938 × 1 = **938 個樣本**

**執行過程**:
```
Batch 0: [sample_0]           → 1 個樣本
Batch 1: [sample_1]           → 1 個樣本
Batch 2: [sample_2]           → 1 個樣本
...
Batch 937: [sample_937]       → 1 個樣本

總計: 938 batches × 1 sample/batch = 938 samples
```

### 示例 2: batch-size=4, calibrate-batches=235

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --calibrate-batches 235 \
    --batch-size 4 \
    ...
```

**結果**:
- 每個 batch 包含 **4 個樣本**
- 使用 **235 個 batches**
- **總校準樣本數** = 235 × 4 = **940 個樣本**（接近 938）

**執行過程**:
```
Batch 0: [sample_0, sample_1, sample_2, sample_3]     → 4 個樣本
Batch 1: [sample_4, sample_5, sample_6, sample_7]     → 4 個樣本
Batch 2: [sample_8, sample_9, sample_10, sample_11]  → 4 個樣本
...
Batch 234: [sample_936, sample_937, sample_938, sample_939] → 4 個樣本

總計: 235 batches × 4 samples/batch = 940 samples
```

### 示例 3: batch-size=16, calibrate-batches=59

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --calibrate-batches 59 \
    --batch-size 16 \
    ...
```

**結果**:
- 每個 batch 包含 **16 個樣本**
- 使用 **59 個 batches**
- **總校準樣本數** = 59 × 16 = **944 個樣本**（接近 938）

**執行過程**:
```
Batch 0: [sample_0, ..., sample_15]        → 16 個樣本
Batch 1: [sample_16, ..., sample_31]       → 16 個樣本
...
Batch 58: [sample_928, ..., sample_943]    → 16 個樣本

總計: 59 batches × 16 samples/batch = 944 samples
```

## 代碼實現

### 關鍵代碼位置

**文件**: `tools/detection3d/centerpoint_quantization.py`

```python
# 第 320 行: 設置 batch_size
cfg.val_dataloader['batch_size'] = args.batch_size

# 第 351-354 行: 傳遞 num_batches
calibrator.calibrate(
    dataloader,
    num_batches=args.calibrate_batches,  # ← 這裡是 batch 數量
    method="mse",
)
```

**文件**: `projects/CenterPoint/quantization/calibration/calibrator.py`

```python
# 第 118-121 行: 實際處理邏輯
pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Calibrating")
for i, batch in pbar:
    if i >= num_batches:  # ← 檢查是否達到指定的 batch 數量
        break
    # 處理這個 batch（包含 batch_size 個樣本）
    model.test_step(batch)
```

## 如何選擇參數組合

### 目標：使用相同數量的樣本進行校準

如果要使用 **938 個樣本**進行校準，可以選擇：

| Batch Size | Calibrate Batches | 總樣本數 | 說明 |
|------------|------------------|---------|------|
| 1 | 938 | 938 | 原始設置，seed 敏感性最高 |
| 2 | 469 | 938 | seed 敏感性降低 |
| 4 | 235 | 940 | **推薦**，平衡點 |
| 8 | 118 | 944 | seed 敏感性較低 |
| 16 | 59 | 944 | seed 敏感性低，但內存占用高 |

### 計算公式

```python
# 給定目標樣本數 total_samples
calibrate_batches = ceil(total_samples / batch_size)

# 例如：目標 938 個樣本
# batch_size = 4 → calibrate_batches = ceil(938/4) = 235
# batch_size = 16 → calibrate_batches = ceil(938/16) = 59
```

## 實際使用建議

### 場景 1: 保持與 CUDA-CenterPoint 一致

CUDA-CenterPoint 使用 `batch_size=4, calibrate_batch=400`:
```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --calibrate-batches 400 \
    --batch-size 4 \
    ...
```
**總樣本數**: 400 × 4 = 1600 個樣本

### 場景 2: 使用固定樣本數（如 938）

```bash
# 選項 A: batch_size=1 (原始)
python ... --calibrate-batches 938 --batch-size 1

# 選項 B: batch_size=4 (推薦)
python ... --calibrate-batches 235 --batch-size 4  # 235×4=940

# 選項 C: batch_size=16 (內存充足時)
python ... --calibrate-batches 59 --batch-size 16   # 59×16=944
```

### 場景 3: 減少 seed 敏感性

使用較大的 batch_size，但保持相同的總樣本數：
```bash
# 原始: batch_size=1, calibrate_batches=938
# 改進: batch_size=4, calibrate_batches=235
# 效果: seed 敏感性從 ⭐⭐⭐⭐⭐ 降到 ⭐⭐⭐⭐
```

## 注意事項

### 1. 樣本數量的微小差異

由於 `calibrate_batches` 必須是整數，可能會有微小差異：

```
目標: 938 個樣本
batch_size=4: 235 batches × 4 = 940 samples (+2)
batch_size=16: 59 batches × 16 = 944 samples (+6)
```

**影響**: 通常可以忽略，對校準結果影響很小。

### 2. 內存限制

- `batch_size` 越大，每個 batch 的內存占用越高
- 需要根據 GPU 內存選擇合適的 `batch_size`

### 3. 校準時間

- 總樣本數相同時，`batch_size` 越大，batch 數量越少
- 但每個 batch 的處理時間可能略長（因為樣本更多）
- **總時間大致相同**

## 總結

### 關鍵公式

```
總校準樣本數 = calibrate-batches × batch-size
```

### 選擇建議

1. **固定總樣本數**（如 938）:
   - `batch_size=1` → `calibrate_batches=938`
   - `batch_size=4` → `calibrate_batches=235` ⭐ **推薦**
   - `batch_size=16` → `calibrate_batches=59`

2. **減少 seed 敏感性**:
   - 使用較大的 `batch_size`（4 或更大）
   - 相應調整 `calibrate_batches` 以保持總樣本數

3. **內存受限**:
   - 使用較小的 `batch_size`（1 或 2）
   - 增加 `calibrate_batches` 以達到目標樣本數
