# 使用 --calibrate-samples 自動計算 Batches

## 新功能說明

現在可以直接指定**總樣本數**，系統會自動根據 `batch-size` 計算需要的 batches 數量，無需手動計算。

## 使用方法

### 方法 1: 指定總樣本數（推薦）⭐

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    --checkpoint work_dirs/centerpoint/best.pth \
    --calibrate-samples 938 \
    --batch-size 4 \
    --calib-seed 0 \
    --output work_dirs/centerpoint_ptq.pth
```

**自動計算結果**:
- 目標: 938 個樣本
- Batch size: 4
- **自動計算**: `ceil(938 / 4) = 235 batches`
- **實際樣本數**: 235 × 4 = 940 個樣本

### 方法 2: 手動指定 batches（原有方式）

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --config ... \
    --checkpoint ... \
    --calibrate-batches 235 \
    --batch-size 4 \
    --output ...
```

## 參數說明

### `--calibrate-samples` (新參數)

- **含義**: 總校準樣本數
- **單位**: 個數（samples）
- **作用**: 自動計算需要的 batches 數量

### `--calibrate-batches` (原有參數)

- **含義**: 校準 batch 數量
- **單位**: 個數（batches）
- **優先級**: 如果同時指定 `--calibrate-samples`，會被自動覆蓋

### `--batch-size`

- **含義**: 每個 batch 的樣本數
- **單位**: 個數（samples per batch）
- **默認值**: 1

## 計算邏輯

```python
if --calibrate-samples is specified:
    calibrate_batches = ceil(calibrate_samples / batch_size)
    actual_samples = calibrate_batches × batch_size
else if --calibrate-batches is specified:
    use calibrate_batches directly
else:
    default to 100 batches
```

## 實際示例

### 示例 1: 使用 938 個樣本，batch_size=4

```bash
--calibrate-samples 938 --batch-size 4
```

**輸出**:
```
Auto-calculated: 938 samples → 235 batches × 4 = 940 samples
```

### 示例 2: 使用 938 個樣本，batch_size=16

```bash
--calibrate-samples 938 --batch-size 16
```

**輸出**:
```
Auto-calculated: 938 samples → 59 batches × 16 = 944 samples
```

### 示例 3: 使用 1600 個樣本，batch_size=4

```bash
--calibrate-samples 1600 --batch-size 4
```

**輸出**:
```
Auto-calculated: 1600 samples → 400 batches × 4 = 1600 samples
```

## 常見使用場景

### 場景 1: 固定樣本數，減少 seed 敏感性

```bash
# 目標: 使用 938 個樣本
# 選項 A: batch_size=1 (原始，seed 敏感性高)
--calibrate-samples 938 --batch-size 1
# → 938 batches × 1 = 938 samples

# 選項 B: batch_size=4 (推薦) ⭐
--calibrate-samples 938 --batch-size 4
# → 235 batches × 4 = 940 samples

# 選項 C: batch_size=16 (內存充足時)
--calibrate-samples 938 --batch-size 16
# → 59 batches × 16 = 944 samples
```

### 場景 2: 與 CUDA-CenterPoint 保持一致

CUDA-CenterPoint 使用 1600 個樣本（400 batches × 4）:
```bash
--calibrate-samples 1600 --batch-size 4
# → 400 batches × 4 = 1600 samples
```

## 注意事項

1. **樣本數量的微小差異**:
   - 由於 `ceil()` 向上取整，實際樣本數可能略多於目標
   - 例如: 目標 938，batch_size=4 → 實際 940（+2）
   - 影響很小，通常可忽略

2. **參數優先級**:
   - 如果同時指定 `--calibrate-samples` 和 `--calibrate-batches`
   - `--calibrate-samples` 優先，會自動覆蓋 `--calibrate-batches`

3. **默認行為**:
   - 如果兩個參數都不指定，默認使用 100 batches

## 完整命令示例

```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config_int8.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --calibrate-samples 938 \
    --batch-size 4 \
    --calib-seed 0 \
    --output work_dirs/centerpoint_ptq_938_batch4_seed0.pth
```

**執行輸出**:
```
Auto-calculated: 938 samples → 235 batches × 4 = 940 samples
================================================================================
CenterPoint PTQ Quantization
================================================================================
Config: projects/CenterPoint/configs/...
Checkpoint: work_dirs/centerpoint/best_checkpoint.pth
Calibration batches: 235
Batch size: 4
...
```

## 優勢

✅ **無需手動計算**: 直接指定樣本數，系統自動計算 batches  
✅ **更直觀**: 樣本數比 batches 數更容易理解  
✅ **減少錯誤**: 避免手動計算錯誤  
✅ **向後兼容**: 仍支持原有的 `--calibrate-batches` 參數
