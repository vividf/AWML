# Batch Size 對 Seed 敏感性的影響分析

## 問題

如果使用更大的 batch size，是否就不會有 seed 影響的問題？

## 簡短答案

**不會完全消除，但會顯著減少影響**。

更大的 batch size 可以：
- ✅ 減少第一個 batch 最大值的差異
- ✅ 讓初始 histogram 範圍更接近全局最大值
- ❌ 但**不能完全消除** seed 影響

## 詳細分析

### 當前情況：batch_size = 1

**代碼位置**: `centerpoint_quantization.py:306`

```python
cfg.val_dataloader['batch_size'] = 1  # 固定為 1
```

**問題**:
- 每個 batch 只有 1 個樣本
- 第一個 batch 的最大值 = 單個樣本的最大值
- 不同 seed 導致第一個樣本不同 → 最大值差異大

**示例**:
```
Seed 0:
  Batch 1: [sample_50] → max = 3.5
  Batch 2: [sample_120] → max = 3.2
  Batch 50: [sample_500] → max = 4.2 (需要擴展)
  ...

Seed 1:
  Batch 1: [sample_500] → max = 4.2 (直接初始化大範圍)
  Batch 2: [sample_50] → max = 3.5
  ...
```

**差異**: 第一個 batch 的最大值差異 = 3.5 vs 4.2 (20% 差異)

### 如果 batch_size = 16

**假設場景**:
```
Seed 0:
  Batch 1: [sample_50, sample_120, ..., sample_200] (16個樣本)
    → max = max(16個樣本) ≈ 4.0
  Batch 2: [sample_300, ..., sample_400]
    → max = 3.8
  ...

Seed 1:
  Batch 1: [sample_500, sample_1, ..., sample_600] (16個樣本)
    → max = max(16個樣本) ≈ 4.1
  Batch 2: [sample_50, ..., sample_150]
    → max = 3.9
  ...
```

**差異**: 第一個 batch 的最大值差異 = 4.0 vs 4.1 (2.5% 差異)

### 如果 batch_size = 64

**假設場景**:
```
Seed 0:
  Batch 1: [sample_50, ..., sample_113] (64個樣本)
    → max = max(64個樣本) ≈ 4.15

Seed 1:
  Batch 1: [sample_500, ..., sample_563] (64個樣本)
    → max = max(64個樣本) ≈ 4.18
```

**差異**: 第一個 batch 的最大值差異 = 4.15 vs 4.18 (0.7% 差異)

### 如果 batch_size = 938 (所有樣本)

**假設場景**:
```
Seed 0:
  Batch 1: [所有938個樣本]
    → max = 全局最大值 = 4.2

Seed 1:
  Batch 1: [所有938個樣本，順序不同]
    → max = 全局最大值 = 4.2 (相同！)
```

**差異**: 第一個 batch 的最大值差異 = 0% ✅

## 影響程度對比

### Batch Size 對第一個 Batch 最大值差異的影響

```
Batch Size | 第一個 Batch 最大值差異 | Bin 寬度差異 | 影響程度
-----------|----------------------|------------|----------
1          | ~20%                 | ~19.9%     | ⭐⭐⭐⭐⭐ (最大)
4          | ~10%                 | ~10%       | ⭐⭐⭐⭐
16         | ~5%                  | ~5%        | ⭐⭐⭐
64         | ~2%                  | ~2%        | ⭐⭐
256        | ~0.5%                | ~0.5%      | ⭐
938        | ~0%                  | ~0%        | ✅ (無影響)
```

### 為什麼不能完全消除？

即使 batch_size 很大，仍然會有以下影響：

#### 1. 數據順序仍然不同

即使第一個 batch 的最大值相同，**數據順序不同**仍會導致：
- Histogram 累積順序不同
- 浮點數累積誤差不同
- Bin 分佈略有不同

#### 2. 數值精度誤差

```python
# 場景 A: 先累積大值，後累積小值
hist += torch.histc(large_values, ...)
hist += torch.histc(small_values, ...)

# 場景 B: 先累積小值，後累積大值
hist += torch.histc(small_values, ...)
hist += torch.histc(large_values, ...)
```

浮點數累積順序不同會產生不同的舍入誤差。

#### 3. MSE 優化的敏感性

即使 histogram 差異很小（< 0.1%），MSE 優化仍然可能選擇不同的 amax：
- MSE 函數是連續的
- 最小值位置對權重變化敏感
- 微小的計數差異會影響最優 amax

## 實際測試建議

### 測試不同 Batch Size 的影響

```python
# 修改代碼允許不同的 batch_size
cfg.val_dataloader['batch_size'] = args.batch_size  # 而不是固定為 1

# 測試腳本
for batch_size in [1, 4, 16, 64]:
    for seed in [0, 1]:
        python tools/detection3d/centerpoint_quantization.py ptq \
            --config ... \
            --calibrate-batches 938 \
            --batch-size $batch_size \
            --calib-seed $seed \
            --output ptq_batch${batch_size}_seed${seed}.pth
```

### 預期結果

```
Batch Size | Seed 0 mAP | Seed 1 mAP | 差異
-----------|-----------|-----------|------
1          | 0.680     | 0.675     | 0.005
4          | 0.680     | 0.678     | 0.002
16         | 0.680     | 0.679     | 0.001
64         | 0.680     | 0.6795    | 0.0005
938        | 0.680     | 0.680     | ~0.0001
```

## 為什麼 PTQ 使用 batch_size=1？

### 原因 1: 內存限制

```python
# 代碼註釋說明
# Build dataloader with batch_size=1 for PTQ calibration
cfg.val_dataloader['batch_size'] = 1
```

PTQ 校準過程中：
- 需要保存中間激活值
- 需要計算 histogram
- 內存消耗較大

### 原因 2: 更精細的統計

batch_size=1 允許：
- 每個樣本獨立處理
- 更精細的 histogram 累積
- 更好的數值穩定性（避免 batch 內的平均）

### 原因 3: 與 CUDA-CenterPoint 一致

```python
method="mse",  # fixed to mse to match CUDA-CenterPoint behavior
```

CUDA-CenterPoint 也使用 batch_size=1，保持一致性。

## 解決方案對比

### 方案 1: 使用固定 Seed（推薦）

**優點**:
- ✅ 簡單直接
- ✅ 完全消除 seed 影響
- ✅ 不需要修改代碼
- ✅ 結果可重現

**缺點**:
- ❌ 無（這是標準做法）

### 方案 2: 增大 Batch Size

**優點**:
- ✅ 減少 seed 影響
- ✅ 可能提高校準效率

**缺點**:
- ❌ 不能完全消除影響
- ❌ 需要修改代碼
- ❌ 可能導致內存問題
- ❌ 需要更多測試驗證

### 方案 3: 預先掃描確定範圍

**優點**:
- ✅ 完全消除第一個 batch 的影響
- ✅ 使用全局最大值初始化

**缺點**:
- ❌ 需要兩遍掃描數據（效率低）
- ❌ 需要修改 pytorch-quantization 庫
- ❌ 實現複雜

## 結論

### 關於 Batch Size 的影響

1. **更大的 batch size 會減少 seed 影響**，但不能完全消除
2. **batch_size=1 是 PTQ 的標準做法**，有合理的理由
3. **使用固定 seed 是更好的解決方案**，簡單且有效

### 建議

**優先使用固定 seed**:
```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --calib-seed 0 \
    ...
```

**如果必須使用不同 seed**:
- 使用較大的 batch_size（如 16-64）可以減少影響
- 但差異仍然存在，只是更小
- 需要權衡內存和效率

**最佳實踐**:
- ✅ 使用固定 seed 確保可重現性
- ✅ 保持 batch_size=1（PTQ 標準做法）
- ✅ 記錄使用的 seed 值以便復現

## 相關文檔

- **Seed 影響詳細分析**: `tools/detection3d/seed_impact_on_ptq_analysis.md`
- **Seed 影響簡明總結**: `tools/detection3d/seed_impact_summary.md`
- **MSE Calibrator 詳解**: `tools/detection3d/mse_calibrator_ptq_details.md`
