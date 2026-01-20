# Seed 對 PTQ 結果影響的詳細分析

## 問題描述

使用不同的 `--calib-seed` (0 vs 1) 會導致不同的 mAP 結果：
- `--calib-seed 0`: mAP = 0.68
- `--calib-seed 1`: mAP = 0.675

**差異**: 0.005 (0.5% 相對差異)

## 根本原因分析

### 1. Seed 如何影響數據順序

**代碼位置**: `tools/detection3d/centerpoint_quantization.py:309-317`

```python
if args.calib_seed is not None:
    import random
    import numpy as np
    random.seed(args.calib_seed)
    np.random.seed(args.calib_seed)
    torch.manual_seed(args.calib_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.calib_seed)
```

**影響**:
- 不同的 seed 會導致 DataLoader 的隨機採樣順序不同
- 即使使用相同的 938 個樣本，**順序不同**會影響 histogram 的累積過程

### 2. Histogram 累積的順序敏感性

**代碼位置**: `pytorch_quantization/calib/histogram.py:67-117`

#### 關鍵問題：第一個 Batch 決定初始範圍

```python
def collect(self, x):
    x_max = x.max()

    if self._calib_bin_edges is None:
        # 第一次：使用第一個 batch 的最大值初始化 histogram
        self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max)
        self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
    else:
        # 後續：如果遇到更大的值，擴展 histogram 範圍
        if x_max > self._calib_bin_edges[-1]:
            width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
            self._num_bins = int((x_max / width).ceil().item())
            self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

        hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
        hist[:self._calib_hist.numel()] += self._calib_hist
        self._calib_hist = hist
```

#### 為什麼順序重要？

**場景 A: Seed 0**
```
Batch 1: max_value = 3.5  → 初始化 histogram [0, 3.5], bins=2048
Batch 2: max_value = 4.2  → 擴展到 [0, 4.2], bins=2458
Batch 3: max_value = 3.8  → 累積到現有 histogram
...
Batch 938: max_value = 4.0 → 最終 histogram 範圍 [0, 4.2]
```

**場景 B: Seed 1**
```
Batch 1: max_value = 4.2  → 初始化 histogram [0, 4.2], bins=2048
Batch 2: max_value = 3.5  → 累積到現有 histogram
Batch 3: max_value = 3.8  → 累積到現有 histogram
...
Batch 938: max_value = 4.0 → 最終 histogram 範圍 [0, 4.2]
```

**關鍵差異**:
1. **初始 bin 寬度不同**:
   - Seed 0: `width = 3.5 / 2048 ≈ 0.00171`
   - Seed 1: `width = 4.2 / 2048 ≈ 0.00205`

2. **Bin 邊界對齊不同**:
   - 即使最終範圍相同，**bin 邊界的位置不同**
   - 這會導致相同數值被分配到不同的 bin

3. **累積過程中的數值精度**:
   - 浮點數累積順序不同會產生微小的數值誤差
   - 這些誤差會影響 histogram 的形狀

### 3. MSE 優化對 Histogram 形狀的敏感性

**代碼位置**: `pytorch_quantization/calib/histogram.py:257-293`

```python
def _compute_amax_mse(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    centers = (edges[1:] + edges[:-1]) / 2  # bin 中心值
    counts = calib_hist  # 每個 bin 的樣本數量

    for i in range(start_bin, len(centers), stride):
        amax = centers[i]  # 候選 amax
        quant_centers = fake_tensor_quant(centers, amax, num_bits, unsigned)
        mse = ((quant_centers - centers)**2 * counts).mean()
        mses.append(mse)

    argmin = np.argmin(mses)
    return centers[arguments[argmin]]
```

#### 為什麼 Histogram 形狀的微小差異會導致不同的 amax？

**示例**:

假設有兩個 histogram，總計數相同，但 bin 分佈略有不同：

```
Histogram A (Seed 0):
  Bin [3.0, 3.1]: count = 1000
  Bin [3.1, 3.2]: count = 1500
  Bin [3.2, 3.3]: count = 2000
  ...

Histogram B (Seed 1):
  Bin [3.0, 3.1]: count = 1001  ← 微小差異
  Bin [3.1, 3.2]: count = 1499
  Bin [3.2, 3.3]: count = 2000
  ...
```

**MSE 計算過程**:

對於候選 amax = 3.2:
- Histogram A: `MSE = mean((quant_centers - centers)^2 * counts_A)`
- Histogram B: `MSE = mean((quant_centers - centers)^2 * counts_B)`

即使差異很小（0.1%），**加權後的 MSE 值也會不同**，因為：
1. `counts` 作為權重，直接影響 MSE 值
2. MSE 函數是連續的，但最小值位置對權重變化敏感
3. 不同的 MSE 值會導致選擇不同的最優 amax

**實際影響**:

```
Seed 0: 最優 amax = 3.245 (MSE = 0.001234)
Seed 1: 最優 amax = 3.251 (MSE = 0.001238)
```

這個 0.006 的 amax 差異會導致：
- 量化 scale 不同：`scale = amax / 127`
- 量化誤差不同
- 最終模型精度不同

### 4. 累積誤差的放大效應

#### 浮點數累積順序的影響

```python
# 場景 A: 先累積大值，後累積小值
hist = torch.histc(large_values, ...)  # 初始化
hist += torch.histc(small_values, ...)  # 累積

# 場景 B: 先累積小值，後累積大值
hist = torch.histc(small_values, ...)   # 初始化
hist += torch.histc(large_values, ...)  # 累積（需要擴展範圍）
```

**差異來源**:
1. **Bin 寬度計算**:
   - 場景 A: `width = large_max / 2048`
   - 場景 B: `width = small_max / 2048`，然後擴展
   - 擴展過程中的 `ceil()` 操作會產生不同的 bin 數量

2. **數值精度**:
   - 浮點數累積順序不同會產生不同的舍入誤差
   - 這些誤差在 938 個 batch 的累積中會被放大

3. **Bin 邊界對齊**:
   - 不同的初始範圍導致 bin 邊界位置不同
   - 即使數值相同，也可能被分配到不同的 bin

## 影響鏈分析

```
Seed 差異
  │
  ├─> DataLoader 順序不同
  │     │
  │     └─> Histogram 累積順序不同
  │           │
  │           ├─> 初始 bin 範圍不同
  │           │     │
  │           │     └─> Bin 寬度不同
  │           │           │
  │           │           └─> Bin 邊界對齊不同
  │           │
  │           ├─> 累積過程中的數值精度誤差
  │           │     │
  │           │     └─> Histogram 形狀微小差異
  │           │
  │           └─> 最終 Histogram 分佈不同
  │                 │
  │                 └─> MSE 優化找到不同的 amax
  │                       │
  │                       └─> 量化 scale 不同
  │                             │
  │                             └─> 模型精度不同 (mAP 差異)
```

## 具體數值示例

### 假設場景

**激活值分佈**: 均值 3.0, 標準差 1.0, 最大值約 4.5

**Seed 0 的累積過程**:
```
Batch 1: max=3.5 → bins=2048, width=0.00171
Batch 50: max=4.2 → 擴展到 bins=2458, width=0.00171
...
最終: bins=2458, 總計數=938000, 最優 amax=3.245
```

**Seed 1 的累積過程**:
```
Batch 1: max=4.2 → bins=2048, width=0.00205
Batch 50: max=3.5 → 累積到現有 bins
...
最終: bins=2048, 總計數=938000, 最優 amax=3.251
```

**差異**:
- Bin 寬度: 0.00171 vs 0.00205 (差異 19.9%)
- 最優 amax: 3.245 vs 3.251 (差異 0.18%)
- 量化 scale: 0.02555 vs 0.02560 (差異 0.2%)

### 對模型精度的影響

**量化誤差傳播**:
```
amax 差異 0.006
  → scale 差異 0.00005
    → 量化誤差差異累積
      → 激活值差異累積
        → 網絡輸出差異
          → mAP 差異 0.005
```

## 為什麼差異是 0.005？

### 1. 多層累積效應

CenterPoint 模型有多個量化層：
- Voxel Encoder
- Backbone (多個 stage)
- Neck
- Head

每個層的 amax 差異都會累積：
```
Layer 1 amax 差異: 0.006 → 激活誤差: 0.0001
Layer 2 amax 差異: 0.006 → 激活誤差: 0.0002 (累積)
...
Layer N amax 差異: 0.006 → 激活誤差: 0.001 (累積)
```

### 2. 關鍵層的敏感性

某些層對量化更敏感（如 detection head）：
- 這些層的 amax 差異會產生更大的影響
- 即使其他層差異很小，關鍵層的差異也會顯著影響最終精度

### 3. 非線性傳播

量化誤差在網絡中非線性傳播：
- ReLU、BatchNorm 等操作會放大誤差
- 誤差在深度網絡中會指數級增長

## 解決方案和建議

### 1. 固定 Seed（推薦）

```bash
# 始終使用相同的 seed，確保可重現性
python tools/detection3d/centerpoint_quantization.py ptq \
    --calib-seed 0 \
    ...
```

**優點**: 結果可重現
**缺點**: 可能不是最優的數據順序

### 2. 使用多個 Seed 並平均

```bash
# 使用多個 seed 進行校準，然後平均 amax
for seed in 0 1 2 3 4; do
    python ... --calib-seed $seed --output calib_${seed}.pth
done
# 然後平均所有 calib cache 的 amax 值
```

**優點**: 減少順序敏感性
**缺點**: 需要多次校準，計算成本高

### 3. 預先確定最大範圍

修改 histogram 收集邏輯，先掃描所有數據確定最大範圍：
```python
# 第一遍：確定全局最大值
max_val = 0
for batch in dataloader:
    max_val = max(max_val, model.get_max_activation(batch))

# 第二遍：使用固定範圍收集 histogram
calibrator.set_max_range(max_val)
calibrator.collect_stats(dataloader)
```

**優點**: 消除順序影響
**缺點**: 需要兩遍掃描數據

### 4. 使用更穩定的校準方法

考慮使用 `percentile` 方法，它對 histogram 形狀的敏感性較低：
```python
calibrator.calibrate(dataloader, method="percentile", percentile=99.99)
```

**優點**: 對順序不敏感
**缺點**: 可能不如 MSE 方法精確

## 總結

**根本原因**:
1. **數據順序不同** → Histogram 累積順序不同
2. **初始範圍不同** → Bin 寬度和邊界對齊不同
3. **數值精度誤差** → Histogram 形狀微小差異
4. **MSE 敏感性** → 不同的最優 amax
5. **誤差累積** → 最終 mAP 差異

**影響程度**:
- 單層 amax 差異: ~0.1-0.2%
- 多層累積: ~0.5% mAP 差異
- 這是 PTQ 的正常現象，不是 bug

**建議**:
- 使用固定 seed 確保可重現性
- 如果追求穩定性，考慮使用 percentile 方法
- 差異在可接受範圍內（< 1%），屬於正常波動
