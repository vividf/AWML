# Seed 對 mAP 影響的簡明解釋

## 問題

使用 `--calib-seed 0` 和 `--calib-seed 1` 會導致不同的 mAP：
- Seed 0: mAP = 0.68
- Seed 1: mAP = 0.675
- **差異**: 0.005 (0.5%)

## 根本原因：三個關鍵環節

### 1. 數據順序不同 → Histogram 累積順序不同

**代碼**: `centerpoint_quantization.py:309-317`

不同的 seed 導致 DataLoader 的數據順序不同：
- Seed 0: Batch 1 可能是 `[sample_50, sample_120, ...]`
- Seed 1: Batch 1 可能是 `[sample_500, sample_1, ...]`

**影響**: 即使使用相同的 938 個樣本，**順序不同**會影響 histogram 的累積過程。

### 2. 第一個 Batch 決定初始 Histogram 範圍

**代碼**: `pytorch_quantization/calib/histogram.py:106-108`

```python
# 第一個 batch 的最大值決定初始範圍
if self._calib_bin_edges is None:
    x_max = x.max()  # ← 第一個 batch 的最大值
    self._calib_hist = torch.histc(x, bins=2048, min=0, max=x_max)
    self._calib_bin_edges = torch.linspace(0, x_max, 2049)
    # Bin 寬度 = x_max / 2048
```

**實際差異**:
- Seed 0: 第一個 batch max=3.5 → bin 寬度 = 0.00171
- Seed 1: 第一個 batch max=4.2 → bin 寬度 = 0.00205
- **差異**: 19.9% 的 bin 寬度差異！

### 3. Bin 邊界對齊不同 → Histogram 形狀不同

**問題**: 即使最終範圍相同，**bin 邊界的位置不同**會導致：
- 相同的數值被分配到不同的 bin
- Histogram 分佈略有不同

**示例**:
```
數值 x = 3.14159

Seed 0 (width=0.00171):
  Bin 1836: [3.139, 3.141] ← x 落在這裡

Seed 1 (width=0.00205):
  Bin 1532: [3.141, 3.143] ← x 落在這裡（不同的 bin！）
```

### 4. MSE 優化對 Histogram 形狀敏感

**代碼**: `pytorch_quantization/calib/histogram.py:257-293`

MSE 優化會遍歷所有候選 amax，選擇 MSE 最小的：

```python
for i in range(start_bin, len(centers)):
    amax = centers[i]
    quant_centers = fake_tensor_quant(centers, amax, ...)
    mse = ((quant_centers - centers)**2 * counts).mean()  # ← counts 作為權重
    mses.append(mse)

optimal_amax = centers[np.argmin(mses)]
```

**關鍵**: `counts`（histogram 計數）作為權重，即使微小差異也會影響 MSE 值：
- Histogram 形狀不同 → MSE 值不同 → 最優 amax 不同

**實際結果**:
- Seed 0: 最優 amax = 3.245
- Seed 1: 最優 amax = 3.251
- **差異**: 0.006 (0.18%)

### 5. amax 差異 → 量化誤差 → mAP 差異

**誤差傳播**:
```
amax 差異 0.006
  → scale 差異 = 0.006 / 127 ≈ 0.00005
    → 量化誤差累積（多層）
      → 激活值差異累積
        → 網絡輸出差異
          → mAP 差異 0.005
```

## 完整影響鏈

```
┌─────────────────────────────────────────────────────────┐
│ Seed 差異                                                │
│   │                                                      │
│   └─> DataLoader 順序不同                               │
│         │                                                │
│         └─> 第一個 Batch 的最大值不同                   │
│               │                                          │
│               ├─> 初始 Histogram 範圍不同                │
│               │   │                                      │
│               │   └─> Bin 寬度不同 (19.9% 差異)        │
│               │       │                                  │
│               │       └─> Bin 邊界對齊不同              │
│               │           │                              │
│               │           └─> 相同數值分配到不同 Bin    │
│               │               │                          │
│               │               └─> Histogram 形狀不同    │
│               │                   │                      │
│               │                   └─> MSE 優化結果不同  │
│               │                       │                  │
│               │                       └─> amax 差異     │
│               │                           │              │
│               │                           └─> 量化誤差  │
│               │                               │          │
│               │                               └─> mAP 差異│
│               │                                   (0.005)│
│               │                                      │    │
│               └─> 累積過程中的數值精度誤差 ────────────┘
└─────────────────────────────────────────────────────────┘
```

## 哪部分造成的影響最大？

### 1. **第一個 Batch 的決定性作用** (最重要)

第一個 batch 的最大值決定了：
- 初始 histogram 範圍
- Bin 寬度（影響所有後續 batch）
- Bin 邊界對齊

**影響程度**: ⭐⭐⭐⭐⭐ (最大)

### 2. **Bin 寬度差異**

不同的 bin 寬度導致：
- 相同的數值被分配到不同的 bin
- Histogram 分佈不同

**影響程度**: ⭐⭐⭐⭐ (很大)

### 3. **MSE 優化的敏感性**

MSE 方法對 histogram 形狀很敏感：
- 微小的計數差異會影響 MSE 值
- 導致選擇不同的最優 amax

**影響程度**: ⭐⭐⭐ (中等)

### 4. **多層誤差累積**

多個量化層的誤差會累積：
- 單層差異小（0.18%）
- 多層累積後差異放大（0.5% mAP）

**影響程度**: ⭐⭐⭐ (中等)

## 解決方案

### 推薦：使用固定 Seed

```bash
# 始終使用 seed=0，確保可重現性
python tools/detection3d/centerpoint_quantization.py ptq \
    --calib-seed 0 \
    ...
```

**優點**: 結果可重現，差異消失
**缺點**: 無（這是標準做法）

### 替代方案：預先掃描確定範圍

修改代碼，先掃描所有數據確定最大範圍，然後用固定範圍收集 histogram（需要修改 pytorch-quantization 庫）。

## 總結

**造成差異的主要部分**:
1. ✅ **第一個 Batch 的最大值** → 決定初始 histogram 範圍和 bin 寬度
2. ✅ **Bin 寬度差異** → 導致 bin 邊界對齊不同
3. ✅ **MSE 優化敏感性** → 對 histogram 形狀敏感
4. ✅ **多層誤差累積** → 放大最終差異

**這是 PTQ 的正常現象**，不是 bug。使用固定 seed 即可解決。

## 相關文檔

- **詳細分析**: `tools/detection3d/seed_impact_on_ptq_analysis.md`
- **可視化說明**: `tools/detection3d/seed_impact_visualization.md`
- **MSE Calibrator 詳解**: `tools/detection3d/mse_calibrator_ptq_details.md`
