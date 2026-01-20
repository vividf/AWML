# MSE Calibrator PTQ 流程圖

## 完整調用鏈

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 入口點: centerpoint_quantization.py                          │
│    run_ptq()                                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CalibrationManager.calibrate(method="mse")                  │
│    calibrator.py:172                                            │
│                                                                 │
│    ├─> set_quantizer_fast()                                    │
│    │   └─> HistogramCalibrator._torch_hist = True             │
│    │                                                             │
│    ├─> collect_stats(dataloader, num_batches)                  │
│    │   └─> 收集 histogram 統計數據                              │
│    │                                                             │
│    └─> compute_amax(method="mse")                              │
│        └─> 計算最優 amax                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. collect_stats() 詳細流程                                     │
│    calibrator.py:95                                             │
│                                                                 │
│    ├─> _enable_calibration_mode()                              │
│    │   ├─> module.disable_quant()  # 禁用假量化                │
│    │   └─> module.enable_calib()   # 啟用統計收集              │
│    │                                                             │
│    ├─> 前向傳播校準數據                                        │
│    │   for batch in dataloader:                                │
│    │       model.test_step(batch)                               │
│    │       └─> TensorQuantizer.forward()                       │
│    │           └─> HistogramCalibrator.collect(x)              │
│    │               ├─> x = x.abs()  # 取絕對值                 │
│    │               └─> torch.histc(x, ...)  # 累積 histogram   │
│    │                                                             │
│    └─> _disable_calibration_mode()                              │
│        ├─> module.enable_quant()   # 啟用假量化                │
│        └─> module.disable_calib()  # 禁用統計收集              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. compute_amax(method="mse")                                   │
│    calibrator.py:145                                            │
│                                                                 │
│    for module in model.named_modules():                         │
│        if isinstance(module, TensorQuantizer):                 │
│            if isinstance(calibrator, HistogramCalibrator):      │
│                module.load_calib_amax(method="mse")              │
│            elif isinstance(calibrator, MaxCalibrator):         │
│                module.load_calib_amax()                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. TensorQuantizer.load_calib_amax(method="mse")                │
│    tensor_quantizer.py:242                                      │
│                                                                 │
│    calib_amax = self._calibrator.compute_amax(method="mse")    │
│    self._amax.copy_(calib_amax)                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. HistogramCalibrator.compute_amax(method="mse")               │
│    histogram.py:124                                             │
│                                                                 │
│    if method == 'mse':                                          │
│        calib_amax = _compute_amax_mse(...)                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. _compute_amax_mse() - 核心 MSE 優化算法                      │
│    histogram.py:257                                            │
│                                                                 │
│    centers = (edges[1:] + edges[:-1]) / 2  # bin 中心值       │
│    counts = calib_hist                                          │
│                                                                 │
│    for i in range(start_bin=128, len(centers), stride=1):     │
│        amax = centers[i]  # 候選 amax                          │
│                                                                 │
│        # 模擬量化過程                                           │
│        quant_centers = fake_tensor_quant(centers, amax, ...)  │
│                                                                 │
│        # 計算加權 MSE                                          │
│        mse = ((quant_centers - centers)**2 * counts).mean()   │
│                                                                 │
│        mses.append(mse)                                        │
│                                                                 │
│    # 選擇 MSE 最小的 amax                                      │
│    argmin = np.argmin(mses)                                     │
│    calib_amax = centers[arguments[argmin]]                     │
│                                                                 │
│    return calib_amax                                            │
└─────────────────────────────────────────────────────────────────┘
```

## MSE 優化算法詳細流程

```
┌─────────────────────────────────────────────────────────────┐
│ 輸入: histogram (calib_hist, calib_bin_edges)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 準備數據                                                  │
│    - centers = bin 中心值                                    │
│    - counts = 每個 bin 的樣本數量                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 遍歷 amax 候選值 (從 start_bin=128 開始)                  │
│                                                              │
│    for each candidate_amax in [centers[128], ..., max]:     │
│        ┌──────────────────────────────────────────────┐    │
│        │ 3. 模擬量化過程                               │    │
│        │    scale = candidate_amax / (2^7 - 1)        │    │
│        │    quantized = round(centers / scale)        │    │
│        │    dequantized = quantized * scale            │    │
│        │    quant_centers = clamp(dequantized, ...)    │    │
│        └──────────────────────────────────────────────┘    │
│                     │                                        │
│                     ▼                                        │
│        ┌──────────────────────────────────────────────┐    │
│        │ 4. 計算量化誤差                               │    │
│        │    error = quant_centers - centers            │    │
│        │    squared_error = error^2                    │    │
│        │    weighted_error = squared_error * counts     │    │
│        │    mse = mean(weighted_error)                 │    │
│        └──────────────────────────────────────────────┘    │
│                     │                                        │
│                     ▼                                        │
│        ┌──────────────────────────────────────────────┐    │
│        │ 5. 記錄 MSE 值                                │    │
│        │    mses.append(mse)                           │    │
│        │    arguments.append(i)                         │    │
│        └──────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 選擇最優 amax                                             │
│    argmin = index_of_minimum(mses)                          │
│    optimal_amax = centers[arguments[argmin]]               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 輸出: optimal_amax (最小化 MSE 的 amax 值)                   │
└─────────────────────────────────────────────────────────────┘
```

## 關鍵概念說明

### 1. Histogram 收集
- **目的**: 統計激活值的分佈
- **方法**: 使用 `torch.histc()` 累積多個 batch 的統計數據
- **特點**:
  - 動態擴展範圍（遇到更大值時自動擴展）
  - 使用絕對值（量化範圍對稱）

### 2. MSE 優化目標
- **目標**: 最小化量化誤差的加權平方和
- **公式**: `MSE = mean((quantized - original)^2 * counts)`
- **平衡**:
  - Clipping error: amax 太小會截斷太多值
  - Rounding error: amax 太大會增加捨入誤差

### 3. 候選 amax 範圍
- **起始點**: `start_bin=128`（避免過小的 amax）
- **終點**: histogram 的最大值
- **步長**: `stride=1`（每個 bin 都嘗試）

### 4. 量化過程模擬
```python
# 量化
scale = amax / (2^(num_bits-1) - 1)  # 8-bit: amax / 127
quantized = round(x / scale)

# 反量化
dequantized = quantized * scale

# 裁剪
dequantized = clamp(dequantized, -amax, amax)
```

## 實際使用範例

```python
# 1. 準備模型和數據
model = init_model(cfg, checkpoint, device="cuda:0")
quant_model(model)  # 插入 Q/DQ 節點
fuse_model_bn(model)  # 融合 BatchNorm

# 2. 創建 CalibrationManager
calibrator = CalibrationManager(model)

# 3. 執行校準（使用 MSE 方法）
calibrator.calibrate(
    dataloader=dataloader,
    num_batches=100,
    method="mse",  # ← 指定 MSE 方法
)

# 4. 保存校準結果
calibrator.save_calib_cache("calib_cache.pth")
torch.save({"state_dict": model.state_dict()}, "quantized_model.pth")
```

## 相關文件

- **詳細文檔**: `tools/detection3d/mse_calibrator_ptq_details.md`
- **入口代碼**: `tools/detection3d/centerpoint_quantization.py:337`
- **CalibrationManager**: `projects/CenterPoint/quantization/calibration/calibrator.py`
- **MSE 實現**: `pytorch_quantization/calib/histogram.py:257`
