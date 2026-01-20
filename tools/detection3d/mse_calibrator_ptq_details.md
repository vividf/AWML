# MSE Calibrator PTQ 詳細實現

本文檔詳細說明如何使用 MSE (Mean Squared Error) Calibrator 進行 PTQ (Post-Training Quantization) 的完整流程和實現細節。

## 目錄
1. [整體流程](#整體流程)
2. [代碼調用鏈](#代碼調用鏈)
3. [MSE 優化算法詳解](#mse-優化算法詳解)
4. [關鍵代碼位置](#關鍵代碼位置)

---

## 整體流程

PTQ 使用 MSE calibrator 的完整流程如下：

```
1. 準備模型
   └─> 插入 Q/DQ 節點 (quant_model)
   └─> 融合 BatchNorm (fuse_model_bn)

2. 收集統計數據 (collect_stats)
   └─> 啟用校準模式 (enable_calib)
   └─> 前向傳播校準數據
   └─> HistogramCalibrator.collect() 累積 histogram

3. 計算 amax (compute_amax with method="mse")
   └─> HistogramCalibrator.compute_amax(method="mse")
   └─> _compute_amax_mse() 執行 MSE 優化

4. 應用量化
   └─> 禁用校準模式 (disable_calib)
   └─> 啟用量化模式 (enable_quant)
```

---

## 代碼調用鏈

### 1. 入口點：`centerpoint_quantization.py`

**文件位置**: `tools/detection3d/centerpoint_quantization.py:337`

```python
calibrator = CalibrationManager(model)
calibrator.calibrate(
    dataloader,
    num_batches=args.calibrate_batches,
    method="mse",  # ← 指定使用 MSE 方法
)
```

### 2. CalibrationManager.calibrate()

**文件位置**: `projects/CenterPoint/quantization/calibration/calibrator.py:172`

```python
def calibrate(
    self,
    dataloader: Any,
    num_batches: int = 100,
    method: str = "mse",
    forward_fn: Optional[Callable] = None,
):
    """Run full calibration pipeline."""
    print(f"Starting calibration with {num_batches} batches, method={method}")

    # 1. 啟用快速 histogram 模式（使用 PyTorch 的 histogram）
    self.set_quantizer_fast()

    # 2. 收集統計數據
    self.collect_stats(dataloader, num_batches, forward_fn)

    # 3. 計算 amax（使用 MSE 方法）
    self.compute_amax(method)
```

**關鍵步驟**:
- `set_quantizer_fast()`: 設置 `HistogramCalibrator._torch_hist = True`，使用 PyTorch 的 `torch.histc` 加速 histogram 計算
- `collect_stats()`: 遍歷校準數據，累積 histogram
- `compute_amax(method="mse")`: 從 histogram 計算最優 amax

### 3. CalibrationManager.collect_stats()

**文件位置**: `projects/CenterPoint/quantization/calibration/calibrator.py:95`

```python
def collect_stats(self, dataloader, num_batches=100, forward_fn=None):
    """Collect activation statistics for calibration."""
    self.model.eval()
    self._enable_calibration_mode()  # 啟用校準模式

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # 前向傳播（會觸發 TensorQuantizer 的 collect）
            if forward_fn is not None:
                forward_fn(self.model, batch)
            elif hasattr(self.model, "test_step"):
                self.model.test_step(batch)
            # ...

    self._disable_calibration_mode()  # 禁用校準模式
```

**校準模式設置**:
- `enable_calib()`: 啟用統計收集，禁用假量化
- `disable_quant()`: 禁用假量化，確保使用原始浮點值

### 4. HistogramCalibrator.collect()

**文件位置**: `pytorch_quantization/calib/histogram.py:67`

```python
def collect(self, x):
    """Collect histogram"""
    # 處理負值（取絕對值）
    if torch.min(x) < 0.:
        x = x.abs()

    x = x.float()

    if self._torch_hist:
        # 使用 PyTorch histogram（GPU 加速）
        x_max = x.max()
        if self._calib_bin_edges is None:
            # 第一次：初始化 histogram
            self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max)
            self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
        else:
            # 後續：累積 histogram
            if x_max > self._calib_bin_edges[-1]:
                # 擴展 bin 範圍
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                self._num_bins = int((x_max / width).ceil().item())
                self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

            hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
            hist[:self._calib_hist.numel()] += self._calib_hist
            self._calib_hist = hist
```

**關鍵點**:
- 使用絕對值（因為量化範圍是對稱的）
- 動態擴展 histogram 範圍（如果遇到更大的值）
- 累積多個 batch 的統計數據

### 5. CalibrationManager.compute_amax()

**文件位置**: `projects/CenterPoint/quantization/calibration/calibrator.py:145`

```python
def compute_amax(self, method: str = "mse"):
    """Compute amax values from collected statistics."""
    for name, module in self.model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # MaxCalibrator 不需要 method 參數
                    module.load_calib_amax()
                else:
                    # HistogramCalibrator 需要 method 參數
                    module.load_calib_amax(method=method)

                # 移動 amax 到模型設備
                if module._amax is not None:
                    module._amax = module._amax.to(self.device)
```

### 6. TensorQuantizer.load_calib_amax()

**文件位置**: `pytorch_quantization/nn/modules/tensor_quantizer.py:242`

```python
def load_calib_amax(self, *args, **kwargs):
    """Load amax from calibrator."""
    strict = kwargs.pop("strict", True)
    if getattr(self, '_calibrator', None) is None:
        raise RuntimeError("Calibrator not created.")

    # 調用 calibrator 的 compute_amax 方法
    calib_amax = self._calibrator.compute_amax(*args, **kwargs)

    if calib_amax is None:
        # 處理錯誤情況
        if not strict:
            calib_amax = torch.tensor(math.nan)
        else:
            raise RuntimeError("Calibrator returned None...")

    # 更新 amax buffer
    if not hasattr(self, '_amax'):
        self.register_buffer("_amax", calib_amax.data)
    else:
        self._amax.copy_(calib_amax)
```

### 7. HistogramCalibrator.compute_amax()

**文件位置**: `pytorch_quantization/calib/histogram.py:124`

```python
def compute_amax(self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
    """Compute the amax from the collected histogram"""

    # 轉換為 numpy（如果使用 torch histogram）
    if isinstance(self._calib_hist, torch.Tensor):
        calib_hist = self._calib_hist.to(torch.int64).cpu().numpy()
        calib_bin_edges = self._calib_bin_edges.cpu().numpy()
    else:
        calib_hist = self._calib_hist
        calib_bin_edges = self._calib_bin_edges

    # 根據方法調用對應的計算函數
    if method == 'entropy':
        calib_amax = _compute_amax_entropy(...)
    elif method == 'mse':
        calib_amax = _compute_amax_mse(...)  # ← MSE 方法
    elif method == 'percentile':
        calib_amax = _compute_amax_percentile(...)
    else:
        raise TypeError("Unknown calibration method {}".format(method))

    return calib_amax
```

---

## MSE 優化算法詳解

### _compute_amax_mse() 實現

**文件位置**: `pytorch_quantization/calib/histogram.py:257`

```python
def _compute_amax_mse(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    """Returns amax that minimizes MSE of the collected histogram"""

    # 1. 準備數據
    counts = torch.from_numpy(calib_hist[:]).float().cuda()
    edges = torch.from_numpy(calib_bin_edges[:]).float().cuda()
    centers = (edges[1:] + edges[:-1]) / 2  # histogram bin 的中心值

    mses = []
    arguments = []

    # 2. 遍歷所有可能的 amax 候選值
    # start_bin=128: 從第 128 個 bin 開始（避免過小的 amax）
    # stride=1: 每個 bin 都嘗試
    for i in range(start_bin, len(centers), stride):
        amax = centers[i]  # 當前候選 amax

        # 3. 計算量化後的 centers
        # fake_tensor_quant: 模擬量化過程（不實際量化，只計算量化值）
        quant_centers = fake_tensor_quant(centers, amax, num_bits, unsigned)

        # 4. 計算 MSE（加權平均）
        # (quant_centers - centers)^2: 量化誤差的平方
        # * counts: 加權（每個 bin 的樣本數量）
        # .mean(): 平均誤差
        mse = ((quant_centers - centers)**2 * counts).mean()

        mses.append(mse.cpu())
        arguments.append(i)

    # 5. 選擇 MSE 最小的 amax
    logging.debug("mses={}".format(mses))
    argmin = np.argmin(mses)
    calib_amax = centers[arguments[argmin]]

    return calib_amax
```

### MSE 算法關鍵點

1. **候選 amax 範圍**:
   - 從 `start_bin=128` 開始（避免過小的 amax 導致過度量化）
   - 到 histogram 的最大值
   - 步長 `stride=1`（每個 bin 都嘗試）

2. **量化誤差計算**:
   ```python
   quant_centers = fake_tensor_quant(centers, amax, num_bits, unsigned)
   mse = ((quant_centers - centers)**2 * counts).mean()
   ```
   - `fake_tensor_quant`: 模擬量化過程，計算量化後的值
   - `(quant_centers - centers)^2`: 量化誤差的平方
   - `* counts`: 加權（考慮每個 bin 的樣本數量）
   - `.mean()`: 計算平均誤差

3. **優化目標**:
   - 最小化量化誤差的加權平方和
   - 平衡 clipping error（截斷誤差）和 rounding error（捨入誤差）

### fake_tensor_quant() 簡化邏輯

```python
def fake_tensor_quant(x, amax, num_bits, unsigned):
    """模擬量化過程"""
    # 計算 scale
    scale = amax / (2**(num_bits-1) - 1)  # 對於 8-bit: amax / 127

    # 量化：round(x / scale)
    quantized = torch.round(x / scale)

    # 反量化：quantized * scale
    dequantized = quantized * scale

    # 裁剪到範圍內
    dequantized = torch.clamp(dequantized, -amax, amax)

    return dequantized
```

---

## 關鍵代碼位置

### AWML 項目代碼

1. **入口點**:
   - `tools/detection3d/centerpoint_quantization.py:337` - `calibrator.calibrate(method="mse")`

2. **CalibrationManager**:
   - `projects/CenterPoint/quantization/calibration/calibrator.py:172` - `calibrate()`
   - `projects/CenterPoint/quantization/calibration/calibrator.py:95` - `collect_stats()`
   - `projects/CenterPoint/quantization/calibration/calibrator.py:145` - `compute_amax()`
   - `projects/CenterPoint/quantization/calibration/calibrator.py:62` - `set_quantizer_fast()`

### pytorch-quantization 庫代碼

1. **TensorQuantizer**:
   - `pytorch_quantization/nn/modules/tensor_quantizer.py:242` - `load_calib_amax()`

2. **HistogramCalibrator**:
   - `pytorch_quantization/calib/histogram.py:67` - `collect()`
   - `pytorch_quantization/calib/histogram.py:124` - `compute_amax()`
   - `pytorch_quantization/calib/histogram.py:257` - `_compute_amax_mse()` ⭐ **核心 MSE 實現**

---

## 總結

MSE Calibrator 在 PTQ 中的使用流程：

1. **準備階段**: 插入 Q/DQ 節點，融合 BN
2. **統計收集**: 前向傳播校準數據，累積 histogram
3. **MSE 優化**:
   - 遍歷 amax 候選值（從 start_bin 到最大值）
   - 對每個候選值計算量化誤差的加權 MSE
   - 選擇 MSE 最小的 amax
4. **應用量化**: 將計算出的 amax 應用到 TensorQuantizer

**MSE 方法的優勢**:
- 平衡 clipping error 和 rounding error
- 考慮數據分佈（通過 histogram 加權）
- 自動找到最優的量化範圍

**注意事項**:
- MSE 優化對 histogram 形狀敏感
- 不同的校準數據順序可能導致不同的結果
- 建議使用固定的 random seed 以確保可重現性
