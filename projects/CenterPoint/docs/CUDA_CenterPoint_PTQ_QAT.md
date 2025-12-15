## CUDA-CenterPoint 量化流程（PTQ / QAT）解讀

說明來源：`CUDA-CenterPoint/qat/tools/{centerpoint_qat_pipline.py, sparseconv_quantization.py}` 與 `qat/README.md`。

### Q/DQ 設計與量化模組
- 使用 NVIDIA `pytorch_quantization` 的 `TensorQuantizer` 作為 Q/DQ。
- 量化包裹：
  - 卷積：`QuantConv2d` / `QuantConvTranspose2d`，input 量化描述子 `calib_method="histogram"`，weight 描述子使用官方 per-channel 8-bit（預設 MaxCalibrator）。
  - 全連接：`QuantLinear`，input `histogram`，weight per-row 8-bit（預設 MaxCalibrator）。
  - 稀疏卷積：`SparseConvolutionQunat`（spconv），input `QuantDescriptor(num_bits=8, calib_method="histogram")`，weight 8-bit per-channel（預設 MaxCalibrator）。
  - Add 節點：注入 `QuantAdd`，對加法輸入做對齊量化，避免輸入尺度不同。
- 模組替換：
  - `quant_sparseconv_module(model)` 走訪模型，將 `spconv.SubMConv3d/SparseConv3d` 換成 `SparseConvolutionQunat`。
  - `quant_add_module(model)` 在 `SparseBasicBlock` 插入 `QuantAdd`。
  - 對於 2D/Linear 亦有 `quant_conv_module` / `quant_linear_module`（在 AWML 對應實作）。
- Fast histogram：`set_quantizer_fast` 會對 `HistogramCalibrator` 設 `_torch_hist=True`，用 torch.histogram 加速。

### 校準（PTQ）行為
- 校準入口：`calibrate_model(model, dataloader, device, batch_processor, num_batch)`。
- 步驟：
  1. 啟用 calibrate 模式（disable fake quant，enable calib）。
  2. 以 `num_batch` 批資料跑前向（調用 `batch_processor_callback`），收集直方圖/範圍。
  3. 關閉 calibrate，啟用 quant。
  4. `compute_amax(model, method="mse")`：
     - `HistogramCalibrator` → `load_calib_amax(method="mse")`（固定 mse）。
     - `MaxCalibrator` → `load_calib_amax()`（max 行為）。
- 因此：Activation 走 Histogram + mse；Weight 走 MaxCalibrator（max）。

### PTQ 流程（centerpoint_qat_pipline.py，帶 `--ptq_mode`）
1. 讀取 config / checkpoint，建模型並移到 CUDA。
2. 替換 spconv / add 為量化版（上節）。
3. 可選 fuse BN（`funcs.layer_fusion_bn`）降低偏差。
4. 開 fast histogram，執行 `calibrate_model(..., method="mse")`。
5.（若 `--ptq_mode`）直接存成 `ptq.pth` 作推論或 ONNX 匯出；不進入訓練。

### QAT 流程（同檔案，不帶 `--ptq_mode`）
1. 前置同 PTQ：量化包裹 + 校準（mse）。
2. 執行 `train_detector_QAT`（帶假量化的前向與反向），可搭配 BN fuse、學習率調低微調。
3. 訓練後的量化權重可直接推論或匯出 ONNX。

### ONNX / TensorRT 匯出
- PTQ/QAT 生成的量化模型可用 `onnx_export/export-scn.py` 匯出 ONNX；匯出前通常先 fuse ReLU。
- TensorRT 推論再由 C++ (`src/tensorrt.cpp` 等) 載入。

### CUDA-CenterPoint vs AWML CenterPoint 差異對照
- **量化描述子（activation / weight）**
  - CUDA-CenterPoint：activation `calib_method="histogram"`；weight 預設 MaxCalibrator（未指定 `calib_method`）。
  - AWML：activation `histogram`；weight MaxCalibrator。
  - 差異：無。

- **amax 方法選擇**
  - CUDA-CenterPoint：寫死 HistogramCalibrator 用 `method="mse"`；MaxCalibrator 用 max；無 CLI。
  - AWML：同樣寫死 HistogramCalibrator 用 `mse`，MaxCalibrator 用 max；已移除 amax CLI。
  - 差異：無。

- **fast histogram 開關**
  - CUDA-CenterPoint：固定開 `_torch_hist=True`。
  - AWML：曾有 `--disable-fast-hist`，現已移除，預設開啟。
  - 差異：行為相同（都開），AWML 以前有開關。

- **Add 節點量化**
  - CUDA-CenterPoint：有 `QuantAdd` 注入 `SparseBasicBlock`（加法輸入對齊量化）。
  - AWML：已補上 `QuantAdd` 模組與 `attach_quant_add(model)`；需在自家殘差模組的 forward 內呼叫 `self.quant_add(x, y)`（預設匹配類名含 BasicBlock/SparseBasicBlock 並附加 quant_add 屬性）。
  - 差異：已對齊功能，但 AWML 需在 forward 實際使用 `quant_add` 才會生效。

- **SparseConv 量化**
  - CUDA-CenterPoint：有 `SparseConvolutionQunat`，並在 spconv 模組自動替換。
  - AWML：未提供 spconv 量化包裹，主針對 2D conv/linear。
  - 差異：有（CUDA 支援 spconv；AWML 無）。

- **BN 融合**
  - CUDA-CenterPoint：PTQ/QAT 流程可選 BN fuse。
  - AWML：PTQ 有 fuse_model_bn；行為一致。
  - 差異：無。

- **Sensitivity 工具**
  - CUDA-CenterPoint：`centerpoint_eval.py sensitivity`。
  - AWML：`centerpoint_quantization.py sensitivity`。
  - 差異：功能類似。

- **CLI / 流程入口**
  - CUDA-CenterPoint：`centerpoint_qat_pipline.py` 驅動 PTQ/QAT；無 amax 選項，校準批次可調。
  - AWML：`centerpoint_quantization.py` 子命令（ptq/sensitivity/qat）；amax 寫死 mse，校準批次可調。
  - 差異：入口檔名與界面不同，行為相同。
