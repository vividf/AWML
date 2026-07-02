# Spconv INT8 實作說明

本文件說明 **spconv 專案** 中 INT8 量化如何實作，以及 **Lidar_AI_Solution** 內如何透過 libspconv 進行 spconv INT8 推論。

---

## 一、spconv 專案中的 INT8 實作

spconv 的 INT8 支援建立在 **PyTorch FX 量化 (torch.ao.quantization)** 之上，並針對稀疏卷積與 `SparseConvTensor` 做了專用擴充。

### 1.1 架構概覽

- **量化流程**：PTQ（Post-Training Quantization）為主，並可選 QAT（Quantization-Aware Training）。
- **數值格式**：
  - **對稱量化**（symmetric）：`quant_min=-128`, `quant_max=127`，`qscheme=per_tensor_symmetric`。
  - **Activation**：per-tensor scale/zero_point。
  - **Weight**：per-channel scale（`default_per_channel_weight_observer`）。
- **底層計算**：INT8 卷積由 **cumm** 的 NVRTC 編譯 kernel 執行（`is_int8_inference=True` 的 algo），PyTorch 本身不負責 CUDA INT8 稀疏卷積。

### 1.2 關鍵模組與檔案

| 路徑 | 說明 |
|------|------|
| `spconv/pytorch/quantization/` | 量化主目錄 |
| `spconv/pytorch/quantization/fake_q.py` | PTQ/QAT 的 QConfig、Observer（SparseHistogramObserver、SparseMinMaxObserver 等），支援 `SparseConvTensor` 的 forward |
| `spconv/pytorch/quantization/core.py` | `quantize_per_tensor()`：對 `SparseConvTensor.features` 或一般 Tensor 做 per-tensor 量化；`quantized_add()`：INT8 residual add |
| `spconv/pytorch/quantization/backend_cfg.py` | 定義 spconv 的 BackendPatternConfig：Conv/ConvBN/ConvBNReLU/ConvAddReLU 等 fusion，以及 `SPCONV_STATIC_LOWER_*` 將 fused 模組對應到 quantized 實作 |
| `spconv/pytorch/quantization/quantized/conv.py` | `SparseConv`：量化後稀疏卷積，權重為 `qint8`，forward 中計算 `channel_scale = (inp_scale * w_scales) / out_scale`，並呼叫 `_conv_forward(..., channel_scale=..., output_scale=...)` |
| `spconv/pytorch/quantization/intrinsic/quantized/conv_relu.py` | `SparseConvReLU`、`SparseConvAddReLU`：fused 量化模組，同樣用 `channel_scale` / `output_scale` 呼叫 `_conv_forward`，並支援 ReLU、residual add |
| `spconv/pytorch/quantization/graph.py` | `transform_qdq()`：把圖中的 `torch.quantize_per_tensor` 換成支援 `SparseConvTensor` 的 `quantize_per_tensor`；`remove_conv_add_dq()`：對 SparseConvAddReLU 移除多餘的 dequant 以利 fusion |
| `spconv/pytorch/ops.py` | 實際卷積實作：當 `features.is_quantized and filters.is_quantized` 時走 INT8 分支，使用 `channel_scale`、`output_scale`，並在 cumm 端選擇 `is_int8_inference=True` 的 algo |

### 1.3 PTQ 流程（以 test/develop/mnist_int8_dev.py 為例）

1. **Prepare**  
   - 使用 `get_default_spconv_qconfig_mapping(is_qat=False)`（內部用 `default_symmetric_spconv_ptq_qconfig`）。  
   - `prepare_fx(model, qconfig_mapping, ..., backend_config=backend_cfg, prepare_custom_config=prepare_cfg)`  
   - 會 fuse 成 spconv 的 intrinsic 模組（如 ConvBNReLU、ConvAddReLU），並插入 Observer（例如 SparseHistogramObserver）。

2. **Calibrate**  
   - 用代表性資料跑一遍 prepared model，讓 Observer 統計 activation 的 min/max 以計算 scale/zero_point。

3. **Convert**  
   - `convert_fx(prepared_model, ...)` 將 observed 模組換成 quantized 模組（如 `SparseConv`、`SparseConvReLU`）。  
   - `transform_qdq(converted_model)`：替換 `quantize_per_tensor` / `quantized.add` 以支援 SparseConvTensor。  
   - `remove_conv_add_dq(converted_model)`：對 Add+ReLU 的 residual 路徑去掉多餘 dequant，讓 INT8 推論用 fused 的 SparseConvAddReLU。

4. **推論**  
   - 輸入先經 `QuantStub` 量化，進入稀疏卷積後全程以 INT8/量化格式計算，最後經 `DeQuantStub` 再轉回 float。  
   - 實際卷積計算在 `ops.py` 的 INT8 分支中呼叫 cumm 的 INT8 kernel（會觸發 NVRTC 編譯）。

### 1.4 數值細節

- **quantize_per_tensor**（core.py）：  
  - 對 `SparseConvTensor` 只量化 `features`，indices 等維持不變。  
- **SparseConv.forward**（quantized/conv.py）：  
  - `inp_scale = input.q_scale()`  
  - `w_scales = self.weight().q_per_channel_scales()`  
  - `channel_scale = (inp_scale * w_scales) / out_scale`  
  - bias 先除以 `out_scale` 再傳入 `_conv_forward`，與 TensorRT 風格的 symmetric per-tensor activation、per-channel weight 一致。  
- **quantized_add**（core.py）：  
  - 兩個量化 tensor 先 dequant 再相加，結果再 quantize 回 qint8，用於 residual 分支的數值一致。

---

## 二、Lidar_AI_Solution 中的 Spconv INT8

Lidar_AI_Solution 的 **3DSparseConvolution** 庫提供一個獨立的 **C++ 推論引擎**，使用預編譯的 **libspconv.so** 載入 ONNX，並可選擇 **FP16 或 INT8** 精度執行稀疏卷積骨幹（SCN）。

### 2.1 角色分工

- **PyTorch / spconv 側**（在別處完成）：  
  - 訓練或取得 FP16 模型後，用 spconv 的 PTQ（或 QAT）流程得到量化模型。  
  - 再匯出為 ONNX（例如 `centerpoint.scn.PTQ.onnx`），內含 INT8 的權重與量化參數（scale/zero_point 等）。  

- **Lidar_AI_Solution 側**：  
  - **不**包含 spconv 的 Python 量化程式碼。  
  - 只負責：用 **libspconv.so** 讀取上述 ONNX，並以 **INT8 或 FP16** 執行推論。

因此「spconv INT8 怎麼做」在 Lidar_AI_Solution 裡指的是：**如何用 libspconv 載入 PTQ ONNX 並開 INT8 推論**。

### 2.2 關鍵程式與介面

- **推論程式**：`libraries/3DSparseConvolution/src/infer.cpp`（以及可選的 `main.cpp`）。  
- **參數**：  
  - `--onnx=...`：ONNX 路徑（例如 PTQ 導出的 SCN）。  
  - `--fp16=true/false`、`--int8=true/false`：選擇精度。  
- **邏輯**（infer.cpp 中）：  
  - `task.int8 = args.at("int8") == "true"`  
  - `task.main_precision = task.int8 ? spconv::Precision::Int8 : spconv::Precision::Float16`  
  - `load_engine_from_onnx(task.onnx_file, task.main_precision, ...)`  
  - 引擎會依 `main_precision` 使用 INT8 或 FP16 kernel 執行 ONNX 中的稀疏卷積等算子。

### 2.3 如何使用 INT8 推論

1. **準備 PTQ ONNX**  
   - 在 PyTorch 環境中，用 spconv 的 PTQ 流程得到量化模型後，匯出為 ONNX（例如 `centerpoint.scn.PTQ.onnx`）。  
   - 文件中提到的範例：nuScenes 上 CenterPoint SCN 的 PTQ 模型，INT8 約 59.15 mAP / 66.45 NDS。

2. **編譯 3DSparseConvolution**  
   - 依 `libraries/3DSparseConvolution/README.md` 編譯 infer 或 main，並確保連結對應的 libspconv.so（依 CUDA 與架構選擇）。

3. **執行 INT8 推論**  
   - 命令中加上 `--int8=true`，並提供對應的 ONNX、feature、indice、grid_size 等參數。  
   - 例如：  
     `./infer --onnx=centerpoint/centerpoint.scn.PTQ.onnx --feature=... --indice=... --grid_size=... --int8=true ...`

4. **效能與驗證**  
   - `workspace/perf-int8.log` 等日誌會記錄各樣本的 voxelization 與 SCN 推論時間。  
   - 可與 FP16 結果比較（如 `perf-float16-*.log`），並用 `tool/compare.py` 比對輸出 tensor。

### 2.4 支援的算子與限制（摘自 3DSparseConvolution README）

- **支援算子**：SparseConvolution（含 Submanifold / Inverse）、Add、ReLU、Add&ReLU、ScatterDense、Reshape、ScatterDense&Transpose 等。  
- **精度**：透過 `--int8` / `--fp16` 切換，由 libspconv 依 ONNX 與 precision 選擇 INT8 或 FP16 實作。  
- **硬體**：需支援 libspconv 所編譯的架構（如 sm_80/sm_86/sm_87 等，依 libspconv 版本說明）。

---

## 三、對照總表

| 項目 | spconv 專案 | Lidar_AI_Solution (3DSparseConvolution) |
|------|-------------|----------------------------------------|
| INT8 實作位置 | PyTorch 側：quantization/* + ops.py + cumm kernel | C++ 推論引擎 + libspconv.so |
| 量化方式 | PTQ（prepare → calibrate → convert）+ 可選 QAT | 使用已匯出的 PTQ ONNX |
| 數值格式 | 對稱、per-tensor activation、per-channel weight | 由 ONNX 與 libspconv 解析 |
| 實際 INT8 計算 | cumm NVRTC INT8 稀疏卷積（ops.py） | libspconv 內建 INT8 kernel |
| 典型用途 | 訓練、校準、匯出 ONNX、Python 端驗證 | 部署時 C++ INT8 推論、效能測試 |

---

## 四、參考檔案清單

**spconv 專案（INT8 實作與範例）**

- `spconv/pytorch/quantization/__init__.py`  
- `spconv/pytorch/quantization/fake_q.py`  
- `spconv/pytorch/quantization/core.py`  
- `spconv/pytorch/quantization/backend_cfg.py`  
- `spconv/pytorch/quantization/quantized/conv.py`  
- `spconv/pytorch/quantization/intrinsic/quantized/conv_relu.py`  
- `spconv/pytorch/quantization/graph.py`  
- `spconv/pytorch/ops.py`（INT8 分支與 `_conv_forward`）  
- `spconv/test/develop/mnist_int8_dev.py`（PTQ 範例）  

**Lidar_AI_Solution（INT8 推論與文件）**

- `libraries/3DSparseConvolution/README.md`  
- `libraries/3DSparseConvolution/src/infer.cpp`（`--int8`、`main_precision`、`load_engine_from_onnx`）  
- `libraries/3DSparseConvolution/tool/compare.py`（結果比對）  
- `libraries/3DSparseConvolution/workspace/perf-int8.log`（INT8 效能日誌）  

以上即為 spconv INT8 在「spconv 專案」與「Lidar_AI_Solution」中的實作與使用方式說明。
