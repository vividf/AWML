# INT8 Inference: spconv 與 Lidar_AI_Solution 支援方式說明與比較

本文件說明 **spconv** 與 **Lidar_AI_Solution** 兩個 repository 如何支援 INT8 推論，並比較兩者是否相同。內容基於實際程式碼閱讀與追蹤。

---

## 一、spconv repository 如何支援 INT8 推論

spconv 的 INT8 支援分為兩層：**Python 量化與模組**（`spconv` repo）與 **C++/CUDA 核心**（`spconv_cpp` / cumm）。

### 1.1 Python 層：`spconv.pytorch.quantization`

路徑：`spconv/spconv/pytorch/quantization/`。

- **用途**：提供 Post-Training Quantization (PTQ) 與 QAT，並將模型轉成使用 **qint8 權重與 activation** 的靜態量化模組，與 TensorRT 相容（symmetric、per-tensor activation、per-channel weight）。
- **主要檔案**：
  - `backend_cfg.py`：Backend 設定、fuse 規則（conv-bn、conv-bn-relu、conv-add-relu 等）、dtype config（qint8）。
  - `fake_q.py`：`get_default_spconv_trt_ptq_qconfig` / `get_default_spconv_trt_qat_qconfig`，對應 symmetric、per-tensor activation、per-channel weight。
  - `core.py`：`quantize_per_tensor` 支援 `SparseConvTensor` 的量化。
  - `quantized/conv.py`：量化後的 `SparseConv`，權重為 `torch.qint8`，並有 `scale`、`zero_point`；forward 時計算 `channel_scale = (inp_scale * w_scales) / out_scale`，再呼叫 `_conv_forward(..., channel_scale=channel_scale, output_scale=out_scale)`。
  - `intrinsic/quantized/conv_relu.py`：Fused SparseConv+ReLU / SparseConv+Add+ReLU 的量化實作。

**PTQ 流程**（對應 `test/develop/mnist_int8_dev.py`）：

1. **prepare**：`qfx.prepare_fx(model, qconfig_mapping, ..., backend_config=backend_cfg, prepare_custom_config=prepare_cfg)`  
   - Fuse conv-bn-relu 等成 intrinsic 模組（如 `SpconvBnReLUNd`、`SpconvBnAddReLUNd`），並插入 observer。
2. **calibrate**：用 representative data 跑一遍，讓 observer 統計 min/max 等。
3. **convert**：`qfx.convert_fx(...)` 將 observer 換成實際量化，intrinsic 換成 quantized 模組（如 `SparseConv`、`SparseConvReLU`）。
4. **後處理**：`transform_qdq`（把 `torch.quantize_per_tensor` 換成 spconv 的 `quantize_per_tensor`）、`remove_conv_add_dq`（對 SparseConvAddReLU 的 residual 輸入去掉多餘的 dequantize）。

因此，**spconv 的 INT8 支援在 Python 側** = 上述 PTQ/QAT 流程 + 量化後的 SparseConv/SparseConvReLU/SparseConvAddReLU，其 forward 會帶著 qint8 的 input/weight 與 scale 資訊往下傳。

### 1.2 從 Python 到 C++：`_conv_forward` 與 is_int8

路徑：`spconv/spconv/pytorch/conv.py`。

- `_conv_forward` 簽名包含 `channel_scale`、`output_scale`（可選）。
- 關鍵邏輯：
  - `is_int8 = input.is_quantized and weight.is_quantized`
  - 若 `is_int8`，則要求 `output_scale`、`channel_scale` 非 None，且目前實作要求提供 `bias`。
  - 若為 int8，residual add 的 scale 用 `add_input.q_scale()` 當 `output_add_scale`。
- 實際呼叫 C++ 時會傳入：
  - `features`（qint8）、`weight`（qint8）、`bias`、`output_scale`、`channel_scale`（per-channel scale）、以及可選的 `output_add`/`output_add_scale`。

也就是說，**真正執行 INT8 卷積的是底層 C++/CUDA**；Python 只負責量化參數與呼叫介面。

### 1.3 C++/CUDA 層：spconv_cpp 的 `is_int8_inference`

路徑：`spconv_cpp/spconv/src/spconvlib/spconv/csrc/sparse/convops/`。

- **Algo 選擇**（`ConvTunerSimple_get_all_available.cc`）：
  - 若 `!bias.empty() && !scale.empty()`：只保留 `desp.is_int8_inference == true` 的 algo。
  - 若 bias/scale 為空：則**排除** `is_int8_inference == true` 的 algo。
- 因此，**當 Python 傳下 int8 的 features/weight 並附上 bias 與 per-channel scale 時**，C++ 會自動選到 INT8 的 kernel（is_int8_inference）。
- **執行**（`ConvGemmOps_implicit_gemm.cc`）：
  - 若 `tune_res.algo_desp.is_int8_inference`：`alpha = output_scale`；bias/scale 在 kernel 內處理；若有 `output_add`，則 `beta = output_add_scale / output_scale`。
- **run_with_tuned_result**（`ConvTunerSimple_run_with_tuned_result.cc`）：
  - 若 `desp.is_int8_inference` 且 `!output_add.empty()`，會設定 `params.output_add = output_add`，在 kernel 內做 fused add。

結論：**spconv 的 INT8 推論** = Python 用 PTQ/QAT 得到 qint8 模型並傳入 scale/bias → 同一套 C++ 引擎根據 dtype 與 bias/scale 選 `is_int8_inference` 的 algo → 在 CUDA 上跑 INT8 sparse convolution（含 fused bias/scale/add）。

---

## 二、Lidar_AI_Solution repository 如何支援 INT8 推論

Lidar_AI_Solution 中與 SCN INT8 相關的部分主要在 **3DSparseConvolution** 這個 library。

### 2.1 架構：libspconv.so + ONNX + infer 工具

- **README**（`libraries/3DSparseConvolution/README.md`）說明：
  - 使用 **libspconv.so** 做 3D sparse convolution 的 inference，支援 **int8 / fp16**。
  - 範例：`centerpoint.scn.PTQ.onnx` + libspconv.so → **INT8**，mAP/NDS 與 PyTorch FakeQuant INT8 接近。
- 因此，**INT8 的實際運算發生在 libspconv.so 內**，而不是 Lidar_AI_Solution 自己實作一套 INT8 kernel。

### 2.2 infer 程式如何啟用 INT8

路徑：`Lidar_AI_Solution/libraries/3DSparseConvolution/src/infer.cpp`。

- `InferenceTask` 有成員：
  - `bool fp16;`
  - `bool int8;`
  - `spconv::Precision main_precision;`
- 從參數載入（約 184–188 行）：
  - `task.fp16 = args.at("fp16") == "true";`
  - `task.int8 = args.at("int8") == "true";`
  - `task.main_precision = task.int8 ? spconv::Precision::Int8 : spconv::Precision::Float16;`
- 跑推論時（約 264–270 行）：
  - `spconv::load_engine_from_onnx(task.onnx_file, task.main_precision, ...)`

也就是說，**Lidar_AI_Solution 的 INT8 支援** = 使用者傳 `--int8=true` → `main_precision = Precision::Int8` → 用 **同一個 libspconv.so** 以 **Int8** 精度建 engine 並執行。INT8 kernel 的選擇與執行邏輯都在 libspconv.so 內部（即與 spconv_cpp 的 `is_int8_inference` 同一套）。

### 2.3 ONNX 與 PTQ 模型來源

- README 表格中的 **centerpoint.scn.PTQ.onnx** 對應「PTQ Model, spconv.so INT8 Inference」。
- 匯出流程在 `tool/centerpoint-export/export-scn.py`：載入 CenterPoint SCN backbone、做 layer fusion、再 `exptool.export_onnx(...)`。PTQ 版本會是先用 spconv 的 PTQ 流程（或相同量化設定）得到量化權重與 scale，再匯出成 ONNX；ONNX 內會帶 quantized 的權重與量化參數。
- libspconv 的 ONNX parser 會讀取這些權重與精度設定，並用 `Precision::Int8` 建 engine，因此會走與 spconv_cpp 相同的 INT8 路徑（is_int8_inference + bias/scale）。

因此，**Lidar_AI_Solution 並沒有自己實作 INT8 卷積**，而是：
- 使用與 spconv 生態相同的 PTQ/量化設定產生 PTQ ONNX；
- 透過 **同一個 C++ engine（libspconv.so）** 並設定 `Precision::Int8` 來跑 INT8。

---

## 三、兩者是否相同？比較與結論

### 3.1 對照表

| 項目 | spconv (Python + spconv_cpp) | Lidar_AI_Solution (3DSparseConvolution) |
|------|-----------------------------|----------------------------------------|
| INT8 量化方式 | PTQ/QAT，symmetric、per-tensor act、per-channel weight（TensorRT 相容） | 使用 PTQ ONNX（與 spconv 相同量化設定） |
| 誰執行 INT8 卷積 | spconv_cpp / cumm 的 C++/CUDA kernel（is_int8_inference） | 同一個 libspconv.so（即 spconv_cpp 的 engine） |
| 誰選擇 INT8 algo | ConvTunerSimple：依 input/weight dtype 與 bias/scale 是否為空，篩選 is_int8_inference | 同一個 engine：依 Precision::Int8 與 ONNX 中的權重/scale 決定 |
| 呼叫介面 | Python：SparseConv forward 傳 qint8 + scale/bias | C++：load_engine_from_onnx(..., Precision::Int8) |
| Fused bias/scale/add | 在 ConvGemmOps / run_with_tuned_result 中支援 | 同一套 kernel 與 params（output_add、scale、bias） |

### 3.2 結論

- **底層 INT8 實作是同一套**：都是 spconv_cpp / libspconv.so 內、標記為 `is_int8_inference` 的 sparse convolution kernel；algo 選擇邏輯（bias/scale 非空時選 int8 algo）也一致。
- **量化格式一致**：皆為 symmetric、per-tensor activation、per-channel weight，與 spconv 的 `get_default_spconv_trt_ptq_qconfig` 及 backend_cfg 一致；Lidar_AI_Solution 的 PTQ ONNX 來自同一套量化流程。
- **差異僅在「誰呼叫」**：
  - **spconv**：在 Python 裡做 PTQ/QAT、得到量化模型，透過 `spconv.pytorch` 的 SparseConv forward 直接呼叫 C++（qint8 + scale/bias）。
  - **Lidar_AI_Solution**：先匯出 PTQ ONNX，再由 C++ infer 程式用 `load_engine_from_onnx(..., Precision::Int8)` 載入同一個 engine，用同一套 kernel 跑 INT8。

因此，**兩者支援的 INT8 推論在本質上是相同的**：同一套量化設定、同一套 C++/CUDA INT8 kernel（is_int8_inference）、同一套 engine；差別只在於一個從 Python 直接呼叫、一個從 ONNX + C++ 載入 engine 並設定 `Precision::Int8`。

---

## 四、關鍵程式碼位置速查

- **spconv Python 量化**  
  - `spconv/pytorch/quantization/__init__.py`、`backend_cfg.py`、`fake_q.py`、`core.py`  
  - `spconv/pytorch/quantization/quantized/conv.py`（SparseConv forward 與 channel_scale/output_scale）
- **spconv Python 偵測 int8 並傳參**  
  - `spconv/pytorch/conv.py`：`_conv_forward` 中 `is_int8 = input.is_quantized and weight.is_quantized` 及 bias/scale/output_add_scale 傳遞
- **spconv_cpp INT8 algo 與執行**  
  - `ConvTunerSimple_get_all_available.cc`：bias/scale 非空時只保留 `is_int8_inference`  
  - `ConvGemmOps_implicit_gemm.cc`：is_int8_inference 時 alpha、beta、output_add 處理  
  - `ConvTunerSimple_run_with_tuned_result.cc`：is_int8_inference 時設定 output_add
- **Lidar_AI_Solution INT8 開關與 engine**  
  - `libraries/3DSparseConvolution/src/infer.cpp`：`task.int8`、`task.main_precision`、`load_engine_from_onnx(..., task.main_precision, ...)`  
  - `libraries/3DSparseConvolution/README.md`：INT8/FP16 模型與 libspconv.so 說明  
  - `libraries/3DSparseConvolution/libspconv/include/spconv/engine.hpp`：`Precision::Int8` 定義與 engine 介面

以上為根據程式碼追蹤得到的 INT8 支援說明與兩者一致性分析。

---

## 五、開源與閉源界線：`load_engine_from_onnx` 與 spconv 的差別

### 5.1 Lidar_AI_Solution：哪些開源、哪些不開源

在 Lidar_AI_Solution 的 3DSparseConvolution 中，**`load_engine_from_onnx` 的「本體」其實是開源的**，但**依賴的 engine 實作是不開源的**：

- **有開源、在 repo 裡**  
  - **ONNX 解析與建圖**：`libraries/3DSparseConvolution/src/onnx-parser.cpp`（以及 `libspconv/parser/onnx-parser.cpp` 同份邏輯）就是 `load_engine_from_onnx` 的實作。
  - 流程可以從程式碼直接看到：
    1. 用 protobuf 讀取 ONNX (`model.ParseFromIstream`)，取得 `model.graph()`。
    2. 呼叫 `spconv::create_engine_builder()` 取得一個 `builder`（此函式實作在 libspconv.so，見下）。
    3. 對每個 input 做 `builder->push_input(name)`。
    4. 對每個 node 依 `node.op_type()` 分派：
       - **SparseConvolution**：從 graph 的 initializer 讀 weight、bias（目前程式碼裡用 `get_initializer_data` 只支援 **FLOAT16** 的 initializer），從 attribute 讀 `weight_dynamic_ranges`、`kernel_size`、`stride`、`activation`、`precision`、`output_precision` 等；呼叫 `builder->push_sparse_conv(..., precision/output_precision 由 attribute "precision"/"output_precision" 決定為 Int8 或 Float16, ...)`。
       - **Add / QuantAdd**：`builder->push_add(..., precision, output_precision, ...)`。
       - **Relu、ScatterDense、Reshape、Transpose**：對應的 `push_relu`、`push_dense`、`push_reshape`、`push_transpose`。
    5. 收集 output，`builder->push_output(...)`。
    6. 最後 `builder->build(precision, sortmask, enable_blackwell, with_auxiliary_stream, stream)`，其中 **`precision` 就是 `load_engine_from_onnx` 的參數**（例如 `--int8=true` 時傳入的 `Precision::Int8`），會覆蓋/驅動整體推論精度。

- **不開源、只有預編譯庫**  
  - **libspconv.so**（路徑如 `libspconv/lib/x86_64_cuda12.8/libspconv.so`）是**預編譯的二進位**，repo 裡**沒有**其原始碼。
  - 該 .so 至少提供：
    - `create_engine_builder()` 的實作；
    - `EngineBuilder` 的實作（`push_sparse_conv`、`push_add`、`push_relu`、`push_dense`、`build` 等）；
    - `Engine` 的實作（`forward`、`input`、`output` 等）。
  - 因此：**「從 ONNX 建出可執行的 engine、以及 engine->forward() 時真正呼叫的 INT8/FP16 kernel」** 都在 libspconv.so 內部，無法從 Lidar_AI_Solution 的 repo 看到對應 C++/CUDA 原始碼。

所以更精確地說：**`load_engine_from_onnx` 的「ONNX 讀取 + 呼叫 builder API 的邏輯」是開源的**；**「builder 與 engine 的實作、以及實際跑 INT8 的 kernel」是不開源的**（在 libspconv.so 裡）。

### 5.2 libspconv.so 大概是怎麼做的（推論）

從開源的 parser 與 spconv_cpp 的開源實作可以合理推論 libspconv.so 內部流程大致是：

1. **create_engine_builder()** 回傳一個實作 `EngineBuilder` 介面的物件。
2. **push_sparse_conv** 等：把當前層的參數（weight、bias、weight_dynamic_ranges、kernel_size、precision 等）存起來，建立對應的「層」描述，不立刻編譯。
3. **build(precision, ...)**：  
   - 依各層的 precision 與全域 `precision` 決定每層用 FP16 還是 INT8；  
   - 為每一層配置權重/scale 等（INT8 時會用到 weight_dynamic_ranges 等做量化）；  
   - 選 algo（對應 spconv_cpp 裡 bias/scale 非空時選 `is_int8_inference` 的邏輯）；  
   - 組成一條可執行的 forward 順序（可能類似 spconv_cpp 的 ConvGemmOps + ConvTunerSimple）。
4. **Engine::forward(stream)**：依序執行每一層，呼叫的應是與 **spconv_cpp 同一套** 的 sparse conv kernel（含 `is_int8_inference` 的 INT8 路徑），只是透過封裝好的 Engine 介面呼叫，而不是從 Python 經由 PyTorch 模組進來。

也就是說：**INT8 的「做法」與 spconv_cpp 一致**（symmetric、per-channel weight、bias/scale 在 kernel 內處理）；**差別只在「誰組裝圖、誰呼叫」**——Lidar_AI_Solution 用 ONNX + 閉源的 builder/engine，spconv 用 Python 模組 + 開源的 spconv_cpp。

### 5.3 spconv repository：是否開源？

**是，spconv 這條路徑是開源的**（在對應的 GitHub 上）：

- **Python**：`spconv` repo 內所有 PTQ/QAT、quantized 模組、`_conv_forward` 等，原始碼都在。
- **C++/CUDA**：`spconv_cpp`（以及依賴的 cumm）repo 內，ConvTunerSimple、ConvGemmOps、algo 選擇（`is_int8_inference`）、run_with_tuned_result 等，原始碼都在；編譯後是自建的 lib，不是預編譯的 libspconv.so。
- **沒有**在 spconv 主 repo 裡提供「從 ONNX 建 Engine 再 forward」的這條 C++ 流程；那是 Lidar_AI_Solution 的做法，且其 engine 實作在 libspconv.so（閉源）。

### 5.4 開源／閉源對照

| 項目 | spconv (Python + spconv_cpp) | Lidar_AI_Solution (3DSparseConvolution) |
|------|-----------------------------|----------------------------------------|
| 量化與 PTQ 流程 | 開源（Python） | 使用與 spconv 相同流程匯出 ONNX，匯出腳本開源 |
| ONNX 解析與 load_engine_from_onnx 邏輯 | 不適用（不走 ONNX engine） | **開源**（onnx-parser.cpp） |
| EngineBuilder / Engine 實作、forward 與 INT8 kernel | 開源（spconv_cpp + cumm，以 PyTorch 延伸形式呼叫） | **閉源**（libspconv.so 預編譯庫） |
| 實際 INT8 運算 | 開源（同一套 is_int8_inference kernel） | 閉源封裝，但推論為同一套 kernel |

**總結**：  
- **Lidar_AI_Solution**：`load_engine_from_onnx` 的「怎麼讀 ONNX、怎麼呼叫 builder」是開源的；「builder 與 engine 怎麼建、怎麼跑 INT8」在 libspconv.so 裡，不開源。  
- **spconv**：從 Python 量化到 C++ INT8 kernel 整條鏈路都是開源的；沒有提供 ONNX → C++ Engine 的開源實作，那是 Lidar_AI_Solution 搭配閉源 libspconv.so 的做法。

---

## 六、spconv 的 libspconv.so 與 Lidar_AI_Solution 的 libspconv.so 是否相同？

**結論：兩者是不同的產品／artifact，不能混為一談。**

### 6.1 spconv repository 裡的 libspconv.so

- **來源**：在 spconv 專案裡，可透過 **純 C++ 建置流程** 自己建出 libspconv.so。  
  - 依 `docs/PURE_CPP_BUILD.md` 與 `example/libspconv/README.md`：先執行 `python -m spconv.gencode` 產生 C++ 程式碼，再經 cmake 編譯得到 **libspconv.so**。  
  - 文件說明：「libspconv + pybindings = core_cc.so in spconv python package」、「spconv python and libspconv use **same c++ code**」——也就是說，**與 Python 套件用的 C++/CUDA 是同一套**（spconv_cpp/cumm），只是打包成獨立 .so，不帶 Python 綁定。
- **API**：文件寫「currently not available」，需對照 Python 與 C++ 原始碼理解；也就是**以 op 為單位的較底層 API**（例如一層一層傳 tensor 做 sparse conv），**沒有** EngineBuilder / Engine / load_engine_from_onnx 這類「ONNX → 建圖 → forward」的高階介面。
- **開源**：可從 spconv + spconv_cpp/cumm 原始碼自行建置，整條鏈路開源。

### 6.2 Lidar_AI_Solution 裡的 libspconv.so

- **來源**：在 Lidar_AI_Solution 的 `libraries/3DSparseConvolution/libspconv/lib/` 下是**預編譯好的 .so**（例如 `x86_64_cuda12.8/libspconv.so`），**repo 內沒有該 .so 的原始碼**，也不會在該 repo 裡從 spconv 的 gencode 重新建這個檔。
- **API**：提供 **EngineBuilder / Engine** 等高階介面——`create_engine_builder()`、`builder->push_sparse_conv`、`builder->build(precision, ...)`、`engine->forward(stream)` 等，專門給 **ONNX 解析 + C++ inference** 使用（與 3DSparseConvolution 的 onnx-parser.cpp 搭配）。
- **開源**：.so 本身為閉源；僅能從行為與文件推測內部可能使用與 spconv 相同或相近的 kernel 設計（例如 is_int8_inference），但 **builder/engine 的實作與建置方式不公開**。

### 6.3 對照表

| 項目 | spconv 的 libspconv.so | Lidar_AI_Solution 的 libspconv.so |
|------|------------------------|-----------------------------------|
| 如何取得 | 從 spconv 原始碼 + gencode + cmake **自己建** | 使用 repo 內**預編譯**的 .so |
| 底層程式碼 | 與 Python 套件**同一套** C++/CUDA（spconv_cpp/cumm） | 推測與 spconv 同源或同設計，但**無法從 repo 確認** |
| 對外 API | 底層 op 級（無 Engine/ONNX 高階 API） | **EngineBuilder + Engine**（ONNX inference 用） |
| 開源與否 | **可自行建置，開源** | **閉源**（僅二進位） |

因此：**名字都叫 libspconv.so，但是兩個不同的「庫」**——  
- **spconv 的**：自己編出來的、同一套 C++ 程式碼、底層 API、開源。  
- **Lidar_AI_Solution 的**：別人編好的、高階 Engine API、給 ONNX 推論用、閉源。  

若你要在 C++ 裡做「ONNX → load_engine_from_onnx → forward」這條路，只能使用 **Lidar_AI_Solution 那份** libspconv.so（或與其相容的預編譯庫）；spconv 專案裡自己建出來的 libspconv.so **沒有** 這些 Engine/ONNX 介面。

---

## 七、AWML create ONNX / Autoware create engine 與 TRT plugin：若要支援 INT8 該如何處理

本節以 **AWML 的 create ONNX**、**Autoware BEVFusion 的 create engine 與 TensorRT plugin** 為脈絡，說明目前流程與 **若要支援 INT8 應如何處理**（依現有程式碼與文件整理）。

### 7.1 目前 AWML「create ONNX」在做什麼

- **位置**：`deployment/projects/bevfusion/export/onnx_export_pipeline.py`（`BEVFusionONNXExportPipeline`）。
- **流程**：
  1. 載入 PyTorch BEVFusion 模型（訓練好的 FP 權重）。
  2. 用一筆 sample 做 voxelization（`voxels`, `coors`, `num_points_per_voxel`）。
  3. 以 `BEVFusionMainBodyWrapper` 包一層，對外介面為 `(voxels, coors, num_points_per_voxel) → (bbox_pred, score, label_pred)`。
  4. 呼叫 `torch.onnx.export(...)` 匯出成 **單一 ONNX**（例如 `bevfusion_lidar.onnx`）。
  5. 對 TopK 做 TensorRT 相容修正（常數 K）。
- **精度**：匯出的是 **FP 模型**（FP32/FP16，依訓練時設定）；**沒有** PTQ、沒有 qint8、沒有 INT8 相關的 attribute 或 initializer。
- **稀疏卷積**：模型內含 `pts_middle_encoder`（sparse encoder），即 SCN（SubMConv3d / SparseConv3d 等）。ONNX 匯出時這些會變成 **自訂 op 或 TRT 不認識的子圖**，需在 TensorRT 端用 **plugin** 接 libspconv 執行。

### 7.2 目前「create engine」與 TRT 設定

- **位置**：  
  - AWML：`deployment/projects/bevfusion/export/tensorrt_export_pipeline.py`（`BEVFusionTensorRTExportPipeline`）＋ `deployment/exporters/common/tensorrt_exporter.py`（`TensorRTExporter`）。  
  - 設定：`deployment/projects/bevfusion/config/deploy_config.py`（`tensorrt_config.precision_policy="auto"`）。
- **流程**：
  1. 讀取上一步產出的 ONNX。
  2. `trt.init_libnvinfer_plugins(trt_logger, "")`（載入 TRT plugin，包含接 libspconv 的 SCN plugin）。
  3. 用 ONNX parser 建 TensorRT network。
  4. 依 **precision_policy** 設定 builder flags（`deployment/configs/schema.py` 的 `PRECISION_POLICIES`）：目前有 `auto`、`fp16`、`fp32_tf32`、`strongly_typed`，**沒有 INT8**。
  5. 設定 optimization profile（shape min/opt/max）。
  6. `builder.build_serialized_network()` 產出 engine 檔（例如 `bevfusion_lidar.engine`）。
- **Autoware BEVFusion**（依 SPCONV_CPP.md）：  
  - 推論為 **C++ 節點 + TensorRT + libspconv.so**；稀疏卷積由 **TensorRT plugin** 連結 libspconv 執行。  
  - 因此「create engine」時，TRT 會遇到 SCN 相關的自訂 op，由 **TRT plugin** 在 runtime 呼叫 libspconv；plugin 與 libspconv 需一致（例如都 FP16，或都要支援 INT8）。

### 7.3 若要支援 INT8，需要動哪些部分？

整體有兩塊：**(1) 稠密部分（Dense）** 用 TensorRT 原生 INT8；(2) **稀疏部分（SCN）** 用 libspconv 的 INT8（或 TRT plugin 內呼叫 libspconv INT8）。下面分「ONNX」「TensorRT / create engine」「TRT plugin / libspconv」三點說明。

#### (1) ONNX 匯出（AWML create ONNX）

- **現狀**：單一 ONNX、FP、內含 SCN。
- **若要 SCN 走 INT8**，有兩種常見做法：
  - **做法 A（與 Lidar_AI_Solution 對齊）**  
    - 先對 **SCN 子網路** 做 PTQ（spconv 的 `prepare_fx` → calibrate → `convert_fx`，得到 qint8 權重與 scale）。  
    - 再匯出 **SCN 專用 ONNX**，格式與 Lidar_AI_Solution 的 parser 一致：  
      - 節點型別為自訂的 `SparseConvolution` 等；  
      - initializer 為權重（例如 FP16 存，或依 parser 支援寫成 int8）；  
      - attribute 帶 `weight_dynamic_ranges`、`precision="int8"`、`output_precision="int8"` 等（見 Lidar_AI_Solution `onnx-parser.cpp`）。  
    - 這樣 **C++ 端** 可用 `load_engine_from_onnx(..., Precision::Int8)` 載入並跑 SCN INT8；**與現有 AWML 單一 BEVFusion ONNX 不同**，會變成「稠密部分一個 ONNX → TRT」「SCN 一個 ONNX → libspconv」的 **雙 ONNX** 流程。
  - **做法 B（維持單一 ONNX給 TRT）**  
    - 維持目前單一 ONNX，但 SCN 對應的自訂 op 在 ONNX 裡帶 **INT8 相關 attribute**（例如 precision、scale 等）。  
    - **TensorRT plugin** 在 build/run 時讀這些 attribute，把權重與 scale 傳給 libspconv，並以 **INT8** 模式呼叫 libspconv（即 plugin 內部等同 `Precision::Int8`）。  
    - 這需要：AWML 匯出時對 SCN 做 PTQ 並把量化參數寫進 ONNX；TRT plugin 實作要能解析這些參數並轉給 libspconv INT8。

- **實作建議**：  
  - 若要走 **Lidar_AI_Solution 那條**（libspconv.so + 專用 ONNX）：在 AWML 新增一條 **SCN-only 匯出**（或從 BEVFusion 拆出 SCN），用 spconv PTQ 得到量化模型，再匯出成與 Lidar_AI_Solution 相同格式的 ONNX（含 `SparseConvolution`、`weight_dynamic_ranges`、`precision="int8"` 等）。  
  - 若維持 **單一 ONNX + TRT**：在現有 `BEVFusionONNXExportPipeline`（或 deploy 用的 wrapper）前加一步 PTQ（僅 SCN 或全模型），匯出時把 INT8 相關的 attribute 寫進 SCN 對應的自訂 op，並與 TRT plugin 約定好格式。

#### (2) TensorRT create engine（AWML / Autoware）

- **現狀**：`precision_policy` 僅有 `auto` / `fp16` / `fp32_tf32` / `strongly_typed`，沒有 INT8；builder 沒有開 `trt.BuilderFlag.INT8`。
- **若要 INT8**：
  - **稠密部分**：  
    - 在 `deployment/configs/enums.py` 新增例如 `PrecisionPolicy.INT8 = "int8"`。  
    - 在 `PRECISION_POLICIES` 裡對應 `builder_config.set_flag(trt.BuilderFlag.INT8)`；若只做 partial INT8，可考慮 `FP16 + INT8`。  
    - TensorRT 的 INT8 通常需要 **calibration**：實作 `trt.IInt8EntropyCalibrator2`（或類似），用 representative data 跑一遍收集 activation 分布，build 時傳給 `builder_config.set_int8_calibrator(...)`。  
  - **設定檔**：在 `deployment/projects/bevfusion/config/deploy_config.py` 的 `tensorrt_config` 可增加選項，例如 `precision_policy="int8"`，並在需要時指定 calibration 資料或 config。
- **注意**：TRT 的 INT8 只對 **TRT 能解析的 op** 有效；**SCN 自訂 op** 仍由 plugin 處理，plugin 內部要自己呼叫 libspconv 的 INT8（見下一段）。

#### (3) TensorRT plugin 與 libspconv（Autoware BEVFusion）

- **現狀**：SCN 在 TRT 裡由 **plugin** 實作，plugin 連結 **libspconv.so** 執行稀疏卷積；目前推論應為 FP16。
- **若要 INT8**：
  - **若使用 Lidar_AI_Solution 的 libspconv.so**：  
    - 該 .so 的 API 是 `load_engine_from_onnx(..., Precision::Int8)`，即 **ONNX 驅動**；不直接從 TRT plugin 呼叫「單一 layer」的 C API。  
    - 因此有兩種整合方式：  
      - **方式一**：SCN 不進 TRT graph，改為 **獨立路徑**——載入一份 SCN 專用 PTQ ONNX，用 `load_engine_from_onnx(..., Precision::Int8)` 建 engine；TRT 只負責稠密部分（可設 INT8）；runtime 先跑 SCN engine，再跑 TRT engine，再合併結果。  
      - **方式二**：TRT plugin 內 **自己建 SCN engine**（例如從 ONNX 或從 TRT 傳進的權重/scale），在 plugin 內呼叫該 engine 的 `forward()`；這時 plugin 要能取得 PTQ 後的權重與 scale（從 TRT 傳入或從檔案讀），並以 **Int8** 建 engine。  
  - **若使用 spconv_cpp 的 libspconv.so**（開源、底層 op API）：  
    - 沒有 `load_engine_from_onnx`，而是 **逐層** 呼叫 sparse conv；plugin 需在 **build/run** 時傳入 qint8 權重、bias、per-channel scale 等，並呼叫對應的 is_int8_inference 路徑。  
    - 則 ONNX 需帶 INT8 權重與 scale，或 plugin 從 TRT 的 weight 與自訂 field 讀取；plugin 內要組出 spconv 需要的 tensor 與參數並呼叫 libspconv 的 INT8 介面。

### 7.4 建議的 INT8 支援路徑（簡要）

| 階段 | 要做的事 |
|------|----------|
| **AWML create ONNX** | (1) 對 SCN 做 PTQ（spconv `prepare_fx` → calibrate → `convert_fx`）；(2) 若要對齊 Lidar_AI_Solution：匯出 SCN 專用 ONNX（`SparseConvolution` + `weight_dynamic_ranges` + `precision="int8"`）；(3) 若維持單一 ONNX：在 SCN 對應自訂 op 寫入 INT8 相關 attribute，供 TRT plugin 使用。 |
| **TensorRT create engine** | (1) 新增 `precision_policy="int8"`（及必要時 `fp16+int8`）；(2) 設定 `BuilderFlag.INT8`；(3) 實作並設定 INT8 calibrator（representative dataset）；(4) 稠密部分由 TRT 做 INT8；SCN 由 plugin 處理。 |
| **TRT plugin / runtime** | (1) 若 SCN 獨立成第二個 engine：用 libspconv.so 的 `load_engine_from_onnx(..., Precision::Int8)` 載入 SCN PTQ ONNX，與 TRT engine 並行或前後執行；(2) 若 SCN 仍在 TRT graph 內：plugin 從權重/attribute 取得量化參數，在 plugin 內以 **Int8** 呼叫 libspconv（或建 SCN engine），並與 TRT 的 INT8 稠密部分銜接。 |

以上依 **AWML create ONNX**、**Autoware create engine**、**TRT plugin** 的現況整理；實際 INT8 支援需再對接你使用的 libspconv 版本（Lidar_AI_Solution 的 .so 或 spconv_cpp 的 .so）與 Autoware 的 plugin 實作。
