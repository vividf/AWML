# Lidar_AI_Solution（開源 New3D）稀疏卷積：ONNX → `spconv::Engine` 與 INT8 路徑

本文件整理 **NVIDIA Lidar_AI_Solution 開源分支** 中 `libraries/New3DSparseConvolution/` 的定位：它如何用 **自訂 ONNX** 與 **自編譯 traveller59 `libspconv`** 做推理，以及 **INT8** 從圖屬性到 CUDA kernel 的完整鏈路。若你主要走 **AWML + TensorRT + ImplicitGemmInt8 plugin**，請對照 `README_INT8_WHERE_AND_HOW.md`：兩邊最後都可能呼叫同一套 **`ConvGemmOps::implicit_gemm`**，但 **圖與引擎邊界完全不同**。**§8** 則把「參考開源後，AWML 推理可怎麼加速」寫成可執行的對照與優先順序。

---

## 1. 這套元件在做什麼？

| 項目 | 說明 |
|------|------|
| **目的** | 在 **CUDA-BEVFusion / Lidar 管線** 裡跑 **3D 稀疏卷積塔**，不依賴 TensorRT 的 ONNX parser 或 TRT 稀疏 plugin。 |
| **`libscn.so`** | 專案對外提供的 **動態庫介面**；語意上可視為 **drop-in** 取代舊版閉源 **`libspconv.so`**（實際替換需對齊 ABI／符號與部署腳本）。 |
| **`libspconv`** | **從原始碼編譯**的 traveller59 spconv C++ 核心（Makefile 以 `-lspconv` 連結），提供 `tv::Tensor`、`SpconvOps`、`ConvGemmOps` 等。 |
| **`spconv::Engine`** | **不是 TensorRT engine**。由 `load_engine_from_onnx` 讀 **protobuf ONNX** → `EngineBuilder` 建圖 → `build()` 產生可在 GPU 上 `execute` 的 **自訂 IR + op 實作**。 |

**典型原始碼根目錄**（本機路徑僅供對照）：

`.../Lidar_AI_Solution_open_source_spconv/libraries/New3DSparseConvolution/`

---

## 2. ONNX → Engine：和 TensorRT 無關的那條路

流程如下：

1. **`src/onnx-parser.cpp`**：`load_engine_from_onnx(onnx_file, precision, stream, mark_all_output)`  
   - 用 **ONNX protobuf** 解析 `graph()`，**不使用** `nvinfer`、**不使用** TensorRT ONNX Parser。  
2. **`create_engine_builder()`** → 對每個 node 呼叫 `push_*`（`SparseConvolution`、`Add` / `QuantAdd`、`Relu`、`ScatterDense`、…）。  
3. **`EngineBuilder::build(precision, stream)`**（**`libspconv/src/engine.cpp`**）  
   - 先做 **圖融合**（見下一節），再對每個存活節點呼叫 **`operation()->configure(precision, tensor_name_to_scale_, parameters)`**。  
4. 回傳的 **`Engine`** 在執行期呼叫各 op 的 `forward`，內部才是 spconv 的 implicit gemm / scatter 等。

因此：**「Engine」在這裡 = 自訂稀疏執行引擎**；與 **TRT 的 `ICudaEngine`** 僅名稱類似，不要混用。

---

## 3. ONNX 與權重格式（Parser 假設）

- **自訂算子**：例如 `SparseConvolution`、`Add` / `QuantAdd`、`Relu`、`ScatterDense`、`Transpose`、`Reshape` 等（以 `onnx-parser.cpp` 為準）。  
- **Initializer**：`get_initializer_data` **強制要求 FP16**（`TensorProto_DataType_FLOAT16`）；權重與 bias 在圖裡仍以 **FP16 blob** 載入，INT8 權重在 **`configure()`** 階段由 GPU kernel 量化寫入（見 §5）。  
- **`SparseConvolution` 節點**（節錄語意）：  
  - `weight_dynamic_ranges`：長度為 **out_channels** 的 float 列表（校準得到的動態範圍，與 QAT/PTQ 工具鏈一致）。  
  - `input_dynamic_range`：單一 float，描述 **輸入特徵**量化刻度相關量。  
  - `precision` / `output_precision`：字串 `"int8"` 或否 → 映射為 `spconv::Precision::Int8` / `Float16`。

---

## 4. `build()` 階段的圖融合（自訂 IR，不是 ONNX QDQ 融合）

在 **`engine.cpp` 的 `EngineBuilderImplement::build`** 中：

- **Add + Relu** → 合併為 **fused Add+ReLU** 節點（減少 launch 與記憶體來回）。  
- **ScatterDense + Transpose + Reshape** → 把 transpose/reshape 併入 scatter 相關設定（必要時帶 `permute` 再 `configure`）。

這些是 **C++ builder 內的硬編碼規則**，與 TensorRT 的 layer fusion 無關。

---

## 5. INT8：從 ONNX 屬性到 `implicit_gemm`（詳細過程）

以下分階段說明，方便對照 **校準 → 匯出 ONNX → `configure` → `forward`**。

### 5.1 語意：動態範圍與「scale」

此堆疊沿用 **對稱量化到 int8（±127）** 的常見約定（與 NVIDIA QAT 範例、`TENSORRT_INT8_GUIDE` 類文件一致）：

- ONNX 裡的 **`input_dynamic_range` / `input0_dynamic_range` / `weight_dynamic_ranges`** 等，代表校準得到的 **|max|** 類資訊（程式註解稱 **QAMAX**）。  
- Builder 在建立節點時，把它們轉成 **張量名 → `scale = dynamic_range / 127`**，存入 **`tensor_name_to_scale_`**：  
  - **稀疏卷積**：`tensor_name_to_scale_[conv 的輸入張量名] = input_dynamic_range / 127`。  
  - **Add / QuantAdd**：兩個輸入張量名各自對應 `a_dynamic_range / 127`、`b_dynamic_range / 127`。

**重要：** 地圖裡鍵是 **ONNX 張量名字串**。若某卷積的 **輸出** 要做 INT8 且需在 `configure` 裡查 **`output_scale_`**，則該 **輸出張量名** 也必須能從 **`tensor_name_to_scale_`** 查到——實務上通常由 **下游節點**（例如下一層的 `Add` 某一腳、或另一層的 `input_dynamic_range`）在 `push_*` 時註冊到 **同一個名字**，否則 `configure` 會 throw「scale not found」。匯出 ONNX 時需保證 **張量命名與動態範圍** 與執行順序一致。

### 5.2 全域與節點精度：`load_engine_from_onnx(..., Precision precision, ...)`

- 呼叫端傳入的 **`precision`** 表示 **整張圖**是否要走 INT8 推理管線。  
- 每個節點另有 **`precision` / `output_precision`**（輸入側／輸出側節點精度）。

### 5.3 `SparseConvolution::configure`（**`libspconv/src/sparseConvImplicit.cu`**）

在 **`build()`** 尾端對每個 op 呼叫 `configure`。對稀疏卷積而言：

**（1）是否啟用 INT8 推理 `int8_inference_`**

- 先設 `int8_inference_ = (全域 precision == Int8)`。  
- 再與節點 I/O 精度組合：若 **輸入精度與輸出精度皆為 FP16**，則 **關閉** INT8 路徑（保留純 FP16 子圖行為）。  
- 若關閉 INT8，但節點上仍標了 INT8 精度，程式會 **降回 FP16** 並打 log，避免與全域模式不一致。

**（2）輸出張量 dtype**

- 若啟用 INT8 且 **`output_precision_ == Int8`**：`output_dtype_ = int8`，並預先配置 **int8 輸出 buffer**。  
- 否則輸出仍為 **FP16**（即使中間用 INT8 計算，見 forward）。

**（3）從 `tensor_name_to_scale_` 取 `input_scale_` / `output_scale_`**

- **`input_scale_`**：必取，鍵為 **卷積輸入張量名**。  
- **`output_scale_`**：僅當 **輸出為 Int8** 時必取，鍵為 **卷積輸出張量名**。  

**（4）權重與 bias 的編譯期準備（僅在 `int8_inference_` 為真時）**

對每個 **輸出通道** \(i\)（`weight_dynamic_range` 逐通道）：

1. `weight_scale[i] = weight_dynamic_range[i] / 127`（與 TRT/spconv 文件中的 per-channel scale 一致）。  
2. **合併刻度**（供 implicit_gemm epilogue 使用）：  
   `scale_[i] = (weight_scale[i] * input_scale_) / output_scale_`  
3. **bias**：由 FP16 讀入後轉 FP32，並做 **`bias_fp32[i] = float(bias_fp16[i]) / output_scale_`**，之後 **`bias_` 指向 FP32**，與 INT8 GEMM 累加後的 **rescale** 對齊。  
4. **權重量化**：在 GPU 上對 **KRSC 布局**權重做 **per-output-channel** 的 `quantize(fp16_value, weight_scale[channel])`，結果 **`weight_` 替換為 int8 tensor**。

至此：**靜態權重已是 int8**；**bias 已是「已除以 output_scale 的 FP32」**；**`scale_` 張量**承載 **(weight_scale × input_scale) / output_scale**。

### 5.4 `SparseConvolution::forward`（執行期）

1. **索引與 mask**：`SpconvOps::get_indice_pairs_implicit_gemm`（與 FP16 路徑相同類型的資料結構）。  
2. **輸入特徵**：  
   - 若 **`int8_inference_`** 且 **節點輸入精度為 FP16**：在 conv 前對 **FP16 特徵**做 **quantize → int8**（使用 `input_scale_`），得到 **`features_int8`**。  
   - 若輸入已是 INT8 語意，則可直接使用對應 buffer（實作以原始碼分支為準）。  
3. **卷積核心**：`ConvGemmOps::implicit_gemm(...)`，其中：  
   - **A/B**：int8 特徵與 int8 權重；  
   - **`output_scale` / `scale_` / bias**：與 §5.3 一致，用於 **int32 累加後的縮放與加 bias**；  
   - **`output_dtype_`**：決定最終寫回 **int8** 或 **FP16**。  
4. **啟用函數**：若 `activation_` 需要，後接對應 fused 或非 fused kernel。

與 **AWML TensorRT sparse INT8** 的對照：**兩邊都是在 plugin 或本 op 內把特徵／權重變成 int8，再進同一 family 的 `implicit_gemm`**；差別在於 **誰解析 ONNX**（這裡是 **自訂 parser + Engine**，AWML 是 **TRT + 自訂 plugin**）。

### 5.5 Add、ReLU、Add+ReLU 的 INT8

- **`sparseAdd.cu`**：在 INT8 模式下對兩路輸入做 **rescale 對齊**後再做元素級運算（細節以實作為準）。  
- **`sparseRelu.cu`**：支援 INT8 輸入輸出路徑。  
- **`sparseFusedAddRelu.cu`**：`build()` 將 **Add+Relu** 融合後走此路徑，減少中間張量與 kernel 次數。

### 5.6 與 AWML 路徑的一頁對照

| 項目 | New3D `libspconv` Engine | AWML TensorRT + ImplicitGemmInt8 |
|------|---------------------------|-------------------------------------|
| 圖載入 | 自訂 ONNX protobuf → `EngineBuilder` | ONNX → TensorRT `INetworkDefinition` |
| INT8 邊界 | 各 op 的 `forward` / fused kernel | Plugin `enqueue` |
| 權重格式 | ONNX FP16 → `configure` 量成 int8 | Plugin 內量化（常見由 initializers + scale 驅動） |
| 核心 GEMM | `ConvGemmOps::implicit_gemm` | 同左（共用 spconv/cumm 思路） |

---

## 6. 與 CUDA-BEVFusion CMake 的關係（避免混用路徑）

開源儲存庫中，**稀疏塔**可能仍見 **`libraries/3DSparseConvolution`**（歷史／閉源 bundle）與 **`libraries/New3DSparseConvolution`**（開源）並存。整合時請確認 **連結的 `.so`、ONNX 匯出腳本、以及 `precision` 旗標** 與你選的目錄一致，避免混到另一條引擎路徑。

---

## 7. 建議閱讀的原始碼路徑（New3D）

| 主題 | 路徑（相對 `libraries/New3DSparseConvolution/`） |
|------|--------------------------------------------------|
| ONNX 載入 | `src/onnx-parser.cpp` |
| Builder / 融合 / `configure` 呼叫 | `libspconv/src/engine.cpp` |
| 稀疏卷積 INT8 `configure` + `forward` | `libspconv/src/sparseConvImplicit.cu` |
| Add / ReLU / fused | `libspconv/src/sparseAdd.cu`、`sparseRelu.cu`、`sparseFusedAddRelu.cu` |
| Engine API | `libspconv/include/spconv/engine.hpp`（或同目錄標頭） |

---

## 8. 參考開源作法後：AWML 推理加速可能方向（詳細）

以下對照 **New3D 自訂 `spconv::Engine`** 與 **AWML（TensorRT + ImplicitGemm／ImplicitGemmInt8）** 的差異，整理 **延遲／吞吐** 上較有機會改善的作法與取捨。實作前請用 **Nsight Systems／TensorRT layer timing** 確認瓶頸在 **稀疏塔、dense BEV、或 CPU 前處理**。

### 8.1 權重量化時機：開源做一次，sparse INT8 目前每 frame 做一次

**New3D**：在引擎 **`build()` → `SparseConvolution::configure()`** 階段，把 **FP16 權重**量成 **int8**，並算好 **`scale_`／bias 的 FP32 融合**（見 §5.3）。之後 **`forward` 只處理會變的特徵與索引**。

**AWML sparse INT8**（`deployment/projects/bevfusion/cpp/int8_plugin/implicit_gemm_int8_plugin.cpp` 的 **`enqueue`**）在 **每次推理** roughly 會做：

1. `launch_compute_w_scales`：由 `channel_scale` 與 `input_scale`／`output_scale` 推出每通道權重刻度。  
2. `launch_quantize_features`：**特徵** FP16 → int8（每 frame 合理，因 voxel／活躍點會變）。  
3. **`launch_quantize_weights_per_channel`**：**權重** FP16 → int8（推理時權重通常 **固定**）。  
4. `launch_fuse_output_scale_into_gemm_scale_bias`：把輸出反量化融進 GEMM 用的 scale／bias。

對 **靜態權重** 而言，(1)(3)(4) 與 New3D 在 **`configure` 一次完成** 的內容重疊，代表 **每層、每 inference 多付一輪全域記憶體讀寫與 kernel**。

**建議改進方向**

- 若該層 **filters 在 TensorRT 圖中為常數**（常見：權重來自 engine constant），可在 plugin 的 **`configurePlugin`／序列化 blob** 階段預算 **device 上 int8 weight + 已融合的 `scale`／`bias`**，`enqueue` **只保留特徵量化 + `implicit_gemm`**。  
- 實作時需處理：**動態 shape** 下 workspace 與權重 buffer 生命週期、**版本化序列化**（換 checkpoint 須重建 engine）、以及 TRT 是否仍把 filters 當 **execution tensor**（若是，則無法假設常數，需以實際 network 為準）。

此項通常是 **「抄開源最大收益、又不必換整條 runtime」** 的切入點。

### 8.2 圖級融合：New3D 在 builder 硬併，TRT 難跨 plugin 邊界

**New3D** 在 **`engine.cpp` 的 `build()`** 內把 **Add+ReLU**、**ScatterDense+Transpose+Reshape** 等合併（§4），減少 **kernel launch 次數** 與 **中間張量** 讀寫。

**AWML** 稀疏段由 **多個自訂 plugin／稀疏 op** 組成；TensorRT 的 **layer fusion** 多半 **無法** 跨過這些邊界自動做出同等效果。

**建議改進方向**

- 在 **ONNX 匯出或 graph surgery** 階段，把穩定出現的 **Add+ReLU、或小型 elementwise 鏈** 併成 **單一 fused plugin**（或擴充現有 kernel），而不是依賴 TRT 預設融合。  
- **Dense BEV 段**（`bevfusion_dense`）仍可善用 TRT 內建融合；並注意 **`deploy_config_split.py`** 已提醒：**`lidar_bev` 的 H、W profile 須對齊真實 grid**，過寬的 dynamic range 會讓引擎長期落在非 optimal shape，間接拖慢推理。

### 8.3 算力路徑：INT8 implicit gemm

開源堆疊與 AWML 底層皆可呼叫 **`ConvGemmOps::implicit_gemm`** 的 **int8 演算法**（`is_int8_inference`，見 `README_INT8_WHERE_AND_HOW.md`）。

若目前僅使用 **FP16 ImplicitGemm plugin** 而 **GPU 世代支援** int8 tensor core 路徑，啟用 **sparse INT8（ImplicitGemmInt8）** 往往是稀疏塔 **吞吐／延遲** 最直觀的升級；**精度**需以 PTQ／校準集驗證。

### 8.4 資料型別與 I／O：FP16 邊界與「層間 dequant–requant」

TensorRT 對 **自訂 op 的 I／O** 常限制為 **FP16**（見 `spconv/docs/TENSORRT_INT8_GUIDE.md` 與 AWML 文件），因此 **層與層之間** 往往是 **FP16**，INT8 僅在 **plugin 內部** 與 **GEMM 的 A／B** 上。

**代價**：相較 New3D 可在圖裡標 **int8 中間張量** 並在 **Add／ReLU** 等節點上延續 int8，AWML 較容易出現 **dequant → 下一層再 quant** 的額外帶寬與 kernel。

**建議改進方向**：在 **單一 plugin 內** 融合多個連續稀疏運算（若圖結構允許），讓 **int8 活動範圍**盡量不要每層都回到 FP16 邊界。

### 8.5 系統與 TensorRT 層（與開源架構無直接對應但常有效）

- **Builder**：啟用／快取 **timing cache**、給足 **workspace**、關閉不必要 verbose；針對實際 **voxel 數** 調整 **profile `opt_shape`**，避免長期用不到 optimal kernel。  
- **執行**：若 shape 與依賴關係允許，可評估 **CUDA Graph** 包一次 **`execute`**（動態 shape 需額外設計）。  
- **Pipeline**：已採 **sparse／dense 分引擎**（`deploy_config_split`）時，可用 **雙 CUDA stream** 在依賴允許下 **重疊** 兩段執行。  
- **前處理**：split 設定中 **voxels 常為 FP32**；若管線與精度允許，可評估 **FP16 voxel 輸入** 以降低 **PCIe／kernel** 成本（需單獨驗證 mAP）。

### 8.6 是否改為「整段稀疏塔 = New3D 式 Engine」？

將 **稀疏子網** 從 TensorRT 抽出、改跑 **自訂 `spconv::Engine`**（與 Lidar 開源相同思路），理論上可減少 **TRT 排程 + 多 plugin 邊界** 開銷，並複製 **configure 時量化 + 圖融合**。

**代價**：維護 **第二套 runtime**、與現有 **Autoware／TRT 部署鏈** 整合成本高。建議在 profile **證明 TRT 邊界與 per-invocation 權重量化** 占顯著比例後，再評估此路線。

### 8.7 小結（優先順序參考）

| 優先級 | 方向 | 說明 |
|--------|------|------|
| 高 | **權重／scale 前置到 configure** | 對應 New3D §5.3，減少 `enqueue` 重複工作。 |
| 高 | **啟用 sparse INT8**（若尚未啟用且 GPU 適合） | 與開源同底層 int8 GEMM。 |
| 中 | **ONNX／plugin 層融合** | 對齊 New3D §4 精神，補 TRT 跨 plugin 不足。 |
| 中 | **Profile 與 timing cache** | 成本低，常見遺漏。 |
| 視需求 | **整段換自訂 Engine** | 架構變動大，需強理由。 |

---

## 9. 相關 AWML 文件

- **`README_INT8_WHERE_AND_HOW.md`**：AWML 與 `spconv_cpp` / TRT plugin 的 INT8 邊界與 `implicit_gemm` 選 kernel 邏輯。  
- **`README_PTQ_INT8_SPCONV_DEPLOYMENT.md`**：PTQ 到部署的端到端說明。

若你希望把本文件連回 **Lidar 官方 repo**，可在該 repo 的 `New3DSparseConvolution/README.md` 放一則 **「AWML 對照說明」** 連結到本檔路徑（雙向維護依專案習慣即可）。
