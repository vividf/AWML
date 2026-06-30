# BEVFusion Deployment: ONNX / spconv / TensorRT Plugin 與 Inference 流程

這份文件整理 AWML 目前 BEVFusion deployment 的三件事：

1. `ONNX`、`spconv`、`TensorRT plugin` 彼此關係。
2. 稀疏 INT8 plugin 相關檔案各自負責什麼。
3. BEVFusion 在 AWML 中從輸入點雲到輸出的推論流程。

本文主要對照以下程式路徑：

- `deployment/projects/bevfusion/export/onnx_export_pipeline.py`
- `deployment/projects/bevfusion/export/sparse_int8_onnx_transform.py`
- `deployment/projects/bevfusion/pipelines/tensorrt.py`
- `deployment/projects/bevfusion/pipelines/bevfusion_pipeline.py`
- `deployment/projects/bevfusion/cpp/int8_plugin/*`
- `autoware.universe/perception/autoware_tensorrt_plugins/src/*`

---

## 1) ONNX、spconv、TensorRT plugin 的關係

### 1.1 高階關係

- **PyTorch (訓練/校準模型)**：原始 BEVFusion 與 `spconv` 模組。
- **ONNX 匯出層**：把模型切成可部署圖（單圖或 split）。
- **TensorRT 引擎層**：把 ONNX 轉成 engine 執行。
- **Plugin 層**：補上 TensorRT 原生不支援或需要客製最佳化的算子（例如 `ImplicitGemm`、`GetIndicePairsImplicitGemm`）。

### 1.2 在 AWML 的 split 模式（最常用）

split 會輸出兩張 ONNX、兩顆 engine：

- `bevfusion_sparse.onnx/.engine`: LiDAR sparse tower
- `bevfusion_dense.onnx/.engine`: backbone/neck/head

匯出與轉換關鍵點：

1. `onnx_export_pipeline.py` 先匯出 sparse/dense ONNX。
2. 若開 `quantization.spconv_int8=True`，`sparse_int8_onnx_transform.py` 會把 sparse ONNX 裡的 `autoware::ImplicitGemm` 改成 `autoware::ImplicitGemmInt8`。
3. TRT 建 engine 時會載入 plugin `.so`，讓上述 custom op 可以被解析與執行。

### 1.3 為什麼同時看到「autoware plugin」與「AWML int8 plugin」

- **autoware plugin (`libautoware_tensorrt_plugins.so`)**：
  - 提供原本的 `ImplicitGemm`、`GetIndicePairsImplicitGemm` 等 op。
- **AWML int8 plugin (`libimplicit_gemm_int8_plugin.so`)**：
  - 提供 `ImplicitGemmInt8`，讓 sparse conv 走 INT8 kernel。

在 `deploy_config_split_int8.py` 的 `tensorrt_config.plugin_libraries` 可看到兩者都會載入。

---

## 2) `sparse_int8_onnx_transform.py` 在做什麼

`deployment/projects/bevfusion/export/sparse_int8_onnx_transform.py` 是 Path-B INT8 的核心：

- 讀 PTQ checkpoint `_amax`。
- 建立每層 `input_scale / output_scale / channel_scale / bias_scaled`。
- 將 ONNX 的 `autoware::ImplicitGemm` 節點替換成 `autoware::ImplicitGemmInt8`。
- 新增 `channel_scale`、`bias_scaled` 兩個 initializer 作為 plugin 輸入。
- 支援 `spconv_int8_fp16_layers`，可把指定層維持 FP16 做精度調整。
- 可選擇 fuse `ImplicitGemm + (Add) + ReLU`，減少圖中額外節點。

一句話：**它負責把「可校準的 sparse FP 圖」轉成「可被 INT8 plugin 執行的 sparse ONNX 圖」。**

---

## 3) TensorRT plugin 檔案逐一說明（AWML `cpp/int8_plugin`）

目錄：`deployment/projects/bevfusion/cpp/int8_plugin`

### 3.1 先看 `ImplicitGemmInt8` 的輸入/輸出 contract（最重要）

`ImplicitGemmInt8` 在 header 中定義為 **7 inputs / 1 output**：

- `input[0] features`：`FP16 [N, C_in]`
- `input[1] filters`：`FP16 [C_out, K1, K2, K3, C_in]`
- `input[2] pair_fwd`：`INT32 [K_vol, num_act_out]`
- `input[3] pair_mask_fwd`：`INT32 [...]`
- `input[4] mask_argsort_fwd`：`INT32 [...]`
- `input[5] channel_scale`：`FP32 [C_out]`
- `input[6] bias_scaled`：`FP32 [C_out]`
- `output[0] out_features`：`FP16 [num_act_out, C_out]`

其中 `pair_*` 與 `mask_*` 不是本 plugin 產生的，而是上游 `GetIndicePairsImplicitGemm` plugin 先生成，然後餵給 `ImplicitGemmInt8`。

### 3.2 Plugin lifecycle（TensorRT 呼叫時序）

實務上建 engine / 跑 inference 時，plugin 的典型呼叫順序：

1. `Creator` 被註冊（`REGISTER_TENSORRT_PLUGIN`）。
2. TRT parser 讀 ONNX custom node 時呼叫 `createPlugin(...)`。
3. 建圖期呼叫 `configurePlugin(...)` 檢查 shape/dtype 合法性。
4. runtime 反覆呼叫 `enqueue(...)` 執行每一層。
5. engine 釋放時 destruct plugin，釋放 cache/event 資源。

AWML 目前 `ImplicitGemmInt8` 是 **constant-only cache mode**：

- 第一次 `enqueue` 會把常量（weights/channel_scale/bias_scaled）轉換並快取到 GPU。
- 後續 `enqueue` 驗證 pointer 與 shape 不可變，直接重用快取。
- 這可降低每幀重複量化權重的開銷。

### 3.3 `enqueue` 內部資料流（細節）

`implicit_gemm_int8_plugin.cpp` 的執行流程可拆成：

1. **feature 量化**
   - 將 `input[0]` 的 FP16 feature 量化成 INT8（workspace 暫存）。
2. **常量快取初始化（首幀）**
   - 量化 FP16 filter -> INT8
   - 根據 `channel_scale/output_scale/input_scale` 計算 `w_scales`
   - 融合 `output_scale` 到 gemm scale/bias（對齊 cumm epilogue 實作）
3. **包裝成 tv::Tensor**
   - 將 feature/filter/pair/mask/out 指標包成 spconv 可用結構。
4. **呼叫 `ConvGemmOps::implicit_gemm`**
   - 真正 sparse GEMM 計算在這裡發生。
5. **輸出與診斷**
   - 產出 FP16 `out_features`
   - 若開啟 timing/debug，輸出各段耗時與 tensor 統計。

### 3.4 Plugin 主體檔案

- `implicit_gemm_int8_plugin.hpp`
  - Plugin 類別與參數定義（`ImplicitGemmInt8Parameters`）。
  - 宣告 TensorRT plugin 介面（`configurePlugin`、`enqueue`、`getWorkspaceSize` 等）。
  - 定義輸入/輸出 contract（7 inputs / 1 output）。

- `implicit_gemm_int8_plugin.cpp`
  - `enqueue` 真正執行邏輯（見上節資料流）。
  - `supportsFormatCombination` / `getOutputShapes` 定義 TRT 對 dtype/shape 的可接受組合。
  - 管理 constant cache（避免每次重做權重量化）。
  - 實作 timing/debug（`BEVFUSION_INT8_GEMM_DEBUG` 與 plugin timing）。
  - 管理 workspace 與 CUDA event 生命週期。

### 3.5 Plugin Creator / 註冊

- `implicit_gemm_int8_plugin_creator.hpp`
  - Creator 類別宣告，提供 TensorRT 反序列化/建構 plugin 的入口介面。

- `implicit_gemm_int8_plugin_creator.cpp`
  - 實作欄位解析（`input_scale`、`output_scale`、`act_type` 等）。
  - `REGISTER_TENSORRT_PLUGIN(ImplicitGemmInt8PluginCreator)`。
  - 這一層負責把 ONNX node attributes 轉成 C++ plugin 參數 struct。

- `plugin_registration.cpp`
  - 顯式 include creator，確保 static registration 生效。

### 3.6 CUDA 量化輔助 kernel

- `quantize_features.cuh`
  - 量化 kernel 的函式宣告。

- `quantize_features.cu`
  - 實作 feature 量化 kernel（FP16 -> INT8）。
  - 實作 weight per-channel 量化 kernel。
  - 實作 `compute_w_scales` 與 `fuse_output_scale_into_gemm_scale_bias` kernel。
  - 主要提供 plugin 在 `enqueue` 內呼叫。

### 3.7 建置系統

- `CMakeLists.txt`
  - 建 `int8_quantize_ops`（CUDA kernels）與 `implicit_gemm_int8_plugin`（plugin so）。
  - 連結 TensorRT、CUDA、spconv。
  - 控制 symbol visibility，避免與 Python spconv 同進程衝突。

### 3.8 常見除錯開關（plugin 相關）

- `BEVFUSION_INT8_GEMM_DEBUG=1`
  - 印出 plugin 輸出統計（min/max/mean 等），用來比對 PyTorch / TRT 中間特徵量級。
- `BEVFUSION_INT8_GEMM_DEBUG_MAX=<N>`
  - 控制最多列印幾次，避免 log 爆量。
- deploy cfg 的 `implicit_gemm_int8_plugin_timing`
  - 會在 ONNX transform 時把 timing attribute 寫進 node，plugin 再依此輸出分段時間。

### 3.9 其他檔案

- `README.md`
  - 這個 plugin 的背景、除錯紀錄、數值修正脈絡。
- `build/`
  - CMake 產物，不是邏輯來源碼。

---

## 4) autoware 端主要 plugin 檔案（和 sparse 路徑直接相關）

目錄：`autoware.universe/perception/autoware_tensorrt_plugins/src`

- `get_indices_pairs_implicit_gemm_plugin.cpp`
  - 建 sparse conv 所需 pair/index 資料結構。
  - 內含 `do_sort` 行為，可由 ONNX attributes 決定（INT8 常關閉 sort）。
  - 這層主要是「索引建立」，不是做卷積本身。

- `get_indices_pairs_implicit_gemm_plugin_creator.cpp`
  - 對應 creator 與欄位解析。

- `implicit_gemm_plugin.cpp`
  - 原生 FP 路徑的 `ImplicitGemm` plugin。
  - 支援 bias 與 activation、可做 timing 輸出。
  - 這層才是「根據 pair/index 做 sparse gemm」。

- `implicit_gemm_plugin_creator.cpp`
  - 對應 creator。

### 4.1 兩個 plugin 的角色切分（很容易搞混）

- `GetIndicePairsImplicitGemm`：
  - 負責「找配對」與產生 index/pair/mask（索引前處理）。
- `ImplicitGemm` / `ImplicitGemmInt8`：
  - 負責「拿到 pair 後做卷積計算」。

可以把它想成：

- 前者是「建立稀疏路徑交通地圖」；
- 後者是「照地圖跑實際矩陣運算」。

---

## 5) BEVFusion inference 怎麼做（AWML deployment 視角）

### 5.1 入口與組裝

- 入口：`deployment/projects/bevfusion/entrypoint.py`
  - 讀 deploy/model config。
  - 套用 `spconv_do_sort` 到 symbolic/export 邏輯。
  - 建立 data loader、evaluator、runner。

### 5.2 Export / Build 階段

- Runner：`deployment/projects/bevfusion/runner.py`
  - 載入 PyTorch 模型（可含 quantization checkpoint）。
  - ONNX export pipeline：`export/onnx_export_pipeline.py`
  - TensorRT export pipeline：`export/tensorrt_export_pipeline.py`

### 5.3 Runtime 推論階段（核心）

通用流程在 `pipelines/bevfusion_pipeline.py`：

1. **Preprocess**
   - 用 `pts_voxel_layer` 把 point cloud 轉成：
     - `voxels`
     - `coors`
     - `num_points_per_voxel`

2. **Run backend**
   - ONNX backend：`pipelines/onnx.py`
   - TensorRT backend：`pipelines/tensorrt.py`

3. **Postprocess**
   - 將 `bbox_pred/score/label_pred` 送進 `bbox_coder.decode`，轉成最終 3D box（座標、尺寸、yaw、速度）。

### 5.4 TensorRT split 具體執行

在 `pipelines/tensorrt.py`（split 模式）：

1. 先跑 `bevfusion_sparse.engine`：
   - 輸入 voxel 三件組
   - 輸出 `lidar_bev`
2. 再跑 `bevfusion_dense.engine`：
   - 輸入 `lidar_bev`
   - 輸出 `bbox_pred / score / label_pred`
3. 收集 stage latency 與可選 profile 資訊。

---

## 6) 一句話總結

- `ONNX` 是中間表示與切分邊界。
- `spconv` 提供 sparse conv 的底層計算能力。
- `TensorRT plugin` 讓 sparse custom op（特別是 INT8 `ImplicitGemmInt8`）能在 TRT engine 中執行。
- BEVFusion inference 在 AWML 是「voxelize -> sparse branch -> dense branch -> decode」的標準鏈路，split 模式把 sparse/dense 清楚拆開，便於調優與除錯。
