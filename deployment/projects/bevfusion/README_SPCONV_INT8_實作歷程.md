# BEVFusion Spconv INT8 實作歷程（中文）

本文記錄 **spconv 稀疏編碼器 INT8** 在 AWML BEVFusion 部署／PTQ 路線上的**實作思路、踩過的坑與對應解法**，並對齊 commit **`b0ad1b1027efa8f49ef6733283f61eeb31c9d25d`**（`chore: testing spconv int8`）所涵蓋的變更。  
更完整的指令與設定欄位說明請見同目錄 **[README_INT8_IMPLEMENTATION.md](./README_INT8_IMPLEMENTATION.md)**。

---

## 一、目標與整體策略

- **目標**：在維持 BEVFusion 點雲分支可用的前提下，讓 **`pts_middle_encoder`（稀疏 3D backbone）** 走 **spconv 官方建議的 FX 圖量化**（`torch.ao.quantization` + spconv backend），並與 **dense 端的 pytorch_quantization（Q/DQ）** 共存於同一 PTQ checkpoint 與 deployment 流程。
- **策略**：
  1. **Sparse 與 Dense 分開處理**：dense 用 TensorQuantizer；sparse 用 **prepare_fx → 校準 → convert_fx → spconv 的 transform_qdq / remove_conv_add_dq**。
  2. **Encoder 必須可被 `torch.fx` 追蹤**：避免在 trace 路徑上對 `SparseConvTensor` 誤用 `nn.ReLU`、`replace_feature` 的動態分支等。
  3. **部署載入時結構需與存檔一致**：PTQ 存的是 FX 轉完後的 `GraphModule`，載入前要先 **fuse 稀疏 BN**、**重建同一套 FX 結構** 再 `load_state_dict`。

---

## 二、INT8 實際在做什麼（流程總覽）

### 2.1 PTQ 階段（`bevfusion_quantization.py`）

當 `deploy_config` 裡 `quantization.spconv_int8=True` 且使用 **FX 版** sparse config（`block_type='basicblock_fx'`）時，大致順序為：

1. 建 FP32 模型；dense 側 fuse BN、插入 Q/DQ（與原本 INT8 流程一致）。
2. 對 **`pts_middle_encoder`**：
   - 先 **SparseConv + BN 融合**（`fuse_spconv_bn_eval` 等）。
   - **`prepare_fx`**：插入 observer / fake quant。
   - 用校準資料跑 encoder（**calibrate**）。
   - **`convert_fx`**：換成實際 INT8 / quantized 模組（搭配 spconv 的 backend config）。
   - **`transform_qdq`**：把圖裡的 `torch.quantize_per_tensor` 等換成 spconv 可處理 **SparseConvTensor** 的版本。
   - **`remove_conv_add_dq`**：依 spconv 範例處理 conv-add-dq 模式。
3. 將轉換後的 encoder **掛回** `model.pts_middle_encoder`，與 dense 量化結果一併 **存成 PTQ checkpoint**。

### 2.2 Deployment 載入階段（`model_loader.py` + `runner.py`）

1. 建模型；dense 側 fuse BN、建 Q/DQ 結構。
2. 若 **`spconv_int8`**：
   - 對 **`pts_middle_encoder` 再做稀疏 BN fusion**（與 PTQ 前一致，否則 key 對不齊）。
   - **不重新校準**，但再跑一次 **prepare_fx → convert_fx → transform_qdq → remove_conv_add_dq**，得到與存檔時**同形狀的 GraphModule**，再 **`load_state_dict`** 灌入 INT8 權重與 scale。
3. 可選：**驗證** sparse 是否像 INT8（見 `verify_spconv_int8_encoder`、`scripts/verify_spconv_int8.py`）。

### 2.3 ONNX 導出（`onnx_export_pipeline.py`）

- 全圖 trace 時，稀疏塔末端仍會執行 **`SparseConvTensor.dense()`** 轉成 BEV feature，**體積極大**，在小 VRAM GPU 上容易 **CUDA OOM**。
- 對策：支援 **`onnx_config.trace_device` / 環境變數 `BEVFUSION_ONNX_TRACE_DEVICE`**，**預設傾向在 CPU 上做 `torch.onnx.export` 追蹤**，結束後把模型移回原 device（見 schema `OnnxConfig.trace_device`）。

---

## 三、實作過程中遇到的主要困難與解法

以下對應 **commit b0ad1b1** 中新增或大幅修改的檔案與設計決策。

### 3.1 FX 追蹤失敗：`prepare_fx` 過不了

**現象**：`replace_feature`、implicit gemm、indice 相關邏輯在 symbolic trace 時爆掉或圖不正確。

**作法**：

- 在 **`projects/SparseConvolution/`**（`sparse_conv.py`、`sparse_functional.py`）對 **`_fx_tracing`** 等路徑加守衛或 stub，讓 **FX 僅為了構圖** 時能走完（與 spconv INT8 指南方向一致）。
- 在 **`deployment/cli/main.py`** 於 import 鏈最前設定 **`SPCONV_FX_TRACE_MODE=1`**，確保部署進程與容器行為一致。

### 3.2 `nn.ReLU` 接到 `SparseConvTensor`（ONNX / trace 崩潰）

**現象**：`TypeError: relu_(): argument 'input' must be Tensor, not SparseConvTensor`。  
**原因**：mmdet3d 的 **`make_sparse_convmodule`** 在 `act` 槽位使用 **`nn.ReLU`**，但 spconv2 的 **`SparseSequential`** 把 **整個 `SparseConvTensor`** 往後傳，ReLU 只認 dense tensor。

**作法**：

- 新增 **`projects/BEVFusion/bevfusion/sparse_convmodule.py`**：API 對齊 mmdet3d，但在 spconv2 下 **`act` 改用 `spconv.pytorch.SparseReLU`**。
- **`sparse_encoder.py`** 改為 **從本專案 `sparse_convmodule` 匯入 `make_sparse_convmodule`**，讓 conv stem / `conv_out` / `make_encoder_layers` 全線一致。
- **`sparse_block_fx.py`**（`SparseBasicBlockFX`）：中間與最後 activation 使用 **SparseReLU**；殘差寫成 **`out + identity`**（同為 `SparseConvTensor`）以利 FX 融合殘差與後續量化圖。

### 3.3 `quantized_add` 只支援 dense，殘差仍為 `SparseConvTensor`

**現象**：`AttributeError: 'SparseConvTensor' object has no attribute 'shape'`（spconv `quantization/core.py` 的 `quantized_add`）。  
**原因**：`transform_qdq` 把 `torch.ops.quantized.add` 指到 spconv 的 **`quantized_add`**，但殘差在圖上仍是 **sparse 包裝**，內部才是 qint8 feature。

**作法**（多層防護，commit 內皆有）：

1. **上游 spconv 原始碼**（若你從 workspace 編譯）：在 **`spconv/pytorch/quantization/core.py`** 將「純 tensor 的加總」抽成 **`_quantized_add_qtensor`**，**`quantized_add` 對 `SparseConvTensor` 改為對 `.features` 運算後 `replace_feature`**。
2. **`spconv/pytorch/quantization/graph.py`**：`transform_qdq` 改為綁定 **`spconv_quant_core.quantized_add`**（模組屬性），避免 `from core import quantized_add` 造成 **patch 後仍指向舊函式**。
3. **AWML 補丁**：**`deployment/projects/bevfusion/quantization/spconv_quantized_add_patch.py`**  
   - **`ensure_spconv_quantized_add_sparse_support()`**：對 **pip 安裝的舊版 spconv** 做 monkey-patch，並同步 **`quantization.graph.quantized_add`**。  
   - **`retarget_graphmodule_quantized_add_calls()`**：已存好的 **GraphModule** 若仍指向舊函式，載入後把 node target 改回目前的 **`core.quantized_add`**。  
   - 在 **`convert_spconv_int8`** 與 **`model_loader`（spconv_int8 載入後）** 呼叫上述邏輯。

### 3.4 Runner / PTQ 腳本參數錯誤或重複轉 FX

**現象**：校準函式傳錯參數、或 encoder 已是 `GraphModule` 又做一次 INT8 轉換。

**作法**：**`runner.py`**、**`bevfusion_quantization.py`**、**`spconv_int8.py`** 整併為單一清晰流程：**`apply_spconv_int8_quantization` → `calibrate_spconv_model` → `convert_spconv_int8`**；若 encoder 已是 FX 結果則 **跳過重複轉換**（依實作版本而定，詳見程式註解）。

### 3.5 PTQ checkpoint 與模型權重排列不一致

**現象**：`load_state_dict` 大量 missing / unexpected；sparse conv 5D weight 維度順序與 FX 模型不一致。

**作法**：**`model_loader.py`** 內 **`_permute_sparse_encoder_weights_to_match_model`**、以及載入前 **`_strip_module_prefix_state_dict`** 等，對齊 **checkpoint 與當前 `pts_middle_encoder` 的 key 與 shape**。

### 3.6 ONNX 追蹤時 GPU OOM（`dense()`）

**現象**：`torch.OutOfMemoryError`，在 **`sparse_encoder._conv_out_to_bev` → `out_tensor.dense()`** 配置出超大 dense grid。

**作法**：**`onnx_export_pipeline.py`** + **`OnnxConfig.trace_device`**（**`deployment/configs/schema.py`**）：**預設用 CPU 做 ONNX trace**（可用環境變數或 config 改回 CUDA）。

### 3.7 Docker / 依賴

**作法**：**`projects/BEVFusion/Dockerfile`** 等補上與量化／FX 相關的環境說明；實際 **`pytorch-quantization`** 安裝指令仍以 **README_INT8_IMPLEMENTATION.md** 為準（含 NVIDIA PyPI index）。

---

## 四、Commit b0ad1b1 變更檔案對照（精簡）

| 區塊 | 檔案 | 角色 |
|------|------|------|
| CLI / 環境 | `deployment/cli/main.py` | `SPCONV_FX_TRACE_MODE` 等 |
| 設定 | `deployment/configs/schema.py` | `OnnxConfig.trace_device` |
| 部署設定 | `deploy_config_int8.py` | INT8 範例、ONNX trace 說明 |
| ONNX | `export/onnx_export_pipeline.py` | 依 `trace_device` 搬模型做 export |
| 載入 | `io/model_loader.py` | sparse fuse、FX 結構重建、permute、驗證、quantized_add retarget |
| Spconv INT8 | `quantization/spconv_int8.py` | prepare / calibrate / convert 主流程 |
| 補丁 | `quantization/spconv_quantized_add_patch.py` | pip spconv + 舊 GraphModule |
| Runner | `runner.py` | 與 quantization 腳本對齊的呼叫方式 |
| 驗證腳本 | `scripts/verify_spconv_int8.py` | 離線檢查 encoder |
| PTQ 入口 | `deployment/quantization/bevfusion_quantization.py` | PTQ 與 spconv INT8 整合 |
| 模型 | `sparse_block_fx.py`、`sparse_convmodule.py`、`sparse_encoder.py`、`bevfusion/__init__.py` | FX 友善 sparse 結構 |
| Config | `bevfusion_*_120m_fx.py` | `block_type=basicblock_fx` |
| 底層 | `SparseConvolution/sparse_conv.py`、`sparse_functional.py` | FX trace 守衛 |
| 文件 | `README_INT8_IMPLEMENTATION.md` | 與本歷程互補的操作說明 |

---

## 五、目前 INT8「長什麼樣子」（概念上）

- **Dense（image backbone / neck / head）**：**FakeQuant + TensorQuantizer**，校準後 `_amax` 進 checkpoint；推論時為 Q/DQ 或對應量化算子（依匯出方式而定）。
- **Sparse（`pts_middle_encoder`）**：在啟用 **`spconv_int8`** 且使用 **`basicblock_fx`** 時，為 **FX 轉換後的量化圖（多為 `GraphModule`）**，權重與 scale 在 **spconv / torch AO** 定義的子模組與 buffer 中；數學上為 **INT8 稀疏卷積路徑 + 殘差上的量化加法**（經 **`quantized_add` sparse 感知** 修正後）。

---

## 六、使用與除錯建議

1. **務必使用 `*_fx.py` 訓練 config**（僅 `block_type` 等差異），與 PTQ / deploy 使用同一條模型結構鏈。
2. **PTQ 與 deploy 的 `deploy_config_int8.py`**：`quantization` 區塊（`spconv_int8`、`ptq_checkpoint`）需與 checkpoint 類型一致。
3. ONNX 追蹤 device：**預設 `auto` = 與模型同 device（多為 CUDA）**。**CUDA 版 spconv** 的 `implicit_gemm` 只吃 **GPU 上的 indices**；若強制 **`trace_device=cpu`**，會出現 **`indices_device=cpu` + `cudaErrorIllegalAddress`**。小 GPU 若因 **`dense()`** OOM，請換 **更大 VRAM** 或 **在別台機器導 ONNX**，不要用「CPU trace + CUDA spconv」組合（除非使用 **CPU 版 spconv**）。
4. 若懷疑 sparse 沒真的 INT8：跑 **`verify_spconv_int8.py`**，並看 log 裡 **`verify_spconv_int8_encoder`** 的欄位（注意 **qint8 param 數為 0 時仍可能合法**，若權重包在 quantized submodule 內）。
5. **`SPCONV_FX_TRACE_MODE` 與一般 FP32 ONNX**：若在 **`deployment/cli/main.py`** 對**所有**專案預設 `SPCONV_FX_TRACE_MODE=1`，spconv 會**關掉** `SparseConvTensor` 裡對 **`indices` 須為 `int32`** 等檢查。此時 ONNX 包裝器若用 **`torch.zeros(...)` 當 batch 維（預設 float32）再與 `coors` 拼接，會得到**非 int32 的 indices**，CUDA `implicit_gemm` 可能 **`cudaErrorIllegalAddress`**，且在 `finally` 裡 `model.to(cuda)` 再報二次錯誤。**現行行為**：僅在命令列含 **`deploy_config_int8`** 或設 **`AWML_SPCONV_INT8=1`** 時預設開 FX 模式；一般 **`deploy_config.py`** 維持嚴格檢查。PTQ 腳本 **`bevfusion_quantization.py`** 開頭仍會 `setdefault(..., "1")`。**ONNX 路徑**已強制 **`coors` / voxel 座標為 `int32`**，避免再誤入 CUDA。
6. 若 INT8 deploy 的 deploy 檔名**不含** `deploy_config_int8`：請 **`export SPCONV_FX_TRACE_MODE=1`** 或 **`export AWML_SPCONV_INT8=1`** 再執行 CLI。
7. **`merge_sort: cudaErrorIllegalAddress`（`implicit_gemm_pair`）** 常見根因：**`indices` 非 `int32` 或非 contiguous** 仍進入 spconv CUDA；或 **`batch_size` 與 `coors[:,0]` 不一致**。對策：`sparse_conv` 在進 kernel 前 **`.contiguous()` + 轉 `int32`**；`extract_pts_feat` ONNX 路徑 **`torch.jit.is_tracing()` 時 clone** 座標／特徵避免 trace 別名；**`batch_size = max(coords[:,0].max()+1, 1)`**；`sparse_encoder` 建 tensor 前同樣保證 dtype/layout。仍失敗時設 **`CUDA_LAUNCH_BLOCKING=1`** 對齊 stack。徹底方案長期是 **torch.export/dynamo** 或 **稀疏段自訂 ONNX op**，因 **legacy `torch.onnx.export` + JIT trace 與 spconv 本質上脆弱**。

---

## 七、與其他文件的關係

| 文件 | 內容 |
|------|------|
| **README_INT8_IMPLEMENTATION.md** | 指令、參數表、Docker、常見錯誤編號 |
| **README_SPCONV_INT8_實作歷程.md**（本文件） | **為什麼這樣改**、**b0ad1b1 脈絡**、**困難與對策** |
| **README_SPCONV_INT8.md**（若存在） | spconv 官方路徑與 API 對照 |

---

*本文描述之行為以仓库内程式為準；commit message 僅為 `chore: testing spconv int8`，實質技術內容以上述檔案與本說明為主。*
