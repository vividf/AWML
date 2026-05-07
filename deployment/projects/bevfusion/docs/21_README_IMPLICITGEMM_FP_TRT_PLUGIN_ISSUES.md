# 21：FP `ImplicitGemm` TRT 外掛——改動動機、行為與常見問題

本文件說明：**為何在 Path B（`ImplicitGemm` + `ImplicitGemmInt8`）管線要動 ONNX 與 TRT 外掛**、**如何使用（CLI／transform／plugin）**、改動後行為為何、以及如何對照症狀排查。可與 **`11`**（Path B plugin）、**`12`**（管線）、**`20`**（方法對照）、**`export/onnx_fuse_implicit_gemm_activation.py`** 一併閱讀。

**閱讀順序**：一、動機 → 二、改動後行為一覽 → **三、使用方式（命令、參數、plugin 編譯）** → 四、實作對照 → 五、背景 → 六～七、問題與速查 → 八、路徑 → 九、索引。

---

## 一、為什麼要做這些改動（目標與動機）

### 1. 為什麼要把 `Relu`／`Add(const)` 熔進 sparse conv（ONNX fuse）

| 現象 | 說明 |
|------|------|
| TensorRT **不會**把標準 **`Relu`**、**`Add`** 與 **`autoware::ImplicitGemm`** 當成單一融合 kernel | 若維持「ImplicitGemm → Relu」或「ImplicitGemm → Add → Relu」，圖上會多 **獨立的啟動函數／記憶體來回**，延遲與 bandwidth 都較差。 |
| 底層 **spconv／`ConvGemmOps::implicit_gemm`** 已支援 **`tv::gemm::Activation`**（含 **ReLU**）與 **bias** | 若在 ONNX／外掛層把激活與「可加上的常數偏置」表達進 **`act_type`** 與 **tensor 輸入**，可在 **同一次 implicit GEMM** 內完成，與訓練時稀疏卷積語意對齊，而不是事後再 patch。 |

**因此**：用 **`onnx_fuse_implicit_gemm_activation.py`** 在 **匯出／後處理 ONNX** 時，把尾端的 **`Relu`** 或 **`Add(常數)+Relu`** 折進 plugin（**`act_type`**、FP16 時可選 **第 6 路 bias**；INT8 時折進 **`bias_scaled`**），目的是 **正確性與效能**，不是單純修報錯。

### 2. 為什麼 FP `ImplicitGemm` 外掛必須支援 **5 與 6** 個 input，且建 engine 時不可 `abort`

| 現象 | 說明 |
|------|------|
| Fuse **`ImplicitGemm → Add(const) → Relu`** 後，ONNX 會多出 **第 6 路 bias** | 若外掛或 fork 仍寫死 **`num_inputs == 5`**（或僅在 **`supportsFormatCombination`** 裡 **`PLUGIN_ASSERT`**），TensorRT 一解析到 **6-input** 節點就 **`Assertion failed … Aborting`**，**引擎無法建立**。 |
| **`IPluginV3OneBuild::supportsFormatCombination`** 的語意是「這個 `(pos, format)` 組合是否允許」 | 建置階段 TRT 會 **枚舉大量組合**；在此處 **`abort`** 不符合 API 預期；應 **`return false`**，由引擎換別組合或報錯，而不是 **整個進程退出**。 |
| **`configurePlugin`** 若對「探路階段尚未定型」的張量過度 **`PLUGIN_ASSERT`** | 同樣會把 **可恢復的建圖失敗**變成 **硬崩潰**；應以 **回傳錯誤碼 + stderr** 為主。 |

**因此**：Autoware 側 **`ImplicitGemm`** 外掛要 **正式支援 5／6 input**，並把 **格式協商／設定**路徑上的 assert 改成 **可否定的回傳值**，目的是 **讓「有 fuse 的 ONNX」能穩定建 engine**，且行為符合 **TensorRT plugin 合約**。

### 3. 為什麼 Path-B 在 **6-input `ImplicitGemm` → `ImplicitGemmInt8`** 時要改 **`sparse_int8_onnx_transform.py`**

| 現象 | 說明 |
|------|------|
| INT8 節點固定是 **7 個輸入**（5 個稀疏 tensor + **`channel_scale`** + **`bias_scaled`**） | 轉換若只取 **`ImplicitGemm` 的前 5 個** sparse 埠，**第 6 路 fuse 進來的常數 bias** 會被 **直接丟棄**。 |
| Checkpoint 裡的 **`bias_scaled`** 通常來自 **conv.bias**，不一定等於「fuse 時額外 **`Add`** 的那一坨常數」 | 若不把第 6 路 **加回** INT8 的 **`bias_scaled`**（按 **`extra / output_scale`** 與 plugin 語意對齊），**INT8 推理數值會與 FP16 fuse 後不一致**。 |

**因此**：在 Path-B transform 裡 **合併第 6 路常數到 `bias_scaled`**，目的是 **FP16 fuse 與 INT8 部署語意連貫**，避免 **靜默錯精度**。

### 4. 為什麼執行期還要談 **第 6 路 bias 的 FP16／FP32**（問題 B）

| 現象 | 說明 |
|------|------|
| ONNX initializer 常為 **float32**，TensorRT 也可能給 **FP32 buffer** | **`ConvGemmOps::implicit_gemm`**／tensorview 依 **輸出 feature dtype** 選模板；若 **輸出為 FP16** 卻把 bias **`tv::Tensor` 建成 float32**，會觸發 **`tensor.h` dtype 檢查**錯誤。 |

**因此**：在 **`enqueue`** 把 bias **對齊 activation／輸出的 dtype**（必要時做數值轉型），目的是 **與 spconv 假設一致**，避免 **執行期 runtime_error**。

### 5. `timing_enabled`／`timing_max_logs` 警告（問題 C）

| 現象 | 說明 |
|------|------|
| ONNX 節點若未帶這兩個 attribute，TRT 會警告 creator 預設 | 多半 **不單獨造成 crash**；若要乾淨 log，可在 export／deploy **烘焙進 ONNX**，或依賴 creator／結構體預設值。 |

---

## 二、改動後的行為一覽（我們得到了什麼）

| 層級 | 改動後預期行為 |
|------|----------------|
| **ONNX（FP16）** | 尾端 **`Relu`** 或 **`Add(const)+Relu`** 可消失；改由 **`act_type`**（與可選 **第 6 路 bias**）表達。 |
| **ONNX（INT8）** | **`ImplicitGemmInt8` 後的常數 Add+Relu** 可折進 **`bias_scaled`**／**`act_type`**。 |
| **FP `ImplicitGemm` 外掛** | **5 或 6** 個輸入皆可建 engine；**格式協商不 abort**；設定失敗 **回傳錯誤碼**。 |
| **`ImplicitGemmInt8` 外掛** | 建置路徑同樣避免 **`supportsFormatCombination`／configure** 硬 abort；**pos** 越界保護。 |
| **Path-B transform** | **6-input FP `ImplicitGemm`** 轉 INT8 時，**fuse bias 合併進 `bias_scaled`**，不無聲遺失。 |

---

## 三、使用方式（建議操作流程）

### 1. 前置：目錄與環境

- 在 **`AWML` 倉庫根目錄**執行下列命令（`python` 需已安裝 **PyTorch**、**ONNX** 等；Path B 量化／deploy 另依 **`deploy_config_split_int8.py`** 可能需要 **mmengine**、**pytorch-quantization** 等）。
- **Autoware TensorRT plugin**：建 sparse engine 前需載入含 **`ImplicitGemm`** 的 **`.so`**；編譯見 **§3.4**。  
- **INT8 `ImplicitGemmInt8` plugin**：AWML **`cpp/int8_plugin`** 產物須與 **`deploy_config`** 裡 **`tensorrt_config.plugin_libraries`** 一致。

### 2. Path B（split ONNX + 稀疏 INT8）管線順序（概念）

典型順序如下（實際檔名以 **`work_dirs`**／deploy 輸出為準）：

1. **（可選）PTQ**：用 **`deployment/quantization/bevfusion_quantization.py ptq`** 與 **`deploy_config_split_int8.py`** 產生含 **`_amax`** 的 **`.pth`**（見該 deploy 檔案開頭註解）。
2. **Deploy 匯出 ONNX／建 engine**：  
   `python -m deployment.cli.main bevfusion <deploy_config_split_int8.py> <mmconfig.py> ...`  
   會依設定產出例如 **`bevfusion_sparse.onnx`**（內為 **`autoware::ImplicitGemm`**，FP16／FP32 稀疏塔）。
3. **稀疏塔換成 `ImplicitGemmInt8` ONNX**：對 **`bevfusion_sparse.onnx`** 跑 **`sparse_int8_onnx_transform.py`**（見 **§3.3**），輸出 Path B 用的 **INT8 稀疏 ONNX**；再以 TensorRT 建 **sparse engine**（載入 **FP + Int8 兩個 plugin** 的設定見 **`deploy_config_split_int8.py`**）。

本文件關注的 **Relu／Add fuse**、**6-input bias 合併**，在步驟 **3** 的 transform 裡 **預設開啟**（見 **§3.3 參數**）。

### 3. `sparse_int8_onnx_transform`（Path B 稀疏 ONNX）

**用途**：讀入 **浮點 `ImplicitGemm`** 的 **`bevfusion_sparse.onnx`**，依 PTQ checkpoint 的 **`_amax`** 與 **encoder state_dict** 產生 **`ImplicitGemmInt8`** 節點；**預設**會呼叫 **`onnx_fuse_implicit_gemm_activation`**：先做 **FP16** 的 **`ImplicitGemm→Add(const)→Relu`**、**`ImplicitGemm→Relu`**，完成 **`ImplicitGemm→ImplicitGemmInt8`** 後再做 **`ImplicitGemmInt8→Add→Relu`**（詳見 **第四節**）。

**範例**（於 **`AWML` 根目錄**）：

```bash
python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \
  --onnx work_dirs/bevfusion/bevfusion_sparse.onnx \
  --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \
  --output work_dirs/bevfusion/bevfusion_sparse_int8.onnx \
  --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  --verbose
```

**常用參數**：

| 參數 | 說明 |
|------|------|
| **`--onnx`** | 輸入：deploy 匯出的 **稀疏** ONNX（含 **`autoware::ImplicitGemm`**）。 |
| **`--checkpoint`** | PTQ **`.pth`**（內含 **`_amax`**；Path B 亦用 **bias** 等）。 |
| **`--output`** | 輸出 ONNX 路徑。 |
| **`--deploy-cfg`** | 載入 **deploy 設定**：**`spconv_int8_fp16_layers`**（哪些層保留 FP **`ImplicitGemm`**）、**`implicit_gemm_int8_plugin_timing`**、**`implicit_gemm_int8_plugin_timing_max_logs`**（烘焙進 **`ImplicitGemmInt8`**，可減少 TRT **timing_*** 警告）、**`spconv_int8_fuse_implicit_gemm_relu`**（控制是否做 ImplicitGemm/ImplicitGemmInt8 的 Relu/Add 融合）。 |
| **`--fp16-layers`** | 逗號分隔 **字串子串**，匹配 **`ImplicitGemm` `node.name`**，該節點不換成 Int8（與 deploy 清單 **合併**）。 |
| **`--verbose`** | 印 stem 匹配、尺度鏈等診斷（stem 衝突時必開）。 |
| **`--audit-report`** | 輸出 JSON，記錄各層 INT8 scale 摘要。 |
| **`--pathb-terminal-absmax`** | 覆寫最後 INT8 層 **`output_scale`** 用的 terminal absmax（checkpoint 缺欄位時）。 |

### 4. 編譯並載入 TensorRT plugin

| 用途 | 作法 |
|------|------|
| **Autoware `ImplicitGemm`（FP）** | 編譯 **`autoware.universe/perception/autoware_tensorrt_plugins`**。容器內可用 **`projects/BEVFusion/plugins/build_plugin_inside_container.sh`**，並設定 **`AUTOWARE_TENSORRT_PLUGINS_SRC`** 指向本機 **`perception/autoware_tensorrt_plugins`**，避免 **git clone** 拿到舊 fork。 |
| **`ImplicitGemmInt8`** | 依 AWML **`deployment/projects/bevfusion/cpp/int8_plugin`** 與專案既有 CMake／README 編譯 **`.so`**，並在 **`deploy_config`** 的 **`tensorrt_config.plugin_libraries`** 與 sparse engine 建置流程中載入。 |

### 5. Deploy CLI（與本文件功能直接相關的片段）

與 **`deploy_config_split_int8.py`** 搭配的典型呼叫（完整選項以 **`deployment.cli.main`** 為準）：

```bash
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  --module main_body
```

實際會跑哪些 **export／transform／trtexec** 階段，依 **`deploy_config`** 與 CLI 子命令而定；請以 **`deploy_config_split_int8.py`** 內 **`tensorrt_config`**、**`spconv_do_sort`**、**`implicit_gemm_int8_plugin_timing`** 等為準。

### 6. 僅處理現成 ONNX（進階）

若已有 **`bevfusion_sparse.onnx`**，不要求 Path B 完整改檔，可在 **Python** 內：

```python
import onnx
from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import (
    fuse_autoware_implicit_gemm_fp16_add_relu,
    fuse_autoware_implicit_gemm_trailing_relu,
)

model = onnx.load("bevfusion_sparse.onnx")
fuse_autoware_implicit_gemm_fp16_add_relu(model)
fuse_autoware_implicit_gemm_trailing_relu(model)
onnx.save(model, "bevfusion_sparse_fused.onnx")
```

**INT8** 節點上的 **`fuse_autoware_implicit_gemm_int8_add_relu`** 通常在 **`sparse_int8_onnx_transform`** 替換為 **`ImplicitGemmInt8`** **之後**再呼叫（見 **`transform_onnx_int8`**）。

---

## 四、實作對照（改了什麼、檔案在哪）

### 1. ONNX fuse（啟動與偏置折入 plugin）

| 檔案 | 內容摘要 |
|------|----------|
| **`deployment/projects/bevfusion/export/onnx_fuse_implicit_gemm_activation.py`** | **FP16**：`ImplicitGemm → Relu` → **`act_type`**；`ImplicitGemm → Add(const) → Relu` → **第 6 路 bias + `act_type`**。**INT8**：`ImplicitGemmInt8 → Add(const) → Relu` → **`bias_scaled`／`act_type`**。輔助：**`_try_get_constant_numpy`** 等。 |
| **`projects/SparseConvolution/sparse_functional.py`**（`ImplicitGemm.symbolic`） | 匯出 **`act_type_i`** 等，與外掛 **`ImplicitGemmParameters::act_type`**／Int8 **`act_type`** 對齊。 |

### 2. FP `ImplicitGemm` 外掛（Autoware）

| 檔案 | 內容摘要 |
|------|----------|
| **`autoware.universe/.../implicit_gemm_plugin.{hpp,cpp}`** | **5／6 input**；**`supportsFormatCombination`**：`pos` 範圍檢查、不支援則 **`return false`**；**`configurePlugin`／`getOutputDataTypes`／`getOutputShapes`**：失敗 **`return -1`** + stderr；**`enqueue`**：第 6 路 bias **dtype 與 activation 對齊**（見 **第六節 問題 B**）。 |

### 3. `ImplicitGemmInt8` 外掛（AWML）

| 檔案 | 內容摘要 |
|------|----------|
| **`deployment/projects/bevfusion/cpp/int8_plugin/implicit_gemm_int8_plugin.{hpp,cpp}`** | **`supportsFormatCombination`**：`NUM_INPUTS`／`NUM_OUTPUTS`／`pos` 邊界；**configure／shape／dtype**：錯誤時回傳碼取代會 **abort** 的 assert。 |

### 4. Path-B：`ImplicitGemm` → `ImplicitGemmInt8`

| 檔案 | 內容摘要 |
|------|----------|
| **`deployment/projects/bevfusion/export/sparse_int8_onnx_transform.py`** | 若 **`len(node.input)==6`**：讀第 6 路常數，**`bias_scaled += extra_fp32 / output_scale`**，再建立 initializer；稀疏埠仍為 **前 5 個**。 |

---

## 五、背景：為什麼同一張 ONNX 會同時看到 5／6／7 個 input

| ONNX 節點 | 典型 `inputs=` | 說明 |
|-----------|----------------|------|
| **`autoware::ImplicitGemm`**（FP16 kernel） | **5** 或 **6** | **6** 來自 **`fuse_autoware_implicit_gemm_fp16_add_relu`**：`ImplicitGemm → Add(const) → Relu` 熔進 plugin，多一路 **bias**。 |
| **`autoware::ImplicitGemmInt8`** | **7** | 5 個稀疏 tensor + **`channel_scale`** + **`bias_scaled`**。 |

**結論**：同一 **`bevfusion_sparse.onnx`** 裡，FP 節點 **有的是 5 input、有的是 6**，屬 **預期**；不是 transform 隨機壞掉。

---

## 六、常見問題與處置

### 問題 A：`Assertion failed: num_inputs == 5`（`implicit_gemm_plugin.cpp` 某行）

**現象**：建 engine 時 **`[TRT] [F] Assertion failed: num_inputs == 5`**，接著 **`Aborting...`**。行號會因分支不同而偏移。

**根因**：見 **第一節 §2**。簡言之：**ONNX 已是 6 input**，但外掛仍 **`PLUGIN_ASSERT(num_inputs == 5)`**（或 **`supportsFormatCombination`** 仍對 6 input abort）。

**處置**：

1. 掃描 ONNX：`domain=="autoware"`、op **`ImplicitGemm`**，確認 **`len(node.input)`** 為 **5 或 6**（FP）／**7**（Int8）。
2. 使用 **`autoware.universe`** 中符合 **第四節 §2** 的外掛原始碼重新編譯。
3. 容器內編譯：`projects/BEVFusion/plugins/build_plugin_inside_container.sh`，設定 **`AUTOWARE_TENSORRT_PLUGINS_SRC`** 指向本機 **`perception/autoware_tensorrt_plugins`**，避免 clone 到舊 fork。

---

### 問題 B：`tensorview/tensor.h`——`expect half`／`float32` dtype 衝突

**現象**：執行期 **`std::runtime_error`**，堆疊含 **`tensor.h`**。

**根因**：見 **第一節 §4**。簡言之：**輸出為 FP16**，第 6 路 bias buffer 仍被 **`tv::Tensor` 標成 float32**，與 **spconv bias kernel** 模板假設不符。

**處置**：在 **`implicit_gemm_plugin.cpp` 的 `enqueue`**，依 TRT 實際 **`DataType`** 建立與 **activation 一致 dtype** 的 bias **`tv::Tensor`**（必要時 **FP32↔FP16 數值轉換**）。具體 helper 以 **`autoware.universe`** 目前提交為準。

---

### 問題 C：警告 `timing_enabled` / `timing_max_logs` not found

**現象**：建 engine 時 TRT 警告 **Attribute … not found in plugin node**。

**根因**：見 **第一節 §5**。多为 ONNX **未烘焙**這兩個欄位。

**處置（可選）**：在 **`sparse_int8_onnx_transform`**／**`deploy_config_split_int8.py`**（如 **`implicit_gemm_int8_plugin_timing`**）把欄位寫進 ONNX；或確認 creator／預設值與所用 TRT 版本相容。**通常不單獨导致 crash**。

---

## 七、症狀速查表

| 症狀 | 主要原因 | 處理方向 |
|------|-----------|----------|
| `num_inputs == 5` assert | 外掛仍只支援 5 input，或 **格式協商仍 assert** | **第四節 §2** 行為 + **`AUTOWARE_TENSORRT_PLUGINS_SRC`** |
| half／float32 dtype assert | **FP16 路徑 + 第 6 路 FP32 bias buffer** | **`enqueue` dtype 對齊**（**第六節 問題 B**） |
| timing attribute 警告 | ONNX 未帶欄位 | 烘焙 attribute 或忽略（**第六節 問題 C**） |
| INT8 與 FP16 fuse 數值不一致 | 6-input fuse bias **未併入 `bias_scaled`** | **第四節 §4**／**第一節 §3** |

---

## 八、相關路徑（repo 內）

| 項目 | 路徑或檔案 |
|------|------------|
| FP **`ImplicitGemm`** plugin | `autoware.universe/perception/autoware_tensorrt_plugins/src/implicit_gemm_plugin.{hpp,cpp}` |
| ONNX fuse | `deployment/projects/bevfusion/export/onnx_fuse_implicit_gemm_activation.py` |
| Path B INT8 ONNX | `deployment/projects/bevfusion/export/sparse_int8_onnx_transform.py` |
| 容器內編譯 plugin | `projects/BEVFusion/plugins/build_plugin_inside_container.sh`（**`AUTOWARE_TENSORRT_PLUGINS_SRC`**） |
| 雙 plugin deploy 範例 | `deployment/projects/bevfusion/config/deploy_config_split_int8.py` |

---

## 九、文件索引

- 與 **`20_METHOD1_TRT_PLUGIN_VS_METHOD2_LIBSPCONV_ENGINE.md`** 互補：本文件說明 **為何與如何** 在 FP／INT8 plugin、ONNX fuse、Path-B transform 上配合；INT8 整體管線仍以 **`11`**、**`12`**、**`20`** 為主。
