# Path B: INT8 ImplicitGemm Plugin for Autoware TensorRT

## Overview

**Path B** modifies the Autoware TensorRT `ImplicitGemm` plugin to run INT8
sparse convolution kernels from `cumm` (the open-source kernel library behind
`spconv`).  Unlike Path A (which uses NVIDIA's closed-source `libspconv.so`),
Path B keeps the entire stack open-source and maintainable.

### Path A vs Path B Comparison

| Aspect                | Path A (libspconv)                     | Path B (cumm INT8 Plugin)             |
|-----------------------|----------------------------------------|---------------------------------------|
| Kernel source         | Closed-source binary (`libspconv.so`)  | Open-source (`spconv` + `cumm`)       |
| Autoware integration  | Requires separate runtime bypass       | Drop-in plugin replacement            |
| Maintainability       | Hard to update/debug                   | Full source access, recompilable      |
| Dependency            | NVIDIA Lidar AI Solution               | Standard `spconv`/`cumm` packages     |
| Plugin interface      | Proprietary ONNX + C++ bridge          | Standard TRT plugin (ONNX compatible) |

## Architecture

### Data Flow

```
Standard FP16 Path:
  [FP16 features] ──→ ImplicitGemm Plugin ──→ [FP16 output]
                       (FP16 cumm kernel)

INT8 Path B:
  [FP16 features] ──→ ImplicitGemmInt8 Plugin ──→ [FP16 output]
                       │                    ↑
                       │  quantize FP16→INT8 │  output_dtype=FP16
                       │  (CUDA kernel)      │  (dequantized)
                       ↓                     │
                    INT8 cumm implicit_gemm ──┘
                    (with channel_scale + bias)
```

The plugin maintains FP16 I/O for seamless TRT integration while running
INT8 kernels internally.  This avoids TensorRT's INT8 tensor dimension
restrictions (≥3D) for the 2D sparse features `[N, C]`.

### Plugin Inputs (7)

| Index | Name                  | Type  | Shape                        |
|-------|-----------------------|-------|------------------------------|
| 0     | `features`            | FP16  | `[N, C_in]`                  |
| 1     | `filters`             | FP16  | `[C_out, K1, K2, K3, C_in]`  |
| 2     | `pair_fwd`            | INT32 | `[K_vol, num_act_out]`        |
| 3     | `pair_mask_fwd`       | INT32 | `[num_act_out, 1]`            |
| 4     | `mask_argsort_fwd`    | INT32 | `[num_act_out]`               |
| 5     | `channel_scale`       | FP32  | `[C_out]`                     |
| 6     | `bias_scaled`         | FP32  | `[C_out]`                     |

### Plugin Attributes

| Name           | Type    | Description                                  |
|----------------|---------|----------------------------------------------|
| `is_subm`      | INT32   | SubManifold convolution flag                 |
| `output_scale` | FLOAT32 | Output activation scale (amax/127)           |
| `input_scale`  | FLOAT32 | Input activation scale (amax/127)            |
| `act_alpha`    | FLOAT32 | Activation parameter (LeakyReLU slope, etc.) |
| `act_beta`     | FLOAT32 | Activation parameter                         |

### INT8 Scale Computation

From PTQ calibration `_amax` values:

```python
input_scale  = input_amax / 127.0      # per-tensor
w_scales[c]  = weight_amax[c] / 127.0  # per-channel
output_scale = output_amax / 127.0     # per-tensor (next layer's input_amax)

channel_scale[c] = (input_scale * w_scales[c]) / output_scale
bias_scaled[c]   = bias[c] / output_scale
```

Inside the kernel (from spconv TENSORRT_INT8_GUIDE):

```
output = alpha × (int8_features @ int8_weights) × channel_scale + bias_scaled
       = output_scale × (feat/inp_s @ w/w_s) × (inp_s × w_s/out_s) + bias/out_s
       ≈ (feat @ w + bias) / out_s × out_s
       = feat @ w + bias    (full-precision equivalent)
```

## Internal Operation (`enqueue`)

1. **Compute weight scales**: `w_scale[c] = channel_scale[c] × output_scale / input_scale`
2. **Quantize features**: FP16 → INT8 via CUDA kernel (`round(fp16 / input_scale)`)
3. **Quantize weights**: FP16 → INT8 per-channel via CUDA kernel
4. **Call `ConvGemmOps::implicit_gemm`** with INT8 features/weights + scale/bias
5. **Output**: FP16 (set via `output_dtype = tv::float16`)

The cumm tuner automatically selects INT8 kernels when it detects INT8
input tensors.

## File Structure

```
deployment/projects/bevfusion/cpp/int8_plugin/
├── CMakeLists.txt                           # Standalone build
├── implicit_gemm_int8_plugin.hpp            # Plugin header
├── implicit_gemm_int8_plugin.cpp            # Plugin implementation
├── implicit_gemm_int8_plugin_creator.hpp    # TRT creator header
├── implicit_gemm_int8_plugin_creator.cpp    # TRT creator implementation
├── quantize_features.cuh                    # CUDA kernel header
├── quantize_features.cu                     # FP16→INT8 quantization kernels
└── plugin_registration.cpp                  # TRT plugin registration

deployment/projects/bevfusion/export/
└── sparse_int8_onnx_transform.py            # ONNX post-processor

deployment/projects/bevfusion/benchmark/
└── build_and_test_int8_plugin.sh            # End-to-end script
```

## Build Instructions

### Prerequisites

- CUDA Toolkit ≥ 11.4
- TensorRT ≥ 10.0
- spconv and cumm (installed via pip or from source)
- CMake ≥ 3.18

### Build

```bash
cd deployment/projects/bevfusion/cpp/int8_plugin
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces `libimplicit_gemm_int8_plugin.so`.

### Loading the Plugin

Add to `deploy_config_split_int8.py` → `tensorrt_config`:
```python
plugin_libraries = [
    '/opt/plugins/libautoware_tensorrt_plugins.so',
    '/path/to/build/libimplicit_gemm_int8_plugin.so',
]
```

**WARNING: Do NOT use `LD_PRELOAD`.**  Loading the plugin at process start
causes C++ symbol conflicts with Python's `spconv` package
(`basic_string::_M_create`).  The `plugin_libraries` mechanism defers
loading until TRT engine build/load, after Python spconv is fully
initialized.

## Usage: End-to-End Pipeline

### Step 1: PTQ Calibration (already done)

```bash
python deployment/quantization/bevfusion_quantization.py ptq \
    --config <config> --checkpoint <fp32_ckpt> \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    --calibrate-samples 256 --sparse-int8-only \
    --output work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth
```

### Step 2: Export Standard FP16 ONNX

```bash
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    <config> --export-only
```

### Step 3: Transform ONNX for INT8

```bash
python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \
    --onnx work_dirs/bevfusion/sparse_encoder.onnx \
    --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \
    --output work_dirs/bevfusion/sparse_encoder_int8_pathb.onnx
```

### Step 4: Build TRT Engine and Evaluate

Use the INT8-transformed ONNX with the ImplicitGemmInt8 plugin loaded.

### All-in-One Script

```bash
bash deployment/projects/bevfusion/benchmark/build_and_test_int8_plugin.sh \
    --config <config> --checkpoint <ptq_ckpt> --onnx <fp16_onnx>
```

## cumm INT8 Kernel Requirements

From `spconv/docs/INT8_GUIDE.md`:

- **Channel alignment**: `input_channel % 32 == 0 && output_channel % 32 == 0`
- **Efficient shapes**: INT8 faster than FP16 when:
  - `C == 32 && K == 64`
  - `C == 64 && K == 32`
  - `C >= 64 && K >= 64`

BEVFusion sparse encoder channels (16→32→64→128→256) mostly satisfy these
requirements starting from the second layer.

## Comparison with Existing FP16 Plugin

The existing `ImplicitGemmPlugin` (version 1) already calls
`ConvGemmOps::implicit_gemm` but only passes:
- FP16 features + FP16 weights
- `output_scale = 1.0` (no-op)
- Empty `scale` and `bias` tensors

The `ImplicitGemmInt8Plugin` extends this by:
1. Adding CUDA quantization kernels (FP16 → INT8)
2. Passing actual `output_scale`, `channel_scale`, and `bias_scaled`
3. Setting `output_dtype = FP16` for dequantized output
4. Requesting workspace for INT8 temporary buffers
5. Using a dedicated INT8 tuner instance

The cumm tuner automatically selects INT8 GEMM kernels when it detects
INT8 input dtype, achieving hardware INT8 throughput on Tensor Cores.

---

## 開發紀錄與補充（依實作順序累加）

以下為 Path B 實作過程中 **逐步** 記下的修正、除錯方式與目前進度。**上文各節維持初稿，不作改寫**；若與本節有出入，以本節實務紀錄為準。

### 模組說明：`deployment.projects.bevfusion.export.sparse_int8_onnx_transform`

此模組是 **Path B 的 ONNX 後處理器**（CLI：`python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform`）。標準 BEVFusion／Autoware 稀疏段匯出後，圖裡是 **`autoware::ImplicitGemm`**（5 個輸入：feature、weight、pair 相關等），數值仍是 **FP16 權重／特徵**；TRT 上的 FP16 plugin 不會自動帶入 PTQ 的 INT8 scale。本腳本把每顆可對齊校準的 `ImplicitGemm` **改成** **`autoware::ImplicitGemmInt8`**（7 個輸入），並寫入與 PTQ 一致的 scale／bias，讓後續載入 **`ImplicitGemmInt8` C++ plugin** 時能跑內部 INT8 kernel。

**輸入**

| 參數 | 用途 |
|------|------|
| `--onnx` | 已匯出的 **FP16** sparse ONNX（含 `ImplicitGemm`），勿對已 INT8 轉換過的檔重跑 |
| `--checkpoint` | PTQ `.pth`，內含 NVIDIA 風格的 `*_quantizer._amax`（輸入／權重校準） |
| `--output` | 寫出的新 ONNX 路徑 |
| `--config` | 選填；目前主流程從 checkpoint 的 `state_dict` 取 bias，多數情況可不填 |

**內部步驟（概念順序）**

1. **`_load_amax_from_checkpoint`**：掃描 `state_dict` 裡所有含 `_amax` 的 tensor。
2. **`_build_layer_scale_map`**：用 regex 對 `pts_middle_encoder.<stem>._input_quantizer._amax` 與 `._weight_quantizer._amax` 分組，得到每層的 `input_amax`、`weight_amax`（權重多為 per-channel）。
3. **推導 `output_amax`**：第 *i* 層的輸出尺度取第 *i+1* 層的 `input_amax`（與常見「下一層輸入量化」鏈一致）；最後一層可無下一層則 `output_scale` fallback。
4. **`_compute_int8_scales`**：依上文 **INT8 Scale Computation** 的公式算 `input_scale`、`output_scale`、`channel_scale[C_out]`、`bias_scaled[C_out]`（bias 從 `state_dict` 的 `…stem….bias` 或 `pts_middle_encoder.<stem>.bias` 取，若無則 bias_scaled 填零）。
5. **遍歷 `graph.node`**：對 `domain==autoware` 且 `op_type==ImplicitGemm` 的節點，用 **`_match_onnx_node_to_layer`** 把 ONNX 張量名（`/` 路徑）與 checkpoint 的 **layer stem**（`.` 路徑）對上（細節見下方 **紀錄 3**）。
6. **改寫節點**：同名屬性保留，並加上 `input_scale`／`output_scale`；新增兩個 **initializer**（`channel_scale`、`bias_scaled`），並為 TensorRT 解析器補上對應的 **`graph.input`（ValueInfo）**（見 **紀錄 4**）。
7. **`onnx.save`** 輸出新圖。

**輸出**：仍為 FP16 I／O 的稀疏 ONNX，但稀疏卷積節點變為 **`ImplicitGemmInt8`**，供 TensorRT 在載入 `libimplicit_gemm_int8_plugin.so` 時對應到 Path B plugin。無法對齊校準的節點（例如未量化的 `conv_out`）會 **維持原 `ImplicitGemm`**，並印 `[warn] Could not match …`。

### 紀錄 1：`ImplicitGemmInt8` C++ plugin 與 CMake

- 新增目錄 `deployment/projects/bevfusion/cpp/int8_plugin/`（見上文 **File Structure**）：`enqueue` 內 FP16→INT8 量化、`ConvGemmOps::implicit_gemm`、FP16 輸出。
- **編譯**：`SPCONV_ALLOC_*` 與 spconv 標頭巨集同名，不可再定義同名 `static constexpr`；改 `#ifndef …` fallback 或沿用標頭定義。
- **`PLUGIN_ASSERT` 在 `noexcept` 內**：不可 `throw`，改 `fprintf` + `std::abort()`，避免 `-Wterminate`。
- **符號可見性**：CMake 設 `CMAKE_CXX_VISIBILITY_PRESET hidden` 等，降低與同 process 內 Python `spconv` 的衝突（plugin 僅在 TRT 載入 library 時載入）。

### 紀錄 2：載入方式 — 禁用 `LD_PRELOAD`

- **現象**：程式一進 Python 即 `ValueError: basic_string::_M_create`（常出現在 `spconv` 測試路徑）。
- **原因**：INT8 plugin `.so` 連結 spconv/cumm，若用 `LD_PRELOAD` 會 **早於** PyTorch 用的 spconv 載入，符號／初始化順序衝突。
- **做法**：**僅**透過 `tensorrt_config.plugin_libraries` 讓 TensorRT 在建 engine／載入時再載入 plugin（上文 **Loading the Plugin** 已警告；`benchmark/build_and_test_int8_plugin.sh` 已移除誤導的 `LD_PRELOAD`）。

### 紀錄 3：`sparse_int8_onnx_transform.py` — checkpoint stem 與 ONNX 節點對齊

- Checkpoint 裡的 stem 形如 `encoder_layers.encoder_layer1.0.conv1`；ONNX 圖路徑多為 `/pts_middle_encoder/.../encoder_layer1/...`，**沒有** `encoder_layers` 前綴。
- **作法**：除完整 stem 外，產生 **只剝掉第一層前綴**（第一個 `.` 之前）的 variant，例如 `encoder_layer1.0.conv1`。
- **錯誤作法**：用過短後綴（如 `0.conv1`）去 substring match → **多層誤配**，`channel_scale` 維度錯（例如多層都變成 16）。
- **驗證**：log 中 `Transformed … ImplicitGemm → ImplicitGemmInt8` 數量應與預期 INT8 層數一致；各層 `channel_scale_shape` 應隨 **C_out** 遞增（16→32→64→…），而非多層重複同一 shape。

### 紀錄 4：TensorRT 解析 ONNX — 新 initializer 需對應 `graph.input`

- 為 `channel_scale` / `bias_scaled` 新增 `graph.initializer` 後，若缺少對應 **`graph.input`（`ValueInfo`，含 shape / elem_type）**，TRT 常報：`INVALID_GRAPH: Failed to import initializer`。
- **作法**：每個新 initializer 以 `helper.make_tensor_value_info` 等補上 **同名 graph input**（與現有 transform 腳本一致版本）。
- **名稱**：若 initializer 名稱含 **`.`**（例如用 layer stem `encoder_layer1.0.conv1` 直接拼接），部分 TensorRT 版本仍會 **Failed to import initializer**。`sparse_int8_onnx_transform` 已將 scale 相關 tensor 名 **sanitize 成僅 `[A-Za-z0-9_]`**（見 `_safe_trt_scale_names`）。

### 紀錄 5：Deploy 與 CLI 實務

- **`export.mode`**（`deployment/configs/enums.py` → `ExportMode`）：`onnx` 只匯 ONNX；`trt` 只建 TensorRT、**不**重匯 ONNX（讀 `export.onnx_path` 下既有檔）；`both` 兩者；`none` 皆不做。手動跑完 INT8 transform 後若要避免覆蓋 ONNX，deploy 設 **`mode="trt"`**（與 `deploy_config_split_int8.py` 用法一致）。
- **上文 Step 2 的 `--export-only`**：目前 **CLI 無此旗標**；匯出與否由 deploy 的 `export.mode` 控制（例如要先匯 FP16 ONNX 可用 `both` 或 `onnx`）。
- **檔名**：建議保留一份 **未經 INT8 transform** 的 FP16 sparse ONNX 作為 transform 的 `--onnx` 輸入；若輸出覆蓋成 `bevfusion_sparse.onnx`，勿再以已 INT8 的檔重跑 transform。
- **`conv_out`**：PTQ 若維持 FP32，最後一顆 `ImplicitGemm` **不**變成 `ImplicitGemmInt8`，log 可能出現 `Could not match … conv_out` —— **屬預期**。

### 紀錄 6：除錯檢查清單

| 症狀 | 方向 |
|------|------|
| `basic_string::_M_create` 於 import / 極早階段 | 檢查是否誤用 `LD_PRELOAD` 載入 plugin |
| 幾乎只配到一層或 `channel_scale` 形狀全錯 | stem／suffix 匹配邏輯；見紀錄 3 |
| `Failed to import initializer` | 新 scale／bias 是否補齊 `graph.input`；見紀錄 4 |
| CMake `SPCONV_ALLOC_FEATURES` 相關錯誤 | 與 spconv 巨集衝突；見紀錄 1 |
| 改過 transform 但行為不變 | 刪除 `deployment/projects/bevfusion/export/__pycache__` 後重跑 |

### 紀錄 7：目前進度（快照）

| 項目 | 狀態 |
|------|------|
| `ImplicitGemmInt8` + 獨立 CMake 編譯 | 已實作；需在目標 CUDA/TRT/spconv 環境驗證 |
| `sparse_int8_onnx_transform.py` | 已實作（stem variant、graph.input、避免短 suffix 誤配） |
| `deploy_config_split_int8.py` 雙 plugin `.so` | 已可配置 |
| PyTorch 端稀疏 INT8 mAP | 見 `9_nvidia_spconv_int8_fix.md` 等既有文件 |
| TRT 稀疏 INT8 端到端數值／延遲 vs PyTorch | **待完整驗證**（建議單 frame `lidar_bev` 對齊） |

### 紀錄 8：`ImplicitGemmInt8` 與 Q/DQ 的關係 — 稀疏 ONNX 有 Q/DQ 的根因與對策

**問題**：匯出的 `bevfusion_sparse.onnx` 裡含有 `QuantizeLinear` / `DequantizeLinear`。`ImplicitGemmInt8` 能不能吃 Q/DQ？

#### 8-1. `ImplicitGemmInt8` 不吃 Q/DQ

Plugin 預期收到的是 **FP16** feature 和 **FP16** weight，自己在 `enqueue` 內用 CUDA kernel（`quantize_features.cu`）做 FP16→INT8。scale 來自 plugin 的 attribute（`input_scale`、`output_scale`）和額外輸入（`channel_scale`、`bias_scaled`），與標準 ONNX Q/DQ 節點完全無關。

#### 8-2. 為什麼我們的稀疏 ONNX 會有 Q/DQ — 根因分析

匯出流程有一個 **FP32 shadow 機制**（`onnx_export_pipeline.py` line 462–496）：偵測到 `pts_middle_encoder` 是 FX `GraphModule` 時，會用 `build_float_sparse_encoder_shadow` 建一個全新的 FP32 encoder 暫替 trace → 乾淨的 FP16 ONNX。

**但我們走的是 NVIDIA TensorQuantizer 路徑，不是 FX 路徑：**

```
model_loader.py  line 413–415:
    _prepare_encoder_for_nvidia_int8(model)
        → apply_nvidia_spconv_int8(sparse_encoder)
        → 對每個 SparseConvolution 加上 _input_quantizer、_weight_quantizer（TensorQuantizer 子模組）
```

`apply_nvidia_spconv_int8` **不做 FX tracing**，encoder 仍然是原始的 `BEVFusionSparseEncoder`（不是 `torch.fx.GraphModule`）。

匯出時 `resolve_sparse_onnx_shadow` 掃描 `pts_middle_encoder.modules()` 找 `isinstance(m, torch.fx.GraphModule)` → **找不到** → 回傳 `None` → **shadow 不觸發**。

結果：帶 `TensorQuantizer` 的 encoder 直接被 `torch.onnx.export` trace，加上前面呼叫的 `setup_quantization_for_onnx_export()`（讓 TensorQuantizer 輸出 `QuantizeLinear` / `DequantizeLinear`），稀疏 ONNX 裡就出現了 Q/DQ。

ONNX 圖裡的結構大致如下：

```
weight_initializer → QuantizeLinear → DequantizeLinear → ImplicitGemm(input[1])
prev_layer_output → QuantizeLinear → DequantizeLinear → ImplicitGemm(input[0])
```

#### 8-3. Q/DQ 存在時，`sparse_int8_onnx_transform` 仍可運作嗎？

**可以運作**，但有冗餘。Transform 的 node matching 是看 `ImplicitGemm` 的 input **名稱**做 substring match，像 `…/_weight_quantizer/DequantizeLinear_output_0` 一樣能配對（docstring 範例就是這種名字）。配對完後改成 `ImplicitGemmInt8`，plugin 拿到的 feature/weight 仍是 **DequantizeLinear 的輸出（FP16）**。

數值面：feature 經過 Q→DQ（float→int8→float，用 `_input_quantizer._amax/127` 做 scale），再被 plugin 用同一個 `input_scale`（同樣取自 `_amax/127`）做 FP16→INT8。因為 **scale 相同**，第二次量化等同 `round(round(x/s)*s / s) = round(x/s)`，精度損失只有第一次 round，**實測無額外損失**。Weight 同理。

但 TRT 會多跑 Q/DQ 節點當獨立 layer（custom plugin 是黑盒，TRT 不會融合進去），有 **不必要的運算與記憶體開銷**。

#### 8-4. 三種對策

| 方案 | 作法 | 優點 | 缺點 |
|------|------|------|------|
| **A. 讓 shadow 也認 NVIDIA 路徑**（**已實作**） | `resolve_sparse_onnx_shadow`：若底下無 `GraphModule` 但 `encoder_has_nvidia_tensor_quantizers` 為真，且具 shadow 屬性或可由 `model.cfg` 補齊 → 回傳 `(pts_middle_encoder, overrides)`，`build_float_sparse_encoder_shadow` 從現有 `state_dict` 拷貝浮點權重 | 稀疏 ONNX 無 Q/DQ | 需重新匯出 ONNX |
| **B. Transform 時剝除 Q/DQ** | 在 `sparse_int8_onnx_transform.py` 中，遇到 `ImplicitGemm` 輸入是 `DequantizeLinear` 的輸出時，追溯到 DQ 的輸入（即 Q 之前的 raw float），將 `ImplicitGemmInt8` 的 input 改指向 Q/DQ 之前 | 不需重新匯出 ONNX | Transform 邏輯更複雜；每層需找 Q→DQ chain |
| **C. 維持現狀（接受 Q/DQ）** | `ImplicitGemmInt8` 照收 DQ 輸出的 FP16，再自己量化 | 不改任何程式碼 | TRT 多跑 Q/DQ 節點；圖不乾淨 |

##### 方案 A：「觸發 shadow」具體是什麼意思？

**「觸發 shadow」** 不是再跑一次 PTQ，而是指在 **`torch.onnx.export` 執行期間**（且僅此期間），匯出管線做與現有 FX 路徑相同的一件事：

1. **暫存** `orig = model.pts_middle_encoder`（你現在這顆：帶 `_input_quantizer` / `_weight_quantizer` 的 encoder）。
2. **換上** `model.pts_middle_encoder = shadow_encoder`，其中 `shadow_encoder` 是一顆 **新建立的純 FP32 `BEVFusionSparseEncoder`**：
   - **拓樸與設定**與真實塔一致（`sparse_shape`、`encoder_channels`… 從 `model.cfg` 或現有模組複製，與 `build_float_sparse_encoder_shadow` 今日用法相同）。
   - **Forward 裡沒有** `TensorQuantizer`，因此 `setup_quantization_for_onnx_export()` 也不會在稀疏塔這段產生 `QuantizeLinear` / `DequantizeLinear`。
3. **權重**：把 **浮點 conv weight / bias** 從 `orig.state_dict()`（或對應 key）灌進 `shadow_encoder`，**不**把 `_amax`、quantizer 子模組當成 ONNX 裡要 trace 的節點。  
   - 今日 `build_float_sparse_encoder_shadow(gm_src, …)` 的實作假設 **`gm_src` 是 `GraphModule`**；方案 A 的工程意義就是：**當 `gm_src` 改成「NVIDIA 路徑的 `BEVFusionSparseEncoder`」時，仍要能建出同一顆 FP32 shadow 並正確 `load_state_dict` 對應層**，必要時在 `sparse_encoder_float_shadow.py` / `onnx_export_pipeline.py` 加分支。
4. **`torch.onnx.export` 跑完**（在 `finally`）把 `model.pts_middle_encoder = orig` **還原**，所以 **記憶體裡的訓練／推理用模型** 仍是帶 quantizer 的版本，只有寫到磁碟的 **稀疏 ONNX 是「無 Q/DQ 的浮點 ImplicitGemm 圖」**。

**一句話**：「觸發 shadow」= **匯出時短暫換成「沒有 TensorQuantizer 的雙胞胎 FP32 稀疏塔」來 trace**，讓 ONNX 裡不要出現 Q/DQ；INT8 語意仍由後續 Path B（`_amax` → `ImplicitGemmInt8`）或 PyTorch 推理路徑負責，與這顆乾淨 ONNX 相容。

**建議先用方案 C 走通端到端驗證**，確認 TRT engine build + mAP 正確後，再用方案 A 或 B 做乾淨版。

#### 8-5. 確認方式

```python
import onnx
model = onnx.load("bevfusion_sparse.onnx")
qdq = [n.name for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
print(f"Q/DQ nodes in sparse ONNX: {len(qdq)}")
# NVIDIA TensorQuantizer 路徑：>0（目前狀態）
# FP32 shadow 觸發（FX 路徑或方案 A）：0
```

#### 8-6. Dense ONNX 的 Q/DQ 是另一回事

`bevfusion_dense.onnx` 如果 backbone／neck／head 有 `pytorch_quantization`，Q/DQ 是 TRT 原生就能處理的（TRT 會融合成 INT8 layer）。這跟稀疏段無關。

### 相關文件（Path B 以外）

- `10_int8_trt_gap_analysis.md`：稀疏 INT8 與 TRT 鴻溝、Path A 思路。
- Path A 產物（與本 plugin 路線獨立）：`export/libspconv_onnx_exporter.py`、`export/export_sparse_encoder_int8.py`、`cpp/libspconv_trt_bridge.cpp` 等。



### 稀疏 ONNX 含 Q／DQ（NVIDIA TensorQuantizer 路徑）

- **現象**：`bevfusion_sparse.onnx` 出現 `QuantizeLinear` / `DequantizeLinear`。
- **根因**：`pts_middle_encoder` 非 FX `GraphModule` 時，舊版 **FP32 shadow** 不觸發，直接 trace 帶 `TensorQuantizer` 的 encoder。
- **修正（scheme A）**：`sparse_encoder_float_shadow.resolve_sparse_onnx_shadow` 在偵測到 `encoder_has_nvidia_tensor_quantizers` 且 shadow 屬性（或 `model.cfg`）可補齊時，同樣走 `build_float_sparse_encoder_shadow`，匯出**無 Q/DQ** 的稀疏圖。詳見 **11** 紀錄 8、**12** §4.4。

### TensorRT：`INVALID_GRAPH: Failed to import initializer`

- **根因**：Path B 為每層新增 `channel_scale` / `bias_scaled` **initializer**；除 `graph.initializer` 外需補 **`graph.input`（ValueInfo）**（見 **11** 紀錄 4）。
- **追加**：部分 TRT 版本對 initializer **名稱含 `.`**（由 layer stem 拼接）匯入失敗。
- **修正**：`sparse_int8_onnx_transform` 內 **`_safe_trt_scale_names`**：tensor 名改為僅 `[A-Za-z0-9_]`，並用 **`_collect_occupied_tensor_names`** 避免撞名。

### Plugin assert：`channel_scale` 維度 ≠ `filters` 的 `C_out`

- **現象**：`ImplicitGemmInt8Plugin` 內 `in[IN_CHANNEL_SCALE].desc.dims.d[0] == in[IN_FILTERS].desc.d[0]` assert → `Aborted`。
- **根因**：**`conv_out`**（最後一顆稀疏卷積，PTQ 常 FP32、無可靠 `_amax`）在 ONNX 上被 **錯誤 stem 匹配**，`channel_scale` 長度變成別層的 `C_out`。
- **修正**：① 路徑／名稱含 **`conv_out`** 的節點 **不轉** `ImplicitGemmInt8`，維持 `ImplicitGemm`。② 對其餘層比對 **filter initializer 5D `shape[0]`** 與 `len(channel_scale)`，不符則 skip 並 `[warn]`。

### TensorRT warning：`Attribute act_alpha / input_scale / … not found in plugin node`

- **現象**：`onnxOpImporters.cpp:6435` 連續 warning；`[ImplicitGemmInt8] ... input_scale=1 output_scale=1`（預設值），**高風險**（未吃到 PTQ 尺度）。
- **說明**：Netron 常把 **`input_scale`（float）** 顯示成 **`input_scale_f`**；若 protobuf 裡 `AttributeProto.name` 真的帶 `_f`／或讀寫時重複後綴，TRT 用 **`input_scale`** 對 `PluginField` 會對不上。
- **修正**：`sparse_int8_onnx_transform` 以 **`_normalize_onnx_attr_field_name`** 讀原 `ImplicitGemm`，並用 **`helper.make_attribute("input_scale", …)`** 等**無後綴標準名**寫入 `ImplicitGemmInt8`；必要時保留 `is_train` / `fp32_accum` / `output_add_scale`。

### `sparse_int8_onnx_transform` 其它行為（先前已實作）

- Checkpoint stem 與 ONNX 路徑：只剝**一層**前綴 variant；**禁止**過短 suffix（如 `0.conv1`）跨層誤配。
- **詳見程式與長文**：`export/sparse_int8_onnx_transform.py`、`11`（含模組說明、紀錄 1–8）、`12`。

### 驗證結果（修正 attribute 命名後）

TRT 匯入時 **warning 消失**，plugin creator 日誌出現**逐層** `input_scale` / `output_scale`（與下一層 input 尺度鏈一致），`is_subm=1`；**最後一顆 Int8 層**可見 `output_scale=1.000000`（與 transform 最後層無 `next input_amax` 時的 fallback 一致）。範例（節錄）：

```text
[ImplicitGemmInt8] .../encoder_layer1.0/conv1/...: is_subm=1 output_scale=0.017025 input_scale=0.023417
[ImplicitGemmInt8] .../encoder_layer1.0/conv2/...: is_subm=1 output_scale=0.052212 input_scale=0.017025
...
[ImplicitGemmInt8] .../encoder_layer4.1/conv2/...: is_subm=1 output_scale=1.000000 input_scale=0.021122
```

（`conv_out` 應仍為 **`ImplicitGemm`**，不會出現在上列 Int8 日誌中。）
