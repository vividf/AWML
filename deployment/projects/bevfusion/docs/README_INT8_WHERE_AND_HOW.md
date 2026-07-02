# AWML 的 INT8 在哪裡、怎麼做？與 spconv／cumm 官方路徑對照

本文件回答三件事：

1. **AWML 到底有沒有「真的」用 INT8？** 還是只有 FP16？  
2. **從 PTQ 到 TensorRT，資料在哪些步驟是 FP16、哪些步驟是 INT8？**  
3. **spconv 官方（Python + `spconv_cpp` + cumm）的 INT8 是怎麼接到同一個 `implicit_gemm` 的？**

相關延伸：`README_PTQ_INT8_SPCONV_DEPLOYMENT.md`（端到端管線）、`README_NEW3D_LIDAR_OPEN_SPCONV.md`（Lidar 開源 New3D：`spconv::Engine`、自訂 ONNX、INT8 鏈路；**§8** 為 AWML 推理加速與開源作法對照）、`cpp/int8_plugin/README.md`（`output_scale`／epilogue）、`11_int8_autoware_plugin.md`。

---

## 1. 一句話結論

| 場景 | 是否「真的」在 GPU 上跑 INT8 稀疏 GEMM？ |
|------|------------------------------------------|
| **PyTorch 評測（AWML NVIDIA PTQ）** | **大多數時間沒有**：forward 主要是 **浮點 + fake quant（TensorQuantizer）**；數學上模擬 INT8 動態範圍，**一般不等於**在 CUDA 裡對每層都做 `int8 × int8` 累加（除非你再接 spconv 的 `qint8` 推理路徑）。 |
| **TensorRT + 僅 `ImplicitGemm`（FP16 plugin）** | **沒有**：稀疏卷積是 **FP16（或 FP32）cumm kernel**，與 PyTorch 端 INT8 校準 **語意分離**。 |
| **TensorRT + sparse INT8 `ImplicitGemmInt8` + `libimplicit_gemm_int8_plugin.so`** | **有**：在 plugin 的 `enqueue` 內把特徵與權重變成 **`tv::Tensor` `int8`**，呼叫 **`ConvGemmOps::implicit_gemm`**，tuner 選 **`is_int8_inference`** 的演算法（與 spconv Python 走 C++ 路徑時相同底層）。 |

所以：**「AWML 專案有沒有用 INT8」取決於你指 PyTorch 還是 TRT、有沒有開 sparse INT8。** PTQ checkpoint 裡的 `_amax` 是 **刻度**；**真正把乘加放在 INT8 上的是 sparse INT8 TRT plugin（以及 spconv 官方的量化推理路徑）。**

---

## 2. AWML：哪裡是 FP16／FP32，哪裡是 INT8？

### 2.1 張量層級（sparse INT8 啟用時）

```
[TRT 引擎邊界]
  voxels / lidar_bev 等：多為 FP32 或 FP16（依 export／engine）
  ↓
[ImplicitGemmInt8 插件邊界]
  輸入 features：FP16
  輸入 filters：FP16（或專案中曾用 FP32 initializer 存整數格點；enqueue 仍會再量化）
  channel_scale / bias_scaled：FP32
  索引 pair_* ：INT32
  ↓
[enqueue 內部 workspace]
  feat_int8、weight_int8：INT8（device 緩衝）
  ↓
ConvGemmOps::implicit_gemm(features=int8, filters=int8, …, output_dtype=FP16)
  ↓
[Int8Inference epilogue：scale×int32_acc + bias → FP16]
  ↓
插件輸出：FP16 寫回 TRT tensor
```

**重點**：TensorRT **自訂 op 的 I/O** 刻意維持 **FP16**（官方 `TENSORRT_INT8_GUIDE.md` 也說明 plugin 對 INT8 tensor 維度有限制）。**INT8 只保證出現在 plugin 內部與 `implicit_gemm` 的 A／B 矩陣上。**

### 2.2 程式位置（AWML）

- **量化 + `implicit_gemm`**：`deployment/projects/bevfusion/cpp/int8_plugin/implicit_gemm_int8_plugin.cpp` 的 `enqueue`  
  - `launch_quantize_features` / `launch_quantize_weights_per_channel` → **INT8 buffer**  
  - `tv::from_blob(..., tv::int8)`  
  - `ConvGemmOps::implicit_gemm(...)`，`output_dtype` = FP16  
- **刻度從 PTQ 進 ONNX**：`export/sparse_int8_onnx_transform.py`（`ImplicitGemm` → `ImplicitGemmInt8` + initializers）  
- **PyTorch 稀疏塔 fake quant**：`deployment/projects/bevfusion/quantization/spconv_int8.py`（`TensorQuantizer`，histogram → `_amax`）

---

## 3. cumm／`spconv_cpp`：INT8 kernel 是怎麼被選上的？

### 3.1 同一個入口：`ConvGemmOps::implicit_gemm`

實作在：

- `spconv_cpp/spconv/src/spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps/ConvGemmOps_implicit_gemm.cc`

關鍵邏輯：**用三個 dtype 查表 tuned algo**（輸入 feature、filter、輸出）：

```45:50:spconv_cpp/spconv/src/spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps/ConvGemmOps_implicit_gemm.cc
  auto tuned_res_exist = conv_tuner.get_tuned_algo(
      kForwardInt,
      int(features.dtype()),
      int(filters.dtype()),
      int(out_features.dtype()),
      out_channel, in_channel, arch);
```

當 **`features` 與 `filters` 都是 `int8`**，且輸出為 **FP16**（或文件允許的組合）時，tuner 會選到 **`algo_desp.is_int8_inference == true`** 的實作（例如 Turing **s8s8f16** 系列）。同一檔案內：

```79:82:spconv_cpp/spconv/src/spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps/ConvGemmOps_implicit_gemm.cc
  float alpha = 1.0;
  if (tune_res.algo_desp.is_int8_inference){
      alpha = output_scale;
  }
```

（AWML plugin 因 epilogue 未乘 `alpha`，改為把 `output_scale` 融進 scale／bias 並傳 `output_scale=1.0f`；見 `cpp/int8_plugin/README.md`。）

### 3.2 cumm 程式碼產生：`is_int8_inference` 是 algo 描述子的一欄

cumm 用 Python 描述所有 conv kernel 變體，再產生 `ConvMainUnitTest_get_all_conv_algo_desp.cc` 等大表。  
在 `cumm/cumm/conv/main.py` 裡，dispatch 會依 **`algo_desp.is_int8_inference`** 等欄位選到**唯一**對應的 CUDA kernel：

```451:456:cumm/cumm/conv/main.py
                                    if_tests = [
                                        f"algo_desp.mask_sparse == {pccm.boolean(ms_ikf_mw[0])}",
                                        f"algo_desp.increment_k_first == {pccm.boolean(ms_ikf_mw[1])}",
                                        f"algo_desp.is_int8_inference == {pccm.boolean(ms_ikf_mw[2])}",
                                        f"algo_desp.dynamic_mask == {pccm.boolean(ms_ikf_mw[3])}",
                                    ]
```

**直覺**：**INT8 與 FP16 不是同一支 kernel**；是否走 INT8 路徑由 **tensor dtype + 已註冊的 algo 表**決定，不是「名叫 int8 但其實 FP16」。

---

## 4. spconv **官方** Python 路徑怎麼做 INT8？

### 4.1 文件主線

- `spconv/docs/INT8_GUIDE.md`：**FX、`prepare_fx`、QAT／PTQ**、量化 `SparseConv` 等。  
- `spconv/docs/TENSORRT_INT8_GUIDE.md`：**scale／bias 公式**、在 **custom plugin** 裡呼叫 `ConvGemmOps::implicit_gemm`（與 AWML sparse INT8 同一 C++ API）。

### 4.2 程式主線：`spconv/pytorch/ops.py` → `implicit_gemm`

當 **`features` 與 `filters` 皆為 quantized（`torch.qint8`）** 時，標記為 INT8 路徑：

```1474:1474:spconv/spconv/pytorch/ops.py
    is_int8 = features.is_quantized and filters.is_quantized
```

走 **C++ `ConvGemmOps.implicit_gemm`** 時，把 torch 張量轉成 `tv::Tensor`，dtype 對應 **int8／qint8**，與 AWML plugin 一樣進入 **同一套** `ConvGemmOps::implicit_gemm`。

Python 側若走較舊的 **纯 `CONV.run_with_tuned_result`** 分支，同樣用 **`tune_res.algo_desp.is_int8_inference`** 決定 `alpha`（`output_scale`）與 bias／residual 行為：

```1631:1640:spconv/spconv/pytorch/ops.py
    alpha = 1.0
    if tune_res.algo_desp.is_int8_inference:
        alpha = output_scale
    with timer.record("implicit_gemm", stream):
        for j in range(num_split):
            beta = 0 if j == 0 else 1
            if bias is not None and not tune_res.algo_desp.is_int8_inference:
                beta = 1
            if output_add is not None and tune_res.algo_desp.is_int8_inference:
                beta = output_add_scale / output_scale
```

**輸出**：INT8 推理時常用 **`torch._empty_affine_quantized`** 當輸出容器（帶 `scale`／`zero_point`），內部儲存仍與 **int8** 對齊；AWML TRT plugin 則直接 **FP16 輸出 buffer**，因為 TRT 邊界不採用 PyTorch 的 QTensor。

### 4.3 官方 scale 公式（與 AWML sparse INT8 一致）

見 `TENSORRT_INT8_GUIDE.md`：

- `scale_for_spconv_implicit_gemm = (input_scale * w_per_channel_scales) / output_scale`  
- `bias_for_spconv_implicit_gemm = bias / output_scale`  

AWML 的 `sparse_int8_onnx_transform` 寫入的 **`channel_scale`／`bias_scaled`** 與此同構（再依 plugin 內是否融 `output_scale` 微調傳入 `implicit_gemm` 的 `output_scale` 參數）。

---

## 5. AWML vs spconv 官方：對照表

| 維度 | spconv 文件／`ops.py` 典型路徑 | AWML |
|------|-------------------------------|------|
| **校準／刻度** | FX + observers、`qint8` 模組 | **NVIDIA `TensorQuantizer` + histogram／MSE**（`spconv_int8.py`） |
| **PyTorch forward** | 可為 **fake quant** 或 **qint8 + implicit_gemm** | 主線為 **fake quant**（不強制每層都跑 `qint8` implicit_gemm） |
| **TRT 稀疏卷積** | 文件建議 **plugin 內**呼叫 `implicit_gemm` | **sparse INT8**：`ImplicitGemmInt8` plugin → **`tv::int8` + `implicit_gemm`** |
| **底層 CUDA** | **cumm** 生成的 conv kernel，`is_int8_inference` | **相同**（連結同一 `spconv`／`cumm` 套件） |
| **輸出型別** | 常為 **affine quantized tensor** | **FP16 tensor**（TRT I/O） |
| **`output_scale` 參數** | 傳入 `implicit_gemm`，與 `alpha` 掛鉤 | 因 epilogue 行為，**融進 scale／bias 後傳 1.0f**（AWML 修復說明見 plugin README） |

---

## 6. 為什麼端到端加速可能只有一點點？

`spconv/docs/INT8_GUIDE.md` 說明：INT8 kernel 對 **通道數、shape** 有要求（例如 `input_channel % 32 == 0 && output_channel % 32 == 0`），且某些 **C、K** 組合才特別快。BEVFusion 稀疏塔前面常有 **5、16、32** 等寬度，**並非每一層都處於「滿速 INT8」形狀**。因此 **sparse encoder 總時間從 13 ms → 11 ms 這類小幅下降** 與「**有走 INT8 API，但沒有全圖都是最理想 INT8 shape**」可以並存。

若要驗證：**Nsight Compute** 看 kernel 名稱是否為 **s8s8f16** 等，比只看總時間更直接。

---

## 7. 參考路徑速查（利於自己對照原始碼）

| 說明 | 路徑（相對各 repo 根目錄） |
|------|---------------------------|
| AWML INT8 TRT plugin | `AWML/deployment/projects/bevfusion/cpp/int8_plugin/implicit_gemm_int8_plugin.cpp` |
| C++ `implicit_gemm` | `spconv_cpp/spconv/src/spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps/ConvGemmOps_implicit_gemm.cc` |
| spconv Python `implicit_gemm` | `spconv/spconv/pytorch/ops.py` |
| cumm conv dispatch（`is_int8_inference`） | `cumm/cumm/conv/main.py` |
| 官方 TRT 文件 | `spconv/docs/TENSORRT_INT8_GUIDE.md` |
| 官方 PyTorch INT8 文件 | `spconv/docs/INT8_GUIDE.md` |

---

*若你本地 `implicit_gemm_int8_plugin.cpp` 的 `supportsFormatCombination` 對 `filters` 要求 FP32，屬 ONNX／TRT 建圖與 initializer dtype 的工程選擇；**`enqueue` 內仍應以 int8 張量呼叫 `implicit_gemm` 才算走 INT8 GEMM。***
