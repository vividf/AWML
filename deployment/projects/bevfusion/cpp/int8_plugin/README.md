# ImplicitGemmInt8 TensorRT Plugin

本目錄實作稀疏卷積 Path B 的 **`ImplicitGemmInt8`** TensorRT plugin：在 `enqueue` 內將 FP16 feature／weight 量化為 INT8，呼叫 spconv 的 `ConvGemmOps::implicit_gemm`，並以 FP16 寫回輸出。

建置方式與整體 INT8 管線說明請見：

- `deployment/projects/bevfusion/docs/11_int8_pathb_autoware_plugin.md`
- `deployment/projects/bevfusion/docs/12_int8_sparse_pipeline_ptq_onnx_trt.md`

---

## TensorRT 輸出錯誤（mAP≈0、`lidar_bev` 異常放大）— 除錯與修復紀錄

### 現象

- **PyTorch（含 `pytorch_quantization` PTQ）**：推論與 mAP 正常（例如 Center Distance mAP 約 0.89）。
- **TensorRT（同一套 INT8 ONNX + 本 plugin）**：mAP 接近 **0**，中間特徵明顯異常。
- 具體觀察：**`sparse_encoder`／`lidar_bev` 等張量在 TRT 端的數量級遠大於 PyTorch**（例如某層 max 從約 5.8 變成數百），而 **voxel 前處理與 numpy 統計與 PyTorch 對齊**，可將問題收斂到 **稀疏 INT8 GEMM／epilogue**，而非影像或體素管線。

### 除錯路徑（摘要）

1. **環境**：Docker 映像若未含 `pytorch_quantization`，需先安裝（例如 NVIDIA PyPI 的 `pytorch-quantization`），否則 PTQ 權重無法載入、PyTorch 路徑也無法作為對照基準。
2. **對照 PyTorch vs TRT**：逐層或關鍵 tensor 比較數值範圍；確認 voxel 輸入一致後，異常集中在 **INT8 `ImplicitGemm` 之後**。
3. **閱讀 spconv／cumm 呼叫鏈**：
   - `ConvGemmOps::implicit_gemm`（`spconv_cpp/.../ConvGemmOps_implicit_gemm.cc`）在 **int8 inference** 時將 `alpha = output_scale` 傳入 tuned conv／GEMM。
   - 實際 Turing **`s8s8f16`** epilogue 在 `Int8Inference`（例如  
     `spconv_cpp/spconv/include/spconvlib/cumm/conv/main/.../out_op/Int8Inference.h`）：
     - **無 source** 路徑呼叫 `output_op(accumulator, bias, scale)`，實作為  
       `intermediate = scale * converted_accumulator + bias`（註解仍寫 `alpha = output scale`）。
     - **`alpha` 成員從未乘入上述運算**；`ApplyOutputOp.h` 亦只把 `bias`／`scale` fragment 傳入，沒有額外套用 `output_op.alpha`。
4. **與 ONNX／Python 匯出公式對齊**：
   - 匯出端使用（概念上）**除以 `output_scale`** 的 per-channel scale 與 bias（`channel_scale ∝ 1/output_scale`，`bias_scaled = bias / output_scale`），並假設最後會再乘上 **`output_scale`** 還原到 FP 域。
   - 若 kernel **少乘一次 `output_scale`**，則數值會近似 **真實線性輸出除以 `output_scale`**，也就是放大约 **`1 / output_scale`** 倍（與現場觀察的數十倍量級異常一致）。

### 根因（結論）

**cumm 的 `Int8Inference` epilogue 沒有把 `ConvGemmOps` 傳入的 `alpha`（即 `output_scale`）乘到結果上**；註解與上層 API 語意不一致。  
PyTorch／QTensor 路徑可能透過其他解量化路徑掩蓋差異，但 **本 plugin 直接寫 FP16 buffer**，缺少這一步就會整條 TRT 稀疏 backbone 錯位。

### 修復方式（plugin 端，不改 ONNX／Python 匯出）

在 **`enqueue` 內、權重量化完成且 `w_scales` 已不再被讀取之後**，於 GPU 上合成傳給 `implicit_gemm` 的 scale／bias：

- `gemm_channel_scale[c] = channel_scale[c] * output_scale`  
  （數學上等價於把「除以 `output_scale`」從匯出公式抵消，還原成 `input_scale * w_scale` 形式。）
- `gemm_bias[c] = bias_scaled[c] * output_scale`  
  （還原成未除以 `output_scale` 的浮點 bias。）

接著呼叫 `ConvGemmOps::implicit_gemm` 時改傳：

- `/*output_scale=*/ 1.0f`  
  避免上層再以為還需要乘一次 `alpha`（實際上 kernel 本來就沒乘）。

實作上：

- 使用 `quantize_features.cu` 內的  
  `launch_fuse_output_scale_into_gemm_scale_bias`（小 kernel，逐 channel 乘上 `output_scale`）。
- **Workspace**：在原有 `feat_int8`、`w_scales`、`weight_int8` 之外，多配置一塊 **`c_out` 個 float** 存放 `gemm_bias`；`w_scales` 緩衝在權重量化後被覆寫為 `gemm_channel_scale`（見 `implicit_gemm_int8_plugin.cpp` 的 layout 與 `getWorkspaceSize`）。

### 相關程式位置

| 項目 | 路徑 |
|------|------|
| Epilogue（`alpha` 未使用） | `spconv_cpp/.../Turing_s8s8f16.../out_op/Int8Inference.h` |
| ApplyOutputOp | `spconv_cpp/.../output/out_ns_apply/ApplyOutputOp.h` |
| `alpha = output_scale` | `spconv_cpp/.../ConvGemmOps_implicit_gemm.cc` |
| 修復：fuse kernel | `quantize_features.cu`／`quantize_features.cuh` |
| 修復：enqueue／workspace | `implicit_gemm_int8_plugin.cpp` |

### 驗證

重編 **`libimplicit_gemm_int8_plugin.so`** 後，以同一 `deploy_config_split_int8`（或等價 TRT 管線）跑 eval：**TRT 端中間特徵量級應與 PyTorch 對齊，mAP 應恢復正常**。

### 後續若擴充功能時的注意點

目前 plugin 對 **`output_add`（殘差）** 傳空 tensor。若未來在 INT8 路徑啟用殘差，`ConvGemmOps` 內對 int8 的 `beta` 會用到 `output_add_scale / output_scale`；在 **`output_scale` 固定為 `1.0f`** 的情況下，需重新檢查殘差分支的 scale 語意是否仍與匯出端一致。
