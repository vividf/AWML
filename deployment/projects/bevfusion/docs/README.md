# BEVFusion deployment notes

Numbered filenames follow **creation time** (filesystem birth time, with mtime then name as tie-break): **smaller number = older document**, **larger = newer** (latest context is toward the end of the list).

## PTQ → ONNX → TensorRT（稀疏 INT8 管線速覽）

1. **PTQ**（`deployment/quantization/bevfusion/quantization/quantize.py ptq`）：Fuse BN →（可選）稠密端插 Q/DQ → **稀疏塔**用 `apply_nvidia_spconv_int8` + `calibrate_spconv_nvidia` 寫入各層 `_input_quantizer._amax` / `_weight_quantizer._amax` →（可選）稠密 `CalibrationManager` → 存 `.pth`。
2. **Deploy 載入**（`io/model_loader.py`）：對 PTQ checkpoint 再掛 NVIDIA quantizer 結構並 `load_state_dict`，還原稀疏刻度。
3. **ONNX**（`export/onnx_export_pipeline.py`）：`torch.onnx.export`；**FP32 shadow** 在 (a) FX `GraphModule` 或 (b) **NVIDIA TensorQuantizer** 稀疏塔（scheme A，`encoder_has_nvidia_tensor_quantizers` + shadow 屬性／cfg 可補）時暫換純浮點 encoder，稀疏圖通常 **無 Q/DQ**。稀疏卷積節點為 **`autoware::ImplicitGemm`**（浮點 I/O）。
4. **sparse INT8（可選）**（`export/sparse_int8_onnx_transform.py`）：把 `ImplicitGemm` 改成 **`ImplicitGemmInt8`** 並寫入 scale，供 `libimplicit_gemm_int8_plugin.so` 在 TRT 內跑 **cumm INT8 `implicit_gemm`**。
5. **TensorRT**（`export/tensorrt_export_pipeline.py`）：對 `components` 裡每個 ONNX 建 engine，`tensorrt_config.plugin_libraries` 載入 Autoware +（sparse INT8）INT8 plugin。

**「INT8 sparse conv」**：PyTorch 端是 **TensorQuantizer + `_amax`（fake quant）**；預設 TRT 上仍是 **FP16 ImplicitGemm** 除非走 sparse INT8。完整步驟、表格與 mermaid 圖見 **[`12_int8_sparse_pipeline_ptq_onnx_trt.md`](./12_int8_sparse_pipeline_ptq_onnx_trt.md)**。

**端到端總覽（PTQ → INT8 spconv → 部署，並對照 spconv / CUDA-BEVFusion）**：**[`README_PTQ_INT8_SPCONV_DEPLOYMENT.md`](./README_PTQ_INT8_SPCONV_DEPLOYMENT.md)**。

**INT8 到底在哪裡、怎麼做（AWML vs spconv／cumm 對照）**：**[`README_INT8_WHERE_AND_HOW.md`](./README_INT8_WHERE_AND_HOW.md)**。

| # | File | Topic |
|---|------|--------|
| 1 | [`1_spconv_int8.md`](./1_spconv_int8.md) | spconv INT8 / libspconv alignment notes |
| 2 | [`2_spconv_int8_deploy.md`](./2_spconv_int8_deploy.md) | Recommended deploy flow for spconv INT8 |
| 3 | [`3_int8_implementation.md`](./3_int8_implementation.md) | Commands, config, Docker, error codes |
| 4 | [`4_spconv_int8_implementation_history_zh.md`](./4_spconv_int8_implementation_history_zh.md) | Implementation history (ZH), pitfalls |
| 5 | [`5_bevfusion_onnx_trt_spconv_int8.md`](./5_bevfusion_onnx_trt_spconv_int8.md) | ONNX / TensorRT / spconv INT8 overview |
| 6 | [`6_bevfusion_split_ptq_int8_progress.md`](./6_bevfusion_split_ptq_int8_progress.md) | Split ONNX + PTQ INT8 progress log |
| 7 | [`7_bevfusion_int8_eval_fixes_and_progress.md`](./7_bevfusion_int8_eval_fixes_and_progress.md) | INT8 eval 修復總覽、mAP≈0 根因、程式更動與進度表 |
| 8 | [`8_int8_fixes_summary.md`](./8_int8_fixes_summary.md) | INT8 PTQ 完整修復總覽：根因、程式變更、移除的 workaround、診斷工具 |
| **9** | [**`9_nvidia_spconv_int8_fix.md`**](./9_nvidia_spconv_int8_fix.md) | **NVIDIA approach fix: FX→TensorQuantizer, histogram+MSE, mAP 0→0.36** |
| **10** | [**`10_int8_trt_gap_analysis.md`**](./10_int8_trt_gap_analysis.md) | **INT8 TRT deployment gap analysis: why no speedup, libspconv Path A implementation** |
| **11** | [**`11_int8_autoware_plugin.md`**](./11_int8_autoware_plugin.md) | **sparse INT8: open-source INT8 ImplicitGemm plugin using cumm kernels（最新）** |
| **12** | [**`12_int8_sparse_pipeline_ptq_onnx_trt.md`**](./12_int8_sparse_pipeline_ptq_onnx_trt.md) | **PTQ → ONNX → TRT 內部流程；稀疏 INT8 在各層如何成立** |
| **13** | [**`13_int8_tensorrt_eval_milestone.md`**](./13_int8_tensorrt_eval_milestone.md) | **里程碑：PyTorch BEV mAP 0.35→0.37；TRT split mAP 仍 0 但 Predict_num 0→400+（改動極關鍵）** |
| **14** | [**`14_trt_split_map_zero_debug.md`**](./14_trt_split_map_zero_debug.md) | **Split TRT 有預測但 mAP=0：管線對照、原因歸納、`BEVFUSION_TRT_*` / `DEBUG_POSTPROCESS` 除錯** |
| **18** | [**`18_SPARSE_PROFILE_INT8_VS_FP16_COMPARISON.md`**](./18_SPARSE_PROFILE_INT8_VS_FP16_COMPARISON.md) | **Priority A：同一輸入下 INT8 vs FP16 sparse engine 對照；為何 ImplicitGemm 時間相近；Nsight 下沉拆 kernel** |
| **21** | [**`21_README_IMPLICITGEMM_FP_TRT_PLUGIN_ISSUES.md`**](./21_README_IMPLICITGEMM_FP_TRT_PLUGIN_ISSUES.md) | **FP `ImplicitGemm` TRT 外掛：`num_inputs` 5/6、fork assert、bias FP32 vs FP16（tensorview／spconv）、timing 警告與處置** |
| **25** | [**`25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md`**](./25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md) | **`coors` 契約對齊 Autoware：為何舊/新 ONNX 都能在 evaluation 正常、與 ROS `x/y/z` 的關聯與邊界** |
| **26** | [**`26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md`**](./26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md) | **`ScatterND -> SECOND` 為何 `original` 與 split-merge ONNX 節點不一樣、分開 trace 為什麼少 shape-plumbing、對數值的實際影響** |
| — | [**`README_PTQ_INT8_SPCONV_DEPLOYMENT.md`**](./README_PTQ_INT8_SPCONV_DEPLOYMENT.md) | **PTQ → ONNX → TRT 全流程詳解；與 spconv 官方、CUDA-BEVFusion（libspconv）對照** |
| — | [**`README_INT8_WHERE_AND_HOW.md`**](./README_INT8_WHERE_AND_HOW.md) | **哪裡 FP16／哪裡真 INT8；sparse INT8 與 `ConvGemmOps::implicit_gemm`、`is_int8_inference`、spconv `ops.py` 對照** |

Python entrypoints, configs, and pipelines live in the parent directory (`deployment/projects/bevfusion/`), alongside this `docs/` folder.
