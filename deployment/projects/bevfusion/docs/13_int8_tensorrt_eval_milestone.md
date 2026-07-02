# sparse INT8（ImplicitGemmInt8）TensorRT 評測里程碑（2026-04）

本文記錄一次端到端修復後，**稀疏塔 INT8（sparse INT8）+ TensorRT** 在評測腳本上首次出現**合理預測量與非零 mAP** 的狀態，便於之後對照「split 雙引擎」等路徑的落差。

---

## 0. 本次改動的重要性（前後對照）

同一套修正（**權重 `axis=(0)` + PTQ／transform／export 使用同一份 checkpoint**）帶來兩條可觀察的訊號：

| 路徑 | 改動前（約略） | 改動後（同批 5 frames、±121 m 等設定） | 意義 |
|------|----------------|----------------------------------------|------|
| **第一段：PyTorch（含 fake quant / 與部署對齊的推理路徑）** | BEV **mAP ≈ 0.35** | BEV **mAP ≈ 0.37**（例：0.3718） | 刻度與權重語意對齊後，**精度小幅上升**，代表稀疏量化不再用錯 per-channel 軸或錯檔。 |
| **第二段：TensorRT（split：稀疏 INT8 engine + 稠密 engine）** | **mAP 仍為 0**，且 **Predict_num ≈ 0**（幾乎沒有有效框） | **mAP 仍為 0**，但各類 **Predict_num 合計約四百多級**（例：car 486、truck 44、pedestrian 42 等） | 指標仍差代表 **雙引擎銜接／座標或閾值** 可能仍有問題；但 **從「完全沒預測」到「有數百個預測」** 說明稀疏塔輸出已不再是全壞／全空，**這次改動對 TRT 路徑同樣關鍵**。 |

整體結論：**這次改動非常重要**——不只抬升 PyTorch 端 mAP，也讓 TensorRT split 路徑從「零預測」變成「有量級正確的預測筆數」，後續才能專心查 mAP=0 的後段因素。

---

## 1. 摘要

- **先前問題**：`sparse_int8_onnx_transform` 若讀到**舊 PTQ**（權重 `_amax` 沿 **C_in / axis=4** 存的 `(1,1,1,1,5)` 等 legacy 形狀），與 sparse INT8 需要的 **每輸出通道 C_out** 刻度不一致，會直接 `ValueError`；若曾用錯 checkpoint，TRT 端也會出現 **mAP≈0、預測異常**。
- **本次關鍵**：
  1. **稀疏權重量化軸**：`apply_nvidia_spconv_int8` 對 5D 權重使用 `QuantDescriptor(..., axis=(0))`，使 `_weight_quantizer._amax` 與 **`[C_out, k, k, k, C_in]`** 的語意一致（見 [`9_nvidia_spconv_int8_fix.md`](./9_nvidia_spconv_int8_fix.md)）。
  2. **產物對齊**：PTQ 產出的 `.pth`、ONNX 匯出載入的 checkpoint、以及 `sparse_int8_onnx_transform --checkpoint` **必須為同一次 PTQ 檔案**（例如皆指向 `*_ptq_sparse_only2.pth`，勿混用少一個後綴的舊檔）。轉換腳本載入後會印 **`[sanity] conv_input.0 weight_amax shape=...`**：sparse INT8 預期形如 `(16,1,1,1,1)`，若為 `(1,1,1,1,5)` 代表仍為舊 checkpoint。
- **結果（同一批 5 frames、距離範圍約 ±121 m）**：
  - **整體 BEV（Center Distance）**：**mAP ≈ 0.372、mAPH ≈ 0.303**；各類別有正常 **Predict_num**，例如 car 678 / truck 762 / pedestrian 683（相對 GT 數量合理數量級）。
  - **Plane Distance 摘要**：mAP ≈ **0.411**、mAPH ≈ **0.342**。
  - TensorRT 建圖時 **ImplicitGemmInt8** 節點依序打印 `input_scale` / `output_scale`，最後一層 `encoder_layer4.1.conv2` 的 `output_scale=1.000000` 與鏈式 dequant 設計一致。
  - **延遲（範例一次跑分）**：端到端 mean ≈ **216 ms**；其中 **Sparse Encoder（sparse INT8）** ≈ **15.95 ms**，Backbone / Neck / Head 等在 **FP16 稠密引擎**側仍佔剩餘時間。

---

## 2. 對照：另一條評測路徑仍 mAP = 0

同一份日誌後段出現**第二次**評測摘要（延遲結構含 **「Dense Engine」**、`Bevfusion` 總時間較短、Sparse Encoder 約 **6 ms**）：

- **mAP / mAPH 全為 0**；car / truck / pedestrian 的 **Predict_num** 與前段不同（例如 car 486、truck 44）。
- **解讀（工作假設）**：此段對應 **split 部署**（稀疏 ONNX+INT8 plugin 與稠密子圖分開建 engine）或 **不同的 tensor 銜接／前處理路徑**；數值上代表 **sparse INT8 稀疏塔本身已能對齊單一路徑下的 PyTorch 語意**，但 **雙引擎或 I/O 綁定** 仍有待與 [`6_bevfusion_split_ptq_int8_progress.md`](./6_bevfusion_split_ptq_int8_progress.md)、[`10_int8_trt_gap_analysis.md`](./10_int8_trt_gap_analysis.md) 對照排查。

建議後續在文件中固定標註評測模式（例如 `unified` vs `split_sparse_dense`），避免兩段數字混讀。

---

## 3. 日誌中仍可忽略的警告

| 現象 | 說明 |
|------|------|
| `Spconv INT8 encoder verification failed`（`model_loader.py`） | 目前驗證邏輯仍以 **FX GraphModule / 特定參數名** 為主；**NVIDIA TensorQuantizer 路徑**可能被判為 `quantized_params=0`，屬**啟發式誤報**，不表示 `_amax` 未載入。仍以 checkpoint `missing=0`、以及 TRT 端輸出統計為準。 |
| `Disable HistogramCalibrator` / `MaxCalibrator` | 推論階段關閉校準器屬預期行為。 |
| `Make sure output label_pred has Int64 binding` | TensorRT 常見 dtype 提示，與本次 mAP 修復無直接矛盾。 |
| spconv `UserWarning`（非 tuple 索引） | PyTorch 2.9 行為預告，與量化正確性無關。 |

---

## 4. 與相關文件的關係

| 文件 | 角色 |
|------|------|
| [`11_int8_autoware_plugin.md`](./11_int8_autoware_plugin.md) | sparse INT8 plugin、ONNX 節點約定 |
| [`12_int8_sparse_pipeline_ptq_onnx_trt.md`](./12_int8_sparse_pipeline_ptq_onnx_trt.md) | PTQ → ONNX → TRT 全流程 |
| [`9_nvidia_spconv_int8_fix.md`](./9_nvidia_spconv_int8_fix.md) | 稀疏 NVIDIA 校準與權重軸修正 |

---

## 5. 可複現檢查清單（精簡）

1. PTQ：`--sparse-int8-only`（或完整 PTQ）產出 `.pth`，確認 log 中 **`_weight_quantizer` 的 amax shape** 為 `(C_out,1,1,1,1)` 這類形狀，而非 `(1,1,1,1,C_in)`。
2. 匯出稀疏 ONNX →（可選）改名 `*_fp16.onnx` → `sparse_int8_onnx_transform` 的 **`--checkpoint` 與步驟 1 為同一檔**。
3. 建 TRT engine 時載入 **ImplicitGemmInt8** 對應的 `plugin_libraries`。
4. 評測時確認與 baseline 使用**同一前處理、同一座標範圍與同一「unified / split」模式**，再比較 mAP。

此文件僅記錄**已觀測到的里程碑數字與解讀**；正式 baseline 請以團隊約定的 config、checkpoint 與完整 val set 為準。
