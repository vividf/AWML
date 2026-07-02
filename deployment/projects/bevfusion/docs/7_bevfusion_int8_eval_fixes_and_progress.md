# BEVFusion INT8 PTQ／部署評估修復總覽與進度

本文彙整 **INT8 PTQ + 部署 PyTorch eval** 相關的問題分析、程式更動與**目前進度**，作為與 [6_bevfusion_split_ptq_int8_progress.md](./6_bevfusion_split_ptq_int8_progress.md) 並行的「修復說明」文件。

---

## 1. 背景問題（時間序）

### 1.1 稠密校準：`Quantize only works on Float Tensor, got Int`

- **現象**：`[5/6]` 對 dense Q/DQ 做 MSE 校準時，forward 在 `pts_middle_encoder` 的 FX 圖內呼叫 `torch.quantize_per_tensor`，輸入為整數 voxel／activation，PyTorch 拒絕量化。
- **根因**：
  - FX `GraphModule` 在 `convert_fx` 時**綁死**當下的 `spconv` `quantize_per_tensor` 呼叫鏈；事後只 patch `spconv.pytorch.quantization.core.quantize_per_tensor` **無法**改變圖裡已 closure 的 callable。
  - 圖內實際仍會走到 **`torch.quantize_per_tensor`**，需在 **`torch` 層**對第一個 tensor 參數做「非 float → `.float()`」的守門（且需支援 `*args, **kwargs` 以相容不同 PyTorch 簽名）。
- **相關實作**：`install_spconv_quantize_per_tensor_float_input_guard()`（見 §3）。

### 1.2 FP32／FP16 eval 正常、INT8 eval mAP ≈ 0

- **對照**：
  - `deploy_config_split.py` + `*_base_120m.py` + FP32 checkpoint → Center Distance BEV mAP 合理（例如約 0.40）。
  - `deploy_config_split_int8.py` + `*_base_120m_fx.py` + PTQ `.pth` → mAP 常為 **0**，預測數量與 GT 對不上（例如 car 大量 pred、truck 幾乎為 0）。
- **已排除／需注意**：
  - 兩條路線使用的 **model config**（`basicblock` vs `basicblock_fx`）與 **checkpoint** 不同，比較時需用 **Preset B**（同 split_int8、`*_fx.py`、FP32 ckpt、`quantization.enabled=False`）確認 **FX 圖本身**是否仍能還原 mAP。
- **根因（主線）**：`_conv_out_to_bev` 出現 **`dense Z=41`、channel 5248 vs 期望 256** 類警告 → **最後一層 sparse `conv_out`（kernel (1,1,3), stride (1,1,2)）在 INT8 路徑下未正確把深度維壓到約 2**，`spatial_shape[-1]` 仍接近 **grid Z（41）**。後續用 **分段 mean** 把 41→2 **不等於**訓練好的加權 `conv_out`，BEV 語意錯誤 → 框飄移 → Center Distance mAP 崩潰。
- **次要因素**：PTQ `load_state_dict` 後新插入的 **`TensorQuantizer`** 若仍停留在「校準預設」（例如 fake quant 關閉），與 `CalibrationManager.calibrate()` 結束後狀態不一致，稠密端行為偏離預期。

### 1.3 日誌中的其他訊息（通常可與根因分開看）

| 訊息 | 說明 |
|------|------|
| `torch/ao/quantization/observer.py`：`must run observer before calculate_qparams` | 多為 FX／tracer 與 `torch.ao` 路徑交互；與 NVIDIA `pytorch_quantization` 不一定同一套 observer。 |
| `Disable HistogramCalibrator` / `Disable MaxCalibrator` | 呼叫 `disable_calib()` 後的預期 log。 |
| `TracerWarning`（shape 轉 Python int） | FX／trace 常見；需與實際數值錯誤區分。 |
| `_conv_out_to_bev: ... Z=41 ... collapsing` | **高優先**：代表 §1.2 的 Z／BEV 路徑異常（應以 `conv_out` 修復或 FP32 fallback 處理）。 |

---

## 2. 目前進度總表（2026-03-31 起算之修復後狀態）

| 項目 | 狀態 | 說明 |
|------|------|------|
| PTQ 腳本跑完（含 sparse 再 dense 校準順序） | **可完成** | `bevfusion/quantization/quantize.py` 已強調先 INT8 sparse 再 dense，避免 amax 對錯 BEV 分佈。 |
| 稠密校準 `got Int` | **已程式修復** | `torch.quantize_per_tensor` guard + voxel／forward 路徑 float 化（見 §3）。 |
| `deploy_config_split` + FP32 + `*_base_120m.py` eval | **正常** | 基準線。 |
| `deploy_config_split_int8` + PTQ + `*_fx.py` eval mAP | **待你方重跑確認** | 已加 **TensorQuantizer 推論模式** + **FP32 `conv_out` fallback**；若 Z 錯誤被修正，預期 **不應再長期 mAP=0**（數值仍可能低於 FP32）。 |
| ONNX／TensorRT 全 INT8 稀疏塔 | **未於本文件範圍保證** | 稀疏段 ONNX 仍多用 FP32 shadow；真 INT8 稀疏推理見其他 doc。 |

---

## 3. 程式更動一覽（檔案與職責）

### 3.1 `deployment/projects/bevfusion/quantization/spconv_int8.py`

- **`install_spconv_quantize_per_tensor_float_input_guard()`**
  - **主要**：patch **`torch.quantize_per_tensor`**，以 `*args, **kwargs` 轉發；若第一個參數為非量化、非浮點 `Tensor`，先 `.float()` 再呼叫原始實作。
  - **原因**：FX 圖內 closure 的是 spconv 包一層後仍呼叫 `torch.quantize_per_tensor` 的路徑；只改 `spconv_qcore` 不足以覆蓋圖內舊 reference。
  - **輔助**：仍 patch `spconv.pytorch.quantization.core.quantize_per_tensor`，供動態查找路徑使用。
- **`calibrate_spconv_model`**：先前對話中曾為稀疏校準 loop 加上 **tqdm**（若目前檔案中保留，便於觀察進度）。

### 3.2 `deployment/quantization/bevfusion/quantization/quantize.py`（脈絡）

- **`install_spconv_quantize_per_tensor_float_input_guard()`** 在 PTQ `run_ptq` 載入模型後呼叫。
- **`_calibrate_dense`**：`import torch`、`_force_float_voxel_inputs`、自訂 `forward_fn` 等，避免校準 batch 整數 voxel 觸發 quantize 錯誤。

### 3.3 `deployment/quantization/calibration/calibrator.py`

- 校準失敗時印出 **第一批次的完整 traceback**，便於定位。

### 3.4 `projects/BEVFusion/bevfusion/bevfusion.py`（脈絡）

- voxel／`extract_pts_feat`：**FP32 voxel**、`batch_size` 用 Python int、`_ensure_float_lidar_bev`／dequantize、`register_pts_middle_encoder_float_input_hook` 等，與 INT8 稀疏塔銜接稠密 Q/DQ 相容。

### 3.5 `deployment/projects/bevfusion/io/model_loader.py`

- **`_set_tensor_quantizers_inference_mode(model)`**（新增）
  - 在 PTQ **`load_state_dict` 之後**對所有 `TensorQuantizer` 執行與 `CalibrationManager._disable_calibration_mode` 相同邏輯：**`enable_quant()` + `disable_calib()`**（無 calibrator 則 `enable()`）。
  - **目的**：部署時 `quant_conv_module` 新插入的 quantizer 預設偏校準态，與 PTQ 存檔當下不一致時可能導致 eval 行為錯誤。
- **`attach_fp32_conv_out_fallback_for_int8_graph`**（呼叫，見 §3.6）
  - 在 `spconv_int8` 路徑、`retarget_graphmodule_*` 之後嘗試掛 hook；成功時 log **Attached FP32 conv_out fallback…**。

### 3.6 `deployment/projects/bevfusion/export/sparse_encoder_float_shadow.py`

- **`_expected_conv_out_last_spatial_dim(z_in, ...)`**  
  - 對應 `BEVFusionSparseEncoder.conv_out`：`kernel_z=3`, `stride_z=2`, `padding_z=0` → `max((z_in + 2*p - k) // s + 1, 1)`。
- **`attach_fp32_conv_out_fallback_for_int8_graph(model, device)`**（新增）
  - 用 **`build_float_sparse_encoder_shadow`** 從 GraphModule state_dict **dequant／對齊**權重，得到 **FP32 `conv_out`**。
  - 在 GraphModule 的 **`conv_out`** 上註冊 pre/post hook：若輸出 **`spatial_shape[-1]`** 與上式預期不符，改以 **FP32 `conv_out(x_in)`** 取代 INT8 輸出。
  - **權衡**：最後一層 sparse conv 在觸發 fallback 時為 **FP32 多算一步**；換取 BEV 幾何與訓練一致。

### 3.7 其他呼叫點

- **`deployment/projects/bevfusion/runner.py`**：`install_spconv_quantize_per_tensor_float_input_guard()`（非 PTQ 路徑替換 encoder 時）。
- **`model_loader._replace_encoder_with_fx_converted_structure`**：內含 guard + `register_pts_middle_encoder_float_input_hook`。

---

## 4. 建議驗證步驟（給使用／CI）

1. **重跑 INT8 eval**（與你先前相同 CLI）：`deploy_config_split_int8.py` + `*_120m_fx.py` + PTQ `.pth`。
2. **看 log**：
   - 是否出現 **`INT8 conv_out Z mismatch… Replacing with FP32 conv_out`**（表示 fallback 有介入）。
   - **`_conv_out_to_bev: Z=41`** 是否消失或減少。
3. **對照 Preset B**：`split_int8` + `*_fx.py` + FP32 ckpt + `quantization.enabled=False` → mAP 應與 split FP 路線同級（證明 FX config 與資料管線）。
4. **仍 mAP≈0 時**：檢查 PTQ load 的 **missing / unexpected keys**、`spconv_ptq_basicblock_fx` 與產生 PTQ 時的 config 是否一致。

---

## 5. 相關文件

| 文件 | 內容 |
|------|------|
| [6_bevfusion_split_ptq_int8_progress.md](./6_bevfusion_split_ptq_int8_progress.md) | Split ONNX + PTQ 長期進度、除錯紀錄 |
| [5_bevfusion_onnx_trt_spconv_int8.md](./5_bevfusion_onnx_trt_spconv_int8.md) | ONNX／TRT／spconv INT8 架構 |
| [3_int8_implementation.md](./3_int8_implementation.md) | 指令、Docker、錯誤代碼 |

---

## 6. 變更摘要（給 code review）

- **INT8 校準／推理**：`torch.quantize_per_tensor` 浮點守門 +（既有）spconv／BEVFusion voxel float 化，解決 `got Int`。
- **PTQ 部署載入**：`TensorQuantizer` 強制進入與校準完成後一致的推論模式。
- **mAP≈0（Z=41）**：以 **FP32 `conv_out` hook fallback** 繞過 spconv INT8 `conv_out` 的 Z 維 `spatial_shape`／下採樣異常，避免錯誤依賴 `_conv_out_to_bev` 的 mean collapse。

若你方重跑後 mAP 已恢復，建議在本文件 **§2 進度表** 自行更新一格「INT8 PyTorch eval：**已驗證 mAP ≈ …**」，並在 [6](./6_bevfusion_split_ptq_int8_progress.md) 中簡註日期與數字，方便之後接手者對齊。
