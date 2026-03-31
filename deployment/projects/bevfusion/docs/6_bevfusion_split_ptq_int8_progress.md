# BEVFusion Split ONNX + PTQ INT8 — 進度、除錯紀錄與明日接續點

本文是 **AWML BEVFusion** 在 **路線 1（sparse / dense 分離 ONNX）+ spconv INT8 + 稠密端 pytorch_quantization** 路線上的 **工作筆記**：目標、已達成項目、改過的檔案、一路遇到的問題與解法，以及 **尚未完成** 的事項（PyTorch eval mAP≈0、TensorRT、Autoware plugin INT8）。

更廣的架構說明與業界對照仍見：

- [5_bevfusion_onnx_trt_spconv_int8.md](./5_bevfusion_onnx_trt_spconv_int8.md)
- [4_spconv_int8_implementation_history_zh.md](./4_spconv_int8_implementation_history_zh.md)

---

## 1. 我們要達成的目標（總覽）

1. **部署形態**：與 Lidar AI Solution 類似，將 BEVFusion **至少**拆成  
   - **稀疏段**：voxels / coors / num_points → BEV feature（`lidar_bev`）  
   - **稠密段**：`lidar_bev` → head 輸出（bbox / score / label 等）  
2. **量化**：稀疏塔走 **spconv FX PTQ INT8**；稠密塔走 **NVIDIA pytorch_quantization（QAT/PTQ 結構 + `_amax` 等）**。  
3. **產物**：能穩定產出 **`bevfusion_sparse.onnx`** 與 **`bevfusion_dense.onnx`**（以及後續對應 TensorRT engine）。  
4. **推理鏈**：最終希望 **AWML / Autoware** 端能用 **帶 INT8 能力的 TensorRT 與 plugin** 跑通；目前認知是 **既有 Autoware TensorRT plugin 對 INT8 支援不足**，需要 **Autoware 端修改後的 TensorRT / plugin** 才能滿足端到端 INT8。  
5. **精度**：PTQ 後在 **PyTorch evaluation** 上應有合理 mAP，再談 ONNX/TRT 對齊。
6. **與 Lidar AI Solution 靠攏**：**部署／libspconv** 路徑再談全幀；**FX PTQ 校準**在 GPU 上必須有 voxel 上限。ONNX／runtime 逐步移向 **稀疏 ONNX + libspconv / 官方 TRT 流程**。

---

## 2. 目前進度（截至本文件撰寫時）

| 項目 | 狀態 | 說明 |
|------|------|------|
| Split ONNX 匯出 | **可完成** | `deploy_config_split_int8.py` + CLI；稀疏段用 FP32 shadow 換圖匯出，稠密段獨立 trace。 |
| 單檔 / split + **FP32** checkpoint、關閉 quantization | **mAP 正常** | 用於證明 **split 管線、voxel、eval 設定** 本身沒壞。 |
| **PTQ .pth** + `quantization.enabled=True` 的 PyTorch eval | **mAP ≈ 0** | **主要未解問題**；指向 PTQ 品質、載入順序、sparse/dense 結構與 ckpt 不一致等。 |
| `bevfusion_dense.onnx` 含 **QDQ** | **已修正根因（程式）** | 見 §5.7；需在實際環境重新匯出驗證。 |
| TensorRT engine 建置 | **尚未完成** | 依賴 ONNX、plugin、精度策略與 **INT8-capable** 建置鏈。 |
| Autoware TensorRT plugin INT8 | **待對接** | 現有 plugin 不支援 INT8 需求時，需 **Autoware 變更後的 TRT/plugin**。 |

---

## 3. 設定與 CLI 入口（快速對照）

- **Split + INT8 deploy 設定**：`deployment/projects/bevfusion/config/deploy_config_split_int8.py`  
  - 內含 **Preset A（PTQ）** 與 **Preset B（FP32 + `quantization.enabled=False`）** 對照實驗說明。  
- **PTQ 產生 checkpoint**：`deployment/quantization/bevfusion_quantization.py`（`ptq` 子命令）。  
- **匯出 ONNX**：`python -m deployment.cli.main bevfusion ...`（見該 config 頂部 docstring）。  
- **環境**：需 **pytorch-quantization**；spconv 路徑常需 **`SPCONV_FX_TRACE_MODE=1`**（於 import spconv 前），細節見 `deploy_config_int8.py` 註解與實作歷程 README。

---

## 4. 已修改／新增重點檔案一覽（與用途）

以下為本專線上 **與 split + PTQ + 匯出／eval 直接相關** 的變更脈絡（非完整 `git diff`，以「明天接續」夠用為準）。

| 檔案 | 用途 |
|------|------|
| `config/deploy_config_split_int8.py` | Split 元件、`quantization`、`tensorrt_config.plugin_libraries`、隔離實驗說明、`spconv_ptq_basicblock_fx` 與 mmconfig 對齊提醒。 |
| `export/onnx_export_pipeline.py` | Split：先匯 sparse，再算 `lidar_bev` 匯 dense；sparse 匯出前可換 **FP32 shadow**；`setup_quantization_for_onnx_export()`；TopK 修正等。 |
| `export/sparse_encoder_float_shadow.py` | PTQ 後 `pts_middle_encoder` 為 GraphModule 時，匯出 ONNX 前暫時換成 **融合後的 FP32 BEVFusionSparseEncoder**，避免 `_empty_affine_quantized`；權重對齊、FX attribute 遺失時從 `model.cfg` 補、`Tensor.copy_` 經 `get_submodule` 避開 spconv hook。 |
| `io/model_loader.py` | PTQ 載入順序：**dense BN fuse + Q/DQ 插入 → 再還原 FX sparse**；`spconv_ptq_basicblock_fx`；sparse 權重 **ICOC↔KRSC** permute；missing/unexpected key 診斷（BN vs `bn1_scale_0`）；`verify_spconv_int8_encoder`；`setup_quantization_for_onnx_export`。 |
| `pipelines/pytorch.py` | Staged eval：neck 後呼叫 **`model._align_lidar_bev_to_head_grid(neck_out)`**，與 `extract_feat` 幾何一致，避免 head `bev_pos` 與 feature map 尺寸不符。 |
| `quantization/spconv_int8.py` | spconv INT8 管線（與 deploy 一致處）。 |
| `deployment/quantization/bevfusion_quantization.py` | PTQ 腳本側：依 `spconv_ptq_basicblock_fx` 呼叫 **`upgrade_pts_middle_encoder_basicblocks_to_fx`**，與 deploy 載入結構對齊。 |
| `deployment/quantization/modules/quant_conv.py` | **`_skip_fake_quant_for_export_trace`**：當 `TensorQuantizer.use_fb_fake_quant` 為 True 時 **不可**在 trace 中略過 quantizer，否則 dense ONNX 無 QDQ。 |

其他參考（repo 內）：`quantization/spconv_quantized_add_patch.py`（FX 圖內 `quantize_per_tensor` / `quantized_add` 與 spconv 相容）、`projects/BEVFusion/.../bevfusion.py` 的 `_align_lidar_bev_to_head_grid`。

---

## 5. 問題與除錯歷程（依主題）

### 5.1 「是 split 害 mAP 掉嗎？」

**作法**：同一 CLI、同一 `*_120m_fx.py`、仍用 split config，只把 checkpoint 換成 **訓練 FP32 .pth**，並設 **`quantization.enabled=False`**。  

**結果**：mAP 正常。  

**結論**：**split 布局、voxel 前處理、AWML eval 管線** 不是 mAP 崩潰主因；問題集中在 **PTQ checkpoint、quantization 載入、INT8 稀疏塔與稠密 QAT 結構是否與 ckpt 一致**。

### 5.2 稀疏 FX 校準：為何一定要裁 voxel（與 Lidar 的差別）

**事實**：`implicit_gemm` / `get_indice_pairs` 在 **N** 個 voxel 上可配置 **~O(N²)** 的 int32 buffer；**N≈9e4** 時可達 **~866 GiB** 量級，與 `--calibrate-samples 1` 無關。**Lidar AI Solution** 走 **libspconv**，與 **PyTorch FX + GPU** 上跑 `prepare_fx` 校準不是同一條記憶體模型。

**AWML 作法**：預設 **`spconv_calib_max_voxels`（及 env）預設 4096**；`bevfusion_quantization.py` 支援 **`--spconv-calib-max-voxels`**，且已修正 **`_load_deploy_quantization_cfg`**：用 **`deploy_cfg.get("quantization")`** 取量化 dict，避免 **`getattr(deploy_cfg, "quantization")`** 在 MMEngine `Config` 上讀不到鍵 → **`spconv_calib_max_voxels` 被忽略** → 日誌顯示無 cap → OOM。

**與 eval**：eval 仍吃整場景 voxel；mAP≈0 時應優先查 **權重／結構與 ckpt 對齊**。

### 5.3 PTQ sparse tower：**BN missing** vs **`bn1_scale_0` unexpected**

**現象**：`load_state_dict(strict=False)` 出現 `pts_middle_encoder...bn1.weight` **missing**，同時 **unexpected** 帶 `…_scale_0` / `…_zero_point_0` 等 FX observer 風格鍵。  

**原因**：**deploy 側還原的 GraphModule 結構** 與 **產生 PTQ .pth 時** 不一致——常見於 **`spconv_ptq_basicblock_fx`** 與 mmconfig（`*_base_120m.py` vs `*_120m_fx.py`）和 **舊版／新版 `bevfusion_quantization.py`** 是否做 **SparseBasicBlock→FX** 不一致。  

**解法方向**：  

- **新流程**：`*_120m_fx.py` + PTQ 用目前腳本 + deploy **`spconv_ptq_basicblock_fx=True`**。  
- **舊 ckpt**：若 PTQ 未走 FX block，deploy 設 **`spconv_ptq_basicblock_fx=False`** 與其對齊。  

`model_loader` 內已加 **明確錯誤訊息** 指引上述兩條路。

### 5.4 稀疏權重 layout：**ICOC vs KRSC**

**現象**：部分 PTQ checkpoint 稀疏 conv 權重為 **(C_in, C_out, K, K, K)**，FX 模型期望 **(C_out, K, K, K, C_in)**（KRSC）。  

**處理**：`model_loader._permute_sparse_encoder_weights_to_match_model` 在載入前 **in-place** 調整 `state_dict` 中對應 5D weight。

### 5.5 ONNX 匯出：`aten::_empty_affine_quantized`

**原因**：`convert_fx` 後稀疏塔為 **qint8 / 量化張量** 圖，標準 `torch.onnx.export` 不支援。  

**作法**：匯出 **僅在 trace 期間** 將 `pts_middle_encoder` 換成 **`sparse_encoder_float_shadow` 重建的 FP32 融合塔**，匯完 **還原** GraphModule。產生的 **`bevfusion_sparse.onnx` 是浮點稀疏圖**，與 **PyTorch 內真 INT8 稀疏推理** 數值 **不完全相同**；真 INT8 稀疏需 PyTorch、libspconv 載權重、或依 spconv `TENSORRT_INT8_GUIDE` 自建翻譯。

### 5.6 FX GraphModule **遺失** `sparse_shape` 等 attribute

**現象**：`convert_fx` / 圖優化後 root 上沒有 shadow 需要的欄位，無法建 FP32 encoder。  

**處理**：從 **轉換前 encoder** 或 **`model.cfg.model['pts_middle_encoder']`** 合併補齊（`copy_sparse_encoder_public_attrs`、`encoder_cfg_overrides_from_bevfusion_model`）。

### 5.7 Dense ONNX **沒有 QDQ 節點**

**根因**：`QuantConv2d` / `QuantConvTranspose2d` 在 **`torch.jit.is_tracing()` 或 `torch.onnx.is_in_onnx_export()`** 時 **略過** `_input_quantizer` / `_weight_quantizer`，導致圖中只剩普通 Conv。  

**與設計矛盾**：`setup_quantization_for_onnx_export()` 已設 **`TensorQuantizer.use_fb_fake_quant = True`**，目的正是讓 trace 產生 **QuantizeLinear/DequantizeLinear**。  

**修正**：在 `quant_conv.py` 的 `_skip_fake_quant_for_export_trace` 中，若 **`use_fb_fake_quant`** 為 True，**不要 skip**，讓 QDQ 進圖。  

**待驗證**：實際環境重新匯出 `bevfusion_dense.onnx`，用 Netron 或 `grep QuantizeLinear` 確認。

### 5.8 Eval staged pipeline：head **bev 尺寸** 與 `bev_pos` 不一致

**現象**：SECOND/FPN 輸出空間解析度與 bbox_head 預期的 **`grid_size // out_size_factor`** 不一致時，transformer decoder 內 **key / key_pos** 長度對不起來。  

**處理**：在 `pipelines/pytorch.py` 的 neck 之後呼叫 **`model._align_lidar_bev_to_head_grid`**，與正式 `extract_feat` 路徑對齊。

### 5.9 Shadow 權重寫入與 spconv `load_state_dict` hook

**處理**：透過 **`get_submodule` + `Tensor.copy_`** 等方式，避免不兼容的 state_dict 路徑與 hook 行為（見 `sparse_encoder_float_shadow.py` 實作與註解）。

### 5.10 Deploy 載入順序：先 dense Q/DQ，再 sparse FX replace

**原因**：若先換 spconv FX 再插 dense QAT，`quant_conv_module` 與 tracing 會與 PTQ ckpt 預期結構脫節，甚至觸發 fallback 後無法把 PTQ `state_dict` 載入純 FP32 模型。  

**實作**：`_load_with_quantization`（PTQ 分支）固定為 **fuse → dense 量化模組插入 → `_replace_encoder_with_fx_converted_structure` → load_state_dict**，並在密集量化前暫時 **disable spconv FX trace mode**（細節見 `model_loader` 註解）。

---

## 6. 尚未解決與明日建議優先順序

### 6.1 PyTorch evaluation mAP ≈ 0（PTQ checkpoint）

**可能方向（建議依序）**  

1. **日誌**：確認 `verify_spconv_int8_encoder` 是否 **通過**；檢查 **missing / unexpected keys** 是否為零或已可解釋。  
2. **一致性**：deploy 的 `*_fx.py` / `spconv_ptq_basicblock_fx` / PTQ 腳本版本 與 **產生 .pth 當下** 完全一致。  
3. **數值抽樣**：同一 sample 上比較 FP32 vs PTQ 的 **lidar_bev / heatmap** 是否全零或爆掉。  
4. **與 MMDet 官方 `predict` 對齊**：若需嚴格一致，可評估 eval 改走 **`model._forward` / `run_bevfusion`**（trade-off：失去細粒度 stage timing）。  

### 6.2 TensorRT 轉換

**依賴**：  

- `bevfusion_sparse.onnx` 內 **spconv / 自定義 op** 是否為目前 **trtexec + plugin** 所能支援；浮點稀疏 ONNX 與 **INT8 engine** 目標可能仍需 **專用 plugin 或 libspconv**。  
- `bevfusion_dense.onnx`：QDQ 修正後，再跑 **explicit INT8 / QDQ** 建置策略。  

### 6.3 Autoware TensorRT plugin 與 INT8

**現況認知**：現有 **libautoware_tensorrt_plugins** 等路徑對 **INT8 路徑** 支援不足；長期需 **Autoware 端修改後的 TensorRT 建置與 plugin** 與 AWML 產物對接。  

**建議**：在 Autoware workspace 追 **implicit_gemm / get_indices_pairs** 等 plugin 的 **I/O dtype、scale、plugin field** 是否宣告 INT8；與 spconv 官方 `TENSORRT_INT8_GUIDE`、MNIST 範例對照。

---

## 7. 相關外部／鄰近 repo 文件

- spconv：`docs/TENSORRT_INT8_GUIDE.md`、`docs/INT8_GUIDE.md`（於 `spconv` 倉庫）。  
- 使用者本機曾開啟之 Autoware plugin 原始碼（範例路徑）：  
  `autoware_tensorrt_plugins/src/implicit_gemm_plugin.cpp`、`get_indices_pairs_implicit_gemm_plugin.cpp`（實際路徑以各 workspace 為準）。

---

## 8. 一句話給明天開工

**ONNX 匯出已打通；接下來的主線是讓 PTQ 模型在 PyTorch eval **先恢復合理 mAP**（結構與 ckpt 嚴格對齊 +  encoder 驗證 + 可選與 MMDet 路徑對齊），再帶著 **含 QDQ 的 dense ONNX** 與 **Autoware 強化後的 TRT/plugin** 攻 TensorRT INT8。**
