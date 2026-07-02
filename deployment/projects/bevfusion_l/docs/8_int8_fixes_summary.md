# BEVFusion INT8 PTQ 修復總覽（2026-04）

本文件完整記錄從 **mAP ≈ 0** 到目前為止對 BEVFusion INT8 PTQ pipeline 所做的所有修復，包含根因分析、程式碼變更、已移除的 workaround，以及目前正在進行的 debug 診斷。

---

## 1. 問題時間線

| # | 問題 | 狀態 |
|---|------|------|
| 1 | INT8 PTQ eval mAP = 0.0000（所有類別） | **已修復** — 有預測輸出 |
| 2 | `RuntimeError: expected 256 channels, got 5248` | **已修復** |
| 3 | `AssertionError: only support int32` | **已修復** |
| 4 | PTQ 校準 GPU OOM（需 `spconv_calib_max_voxels` 裁切 voxel） | **已修復** — 不再需要裁切 |
| 5 | mAP 仍然很低（≈ 0.0002） | **進行中** — 已加入診斷 |

---

## 2. 根因分析與修復

### 2.1 修復 1：spatial_shape 未正確計算（5248 channels 錯誤）

**檔案**：`projects/SparseConvolution/sparse_conv.py`

**根因**：`SparseConvolution._conv_forward` 中，當 `_fx_tracing=True` 時有一段條件邏輯會**跳過**非 submanifold 卷積的 `out_spatial_shape` 計算，直接設為 `out_spatial_shape = spatial_shape`（不做 stride 降維）。這導致 FX trace 出的 GraphModule 中 `conv_out`（stride=(1,1,2)）不會把深度維 41 降為 ~20，最終 `_conv_out_to_bev` 產出 `[1, 5248, 1440, 1440]` 而非 `[1, 256, 180, 180]`。

**修正**：移除 `_fx_tracing` 條件分支，改為**始終**計算 `out_spatial_shape`：

```python
# 修正後（適用所有 tracing mode）
if not self.subm:
    if self.transposed:
        out_spatial_shape = ops.get_deconv_output_size(...)
    else:
        out_spatial_shape = ops.get_conv_output_size(...)
else:
    out_spatial_shape = spatial_shape
```

`spatial_shape` 來自 `SparseConvTensor`，始終是 concrete Python list（不是 FX Proxy），因此 `get_conv_output_size` 在任何 tracing mode 下都能正確執行。

### 2.2 修復 2：自定義 SparseConv 類型註冊為 non_traceable

**檔案**：`deployment/projects/bevfusion_l/quantization/spconv_int8.py` — `apply_spconv_int8_quantization()`

**根因**：專案使用的 `projects/SparseConvolution/sparse_conv.py` 中的 `SparseConv3d`、`SubMConv3d`、`SparseConvolution` 不在 spconv 內建的 `DEFAULT_SPARSE_CONV_TYPES` 中。FX `prepare_fx` 會 trace **進入**這些模組的 `_conv_forward`，導致：
1. spatial_shape 計算錯誤（修復 1 的觸發條件）
2. Observer 被插入到每個 `_conv_forward` 的中間 tensor 上，校準時產生 **O(N²) GPU 記憶體**消耗 → OOM

**修正**：在 `prepare_fx` 之前，將自定義類型加入 `prepare_custom_config.non_traceable_module_classes`：

```python
from projects.SparseConvolution.sparse_conv import SparseConv3d, SubMConv3d, SparseConvolution

for cls in (SparseConv3d, SubMConv3d, SparseConvolution):
    if cls not in prepare_custom_config.non_traceable_module_classes:
        prepare_custom_config.non_traceable_module_classes.append(cls)
```

**副作用**：這同時解決了 OOM 問題——FX 現在只在 module 邊界放 observer，不深入 `_conv_forward` 內部，記憶體消耗與正常推理相當。

### 2.3 修復 3：保持 SPCONV_FX_TRACE_MODE=True

**檔案**：`deployment/projects/bevfusion_l/io/model_loader.py`、`deployment/projects/bevfusion_l/quantization/spconv_int8.py`

**根因**：FX GraphModule 在 `SPCONV_FX_TRACE_MODE=True` 下 trace 完成，圖內的 dequantize 操作會產生 `float32` 的 indices。如果在推理時關閉此 flag，`SparseConvTensor.__init__` 會嚴格檢查 indices 必須為 `int32` → `AssertionError: only support int32`。

**修正**：移除了之前在 `convert_spconv_int8()` 和 `model_loader.py` 中呼叫 `_disable_spconv_fx_trace_mode()` 的程式碼。FX GraphModule 的推理環境必須與 trace 環境一致。

### 2.4 修復 4：移除 voxel cap（OOM workaround）

**已刪除的程式碼**：

| 檔案 | 移除內容 |
|------|----------|
| `spconv_int8.py` | `_DEFAULT_SPCONV_CALIB_MAX_VOXELS`、`_default_spconv_calib_max_voxels()`、`cap_voxels_for_spconv_calibration()`、`calibrate_spconv_model` 的 `max_voxels_per_sample` 參數、`import os` |
| `bevfusion/quantization/quantize.py` | `_resolve_spconv_calib_voxel_cap()`、`--spconv-calib-max-voxels` CLI 參數、`spconv_calib_max_voxels_cli` 參數 |
| `deploy_config_split_int8.py` | `spconv_calib_max_voxels` 設定（Preset A 與 Preset C） |
| `deploy_config_int8.py` | `spconv_calib_max_voxels=4096` |
| `runner.py` | `max_vox = quantization.get("spconv_calib_max_voxels")` 及 `max_voxels_per_sample=max_vox` |

**為何 OOM 消失**：修復 2 使 FX 將自定義 SparseConv 視為 opaque leaf module → observer 只在 module input/output → 校準時記憶體 ≈ 正常推理 → 完整場景 voxel（50k–120k）不再 OOM。Voxel cap 是 bug 的 workaround，不是根本需求。

### 2.5 其他先前修復（保留中）

| 修復 | 檔案 | 說明 |
|------|------|------|
| `install_spconv_quantize_per_tensor_float_input_guard` | `spconv_int8.py` | Patch `torch.quantize_per_tensor` 確保 float 輸入；解決 `Quantize only works on Float Tensor, got Int` |
| `_set_tensor_quantizers_inference_mode` | `model_loader.py` | PTQ load 後強制 TensorQuantizer 進入推論模式（與 CalibrationManager 一致） |
| `_conv_out_to_bev` 簡化 | `sparse_encoder.py` | 移除 adaptive_avg_pool3d Z-collapse workaround，改為正確的 permute + view |

---

## 3. 目前診斷（mAP ≈ 0.0002 調查中）

模型已能產生預測（Predict_num > 0），但 mAP 仍極低。已加入以下診斷（全部使用 `print()` 確保輸出可見）：

### 3.1 PTQ 校準端

**`spconv_int8.py` — `_report_observer_stats()`**
- 校準結束後檢查所有 observer 是否被實際啟動
- 印出 `[observer-summary] X/Y observers calibrated`
- 列出未校準的 observer 名稱

**`bevfusion/quantization/quantize.py` — `_report_converted_scale_buffers()`**
- `convert_fx` 後檢查 scale/zero_point buffer 是否有效（≠ 1.0）
- 印出 `[ptq-scale-check]` 統計

**`bevfusion/quantization/quantize.py` — 儲存後驗證**
- 印出 `[save-check]` 確認 checkpoint 中 scale/zp key 數量與值

### 3.2 評估載入端

**`model_loader.py` — `_verify_spconv_scale_buffers()`**
- `load_state_dict` 後比對 model 的 scale buffer 與 checkpoint 的值是否一致
- 印出 `[spconv-scale-check]` — 若有 key name 不匹配會顯示 PROBLEM

**`model_loader.py` — `load_state_dict` 結果**
- 印出 `[load-state-dict]` — missing/unexpected key 數量，特別區分 sparse vs other

### 3.3 推理端

**`pytorch.py` — 每 stage 的 tensor 統計**（前 2 個 frame）
- `[debug] voxel_features_input`: shape, min, max, mean, std, nonzero ratio
- `[debug] sparse_encoder_output`: 同上
- `[debug] backbone_out[i]`: 同上
- `[debug] neck_out[i]`: 同上
- `[debug] head_heatmap_raw`: 同上
- `[debug] head_center`: 同上
- `[debug] head_dim`: 同上

### 3.4 診斷判讀指引

| 標籤 | 正常 | 問題指標 |
|------|------|----------|
| `[observer-summary]` | 大部分 observer 已校準 | 多數 UNCALIBRATED → 校準資料未流經這些節點 |
| `[ptq-scale-check]` | scale ≠ 1.0 | scale = 1.0 → convert_fx 遺失校準 |
| `[save-check]` | scale/zp key 存在且值非 1.0 | 缺少 key 或值為 1.0 |
| `[spconv-scale-check]` | `loaded from ckpt` = `scale buffers` | key name 不匹配 → FX 圖結構在 PTQ 與 eval 間不一致 |
| `[load-state-dict]` | sparse missing = 0 | sparse missing > 0 → checkpoint 結構不匹配 |
| `sparse_encoder_output` | mean ~ O(1)，合理 std | 全零 / 極大值 / NaN → 量化損壞 |
| `head_heatmap_raw` | 有大的負值（sigmoid 前） | 接近零 → 模型無信心 |

---

## 4. 所有修改過的檔案清單

### 核心修復

| 檔案 | 變更類型 | 說明 |
|------|----------|------|
| `projects/SparseConvolution/sparse_conv.py` | **Bug fix** | `_conv_forward` 始終計算 `out_spatial_shape`（不再因 `_fx_tracing` 跳過） |
| `deployment/projects/bevfusion_l/quantization/spconv_int8.py` | **Bug fix + 清理** | 新增 `non_traceable_module_classes`；移除 voxel cap 相關程式碼；新增 `_report_observer_stats` 診斷 |
| `deployment/projects/bevfusion_l/io/model_loader.py` | **Bug fix + 診斷** | 不再 disable SPCONV_FX_TRACE_MODE；新增 `_verify_spconv_scale_buffers`、`load_state_dict` 結果印出 |
| `projects/BEVFusion/bevfusion/sparse_encoder.py` | **Bug fix** | `_conv_out_to_bev` 簡化：移除 Z-collapse workaround |

### 清理（移除 workaround）

| 檔案 | 變更類型 | 說明 |
|------|----------|------|
| `deployment/quantization/bevfusion/quantization/quantize.py` | **清理 + 診斷** | 移除 `_resolve_spconv_calib_voxel_cap`、`--spconv-calib-max-voxels` CLI；新增 `_report_converted_scale_buffers`、`[save-check]` |
| `deployment/projects/bevfusion_l/config/deploy_config_split_int8.py` | **清理** | 移除 `spconv_calib_max_voxels` |
| `deployment/projects/bevfusion_l/config/deploy_config_int8.py` | **清理** | 移除 `spconv_calib_max_voxels` |
| `deployment/projects/bevfusion_l/runner.py` | **清理** | 移除 `max_voxels_per_sample` 傳遞 |

### 診斷（暫時性，debug 完成後可移除）

| 檔案 | 變更類型 | 說明 |
|------|----------|------|
| `deployment/projects/bevfusion_l/pipelines/pytorch.py` | **診斷** | 新增 `_tensor_stats` 工具函式；每 stage print tensor 統計（前 2 frame） |

---

## 5. 執行指令

### PTQ（產生 INT8 checkpoint）

```bash
python -m deployment.projects.bevfusion_l.quantization.quantize ptq \
    --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
    --checkpoint work_dirs/bevfusion/bevfusion_epoch_30.pth \
    --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \
    --calibrate-samples 256 --batch-size 1 --calib-seed 0 \
    --sparse-int8-only \
    --output work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth
```

### 評估（用 PTQ checkpoint）

```bash
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

### FP32 基準線（驗證 pipeline 正確性）

```bash
# 在 deploy_config_split_int8.py 中：
# 1. 改 checkpoint_path 為 FP32 .pth
# 2. 設 quantization = dict(enabled=False)
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

---

## 6. 架構圖：INT8 PTQ 資料流

```
[PTQ 校準]
  FP32 checkpoint → build model → fuse BN
    → apply_spconv_int8_quantization (prepare_fx, non_traceable custom convs)
      → calibrate_spconv_model (full voxels, no cap)
        → _report_observer_stats
      → convert_spconv_int8 (convert_fx → transform_qdq → remove_conv_add_dq)
        → _report_converted_scale_buffers
    → save PTQ checkpoint (state_dict with scale/zp buffers)

[評估載入]
  build model → fuse BN
    → _replace_encoder_with_fx_converted_structure
      (apply_spconv_int8_quantization + convert_spconv_int8, no calibration)
    → load_state_dict(PTQ checkpoint, strict=False)
      → [load-state-dict] print missing/unexpected
      → _verify_spconv_scale_buffers
    → _set_tensor_quantizers_inference_mode

[推理]
  voxel_features → pts_middle_encoder (INT8 FX GraphModule)
    → _conv_out_to_bev → spatial_features [1, 256, 180, 180]
      → pts_backbone → pts_neck → bbox_head → predictions
```

---

## 7. 相關文件

| 文件 | 內容 |
|------|------|
| [6_bevfusion_split_ptq_int8_progress.md](./6_bevfusion_split_ptq_int8_progress.md) | Split ONNX + PTQ 長期進度 |
| [7_bevfusion_int8_eval_fixes_and_progress.md](./7_bevfusion_int8_eval_fixes_and_progress.md) | 早期修復紀錄（部分已被本文件取代） |
| [3_int8_implementation.md](./3_int8_implementation.md) | 指令、Docker、錯誤代碼 |
