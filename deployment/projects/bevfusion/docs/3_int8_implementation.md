# BEVFusion INT8 量化實作說明

本文件說明 AWML BEVFusion INT8 部署的**完整流程**、**遇到的困難與解法**、**本次更動摘要**，以及**如何執行 PTQ 與 INT8 deployment**。

---

## 本次更動摘要（PTQ 流程修正與驗證）

以下為完成 PTQ 流程並通過 Docker 驗證時所做的更動與步驟。

### 更動檔案與內容

| 檔案 | 更動內容 |
|------|----------|
| **`deployment/quantization/bevfusion_quantization.py`** | (1) **Spconv INT8 走 FX 流程**：PTQ 時若 `quantization.spconv_int8=True`，執行 **prepare_fx → calibrate → convert_fx → transform_qdq**，並將 `model.pts_middle_encoder` 替換為轉換後的 quantized 模組後存檔；sparse 在 PyTorch 端即為 quantized 模組，可正確讀取與融合。<br>(2) **Config 要求**：sparse encoder 須為 **FX 可追蹤**，請使用 **`block_type='basicblock_fx'`** 的 config（例如 `bevfusion_*_120m_fx.py`），否則 prepare_fx 會因 `replace_feature`/control flow 失敗。<br>(3) **DataLoader**：校準用 dataloader 設定 `num_workers=min(原值, 4)`、`persistent_workers=False`。 |
| **`deployment/projects/bevfusion/io/model_loader.py`** | (1) **PTQ 載入前先 fuse 稀疏 BN**：載入 PTQ checkpoint 時，除 dense BN fusion 外，**先對 `pts_middle_encoder` 做 spconv BN fusion**（`_fuse_spconv_bn(model)`），再（若 `spconv_int8`）**重建 FX 轉換結構**（prepare_fx + convert_fx，不校準），再 `load_state_dict`，使 state_dict key 對齊。<br>(2) **`_fuse_spconv_bn`**：呼叫 `spconv_int8._fuse_spconv_bn_in_encoder`。<br>(3) **`_replace_encoder_with_fx_converted_structure`**：PTQ + spconv_int8 時，先將 encoder 換成 FX 轉換後的結構再載入權重。 |
| **`deployment/projects/bevfusion/config/deploy_config_int8.py`** | **Export 模式**：`export.mode` 設為 `"none"`（僅 PyTorch 推論、不導出 ONNX/TRT），方便驗證 PTQ 載入與 mAP。 |
| **`3_int8_implementation.md`** | 新增「本次更動摘要」、Docker 範例加入 `--shm-size=8g` 與 `pytorch-quantization` 安裝步驟、補充 4.6 / 4.7 問題與解法。 |

### 驗證步驟（Docker 內）

1. **安裝 pytorch_quantization**（容器內每次 `docker run` 為新環境，需在指令內安裝或建新 image）：
   ```bash
   pip install --no-cache-dir --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple pytorch-quantization==2.1.3
   ```

2. **Step 1：PTQ**（需足夠 shared memory，建議 `--shm-size=8g`）：
   ```bash
   docker run --rm --gpus all --shm-size=8g -v $(pwd):/workspace -w /workspace awml-bevfusion:full \
     bash -c "pip install ... pytorch-quantization==2.1.3 && \
     python deployment/quantization/bevfusion_quantization.py ptq \
       --config projects/BEVFusion/configs/.../bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
       --checkpoint work_dirs/bevfusion/epoch_30.pth \
       --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \
       --calibrate-samples 19 --batch-size 1 --calib-seed 0 \
       --output work_dirs/bevfusion/epoch_30_ptq.pth"
   ```
   輸出：`epoch_30_ptq.pth`、`epoch_30_ptq.calib`、`epoch_30_ptq.spconv_scales`。

3. **Step 2：Deploy**（`deploy_config_int8.py` 中 `checkpoint_path` 指向 PTQ 檔、`quantization.ptq_checkpoint=True`）：
   ```bash
   docker run --rm --gpus all --shm-size=8g -v $(pwd):/workspace -w /workspace awml-bevfusion:full \
     bash -c "pip install ... pytorch-quantization==2.1.3 && \
     python -m deployment.cli.main bevfusion \
       deployment/projects/bevfusion/config/deploy_config_int8.py \
       projects/BEVFusion/configs/.../bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
       --module main_body"
   ```

4. **預期結果**：載入 PTQ 後無 84 missing / 21 unexpected keys；mAP 為合理數值（例如 Center Distance BEV mAP ≈ 0.90），且 Stage-wise Latency Breakdown 正常輸出。

---

## 一、完整 INT8 量化流程

BEVFusion INT8 量化分為**兩個階段**，與 CenterPoint 相同：

### 階段一：PTQ（Post-Training Quantization）— 離線校準

使用 `bevfusion_quantization.py ptq` 產生含校準 `_amax` 值的 PTQ checkpoint。

**流程：**
1. 載入 FP32 BEVFusion 模型
2. Fuse BatchNorm（dense 部分用 `fuse_model_bn`，sparse 部分用 `fuse_spconv_bn_eval`）
3. 插入 Q/DQ nodes（`Conv2d → QuantConv2d + TensorQuantizer`），插入在 backbone/neck/head
4. Calibrate：跑 N 個 calibration samples 過模型，TensorQuantizer 收集 histogram 統計量並計算 `_amax`（MSE 最佳化）
5. 可選：spconv INT8（**FX 流程**）— 使用 **`block_type='basicblock_fx'`** 的 config 時：**prepare_fx → calibrate → convert_fx + spconv backend_cfg → transform_qdq**，並以轉換後的 quantized 模組替換 `pts_middle_encoder`，再存檔（state_dict 內含 sparse INT8 權重與 scale）。
6. 存 PTQ checkpoint（`state_dict` 內含 dense 的 `_amax` 值與 sparse 的 quantized encoder）

### 階段二：Deployment — 載入 PTQ Checkpoint 推論

使用 `deployment.cli.main bevfusion` 載入 PTQ checkpoint 部署：
1. 建構模型結構
2. **Fuse BN（dense + sparse）** → 插入 Q/DQ nodes（重建與 PTQ 相同的 quantized 結構）
3. 載入 PTQ checkpoint（`_amax` 值自動恢復到 TensorQuantizer）
4. 若 PTQ 使用 spconv_int8（FX 流程）：載入前先 **fuse 稀疏 BN**，再 **prepare_fx + convert_fx** 重建 encoder 結構，然後 `load_state_dict` 載入權重與 scale。
5. 推論 / ONNX 導出 / TensorRT build

---

## 二、如何執行

### 2.1 環境需求

- **Docker**: `awml-bevfusion:full` 映像
- **pytorch_quantization**: 需安裝才能執行 PTQ 和 dense INT8（建議指定版本以可重現）
  ```bash
  pip install --no-cache-dir --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple pytorch-quantization==2.1.3
  ```
- 若未安裝 pytorch_quantization：dense 部分會維持 FP32，僅 sparse encoder 可做 spconv INT8

### 2.2 Step 1: 產生 PTQ Checkpoint

```bash
python deployment/quantization/bevfusion_quantization.py ptq \
    --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --checkpoint work_dirs/bevfusion/epoch_30.pth \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \
    --calibrate-samples 256 \
    --batch-size 1 \
    --calib-seed 0 \
    --output work_dirs/bevfusion/epoch_30_ptq.pth
```

**參數說明：**
| 參數 | 說明 |
|------|------|
| `--config` | BEVFusion 訓練 config（用來建構模型與 dataloader）|
| `--checkpoint` | 原始 FP32 checkpoint |
| `--deploy-cfg` | 部署 config（讀取 `quantization` 設定：哪些部份要量化、sensitive layers 等）|
| `--calibrate-samples` | 校準樣本數（建議 256 以上）|
| `--batch-size` | 校準 batch size |
| `--calib-seed` | 隨機種子（確保可重現）|
| `--output` | 輸出的 PTQ checkpoint 路徑 |
| `--skip-spconv-int8` | 可選：跳過 spconv INT8 校準 |

**Spconv INT8（FX 流程）**：須使用 **FX 可追蹤** 的 sparse encoder config，即 **`block_type='basicblock_fx'`**。請使用 `bevfusion_*_120m_fx.py`（與 base 120m 僅差 `pts_middle_encoder.block_type='basicblock_fx'`），checkpoint 可沿用原 basicblock 權重（多出的 `relu_final` 無額外參數）。

**輸出：**
- `work_dirs/bevfusion/epoch_30_ptq.pth` — PTQ checkpoint（dense 的 calibrated `_amax`；若啟用 spconv_int8 則 sparse encoder 為 FX 轉換後的 INT8 模組）
- `work_dirs/bevfusion/epoch_30_ptq.calib` — dense calibration cache

**為何 sparse encoder 還是 ~12ms、沒有明顯加速？**  
spconv 官方文件（`docs/INT8_GUIDE.md`）註明：**INT8 kernel 僅在 `input_channel % 32 == 0` 且 `output_channel % 32 == 0` 時啟用**；且只有在部分 shape（如 C≥64 且 K≥64）下 INT8 才明顯快於 FP16。目前 SECOND/BEVFusion 的 `default_lidar_second_secfpn_120m` 使用 **encoder_channels=((16, 16, 32), (32, 32, 64), ...)**，**第一階段為 16 channel**，16 % 32 ≠ 0，因此這些層會 **fallback 到 FP16**，整體 sparse 延遲可能與未量化時相近。若希望 sparse 部分有明顯 INT8 加速，需使用 **全部 channel 為 32 的倍數** 的 encoder（例如 32/64/128），並重新訓練或使用對應 checkpoint；現成 16-channel 架構在 spconv 下無法讓多數層使用 INT8 kernel。

### 2.3 Step 2: 使用 PTQ Checkpoint 部署

確保 `deploy_config_int8.py` 設定：
```python
checkpoint_path = "work_dirs/bevfusion/epoch_30_ptq.pth"

quantization = dict(
    enabled=True,
    ptq_checkpoint=True,   # <-- 關鍵：告知 loader 這是 PTQ checkpoint
    fuse_bn=True,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    spconv_int8=True,
    ...
)
```

然後執行：
```bash
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion/config/deploy_config_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --module main_body
```

**驗證 sparse encoder 是否真的用 INT8**  
- **載入時自動檢查**：當 PTQ + `spconv_int8=True` 時，`model_loader` 載入後會呼叫 `verify_spconv_int8_encoder`，在 log 中會看到 `Spconv INT8 encoder verified`（GraphModule + qint8 參數）或 `Spconv INT8 encoder verification failed`。  
- **手動驗證與測速**：在專案根目錄（已安裝 torch/CUDA 與 spconv）執行：
  ```bash
  python deployment/projects/bevfusion/scripts/verify_spconv_int8.py \
    deployment/projects/bevfusion/config/deploy_config_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
    --timing-runs 50
  ```
  會印出 encoder 是否為 GraphModule、qint8 參數數量、以及僅 sparse encoder 的 median 延遲（ms）。可與未 PTQ 模型比較，確認 INT8 有無生效。

### 2.4 Docker 完整範例

映像內未預裝 `pytorch_quantization`，需在指令中安裝；另建議加上 `--shm-size=8g` 避免校準時 DataLoader 觸發 Bus error（/dev/shm 不足）。

```bash
cd /path/to/AWML

# Step 1: PTQ（--shm-size=8g 避免 shared memory 不足）
docker run --rm --gpus all --shm-size=8g \
  -v $(pwd):/workspace -w /workspace \
  awml-bevfusion:full \
  bash -c "pip install --no-cache-dir --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple pytorch-quantization==2.1.3 && \
  python deployment/quantization/bevfusion_quantization.py ptq \
    --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --checkpoint work_dirs/bevfusion/epoch_30.pth \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \
    --calibrate-samples 256 --batch-size 1 --calib-seed 0 \
    --output work_dirs/bevfusion/epoch_30_ptq.pth"
# 註：PTQ 會做 sparse 的 BN fusion + 手動 scale 收集（輸出 .spconv_scales），但**不做** spconv FX 量化
# （因 4.3 control flow 問題）；故推論時 Sparse Encoder 仍為 FP。若只要 dense INT8 可加 --skip-spconv-int8。

# Step 2: Deploy（使用 PTQ checkpoint）
docker run --rm --gpus all --shm-size=8g \
  -v $(pwd):/workspace -w /workspace \
  awml-bevfusion:full \
  bash -c "pip install --no-cache-dir --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple pytorch-quantization==2.1.3 && \
  python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion/config/deploy_config_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --module main_body"
```

---

## 三、架構與檔案

### 3.1 新增 / 修改檔案

| 檔案 | 角色 | 說明 |
|------|------|------|
| `deployment/quantization/bevfusion_quantization.py` | **PTQ 腳本** | 類似 `centerpoint_quantization.py`，負責插入 Q/DQ → Calibrate → 存 PTQ checkpoint |
| `deployment/projects/bevfusion/quantization/spconv_int8.py` | **稀疏量化** | 手動 BN 融合 + activation observer + scale 計算 + `SpconvInt8EncoderWrapper` |
| `deployment/projects/bevfusion/quantization/__init__.py` | **模組入口** | 匯出 `apply_spconv_int8_quantization`、`calibrate_spconv_model` |
| `deployment/projects/bevfusion/io/model_loader.py` | **模型載入** | 支援 PTQ checkpoint 載入（`ptq_checkpoint=True` → 先 fuse dense+sparse BN、建 Q/DQ 結構再載 state_dict）；提供 `_fuse_spconv_bn` 以對齊 PTQ 存檔結構）|
| `deployment/projects/bevfusion/runner.py` | **執行器** | 讀取 quantization config，呼叫 `_apply_spconv_int8` 做 runtime spconv 校準 |
| `deployment/projects/bevfusion/pipelines/pytorch.py` | **Per-block 延遲** | `_run_bevfusion_with_breakdown` 分階段計時 |
| `deployment/projects/bevfusion/config/deploy_config_int8.py` | **部署設定** | `ptq_checkpoint=True`、checkpoint 路徑、量化旗標 |
| `deployment/projects/bevfusion/entrypoint.py` | **入口** | 印出量化設定日誌 |

### 3.2 量化組件對照

| BEVFusion 組件 | 模組名稱 | 量化方式 | 說明 |
|---|---|---|---|
| Voxel Encoder | `pts_voxel_encoder` | (未量化) | 含 Linear 層，可選 `quant_linear` |
| Sparse Encoder | `pts_middle_encoder` | **結構 only**（BN fusion + scale） | **PyTorch 與 TensorRT 皆未跑 INT8 kernel**；延遲與 FP16 相近，見 3.2.1 |
| Backbone | `pts_backbone` (SECOND) | pytorch_quantization | Conv2d → QuantConv2d + TensorQuantizer |
| Neck | `pts_neck` (SECONDFPN) | pytorch_quantization | Conv2d/ConvTranspose2d → Quant 版本 |
| Head | `bbox_head` (BEVFusionHead) | pytorch_quantization | Conv2d → QuantConv2d |

**注意：** BEVFusion 使用 `bbox_head`（非 CenterPoint 的 `pts_bbox_head`），`bevfusion_quantization.py` 已處理此差異。

#### 3.2.1 Sparse Encoder INT8 現況（PyTorch 與 TensorRT）

您提供的 **Stage-wise Latency Breakdown**（例如 Sparse Encoder INT8 11.83 ms vs FP16 12.69 ms 幾乎相同）是 **TensorRT** 推論結果，不是 PyTorch。

- **TensorRT**：main_body 以單一 ONNX 匯出（含 sparse encoder → backbone → neck → head）。目前 `SpconvInt8EncoderWrapper` 在 forward 時不做了輸入／權重的 qint8 轉換，導出的 ONNX **沒有**在 sparse conv 前後插入 QuantizeLinear/DequantizeLinear，因此 TensorRT 建 engine 時 sparse 區塊仍以 FP16 執行，Sparse Encoder 延遲與 FP16 config 相近。
- **PyTorch**：同上，wrapper 僅做 BN 融合與 scale 收集，forward 仍傳 float，spconv 底層走 FP，延遲也與 FP16 相同。
- **結論**：不論 PyTorch 或 TensorRT，Sparse Encoder 目前皆**未**執行 INT8 kernel；Backbone / Neck / Head 的 INT8（dense 部分）在兩者皆會生效。若要讓 sparse 真正跑 INT8，需依 spconv 官方流程在 **PyTorch 端**用 FX 做 Q/DQ 融合（見下方 3.2.2），**僅在 ONNX 插入 Q/DQ 無法**讓 TensorRT 對 sparse plugin 做融合。

#### 3.2.2 插入 Q/DQ 後 spconv INT8 能否正確讀取／融合？（依 spconv 官方說明）

依 **spconv** 倉庫（`docs/TENSORRT_INT8_GUIDE.md`、`spconv/pytorch/quantization/graph.py`、`backend_cfg.py`）的設計：

- **TensorRT 不會對 custom plugin 做 Q/DQ 融合**  
  > "There is a important drawback in tensorrt int8: **tensorrt won't fuse QDQ for custom int8 plugins**. So we must **fuse QDQ by ourself (in pytorch)**, and keep QDQ in regular layers such linear and convolution."

  因此：若只在 **ONNX** 裡對 sparse conv 前後插入 QuantizeLinear/DequantizeLinear，TensorRT **不會**把這些 Q/DQ 與 sparse 自定義層融合成 INT8；plugin 端會收到 DQ 後的 float，無法「正確讀取融合」。

- **Spconv 的作法是：在 PyTorch 裡用 FX 先融合 sparse 的 Q/DQ**  
  1. **prepare_fx**：fuse Conv-BN-ReLU、掛 observer。  
  2. **Calibrate**：跑校準資料。  
  3. **convert_fx**：用 spconv 的 `SPCONV_STATIC_LOWER_FUSED_MODULE_MAP` / `SPCONV_STATIC_LOWER_MODULE_MAP`，把 **sparse** 的 fused 模組換成 **quantized 模組**（如 `snniq.SparseConvReLU`），這些模組本身吃 qint8、出 qint8，圖上不再有獨立的 Q/DQ 節點包住 sparse conv；**dense** 的 Linear/Conv 則從 lower map 拿掉，保留 Q/DQ。  
  4. **transform_qdq**：把圖中的 `torch.quantize_per_tensor` 換成 spconv 的 `quantize_per_tensor`（支援 `SparseConvTensor`）。  
  5. 再將此 PyTorch 模型轉 ONNX / TensorRT；sparse 部分已是「已融合」的 quantized 模組，TRT plugin 收到的是 INT8 + scale，由 plugin 自己實作 INT8 運算。

- **結論**  
  - **僅插入 Q/DQ**（例如只在 ONNX 或只在 wrapper 外層加 QuantizeLinear/DequantizeLinear）：**無法**讓 TensorRT 或 spconv 對 sparse 做「正確讀取融合」；TRT 不融合 custom plugin 的 Q/DQ。  
  - **要能正確讀取與融合**：必須走 spconv 的 **FX 流程**（prepare_fx → calibrate → convert_fx + spconv backend_cfg → transform_qdq），讓 sparse 在 PyTorch 端變成 quantized 模組，再導出。BEVFusion 的 `pts_middle_encoder` 若含 control flow / `replace_feature` 等，需先改寫成可被 `torch.fx` 追蹤的結構（見 spconv `docs/INT8_GUIDE.md` 的 Prepare model 與 residual 寫法），才能套用上述流程。

### 3.3 PTQ vs CenterPoint 對照

| 步驟 | CenterPoint | BEVFusion |
|------|-------------|-----------|
| PTQ 腳本 | `centerpoint_quantization.py ptq` | `bevfusion_quantization.py ptq` |
| BN 融合 | `fuse_model_bn(model)` 整個模型 | Dense: `fuse_model_bn(submodule)` 分別；Sparse: `fuse_spconv_bn_eval` |
| Q/DQ 插入 | `quant_model(model, ...)` | 手動對 `pts_backbone`/`pts_neck`/`bbox_head` 呼叫 `quant_conv_module` |
| Calibration | `CalibrationManager.calibrate(dataloader, ...)` | 同 CenterPoint（dense 部分）；spconv 用手動 observer |
| Head 模組名 | `pts_bbox_head` | `bbox_head` |
| Sparse INT8 | 無（CenterPoint 只量化 dense） | 有（spconv BN fusion + manual calibration） |

---

## 四、遇到的困難與解法

### 4.1 無效的 export mode：`pytorch_only`

- **現象**：`ValueError: Invalid export mode 'pytorch_only'`
- **解法**：改為 `export.mode="none"` 或 `"both"`

### 4.2 缺少 pytorch_quantization 導致崩潰

- **現象**：`pytorch-quantization is required for quantization support`
- **原因**：`awml-bevfusion:full` 映像未預裝 NVIDIA `pytorch_quantization`
- **解法**：
  - `_apply_dense_quantization` 加 `try/except`，缺少時 skip dense Q/DQ
  - `_fuse_dense_bn_standalone` 作為不依賴 pytorch_quantization 的 fallback
  - 外層 `_load_with_quantization` 加 `try/except` fallback 到 FP32
  - **根本解法**：在 Docker 映像中安裝 `pip install pytorch-quantization`

### 4.3 FX 圖量化失敗：control flow（因此 PTQ 未做 spconv「真正 INT8」準備）

- **現象**：`symbolically traced variables cannot be used as inputs to control flow`
- **原因**：`BEVFusionSparseEncoder.forward` 有 Python if/loop/SparseConvTensor 操作，無法被 `torch.fx` 完整追蹤
- **解法**：放棄 spconv 官方的 FX 流程（prepare_fx → convert_fx → transform_qdq），改用手動方式：
  - `fuse_spconv_bn_eval` 做 BN 融合
  - `_ActivationObserver` 用 forward hook 收集 min/max，算出 scale 存成 `.spconv_scales`
  - `SpconvInt8EncoderWrapper` 包住 encoder（僅結構與 scale，forward 仍傳 float）
- **對 PTQ 的影響**：目前 PTQ **有做** sparse 的 BN fusion 與手動 scale 校準（輸出 `.spconv_scales`），但**沒有**做 FX convert，因此推論時 Sparse Encoder 仍為 FP，不會跑 INT8 kernel。若不需要 sparse 的 scale 檔，可下 PTQ 時加 `--skip-spconv-int8` 只做 dense 量化。

### 4.4 fuse_spconv_bn_weights 參數錯誤

- **現象**：`TypeError: fuse_spconv_bn_weights() missing 5 positional arguments`
- **原因**：`fuse_spconv_bn_weights` 需要張量參數，傳入了 module
- **解法**：改用 `fuse_spconv_bn_eval(conv, bn)` 接受 module 對象

### 4.5 之前缺少 PTQ 步驟（本次新增）

- **現象**：Dense 部分插入的 Q/DQ nodes 沒有經過 calibration（無 `_amax`），等於空的 quantizer
- **原因**：之前直接在 deployment 時建 Q/DQ nodes，但 FP32 checkpoint 不含 `_amax`
- **解法**：
  - 新增 `bevfusion_quantization.py` 做完整 PTQ（BN fusion → 插入 Q/DQ → calibrate → 存 checkpoint）
  - `model_loader.py` 支援 `ptq_checkpoint=True`：先建 Q/DQ 結構再載 PTQ state_dict
  - `deploy_config_int8.py` 新增 `ptq_checkpoint` 選項

### 4.6 PTQ 載入時 state_dict key 不一致（84 missing / 21 unexpected）

- **現象**：載入 PTQ checkpoint 後出現 84 個 missing keys（皆為 `pts_middle_encoder` 的 BN 參數）、21 個 unexpected keys（sparse conv 的 bias）。
- **原因**：PTQ 存檔時已對 sparse encoder 做 BN fusion（BN 被融合進 conv、替換成 Identity），state_dict 沒有 BN 的 key、且 fused conv 有 bias；deployment 端若只 fuse dense BN、未 fuse 稀疏 BN，模型結構與 PTQ 不一致，導致 key 對不上。
- **解法**：載入 PTQ 前，**先對 `pts_middle_encoder` 做 spconv BN fusion**（`_fuse_spconv_bn(model)`），再 `load_state_dict`，結構即與 PTQ 一致。

### 4.7 DataLoader Bus error / No space left on device（Docker）

- **現象**：PTQ 校準時 `RuntimeError: unable to write to file </torch_xxx>: No space left on device` 或 `DataLoader worker is killed by signal: Bus error`。
- **原因**：Docker 預設 `/dev/shm` 較小，多 worker 時 PyTorch 共享 tensor 會寫入 shm，導致空間不足。
- **解法**：
  - 執行容器時加 `--shm-size=8g`。
  - PTQ 腳本內將校準用 dataloader 的 `num_workers` 上限設為 4、`persistent_workers=False`。

### 4.8 Sparse Encoder INT8 與 FP16 延遲幾乎相同（TensorRT 結果）

- **現象**：使用 **TensorRT** 跑 INT8 與 FP16 時，Stage-wise Latency Breakdown 中「Sparse Encoder」延遲相近（例如 11.83 ms vs 12.69 ms）。
- **說明**：此 breakdown 來自 **TensorRT** pipeline（`pipelines/tensorrt.py` 的 layer profiler 彙整），不是 PyTorch。main_body 單一 ONNX 內含 sparse encoder，但匯出時 sparse 部分沒有 Q/DQ，TensorRT 以 FP16 執行該區塊，故 INT8/FP16 的 Sparse Encoder 時間幾乎相同。
- **預期**：在未於 ONNX 中為 sparse 子圖加入 Q/DQ 或使用 sparse INT8 plugin 前，Sparse Encoder 階段不會因 INT8 config 而明顯加速；dense 的 Backbone / Neck / Head 仍會以 INT8 加速。

---

## 五、Per-Block 延遲分析

**PyTorch**：`pipelines/pytorch.py` 的 `_run_bevfusion_with_breakdown` 將推論拆成：

| Stage | 說明 |
|---|---|
| Voxel Encoder | `pts_voxel_layer` + `pts_voxel_encoder` |
| Sparse Encoder | `pts_middle_encoder`（SCN） |
| Backbone | `pts_backbone`（SECOND） |
| Neck | `pts_neck`（SECONDFPN） |
| Head | `bbox_head` 推論 |
| Post Scoring | Head 後處理 |

每個 stage 使用 `torch.cuda.synchronize()` + `time.perf_counter()` 確保 GPU 計時準確。

**TensorRT**：`pipelines/tensorrt.py` 使用 `IProfiler` 收集 engine 內各 layer 執行時間，再以 `_aggregate_trt_layers_to_stages` 依 layer 名稱對應到 Sparse Encoder / Backbone / Neck / Head 等 stage；您看到的 Stage-wise Latency Breakdown（INT8 vs FP16）即為 TensorRT 推論結果。

---

## 六、快速指令速查

```bash
# 1. PTQ（產生 calibrated checkpoint）
python deployment/quantization/bevfusion_quantization.py ptq \
    --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --checkpoint work_dirs/bevfusion/epoch_30.pth \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \
    --calibrate-samples 256 --batch-size 1 --calib-seed 0 \
    --output work_dirs/bevfusion/epoch_30_ptq.pth

# 2. Deploy（使用 PTQ checkpoint）
python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion/config/deploy_config_int8.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --module main_body
```
