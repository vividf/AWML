# BEVFusion INT8 量化實作說明

本文件說明 AWML BEVFusion INT8 部署的**完整流程**、**遇到的困難與解法**、**本次更動摘要**，以及**如何執行 PTQ 與 INT8 deployment**。

---

## 本次更動摘要（PTQ 流程修正與驗證）

以下為完成 PTQ 流程並通過 Docker 驗證時所做的更動與步驟。

### 更動檔案與內容

| 檔案 | 更動內容 |
|------|----------|
| **`deployment/quantization/bevfusion_quantization.py`** | (1) **Spconv 校準**：PTQ 時不再用 `apply_spconv_int8_quantization` 包裝 encoder 再存檔（會改變 state_dict key 前綴 `pts_middle_encoder.*` → `pts_middle_encoder.encoder.*`）。改為只呼叫 `calibrate_spconv_model` 取得 scales，並存成 **`.spconv_scales`** 檔（與 PTQ checkpoint 同檔名、副檔名 `.spconv_scales`），deployment 時由 runner 讀取 scales 並套用 wrapper。<br>(2) **DataLoader**：校準用 dataloader 設定 `num_workers=min(原值, 4)`、`persistent_workers=False`，避免 Docker 內 `/dev/shm` 不足導致 Bus error。 |
| **`deployment/projects/bevfusion/io/model_loader.py`** | (1) **PTQ 載入前先 fuse 稀疏 BN**：載入 PTQ checkpoint 時，除 dense BN fusion 外，**先對 `pts_middle_encoder` 做 spconv BN fusion**（`_fuse_spconv_bn(model)`），再 `load_state_dict`，這樣模型結構與 PTQ 存檔時一致（PTQ 已 fuse 稀疏 BN），state_dict key 才能對齊。<br>(2) **`_fuse_spconv_bn`**：新增函式，內部呼叫 `spconv_int8._fuse_spconv_bn_in_encoder`。<br>(3) **除錯日誌**：PTQ 載入後記錄 `missing_keys` / `unexpected_keys` 數量、state_dict 中 `_amax` 數量、以及有多少個 TensorQuantizer 已載入 amax。 |
| **`deployment/projects/bevfusion/config/deploy_config_int8.py`** | **Export 模式**：`export.mode` 設為 `"none"`（僅 PyTorch 推論、不導出 ONNX/TRT），方便驗證 PTQ 載入與 mAP。 |
| **`README_INT8_IMPLEMENTATION.md`** | 新增「本次更動摘要」、Docker 範例加入 `--shm-size=8g` 與 `pytorch-quantization` 安裝步驟、補充 4.6 / 4.7 問題與解法。 |

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
5. 可選：spconv INT8 calibration — 呼叫 `calibrate_spconv_model` 收集 activation 統計並計算每層 scale，**存成 `{output}.spconv_scales`**（不在此時包裝 encoder，以免 state_dict key 變更）
6. 存 PTQ checkpoint（`state_dict` 內含 dense 的 `_amax` 值；sparse 的 scales 在 `.spconv_scales`）

### 階段二：Deployment — 載入 PTQ Checkpoint 推論

使用 `deployment.cli.main bevfusion` 載入 PTQ checkpoint 部署：
1. 建構模型結構
2. **Fuse BN（dense + sparse）** → 插入 Q/DQ nodes（重建與 PTQ 相同的 quantized 結構）
3. 載入 PTQ checkpoint（`_amax` 值自動恢復到 TensorQuantizer）
4. 可選：spconv INT8 — 若 PTQ 有產出 `.spconv_scales`，runner 可載入並套用；否則用 runtime 校準資料再呼叫 `apply_spconv_int8_quantization` 包裝 encoder
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

**輸出：**
- `work_dirs/bevfusion/epoch_30_ptq.pth` — PTQ checkpoint（包含 dense 的 calibrated `_amax`、sparse 已 fuse BN 的權重）
- `work_dirs/bevfusion/epoch_30_ptq.calib` — dense calibration cache
- `work_dirs/bevfusion/epoch_30_ptq.spconv_scales` — 若未 `--skip-spconv-int8`，sparse encoder 各層 activation scale（deployment 時可載入使用）

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
| Sparse Encoder | `pts_middle_encoder` | spconv INT8 (手動) | BN 融合 + activation observer + wrapper |
| Backbone | `pts_backbone` (SECOND) | pytorch_quantization | Conv2d → QuantConv2d + TensorQuantizer |
| Neck | `pts_neck` (SECONDFPN) | pytorch_quantization | Conv2d/ConvTranspose2d → Quant 版本 |
| Head | `bbox_head` (BEVFusionHead) | pytorch_quantization | Conv2d → QuantConv2d |

**注意：** BEVFusion 使用 `bbox_head`（非 CenterPoint 的 `pts_bbox_head`），`bevfusion_quantization.py` 已處理此差異。

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

### 4.3 FX 圖量化失敗：control flow

- **現象**：`symbolically traced variables cannot be used as inputs to control flow`
- **原因**：`BEVFusionSparseEncoder.forward` 有 Python if/loop/SparseConvTensor 操作
- **解法**：放棄 FX graph mode，改用手動方式：
  - `fuse_spconv_bn_eval` 做 BN 融合
  - `_ActivationObserver` 用 forward hook 收集 min/max
  - `SpconvInt8EncoderWrapper` 包住 encoder

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

---

## 五、Per-Block 延遲分析

`pipelines/pytorch.py` 的 `_run_bevfusion_with_breakdown` 將推論拆成：

| Stage | 說明 |
|---|---|
| Voxel Encoder | `pts_voxel_layer` + `pts_voxel_encoder` |
| Sparse Encoder | `pts_middle_encoder`（SCN） |
| Backbone | `pts_backbone`（SECOND） |
| Neck | `pts_neck`（SECONDFPN） |
| Head | `bbox_head` 推論 |
| Post Scoring | Head 後處理 |

每個 stage 使用 `torch.cuda.synchronize()` + `time.perf_counter()` 確保 GPU 計時準確。

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
