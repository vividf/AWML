# BEVFusion Spconv INT8 整合計畫

本文件先整理 **AWML CenterPoint 量化架構**，再據此訂出 **BEVFusion 如何以相同模式整合 INT8 / spconv PTQ**，並給出實作計畫。

---

## 一、AWML CenterPoint 量化架構（現狀）

### 1.1 設計要點

- **單一來源**：`deploy_config` 的 `quantization` 區塊是 PTQ 與 deployment 共用的唯一設定來源。
- **Loader 流程**：build model → (若啟用) fuse_bn → quant_model() → 可選載入 calib cache → load PTQ checkpoint → 驗證 amax → 關閉 sensitive layers → 設定 ONNX 匯出。
- **PTQ 腳本**：`deployment/quantization/centerpoint_quantization.py ptq` 使用同一個 `--deploy-cfg`（內含 `quantization`），產出 `.pth`，deployment 直接載入該 checkpoint。
- **數值與運行時**：CenterPoint 為 **dense**（Pillar + backbone + neck + head），使用 **pytorch_quantization**（TensorQuantizer、Q/DQ）；匯出 ONNX 後由 TensorRT 做 INT8 推論。

### 1.2 關鍵檔案與資料流

| 項目 | 說明 |
|------|------|
| `deploy_config` | `quantization`: enabled, mode, fuse_bn, quant_backbone/neck/head/voxel_encoder, sensitive_layers, skip_*, calib_cache_path |
| `entrypoint.py` | 讀取並 log `quantization_cfg` |
| `runner.load_pytorch_model()` | `quantization = self.config.deploy_cfg.get("quantization")`，傳入 model loader |
| `io/model_loader.py` | `build_model_from_cfg(..., quantization=...)`；enabled 時走 `_load_quantized_checkpoint` |
| `_load_quantized_checkpoint` | fuse_bn → quant_model() → load calib_cache（可選）→ load checkpoint → 驗證/關閉 sensitive → setup_quantization_for_onnx_export() |
| `deployment/quantization` | fuse_model_bn, quant_model, CalibrationManager, quantize_ptq；皆為 **dense** 模組 |
| PTQ 腳本 | `centerpoint_quantization.py ptq --deploy-cfg <int8_config> --output ptq.pth` |

### 1.3 與 BEVFusion 的差異

- CenterPoint：全 dense，僅需 **pytorch_quantization**。
- BEVFusion：**Lidar 為 spconv（稀疏）** + dense neck/head；若要「真正 spconv INT8」需 **spconv PTQ**（torch.ao.quantization + cumm INT8），而 dense 部分可沿用 pytorch_quantization。

---

## 二、BEVFusion 整合策略（對齊 CenterPoint 架構）

### 2.1 雙模式

與 CenterPoint 一樣用 **deploy_config.quantization** 驅動，但 BEVFusion 需要兩種模式：

| 模式 | 用途 | 稀疏 backbone | Dense 部分 | 產物 | 推論 |
|------|------|----------------|------------|------|------|
| `pytorch_quantization` | TensorRT INT8、與現有 lean/quantize 一致 | Fake quant（SparseConvolutionQunat） | TensorQuantizer | ONNX (Q/DQ) → TRT engine | TensorRT |
| `spconv_ptq` | 真正 spconv INT8（cumm kernel） | spconv prepare/convert（量化模組） | 可選 quant 或 FP16 | PTQ checkpoint | PyTorch backend |

### 2.2 建議優先順序

1. **Phase 1**：實作 **pytorch_quantization** 模式，與 CenterPoint 一致地「deploy_config + model_loader + PTQ 腳本」一鍵流程，產出 ONNX → TensorRT INT8（稀疏部分仍為 plugin，是否 INT8 取決於 plugin）。
2. **Phase 2**：實作 **spconv_ptq** 模式：載入 spconv PTQ 產生的 checkpoint，僅走 PyTorch backend，真正使用 cumm INT8。

---

## 三、實作計畫（Phase 1 為主）

### Step 1：deploy_config 支援 quantization（對齊 CenterPoint）

- **檔案**：`deployment/projects/bevfusion/config/deploy_config.py`（及可選的 `deploy_config_int8.py`）。
- **內容**：新增可選區塊 `quantization`，例如：
  - `enabled`: bool  
  - `mode`: `"pytorch_quantization"` | `"spconv_ptq"`  
  - `fuse_bn`: bool（預設 True）  
  - `quant_lidar_backbone`, `quant_neck`, `quant_head`: bool（用於 pytorch_quantization）  
  - `sensitive_layers`: list（要跳過量化的層）  
  - `calib_cache_path`: str | None  
- **行為**：未設定或 `enabled=False` 時維持現有行為（只 load checkpoint，不量化）。

### Step 2：BEVFusion entrypoint 讀取並傳遞 quantization

- **檔案**：`deployment/projects/bevfusion/entrypoint.py`。
- **內容**：  
  - 讀取 `quantization_cfg = deploy_cfg.get("quantization", None)`。  
  - 若存在且 enabled，log「Quantization: mode (enabled)」（與 CenterPoint 一致）。  
- **不需改**：`BaseDeploymentConfig` 不必把 quantization 納入 schema，沿用 `deploy_cfg.get("quantization")` 即可。

### Step 3：BEVFusion runner 將 quantization 傳入 model loader

- **檔案**：`deployment/projects/bevfusion/runner.py`。
- **內容**：在 `load_pytorch_model()` 中：
  - `quantization = self.config.deploy_cfg.get("quantization", None)`  
  - 呼叫 `build_bevfusion_model(..., quantization=quantization)`。

### Step 4：BEVFusion model_loader 支援量化載入（Phase 1：pytorch_quantization）

- **檔案**：`deployment/projects/bevfusion/io/model_loader.py`。
- **新增**：`build_bevfusion_model(model_cfg, checkpoint_path, device, quantization=None)`。
- **邏輯**：  
  - 若 `quantization` 為 None 或 `enabled=False`：維持現狀（build → load_checkpoint）。  
  - 若 `enabled=True` 且 `mode == "pytorch_quantization"`：  
    1. Build 原始 BEVFusion model。  
    2. （可選）fuse_bn：僅對 dense 部分呼叫 `fuse_model_bn`；lidar 若有 BN 需對應現有 `projects/BEVFusion/scripts/lean/quantize.py` 的 fusion 邏輯（或抽出共用函數）。  
    3. 對 lidar sparse 套用 `quantize_encoders_lidar_branch`（來自 lean/quantize.py 或抽到 deployment 的 bevfusion_quantization 模組）；對 camera/dense 套用 `quant_model`（deployment.quantization）或 lean 的 camera 量化。  
    4. 若有 `calib_cache_path`，載入 calibration cache。  
    5. `load_checkpoint` 載入 PTQ 的 .pth。  
    6. 將 TensorQuantizer 的 amax 搬到正確 device、驗證、關閉 sensitive_layers。  
    7. 呼叫 `setup_quantization_for_onnx_export()`（與 CenterPoint 相同），以便 ONNX 匯出 Q/DQ。  
  - 若 `mode == "spconv_ptq"`（Phase 2）：  
    - 方案 A：Build  float model → 依 deploy_config 執行 spconv prepare_fx → 載入 calibration 快取 → convert_fx → load state_dict（需 PTQ 腳本產出與 loader 相容的 checkpoint）。  
    - 方案 B：PTQ 腳本儲存完整 converted model；loader 用 `torch.load` 載入整個模型（較簡單，但須固定 PTQ 與 deployment 的 PyTorch/spconv 版本）。

### Step 5：PTQ 腳本（pytorch_quantization 模式）

- **選項 A**：擴充 `deployment/quantization/centerpoint_quantization.py`，新增子指令 `bevfusion-ptq`，或  
- **選項 B**：新增 `deployment/quantization/bevfusion_quantization.py`（建議），介面與 CenterPoint 一致：  
  - `ptq` 子指令：`--config`, `--checkpoint`, `--deploy-cfg`（必填，內含 `quantization`）, `--calibrate-samples`, `--output`。  
  - 流程：載入 BEVFusion config + checkpoint → 套用與 model_loader **相同**的 fuse + quant（lidar 用 lean 的 quantize_encoders_lidar_branch，dense 用 quant_model 或 lean camera 分支）→ 用 deployment 的 CalibrationManager 或 lean 的 calibrate_model 跑 calibration → 存 `{'state_dict': model.state_dict()}` 到 `--output`。  
- **單一來源**：PTQ 與 deployment 共用同一份 `deploy_config.quantization`，避免結構不一致。

### Step 6：ONNX / TensorRT 匯出

- **現狀**：BEVFusion 已有一條龍 ONNX → TensorRT；需確認當模型帶有 TensorQuantizer 時，匯出時會產生 Q/DQ 節點（與 CenterPoint 相同：`use_fb_fake_quant` 已在 loader 設定）。  
- **若有缺**：在 BEVFusion ONNX export pipeline 中確保使用已設定好 `use_fb_fake_quant` 的 model，必要時在 export 前再呼叫一次 `setup_quantization_for_onnx_export()`。

### Step 7：文件與範例 config

- **README**：更新 `README_SPCONV_INT8_DEPLOY.md`，說明：  
  - CenterPoint 與 BEVFusion 共用「deploy_config.quantization + model_loader + PTQ 腳本」架構。  
  - BEVFusion 兩種模式：`pytorch_quantization`（TensorRT INT8）、`spconv_ptq`（PyTorch 推論、真正 spconv INT8）。  
  - 使用步驟：撰寫/選用 `deploy_config_int8_*.py` → 執行 PTQ 腳本產出 .pth → 同一 deploy_cfg 跑 deployment。  
- **範例**：新增 `deployment/projects/bevfusion/config/deploy_config_int8_pytorch_quant.py`，內含 `quantization = dict(enabled=True, mode="pytorch_quantization", ...)`，並註解各欄位用途。

---

## 四、Phase 2：spconv_ptq 模式（簡要）

- **目標**：載入由 spconv PTQ（prepare_fx → calibrate → convert_fx）產生的 checkpoint，僅用 PyTorch backend 推論，真正跑 cumm INT8。  
- **PTQ 產物**：需與 loader 約定格式，二選一：  
  - 只存 `state_dict`（loader 需能建出相同 converted 結構並 load_state_dict），或  
  - 存整個 converted model（loader 直接 `torch.load` 取模型）。  
- **Loader**：當 `mode == "spconv_ptq"` 時，若採用「存整個模型」：在 model_loader 內 `torch.load(checkpoint_path)` 取得模型並搬到 device；若採用 state_dict，則需在 deployment 內實作「build converted BEVFusion graph」或依賴 PTQ 腳本一併輸出一個描述結構的 config，再在 loader 裡建出相同 graph 後 load_state_dict。  
- **Export**：spconv_ptq 模式可標註為「僅 PyTorch backend」；若未來要 ONNX，需為 spconv 量化 op 註冊 custom symbolic。

---

## 五、對照表與檢查清單

| 項目 | CenterPoint | BEVFusion（計畫） |
|------|-------------|-------------------|
| deploy_config 區塊 | `quantization` | 同左，多 `mode`（pytorch_quantization / spconv_ptq） |
| entrypoint | 讀取並 log quantization | 同左 |
| runner | 傳 quantization 給 loader | 同左 |
| model_loader | build → fuse_bn → quant_model → load .pth → setup export | build → (依 mode) fuse + quant（lidar + dense）→ load .pth → setup export；spconv_ptq 另案 |
| PTQ 腳本 | centerpoint_quantization.py ptq | bevfusion_quantization.py ptq（或擴充既有） |
| 數值棧 | 全 pytorch_quantization | pytorch_quantization（dense + 可選 lidar fake quant）或 spconv PTQ（lidar） |
| 產物 | ONNX (Q/DQ) + TRT engine | 同左（pytorch_quantization）；spconv_ptq 僅 PyTorch |

**實作檢查清單（Phase 1）**

- [ ] deploy_config 新增可選 `quantization`，含 `enabled`, `mode`, `fuse_bn`, `quant_*`, `sensitive_layers`, `calib_cache_path`。  
- [ ] entrypoint 讀取並 log quantization。  
- [ ] runner 將 `deploy_cfg.get("quantization")` 傳入 `build_bevfusion_model`。  
- [ ] model_loader 實作 `quantization` 參數；`pytorch_quantization` 路徑：fuse + quant（lidar + dense）→ load PTQ .pth → amax 驗證 → setup_quantization_for_onnx_export。  
- [ ] 抽出或共用 lean/quantize.py 的 lidar/camera quant + calibration，供 PTQ 腳本與 model_loader 使用。  
- [ ] 新增 PTQ 腳本（bevfusion_quantization.py 或擴充既有），以 deploy_cfg 為單一來源產出 PTQ .pth。  
- [ ] 驗證 ONNX 含 Q/DQ、TensorRT 可建 INT8 engine。  
- [ ] 更新 README 與範例 deploy_config_int8_*.py。

---

以上計畫讓 BEVFusion 的 INT8 整合在「設定來源、loader 流程、PTQ 腳本」上與 CenterPoint 對齊，並保留未來以 `spconv_ptq` 支援真正 spconv INT8 的擴充方式。
