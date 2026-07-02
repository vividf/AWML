# TensorRT split 路徑：有 Predict_num 但 mAP=0 — 管線說明與除錯

本文說明 **unified（單顆 `bevfusion_main_body` engine）** 與 **split（`bevfusion_sparse` + `bevfusion_dense`）** 的差異，歸納 **mAP=0 且仍有數百個預測** 的常見原因，並列出倉庫內可開關的 **debug 環境變數**（見 [`pipelines/tensorrt.py`](../pipelines/tensorrt.py)、[`pipelines/bevfusion_pipeline.py`](../pipelines/bevfusion_pipeline.py)）。

---

## 1. 兩條 TensorRT 路徑對照

| 項目 | Unified（`deploy_config_int8.py` → `main_body`） | Split（`deploy_config_split_int8.py`） |
|------|--------------------------------------------------|----------------------------------------|
| Engine | 單一 `bevfusion_main_body.engine`，voxel → 稀疏 → backbone → neck → head 全在同一圖 | `bevfusion_sparse.engine`（輸出 BEV）+ `bevfusion_dense.engine`（BEV → 偵測） |
| 邊界張量 | 無 Python 邊界，中間結果不落地 | **`lidar_bev`**：`[N,C,H,W]` 由稀疏 engine 寫出、再餵進稠密 engine（見 [`tensorrt.py`](../pipelines/tensorrt.py) `run_bevfusion`） |
| 與 PyTorch 分段對照 | 等同一次 `forward` 內部 | 應等同 [`pytorch.py`](../pipelines/pytorch.py)：**voxel reduce → pts_middle_encoder → pts_backbone → pts_neck → `_align_lidar_bev_to_head_grid` → bbox_head**；split 時 **對齊必須落在某一側的 ONNX 裡**（通常 dense 子圖從已對齊的 BEV 開始，或與 export 定義一致） |

前處理（voxelization）兩條路共用 [`bevfusion_pipeline.preprocess`](../pipelines/bevfusion_pipeline.py)，**coors** 會在 PyTorch 路徑補 batch 維度；TensorRT 稀疏 engine 吃到的 **voxels / coors / num_points** 與 unified 相同。

---

## 2. 為何「很多 Predict_num」仍可能 mAP=0？

mAP 需要預測框與 GT 在 **同一座標系、尺度、類別** 下達到足夠 IoU。以下情況常出現 **Predict_num ≫ 0 但 AP 全 0**：

1. **框都在錯的位置**（仍會被算進 `Predict_num`）  
   - 稀疏或稠密其中一段的 **數值刻度／layout** 與訓練不一致（例如 BEV 通道順序、H/W 翻轉、與 `bbox_coder` 假設不符）。  
   - **雙引擎版本不一致**：sparse ONNX 與 dense ONNX / engine 來自不同次 export 或不同 checkpoint。

2. **`lidar_bev` 形狀對了但語意錯**  
   - 例如 TRT 期望 `[1,256,180,180]`，實際餵入 shape 相同卻是 **另一種排列**（少見，多為 shape 直接錯）。

3. **稠密 engine 輸出語意與 `postprocess` 不一致**  
   - `bbox_pred` 須為 `[10, num_proposals]`（或可被轉成此形狀）、`score` / `label_pred` 長度一致。  
   - 若 ONNX 輸出順序與 `deploy_cfg.components.bevfusion_dense.io.outputs` 不一致，會依 name 重排；若 **name 對錯 tensor**，解碼會錯。

4. **類別或 score 異常**  
   - `label_pred` 超出 `[0, num_classes)` 會讓 heatmap 對應錯誤；score 全極低或分佈與 PyTorch 差太多會導致 **幾乎無真實匹配**。

5. **評測腳本連跑兩種 backend**  
   - 日誌裡第一段高 mAP、第二段 mAP=0 時，請確認第二段是否為 **split TRT**；比較時應固定 **同一 backend、同一 engine 目錄**。

---

## 3. 除錯用環境變數（建議順序）

以下皆在 **執行評測 / inference 的 process** 內設定（例如 `export BEVFUSION_TRT_LOG_IO=1`）。

### 3.1 TensorRT split I/O 與張量統計

| 變數 | 作用 |
|------|------|
| `BEVFUSION_TRT_LOG_IO=1` | 載入 split engine 後印出 **每顆 engine 的 input/output 名稱、shape、dtype**（確認與 `deploy_config_split_int8.py` 的 `lidar_bev` / `bbox_pred` 等一致）。 |
| `BEVFUSION_TRT_DEBUG_SPLIT=1` | 前 `BEVFUSION_TRT_DEBUG_SPLIT_FRAMES` 個 frame（預設 **2**）印：**稀疏輸出 BEV 統計**、**dense 綁定的 input 名稱與 shape 對齊檢查**、**dense 各輸出統計**（含 `bbox_pred` 前兩列範圍、`label_pred` unique、`score` 閾值計數）。 |
| `BEVFUSION_TRT_DEBUG_SPLIT_FRAMES` | 整數，預設 `2`；設 `0` 可關閉 frame 統計（仍可依賴 `LOG_IO`）。 |

**程式修正（重要）**：稠密 engine 的輸入名稱改為依 **`deploy_cfg` 的 `bevfusion_dense.io.inputs` 順序** 與 engine 做匹配，不再假設「TRT 回傳的第一個 input」即正確。若 engine 名稱與 config 不一致會 **warning**，避免 silent bind 錯 tensor。

### 3.2 後處理 / 解碼

| 變數 | 作用 |
|------|------|
| `BEVFUSION_DEBUG_POSTPROCESS=1` | 在 **解碼前** 印：`score` / `label` / feature-space center 的範圍與閾值計數；在 **解碼後** 印：metric **3D center** 的 min/max（與 `point_cloud_range`、GT 分佈對照；若範圍離譜，幾乎可斷定幾何錯）。 |
| `BEVFUSION_DEBUG_POSTPROCESS_FRAMES` | 預設 `2`，可加大。 |

### 3.3 與 PyTorch 對齊的建議流程

1. 同一 frame、同一資料：`BEVFUSION_TRT_DEBUG_SPLIT=1` 看 **sparse 輸出** min/max/mean。  
2. 用 PyTorch pipeline 的 `[debug]`（見 [`pytorch.py`](../pipelines/pytorch.py)）對 **同一 frame** 的 `sparse_encoder_output` 對照；若差異極大 → 問題在 **稀疏 TRT / INT8 scale / ONNX**。  
3. 若 sparse 接近、dense 後 `bbox_pred` / `score` 與 PyTorch 差很大 → 查 **dense engine / export 子圖**（是否含 align、是否同一 checkpoint）。  
4. 若 raw tensor 接近但 **decoded metric center** 離譜 → 查 **`bbox_coder` 與 head 輸出是否一致**（或 TRT 輸出是否被當成錯的 layout）。

**更細的稀疏塔對齊**（INT8 scale 懷疑與 BN/ReLU 圖不一致時）見 **§6**。

---

## 4. sparse INT8：split 時 `lidar_bev` 巨大、`bbox_pred` NaN

若 log 出現 **`lidar_bev` max ~1e4–1e5**（PyTorch BEV 約 **個位數～十**）、稠密 **`bbox_pred` NaN** / `score` **-inf**：

- **常見根因**：(1) 最後 INT8 層 **`output_scale` 與 FP32 `conv_out` 邊界不一致** — 用 `pts_middle_encoder._sparse_tail_absmax`（PTQ 內建）+ `sparse_int8_onnx_transform` 的 `sparse_tail_absmax`；(2) **ONNX 裡 Gemm 之間缺 BN/ReLU** 與 PTQ 校準圖不一致（見 §6）；(3) **NVIDIA forward 曾永久改寫 `weight`**（已改為 no_grad 下暫存還原，需自 **FP32 訓練 ckpt** 重跑 PTQ）。

---

## 6. ONNX：ImplicitGemm 之間是否與 PyTorch BN/ReLU 一致？

**PyTorch**（`make_sparse_convmodule`）：每段為 **conv → BN1d → SparseReLU**；`SparseBasicBlock` 另有 **conv2 → bn → (+shortcut) → relu**。下一層 conv 的 **`_input_quantizer`** 看到的是 **經過 BN+ReLU（與殘差）後** 的分佈。

**TRT sparse INT8** 的 `input_scale` / `output_scale` 鏈假設：上一顆 **ImplicitGemm(Int8)** 的 FP16 輸出，與 PyTorch 裡 **下一個 SparseConv 輸入** 的動態範圍一致。若 ONNX 在兩顆 Gemm 之間 **少了 ReLU 或 BN 等價運算**，校準與引擎會 **語意不一致**。

### 6.1 列印 ONNX 反向鏈（建議）

```bash
python deployment/projects/bevfusion_l/benchmark/debug_sparse_onnx_implicitgemm_topo.py \
  --onnx work_dirs/bevfusion_split_int8_deployment/onnx/bevfusion_sparse.onnx
```

對每一顆 `autoware::ImplicitGemm` / `ImplicitGemmInt8`，從 **feature 輸入**沿 `input[0]` 往回追，直到碰到上一顆 Gemm。中間應出現 **ReLU**（及實作 BN 的相關 op，視匯出而定）。若兩顆 Gemm **幾乎背對背**、中間沒有應有的非線性/歸一化，就要懷疑 **export / fuse_bn** 與 **PTQ 圖** 是否對齊。

### 6.2 `conv_out` **前** sparse feature 統計（PyTorch）

TRT log 裡的 **`lidar_bev`** 已是 **`conv_out` + dense/scatter 後** 的 BEV，**不可**直接與「sparse INT8 `max|features|` before conv_out」比較。

在 PyTorch 推理路徑對 **`pts_middle_encoder.conv_out`** 註冊 `forward_pre_hook`，印 **進 `conv_out` 的 `SparseConvTensor.features`**（與 PTQ sparse INT8 收集一致）：

- 腳本：[`benchmark/debug_sparse_pt_conv_out_stats.py`](../benchmark/debug_sparse_pt_conv_out_stats.py) 內 `ConvOutFeatureStatsHook`。

同一組 `voxels, coors, batch_size` 下，`abs_max` 應與 PTQ 的 **`[nvidia-calib] sparse INT8: max |sparse features| before conv_out`** 同數量級。若 PyTorch 正常而 TRT `lidar_bev` 仍爆 → 問題在 **INT8 主幹或 `conv_out` FP Gemm**；若 PyTorch 已異常 → 先查 **checkpoint / 量化載入**。

---

## 5. 相關設定與程式入口

- Split 元件 I/O 名稱與 profile：[`deploy_config_split_int8.py`](../config/deploy_config_split_int8.py)  
- TRT 推理：`BEVFusionTensorRTPipeline.run_bevfusion` in [`pipelines/tensorrt.py`](../pipelines/tensorrt.py)  
- 解碼：`BEVFusionDeploymentPipeline.postprocess` in [`pipelines/bevfusion_pipeline.py`](../pipelines/bevfusion_pipeline.py)  

更完整的 PTQ → ONNX → TRT 流程圖見 [`12_int8_sparse_pipeline_ptq_onnx_trt.md`](./12_int8_sparse_pipeline_ptq_onnx_trt.md)；sparse INT8 插件見 [`11_int8_autoware_plugin.md`](./11_int8_autoware_plugin.md)。
