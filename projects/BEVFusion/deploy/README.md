# BEVFusion 部署流程說明

本文件說明如何將訓練好的 BEVFusion 模型（PyTorch）匯出為 ONNX，以供後續 TensorRT 等推理引擎使用。以下以 **LiDAR-only**、**main_body** 模組為例，說明完整指令與各步驟意義。

---

## 一、快速開始：完整指令

在專案根目錄執行：

```bash
python projects/BEVFusion/deploy/torch2onnx.py \
  projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only_tensorrt_dynamic.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  work_dirs/bevfusion/epoch_30.pth \
  --device cuda:0 \
  --work-dir work_dirs/bevfusion/ \
  --module main_body
```

> **注意**：`--module` 必須為 `main_body`（不是 `main_bod`）。可選值為：`main_body`、`image_backbone`、`camera_bev_only`。

---

## 二、指令參數說明

| 參數 | 必填 | 說明 |
|------|------|------|
| `deploy_cfg` | ✓ | **部署設定檔**路徑，定義 ONNX/TensorRT 的輸入輸出與 dynamic axes。 |
| `model_cfg` | ✓ | **模型訓練設定檔**路徑，與訓練時使用的 config 一致，用來建構模型結構與載入資料 pipeline。 |
| `checkpoint` | ✓ | **權重檔**路徑（例如 `epoch_30.pth`）。 |
| `--work-dir` | 否 | 輸出目錄，ONNX 會寫入此目錄；預設為當前工作目錄。 |
| `--device` | 否 | 轉換時使用的裝置，例如 `cuda:0` 或 `cpu`；預設 `cpu`。 |
| `--module` | ✓ | 要匯出的子模組：`main_body` / `image_backbone` / `camera_bev_only`。 |
| `--sample_idx` | 否 | 從驗證/測試資料集中取第幾筆樣本做 export 時的 shape 推導；預設 `0`。 |
| `--log-level` | 否 | 日誌等級，如 `INFO`、`DEBUG`；預設 `INFO`。 |

- **deploy_cfg**：決定 ONNX 的 `input_names`、`output_names`、`dynamic_axes`、`opset_version`、`save_file` 等。
- **model_cfg**：決定 BEVFusion 的 backbone、head、voxel 設定，以及 `num_proposals` 等，必須與 checkpoint 訓練時一致。
- **checkpoint**：對應上述 model_cfg 訓練出來的 `.pth`。
- **--module main_body**：匯出「主體網路」（LiDAR-only 或 LiDAR+Camera 的偵測頭），對應你提供的 deploy 與 model config。

---

## 三、部署流程概覽

```
torch2onnx.py
    │
    ├─ 1. parse_args()          解析命令列
    ├─ 2. setup_configs()       合併 deploy_cfg + model_cfg，載入 ONNX 設定、voxel 設定、work_dir
    ├─ 3. Torch2OnnxExporter    建立 exporter
    └─ 4. exporter.export()     執行匯出
            │
            ├─ 4.1 ExportBuilder.build()
            │       ├─ _build_model_data()     載入 checkpoint、建構 PyTorch 模型、取一筆 sample 得到 model_inputs
            │       ├─ _build_backend()        從 deploy_cfg 讀取 backend（如 tensorrt）
            │       ├─ _build_ir_configs()     從 onnx_config 組出 input_names, output_names, dynamic_axes, opset_version
            │       ├─ _build_context_info()    組出 RewriterContext 所需參數
            │       └─ _build_patched_model()  對模型做 mmdeploy patch，得到可匯出的 patched_model
            │
            ├─ 4.2 _export_model()
            │       └─ _export_main_body()     用 TrtBevFusionMainContainer 包裝 patched_model，呼叫 torch.onnx.export
            │                                   輸出暫存為 *_temp_to_be_fixed.onnx
            │
            └─ 4.3 _fix_onnx_graph()           用 onnx_graphsurgeon 修正 TopK 的 K 為常數（從 model_cfg.num_proposals）
                                                另存為最終 .onnx（由 deploy_cfg.onnx_config.save_file 決定檔名）
```

---

## 四、BEV 架構與 TopK 位置

### 4.1 整體資料流（LiDAR-only）

部署時匯出的 **main_body** 對應的是「點雲 → BEV 特徵 → 偵測頭 → 3D 框」這條路徑。架構與資料流如下（TopK 所在處會標註）：

```
輸入（ONNX）
  voxels [M, max_pts, C]   coors [M, 3]   num_points_per_voxel [M]
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Point Cloud / Voxel 特徵提取                                        │
│  • pts_voxel_encoder (HardSimpleVFE)：voxel 內特徵聚合               │
│  • pts_middle_encoder (Sparse 3D CNN)：3D 稀疏卷積 → 偽圖像特徵       │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  BEV Backbone + Neck                                                │
│  • pts_backbone (SECOND)：2D 卷積提取 BEV 特徵                        │
│  • pts_neck (SECONDFPN)：多尺度融合，輸出 fusion_feat [B, 512, H, W]  │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  BEVFusionHead（bbox_head）                                          │
│  1) shared_conv：fusion_feat → hidden [B, C, H, W]                   │
│  2) heatmap_head：預測 dense_heatmap [B, num_classes, H, W]          │
│  3) heatmap.sigmoid() + 類 NMS（max_pool 取 local max）               │
│  4) 攤平 heatmap → [B, num_classes*H*W]                              │
│                                                                     │
│  ★ TopK 所在處（語意：取前 K 大）                                     │
│     argsort(descending=True)[..., :num_proposals]                    │
│     → 等價 ONNX 的 TopK，選出分數最高的 num_proposals 個 BEV 位置      │
│     → 得到 top_proposals_index / top_proposals_class                  │
│                                                                     │
│  5) 用上述 index 從 fusion_feat 做 gather → query_feat, query_pos     │
│  6) Transformer Decoder：query_feat 為 Q，fusion_feat 為 K/V        │
│  7) prediction_heads：預測 center, height, dim, rot, vel, heatmap    │
│  8) 後處理 → bbox_pred, score, label_pred                            │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
輸出（ONNX）
  bbox_pred [num_proposals, 10]   score [num_proposals]   label_pred [num_proposals]
```

### 4.2 各階段對應的程式位置

| 階段 | 對應程式 / 模組 |
|------|------------------|
| Voxel 特徵 | `bevfusion.py`：`extract_pts_feat()` → `pts_voxel_encoder`、`pts_middle_encoder` |
| BEV Backbone / Neck | `bevfusion.py`：`extract_feat()` → `pts_backbone`、`pts_neck` |
| Heatmap + NMS | `bevfusion_head.py`：`forward_single()` 中 `heatmap_head`、`max_pool2d`、`heatmap * (heatmap == local_max)` |
| **TopK（取前 K 個 proposal）** | **`bevfusion_head.py`：`forward_single()` 約 279–282 行**<br>`heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]` |
| Query 特徵與 Decoder | `bevfusion_head.py`：`fusion_feat_flatten.gather(...)`、`self.decoder[i](query_feat, key=fusion_feat_flatten, ...)` |
| 預測頭與後處理 | `bevfusion_head.py`：`prediction_heads`、`containers.py` 的 `postprocessing()` |

### 4.3 TopK 在架構中的角色

- **輸入**：攤平後的 heatmap（shape 為 `[B, num_classes*H*W]`），每個位置對應一個 BEV 網格在某一類別上的分數。
- **作用**：從所有 (類別 × 位置) 中選出分數最高的 **num_proposals** 個（例如 500 個），作為後續 Transformer Decoder 的 **query**（即 proposal）。
- **輸出**：這 K 個位置的索引與類別，用來從 BEV 特徵圖上 **gather** 出 query 特徵與位置編碼，再送入 Decoder 預測每顆 query 的 3D 框與分數。

因此 TopK 位於 **BEV 特徵 → heatmap → NMS 之後、Transformer Decoder 之前**，是「由 dense heatmap 篩選出少量 proposal」的關鍵一步；ONNX 匯出時會變成一個 TopK 節點，部署時需將 K 固定為 `num_proposals` 常數以相容 TensorRT（見下方「六、TopK 算子說明」）。

---

## 五、各步驟說明

### 5.1 `setup_configs()`（utils.py）

- 讀取 **deploy_cfg** 與 **model_cfg**，合併成一份設定。
- 從 deploy_cfg 取得 **onnx_config**（input_names、output_names、dynamic_axes 等）。
- 若有 **voxelize_cfg**（在 model_cfg 中），會轉成 `data_preprocessor.voxel_layer`，供後續資料前處理與 voxel 輸入使用。
- 建立 **SetupConfigs**，包含：deploy_cfg、model_cfg、checkpoint_path、device、work_dir、sample_idx、**module**、onnx_cfg、extract_pts_inputs 等。

### 5.2 `ExportBuilder.build()`（builder.py）

- ** _build_model_data()**  
  - 用 model_cfg 建立 runner，載入 `test_dataloader`，取第 `sample_idx` 筆資料。  
  - 用 mmdeploy 的 `build_task_processor` 與 `checkpoint_path` 建出 PyTorch 模型。  
  - 透過 `task_processor.create_input(..., extract_pts_inputs=...)` 得到一組 **model_inputs**（voxels, coors, num_points_per_voxel, points, imgs, ...）。  
  - 組出 **ModelData**（model_inputs、torch_model、input_metas）。

- ** _build_ir_configs()**  
  - 從 `setup_configs.onnx_cfg` 取出 input_names、output_names、dynamic_axes、opset_version、verbose、keep_initializers_as_inputs，供 `torch.onnx.export` 使用。

- ** _build_patched_model()**  
  - 使用 mmdeploy 的 `patch_model`，依 deploy_cfg 與 backend 對模型做改寫，得到 **patched_model**，並設為 eval、搬到指定 device。

### 5.3 `_export_main_body()`（exporter.py）

- 使用 **TrtBevFusionMainContainer(patched_model)** 包裝模型，其 `forward` 介面為：
  - 輸入：`voxels`, `coors`, `num_points_per_voxel`（LiDAR-only 時僅這三個）；若為 camera+lidar 還會接 points、lidar2img、img_aug_matrix、geom_feats、kept、ranks、indices、image_feats。
  - 輸出：`bbox_pred`, `score`, `label_pred`（與 deploy_cfg 的 `output_names` 一致）。
- 使用 **model_data** 中的那一筆 sample 的 tensor（voxels, coors, num_points_per_voxel 等）當作 `torch.onnx.export` 的範例輸入。
- 匯出時寫入 **work_dir** 下、檔名為 onnx_config.save_file 但副檔名先改為 `_temp_to_be_fixed.onnx`。

### 5.4 `_fix_onnx_graph()`（exporter.py）

- 讀取剛匯出的 `*_temp_to_be_fixed.onnx`。
- 使用 **onnx_graphsurgeon** 找到圖中的 **TopK** 節點，將 TopK 的 K 從動態輸入改為 **常數**，其值來自 **model_cfg["num_proposals"]**（例如在 t4 120m 設定中由 `default_lidar_second_secfpn_120m.py` 設定為 500）。
- 修正後寫入最終 ONNX 檔（例如 `bevfusion_lidar.onnx`），並保留在 `--work-dir` 指定的目錄下。

因此：**model_cfg 必須包含 bbox_head 的 num_proposals**（通常透過 `_base_` 繼承 default 的 bbox_head），否則 `_fix_onnx_graph` 會報錯。

---

## 六、TopK 算子說明

部署流程中會對 ONNX 圖內的 **TopK** 節點做一次修正（`_fix_onnx_graph`）。本節說明 TopK 在模型中的角色、為何會出現在 ONNX 裡，以及為何需要手動修正。

### 6.1 在模型中的語意

在 **BEVFusionHead** 的推理流程中，heatmap 經過 NMS 後，需要從「所有類別 × 所有 BEV 網格」中選出分數最高的 **前 K 個**位置，作為後續 Transformer Decoder 的 **query**（即 proposal）。  
這一步等價於「在攤平的 heatmap 上取前 K 大的索引」。  
在程式裡是用 **argsort + slice** 實作的（`bevfusion_head.py`）：

```python
# top num_proposals among all classes
top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
```

也就是：先對 heatmap 做 `argsort(..., descending=True)` 得到「由大到小」的索引，再取前 `num_proposals` 個。語意上就是「取前 K 大」，K = `num_proposals`（例如 500）。

### 6.2 為何會變成 ONNX 的 TopK？

PyTorch 匯出 ONNX 時，會把「取前 K 大索引」這類 pattern（`argsort` + slice）轉成 ONNX 的 **TopK** 算子，因為兩者等價：

- **輸入**：一個 tensor（例如攤平後的 heatmap）
- **輸出**：前 K 個最大的 **values** 與 **indices**
- **K**：要取幾個，對應我們的 `num_proposals`

因此 ONNX 圖中會出現一個 **TopK** 節點，其第二個輸入就是「K」。

### 6.3 為何要修正 TopK？（`_fix_onnx_graph`）

ONNX 的 TopK 規定：

- 第 1 個輸入：要處理的 tensor（動態 shape 沒問題）
- 第 2 個輸入：**K**（可以是常數，也可以是某個 tensor）

PyTorch 匯出時，`self.num_proposals` 往往會被當成「來自某個節點」的 tensor 或動態值，因此 **K 在 ONNX 裡會變成「動態輸入」**，而不是一個常數。  
**TensorRT 對 TopK 的 K 為動態的支援不佳**，會有相容性問題。因此部署流程中多了一步：**在 ONNX 圖上把 TopK 的 K 改成常數**，並把輸出 shape 標成 `[1, K]`，這樣 TensorRT 才能穩定使用。

`exporter.py` 的 `_fix_onnx_graph()` 做的就是這件事：

```python
# Fix TopK
topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
assert len(topk_nodes) == 1
topk = topk_nodes[0]
k = self.setup_configs.model_cfg.get("num_proposals", None)  # 例如 500
topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))  # K 改為常數
topk.outputs[0].shape = [1, k]
topk.outputs[1].shape = [1, k]
# ... 再寫回最終 .onnx
```

- 在圖中找到唯一的 **TopK** 節點  
- 從 **model_cfg** 讀出 `num_proposals` 作為 K（例如 500）  
- 用 **onnx_graphsurgeon** 把 TopK 的第二個輸入改成常數 `K`  
- 把 TopK 的兩個輸出的 shape 設成 `[1, k]`、dtype 設好  

導出的 ONNX 就變成「K 為常數的 TopK」，適合給 TensorRT 使用。

### 6.4 總結對照

| 項目 | 說明 |
|------|------|
| **語意** | 從 heatmap 中選出分數最高的 **num_proposals** 個位置，當作 detection 的 query。 |
| **程式** | `argsort(..., descending=True)[..., :num_proposals]`（等價於「取前 K 大」）。 |
| **ONNX** | 被匯出成一個 **TopK** 節點，K = num_proposals。 |
| **問題** | 匯出時 K 常變成動態輸入，TensorRT 不友善。 |
| **修正** | 用 `_fix_onnx_graph()` 把 K 改成常數（來自 `model_cfg["num_proposals"]`），並固定輸出 shape。 |

因此：**TopK = 在 heatmap 上做「取前 K 大」的運算，K 就是 config 裡的 `num_proposals`；部署時把 ONNX 裡的 TopK 的 K 固定成常數，以符合 TensorRT 需求。**

---

## 七、重要設定檔說明

### 7.1 部署設定：`bevfusion_main_body_lidar_only_tensorrt_dynamic.py`

- **codebase_config**：`mmdet3d`、任務為 VoxelDetection。
- **custom_imports**：載入 `projects.BEVFusion.deploy`、`projects.BEVFusion.bevfusion`、`projects.SparseConvolution`。
- **backend_config**：  
  - `type="tensorrt"`，`max_workspace_size=1<<32`。  
  - **model_inputs** 定義動態 shape：voxels 的 `[M, 10, 4]`（4 為特徵維，無 intensity 時為 4；有 intensity 時為 5，需改用 `bevfusion_main_body_lidar_only_intensity_tensorrt_dynamic.py`），coors、num_points_per_voxel 的 M 與 voxels 一致，min/opt/max 分別為 1 / 64000 / 256000。
- **onnx_config**：  
  - `save_file="bevfusion_lidar.onnx"`  
  - `input_names=["voxels", "coors", "num_points_per_voxel"]`  
  - `output_names=["bbox_pred", "score", "label_pred"]`  
  - `dynamic_axes` 對上述三個 input 的第 0 維命名為 `voxels_num`  
  - `opset_version=17`，`verbose=True`

### 7.2 模型設定：`bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py`

- 繼承 base 的 runtime、dataset（j6gen2_base）、pipeline（lidar intensity 120m）、**models（default_lidar_second_secfpn_120m）**、scheduler、misc。
- **model**：type=BEVFusion，包含 voxelize_cfg、pts_voxel_encoder、pts_middle_encoder、bbox_head；bbox_head 的 `num_proposals` 等由 default 設定（例如 500）。
- 指定 **data_root**、**work_dir**、train/val/test dataloader、evaluator。  
匯出時主要用到的是 **模型結構** 與 **num_proposals**，以及 **一筆 sample**（由 test_dataloader 的 `sample_idx` 取得）來推導 shape。

### 7.3 LiDAR 有 Intensity 時

若點雲特徵包含 intensity（例如 5 維），應改用：

- **deploy_cfg**：`projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only_intensity_tensorrt_dynamic.py`  
  - 其中 voxels 的 shape 為 `[M, 10, 5]`。

model_cfg 仍使用對應的訓練 config（例如 120m intensity 版本），保證 voxel 維度與訓練一致。

---

## 八、輸出檔案

- 最終 ONNX：`{work_dir}/{save_file}`，例如 `work_dirs/bevfusion/bevfusion_lidar.onnx`。  
- 中間檔（可於除錯時保留）：`{work_dir}/bevfusion_lidar_temp_to_be_fixed.onnx`（若腳本未刪除）。

---

## 九、常見問題

1. **`num_proposals is not found in the model configs`**  
   - model_cfg 或其所繼承的 base 必須在 bbox_head 裡提供 `num_proposals`（例如在 `default_lidar_second_secfpn_120m.py` 中為 500）。

2. **voxel 維度 4 與 5**  
   - 無 intensity：使用 `bevfusion_main_body_lidar_only_tensorrt_dynamic.py`（voxels 最後一維 4）。  
   - 有 intensity：使用 `bevfusion_main_body_lidar_only_intensity_tensorrt_dynamic.py`（voxels 最後一維 5）。

3. **CUDA out of memory**  
   - 可改用 `--device cpu` 做 ONNX 匯出（較慢但省 GPU 記憶體），或減少 `backend_config.model_inputs` 中的 opt/max voxel 數量。

4. **`--module` 寫錯**  
   - 必須是 `main_body`（全寫），不能是 `main_bod`。  
   - 其他可選：`image_backbone`（只匯出影像 backbone）、`camera_bev_only`（僅 camera BEV 子網路）。

---

## 十、其他可用的 deploy config

- `bevfusion_main_body_lidar_only_tensorrt_dynamic.py` — LiDAR-only，voxel 特徵維 4。
- `bevfusion_main_body_lidar_only_intensity_tensorrt_dynamic.py` — LiDAR-only，voxel 特徵維 5（含 intensity）。
- `bevfusion_main_body_with_image_tensorrt_dynamic.py` — LiDAR + Camera 主體。
- `bevfusion_camera_backbone_tensorrt_dynamic.py` — 僅影像 backbone。
- `bevfusion_camera_point_bev_tensorrt_dynamic.py` — Camera + Point BEV。

依你的模型輸入（是否含影像、是否含 intensity）選擇對應的 deploy_cfg，並搭配對應的 model_cfg 與 checkpoint 即可。
