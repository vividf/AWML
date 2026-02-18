# StreamPETR（AWML 版本）網路架構與 Deploy / ONNX 拆分說明

這份文件聚焦在 `projects/StreamPETR/configs/default/vov_flash_480x640_baseline.py`（約 L66-L185）所定義的 **Petr3D + StreamPETRHead** 架構，以及目前 `projects/StreamPETR/deploy/torch2onnx.py` 為什麼要把模型拆成：

- `extract_img_feat.onnx`
- `position_embedding.onnx`
- `pts_head_memory.onnx`

並說明它們分別對應網路中的哪個部份。

---

## 1. 模型總覽：Petr3D = Image Encoder + (Aux 2D) + StreamPETR 3D Head

在此 config 中，`model.type="Petr3D"`（detector）底下主要由三塊組成：

- **(A) Image Encoder（影像特徵抽取）**
  - `img_backbone`: `VoVNet`（`spec_name="V-99-eSE"`）
  - `img_neck`: `CPFPN`（輸出 2 個 scale；但 `Petr3D` 會選其中 `position_level` 對應的一層做後續 3D head）
  - 產生每個 camera 的 feature map（通常 stride=16，所以 H/16、W/16）

- **(B) Aux 2D ROI head（可選/輔助）**
  - `img_roi_head`: `mmdet.FocalHead`
  - 用來提供 2D supervision / 或 top-k token 選擇（訓練時較常用）
  - 重要：`Petr3D.forward_roi_head()` 在 **推論時**（若 `aux_2d_only=True` 且 `not training`）會直接回傳 `topk_indexes=None`，等於推論不依賴這個 head。

- **(C) 3D Head：`StreamPETRHead`（pts_bbox_head）**
  - 吃 image feature tokens，做：
    - **Position embedding**（把 2D 像素網格 + 深度 bins back-project 成 3D 位置編碼）
    - **Temporal memory**（streaming 記憶庫：上一幀的 object queries / proposals）
    - **Temporal Transformer decoder**（多層 decoder）輸出 3D boxes + scores

---

## 2. 重要張量流（對應 deploy 拆分的切點）

把一次 streaming inference（單幀、batch=1）抽象成三個 stage：

### Stage 1：`extract_img_feat`（影像 encoder）

- **輸入**：`img`，shape 類似 `(B, num_cameras, 3, H, W)`
- **輸出**：`img_feats`，shape 類似 `(B, num_cameras, C, H/stride, W/stride)`

在 AWML 版本中，這段實作於 `Petr3D.extract_img_feat()`：

- 可能套用 grid mask augmentation（若 `use_grid_mask=True`）
- `img_backbone(img)` -> `img_neck(img_feats)` -> 取 `position_level` 的那一層 feature
- reshape 回 `(B, len_queue, num_cams, C, H, W)`；deploy export 時 `len_queue=1`，所以會 squeeze 成 `(B, num_cams, C, H, W)`


### Stage 2：`position_embedding`（位置編碼 / cone）

`StreamPETRHead` 需要每個 token 的位置編碼 `pos_embed`，以及一個用於 spatial alignment 的輔助特徵 `cone`。

這段邏輯在 `StreamPETRHead.position_embeding()`，其核心概念是：

- 先建立每個 feature map token 對應的 2D 網格座標（pixel location）
  - 在 deploy wrapper 裡由 `Petr3D.prepare_location()` 產生 `location`
- 對每個 2D location + 一組 depth bins（`coords_d`）形成 `(x, y, d, 1)` 的相機座標點
- 用 `img2lidar`（或 `lidar2img.inverse()`）把它 back-project 到 LiDAR/world frame 得到 `coords3d`
- 將 `coords3d` normalize 到 `position_range` 的 [0,1] 範圍後，做 `inverse_sigmoid` + MLP（`position_encoder`）得到 `pos_embed`
- 同時計算 `cone`（由內參縮放後的 fx/fy + 部分 coords3d 組合）供後面的 `spatial_alignment` 使用

**輸入（deploy/export 定義）**

- `img_metas_pad`: padding 後的影像尺寸（export 時用 `[height, width, 3]` 這種打包方式）
- `img_feats`: stage1 的輸出
- `intrinsics`: 相機內參矩陣
- `img2lidar`: 相機到 LiDAR 的外參（或同等變換）

**輸出**

- `pos_embed`
- `cone`


### Stage 3：`pts_head_memory`（StreamPETRHead transformer + memory update）

這段是整個 3D 偵測 head 的主體（decoder、cls/reg heads、temporal memory update）。

在 deploy wrapper（`TrtPtsHeadContainer`）中，這段刻意 **不在圖內做 position embedding**，而是改成把 `pos_embed` / `cone` 當作輸入（也就是 Stage2 的輸出），然後：

- `memory_embed()`：把 image token feature 投影到 `embed_dims`
- `spatial_alignment(memory, cone)`
- `featurized_pe(pos_embed, memory)`
- 產生 learnable `reference_points`、`query_pos`、`tgt`
- `temporal_alignment()`：把上一幀 memory bank（`pre_memory_*`）注入 decoder 的 temporal tokens
- Transformer decoder 輸出 `outs_dec`
- cls/reg branches 產生 `all_cls_scores`、`all_bbox_preds`
- **post_update_memory**：從當前輸出挑 top-k proposals，更新並輸出下一幀要用的 memory bank（`post_memory_*`）

**輸入（deploy/export 定義）**

- `x`: `(B, num_cameras, C, H/stride, W/stride)`（即 `img_feats`）
- `pos_embed`, `cone`: Stage2 輸出
- `data_timestamp`, `data_ego_pose`, `data_ego_pose_inv`: streaming 用的時間與車體姿態
- `pre_memory_embedding`, `pre_memory_reference_point`, `pre_memory_timestamp`, `pre_memory_egopose`, `pre_memory_velo`: 上一幀（或前 N 個 proposals）的記憶庫狀態

**輸出（deploy/export 定義）**

- `all_cls_scores`, `all_bbox_preds`（並做過 flatten/transpose 以利 C++ 處理）
- `post_memory_embedding`, `post_memory_reference_point`, `post_memory_timestamp`, `post_memory_egopose`, `post_memory_velo`
- 以及一些除錯/中間張量（`reference_points`, `tgt`, `outs_dec`, ...），主要是為了 TRT/ONNX 調試與對齊

---

## 3. 為什麼 deploy 要拆成三個 ONNX？

目前 `projects/StreamPETR/deploy/torch2onnx.py` 的設計就是以 **「三個清楚的 functional boundary」** 來拆：

### 3.1 `extract_img_feat.onnx`：只做影像 encoder（backbone+neck）

對應網路：`Petr3D.extract_img_feat()`

拆出來的理由（實務上最常見）：

- **效能/引擎管理**：image backbone+neck 通常是最重的一段，獨立成一個 TRT engine 比較好做 FP16/INT8、也方便 profile
- **IO 更乾淨**：這段只依賴 raw image，不牽涉時間戳、ego pose、memory bank


### 3.2 `position_embedding.onnx`：把幾何（intrinsics/extrinsics）相關的 position embedding 獨立出來

對應網路：

- `Petr3D.prepare_location()`（產 feature grid 的 pixel locations）
- `StreamPETRHead.position_embeding()`（把 location + depth bins back-project 成 `pos_embed` / `cone`）

拆出來的理由：

- **幾何計算圖通常「大但規律」**：大量 reshape/matmul/normalize，跟 transformer 主體的 attention/FFN 性質不同
- **方便處理 export 相容性**：`torch2onnx.py` 在這個 section 額外註冊了 `onnxruntime.tools.pytorch_export_contrib_ops`，代表這段更容易遇到 ONNX op 相容性問題，因此獨立管理
- **更好維護**：pos embedding 常因座標系、pad、stride、depth bins 設定而調整，拆開能降低改動對主幹 engine 的影響


### 3.3 `pts_head_memory.onnx`：把 temporal transformer head + memory update 獨立成「可 stateful 的一步」

對應網路：`StreamPETRHead.forward()` 的推論主幹（但 deploy wrapper 改成外部餵 `pos_embed`/`cone`）

拆出來的理由：

- **Streaming 是 stateful 問題**：head 需要把上一幀 memory bank 當輸入、並輸出更新後的 memory bank；拆成獨立 onnx 能讓 C++ 端清楚管理 state
- **避免把 position embedding 內嵌進 head**：若把 stage2 也塞進 head，會讓 head 的 ONNX graph 更大、更難 debug，也更容易因幾何/shape 問題導致 TRT build 困難
- **推論配置與訓練不同**：export 時會把 `with_dn=False`（關掉 denoising query），這種「推論專用改動」集中在 head export 更直觀

---

## 4. Deploy 現在怎麼做（AWML 版本流程）

`projects/StreamPETR/README.md` 已列出三條 export 指令，背後的細節在 `projects/StreamPETR/deploy/torch2onnx.py`：

- 讀 config、build runner/model
- 讀 checkpoint（若有給）
- **把 FlashAttention 替換成一般 attention**（ONNX export 需求）
- 依 `--section` 建 wrapper module：
  - `extract_img_feat` -> `TrtEncoderContainer`
  - `position_embedding` -> `TrtPositionEmbeddingContainer`
  - `pts_head_memory` -> `TrtPtsHeadContainer`
- `torch.onnx.export(...)` 產生 `{section}.onnx`
- 再用 `onnxsim.simplify` 產生 `simplify_{section}.onnx`

---

## 5. ONNX 與網路模組對應表（最重要）

- **`extract_img_feat.onnx`**
  - **對應**：`Petr3D.extract_img_feat()`（`img_backbone` + `img_neck` + 選定 feature level + reshape）
  - **產物**：每個 camera 的 2D feature map（後續會 flatten 成 tokens）

- **`position_embedding.onnx`**
  - **對應**：`Petr3D.prepare_location()` + `StreamPETRHead.position_embeding()`
  - **產物**：
    - `pos_embed`: transformer cross-attn 用的 per-token position embedding
    - `cone`: `spatial_alignment` 需要的幾何輔助特徵

- **`pts_head_memory.onnx`**
  - **對應**：`StreamPETRHead` 的推論主幹（temporal alignment + transformer decoder + cls/reg + memory update）
  - **產物**：
    - 當前幀 detections（scores/boxes）
    - 下一幀要用的 memory bank（embedding / reference points / timestamp / egopose / velocity）

---

## 6. ONNX I/O（以 `deploy/torch2onnx.py` 為準）

這裡的名稱與順序，直接對齊 `projects/StreamPETR/deploy/torch2onnx.py` 裡 `input_names` / `output_names`，方便 deploy 端（C++ / TensorRT）接線。

### 6.1 `extract_img_feat.onnx`

- **Inputs**
  - `img`: `(1, num_cameras, 3, H, W)`
- **Outputs**
  - `img_feats`: `(1, num_cameras, C, H/stride, W/stride)`

### 6.2 `position_embedding.onnx`

- **Inputs**
  - `img_metas_pad`: `(3,)`，內容為 `[pad_h, pad_w, 3]`（export 時用 float tensor 包裝）
  - `img_feats`: `(1, num_cameras, C, H/stride, W/stride)`
  - `intrinsics`: `(1, num_cameras, 4, 4)`
  - `img2lidar`: `(1, num_cameras, 4, 4)`
- **Outputs**
  - `pos_embed`: `(1, num_tokens, embed_dims)`，其中 `num_tokens = num_cameras * (H/stride) * (W/stride)`
  - `cone`: `(1, num_tokens, 8)`

### 6.3 `pts_head_memory.onnx`

- **Inputs**
  - `x`: `(1, num_cameras, C, H/stride, W/stride)`（同 `img_feats`）
  - `pos_embed`: `(1, num_tokens, embed_dims)`
  - `cone`: `(1, num_tokens, 8)`
  - `data_timestamp`: `(1,)`（double）
  - `data_ego_pose`: `(1, 4, 4)`
  - `data_ego_pose_inv`: `(1, 4, 4)`
  - `pre_memory_embedding`: `(1, memory_len, embed_dims)`
  - `pre_memory_reference_point`: `(1, memory_len, 3)`
  - `pre_memory_timestamp`: `(1, memory_len, 1)`
  - `pre_memory_egopose`: `(1, memory_len, 4, 4)`
  - `pre_memory_velo`: `(1, memory_len, 2)`
- **Outputs**
  - `all_cls_scores`: 方便 C++ 處理而做過 reshape/transpose 的分類 logits
  - `all_bbox_preds`: 方便 C++ 處理而做過 reshape/transpose 的 bbox regression（含中心點等欄位）
  - `post_memory_embedding`, `post_memory_reference_point`, `post_memory_timestamp`, `post_memory_egopose`, `post_memory_velo`
  - 以及若干中間張量（`reference_points`, `tgt`, `outs_dec`, ...）用於 debug / 對齊
