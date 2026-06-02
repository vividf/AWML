# 26: 為什麼 `ScatterND -> SECOND` 在不同 ONNX 不是一模一樣

本文件說明一個常見疑問：

- `original`（舊 main-body 單圖）ONNX 在 `ScatterND` 後常看到  
  `Transpose -> Shape -> Gather -> Unsqueeze -> ...`
- `bevfusion_deployment_2_7_fp16_opt_merged_flip_no_opt`（split export 再 merge）ONNX  
  常只看到短鏈（例如 `Transpose -> Reshape -> Conv`）

這不是單純「誰對誰錯」，而是 **trace 邊界** 與 **圖組合方式** 不同造成的 ONNX 表達差異。

---

## 1. `ScatterND` 後面那串在做什麼

在 sparse encoder，`conv_out.dense()` 之後要把 5D 張量整理成 BEV 4D：

```20:36:projects/BEVFusion/bevfusion/sparse_encoder.py
x = out_tensor.dense()                      # (N, C, H, W, D)
t = x.permute(0, 1, 4, 2, 3).contiguous()  # (N, C, D, H, W)
return t.view(n, c * int(d), int(h), int(w))
```

PyTorch 的 `permute + view` 在 ONNX 常被展開成：

- `Transpose`（維度重排）
- `Shape/Gather`（抽出 N/C/H/W/D）
- `Unsqueeze/Concat`（組新 shape 向量）
- `Reshape`

所以在 `original` 裡看到 `Transpose -> Shape -> Gather -> Unsqueeze` 是合理的。

---

## 2. 為什麼 split trace 後看起來「少很多」

### 2.1 匯出邊界不同（核心）

`original` 是「單次 trace 全主圖」：  
`voxels/coors -> sparse -> dense backbone/neck/head` 一次畫完。

split export 是兩次 trace：

1. sparse 子圖：`voxels/coors -> lidar_bev`
2. dense 子圖：`lidar_bev -> backbone/neck/head`

對應程式：

- sparse wrapper：`BEVFusionSparseWrapper`
- dense wrapper：`BEVFusionDenseWrapper`
- merge：`onnx.compose.merge_models`

```223:231:/home/yihsiangfang/ml_workspace/AWML/deployment/projects/bevfusion/export/onnx_export_pipeline.py
def _export_split(...):
    """Export ``bevfusion_sparse.onnx`` and ``bevfusion_dense.onnx``."""
```

### 2.1.1 分開 trace vs 一起 trace：各自用什麼方式

- **一起 trace（single-file）**
  - 入口：`BEVFusionONNXExportPipeline.export()` 的 single-file 路徑
  - 包裝：`BEVFusionMainBodyWrapper`
  - 方法：呼叫一次 `self._export_to_onnx(...)`，直接把  
    `voxels/coors/num_points_per_voxel -> bbox_pred/score/label` 全流程 trace 成同一張 ONNX
  - 特性：shape 推導鏈通常完整保留在同一張圖內（較常看到 `Shape/Gather/Unsqueeze`）

- **分開 trace（split export）**
  - 入口：`BEVFusionONNXExportPipeline._export_split()`
  - 包裝與方法：
    1. sparse 段：`BEVFusionSparseWrapper` + `self._export_to_onnx(..., wrapper="sparse")`
    2. dense 段：先用 sparse wrapper 跑出 `lidar_bev`，再用 `self._export_dense_to_onnx(...)` 匯出 dense
    3. 合併段：`onnx.compose.add_prefix + onnx.compose.merge_models + cleanup().toposort()`
  - 特性：merge 後常看到更短的 shape 鏈，因為跨子圖的中介 shape plumbing 容易被折疊

### 2.2 graph compose + cleanup 會消掉中介 shape plumbing

split merge 之後會做 `cleanup().toposort()`，一些跨段的中介 shape 節點會被折疊或改寫：

```488:490:/home/yihsiangfang/ml_workspace/AWML/deployment/projects/bevfusion/export/onnx_export_pipeline.py
merged_graph.cleanup().toposort()
onnx.save_model(gs.export_onnx(merged_graph), str(merged_path))
```

因此你在 merged 圖常看到更短鏈（例如 `Transpose -> Reshape -> Conv`），而不是單圖 trace 時那種長 shape plumbing。

---

## 3. 為什麼「分開 trace」會比較少 `Shape/Gather/Unsqueeze`

不是因為模型少算了，而是因為：

1. dense 子圖把 `lidar_bev` 當作**外部輸入張量**，不需要再從 sparse 內部形狀推導。
2. merge 是「圖級連接」而非「重新 trace 一次全圖」，因此不會強迫保留舊單圖 trace 的所有動態 shape 子圖。
3. cleanup 可刪掉已無引用或可靜態化的 shape-path。

簡單說：  
**同一個數學流程，ONNX 可以有不同等價寫法；split trace + compose 通常更短。**

---

## 4. 這會不會造成數值不同？

### 4.1 理論上

僅就 `Transpose/Shape/Gather/Unsqueeze/Reshape` 這類 layout/shape op：

- 應是語義等價變換
- 不涉及乘加，不應引入浮點誤差

### 4.2 實務上（真正會讓 mAP 掉到 0 的通常不是這串）

真正高風險通常是契約不一致，例如：

- `coors` 欄位順序契約（`xyz` vs `zyx`）
- BN fold 與 bias 契約（特別是 shadow export 路徑）
- head grid 尺寸契約（`grid_size // out_size_factor`，常見 180x180）

也就是：  
**節點長相不同通常不是主因；契約對不齊才是主因。**

---

## 5. 如何判斷是不是「只是圖長相差異」

建議同一筆 sample 做分段比對：

1. 比 `lidar_bev`（sparse 輸出）統計
2. 比 `bbox_pred/score/label` 分布
3. 若需要，再比最終 mAP

如果中間張量與最終輸出一致，表示只是 ONNX 表達不同；  
若差異巨大，優先檢查 `coors`/BN/head-grid 契約。

---

## 6. 結論

`ScatterND` 後面不一模一樣，主要是 **export 策略不同**：

- 單圖 trace：保留較多動態 shape plumbing
- split trace + merge：圖更模組化，shape 子圖更容易被折疊

這種差異本身通常可接受；是否正確應以 **契約一致性** 與 **數值驗證** 為準，而不是只看節點數量或節點名稱是否一樣。
