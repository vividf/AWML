# BEVFusion 稀疏編碼器全流程 ＋ spconv DDS 加速 — 中文詳解

> 本文從頭講解 LiDAR BEVFusion 的推論流程，深入 `GetIndicePairsImplicitGemm` 與 `ImplicitGemm`
> 這兩個 spconv 自訂 op 的輸入/輸出與內部運作（依據 `spconv` / `spconv_cpp` 原始碼），再說明
> trainStation/DDS 優化做了什麼加速、為什麼可行，最後說明 `pilot-auto.x2` 的
> `autoware_bevfusion` 在 preprocess / inference 各做了什麼、加速最終怎麼來的。
>
> 配合閱讀：
> - 設計與里程碑（含 profile 數據、export 手術細節）：[`34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md`](34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md)
> - 流程圖：[`BEVFusion_spconv_DDS_flow.png`](BEVFusion_spconv_DDS_flow.png) / [`.svg`](BEVFusion_spconv_DDS_flow.svg)
> - 參考原始碼：`spconv_cpp/spconv/src/spconvlib/spconv/csrc/...`、AWML `projects/SparseConvolution/`、
>   `autoware_bevfusion/`

---

## 0. 名詞速查（先看這個再往下讀）

| 名詞 | 意思 |
|------|------|
| voxel / 體素 | 把點雲格子化後「有點」的格子。稀疏網路只處理這些被佔用的格子。 |
| active site / 活躍體素 | 某一層中實際存在（非空）的體素。下採樣後數量會變。 |
| rulebook / 規則簿 | 稀疏卷積的索引表：記錄「哪個輸入體素、透過哪個 kernel 位置、貢獻到哪個輸出體素」。 |
| `N` (num_act_out) | 某層下採樣後的活躍體素數。**這個數字要等 GPU 算完才知道** → DDS 的根源。 |
| `KV` | kernel 體積 = `prod(ksize)`，3×3×3 → 27；`conv_out` 1×1×3 → 3。 |
| subm（submanifold） | 不改變活躍體素集合的稀疏卷積（輸出座標 = 輸入座標）。**無 DDS**。 |
| downsample | stride>1 的稀疏卷積，活躍體素數會變（變少）。**有 DDS**。 |
| DDS | Data-Dependent Shape，形狀要等執行時才知道。TensorRT 遇到它必須把形狀從 GPU 拷回 CPU。 |
| trainStation | TensorRT 內部 Myelin 對「被 DDS 切開的執行段」的命名；profile 上看得到。 |

---

## 1. BEVFusion 整體推論流程（LiDAR-only）

本專案部署的是 **LiDAR-only** 的 BEVFusion（split export 要求沒有相機分支）。一幀的流程：

```
點雲 (concatenated/pointcloud)
   │
   ▼  ① preprocess
voxelization：把點雲格子化 → voxels(特徵) + coors(座標[z,y,x]) + num_points_per_voxel
   │
   ▼  ② sparse middle encoder  (本文主角，spconv 3D 稀疏卷積)
pts_middle_encoder = BEVFusionSparseEncoder
   │            輸出：稠密化的 BEV 特徵 (lidar_bev) [1, C, H, W]
   ▼  ③ dense BEV backbone + neck
2D 卷積骨幹 → BEV 特徵圖
   │
   ▼  ④ detection head + postprocess
產生 3D 偵測框 → /objects
```

部署時 ② 與 ③④ 被拆成兩個 ONNX/engine：

- **sparse engine**（`bevfusion_sparse`）＝ 只有 `pts_middle_encoder`，輸入 `voxels/coors/num_points_per_voxel`，輸出 `lidar_bev`。
- **dense engine**（`bevfusion_dense`）＝ 後面的 2D backbone + head。

本文與 DDS 優化只動 **sparse engine**。

### 1.1 sparse middle encoder 的層結構

`sparse_shape = [1440, 1440, 41]`，kernel=3（除非註明）：

| 階段 | 內容 | 型別 | stride | 改變座標？ |
|------|------|------|--------|-----------|
| conv_input | 1 層 | SubMConv3d | 1 | 否 |
| encoder_layer1 | subm, subm + 下採樣 | SubM×2 + SparseConv3d | (2,2,1) | 只有下採樣那層 |
| encoder_layer2 | subm, subm + 下採樣 | SubM×2 + SparseConv3d | (2,2,1) | 只有下採樣那層 |
| encoder_layer3 | subm, subm + 下採樣 | SubM×2 + SparseConv3d | (2,2,1) | 只有下採樣那層 |
| encoder_layer4 | subm, subm | SubM×2 | 1 | 否 |
| conv_out | 1 層 | SparseConv3d k=(1,1,3) | (1,1,2) | 是 |

整個 encoder 在 ONNX 裡共 **21 個 `GetIndicePairsImplicitGemm` + 21 個 `ImplicitGemm`**：

- **4 個下採樣**（`encoder_layer1.2 / layer2.2 / layer3.2 / conv_out`，`subm=0`）→ 有 DDS → 造成 trainStation。
- **17 個 submanifold**（`subm=1`）→ 無 DDS。

空間尺度隨下採樣縮小：`1440 → 720 → 360 → 180`。

---

## 2. 為什麼稀疏卷積要拆成「索引」與「GEMM」兩個 op

### 2.1 稠密 vs 稀疏

稠密 3D 卷積會掃過 1440×1440×41 ≈ 8500 萬個格子，但點雲其實只佔用其中極少數（幾萬個）。稠密做法 99% 的算力都浪費在空格子上。

### 2.2 spconv 的 gather-GEMM 模型

稀疏卷積只對「活躍體素」算。一個稀疏卷積在數學上等於：

```
對每個活躍的「輸出體素 j」：
    out[j] = Σ_k  W[k] · in[ 對應到 (j, k) 的輸入體素 ]
             （k 跑過 kernel 的 KV 個位置；某些 k 沒有對應輸入就跳過）
```

把它整理成矩陣形式，就是一連串「**gather（依索引蒐集輸入列）→ 小矩陣乘法 → scatter（寫回輸出列）**」。

### 2.3 為什麼分兩個 op

關鍵觀察：**上式裡「哪個輸入體素對應到 (j, k)」完全由幾何決定（哪些格子被佔用 + kernel 形狀），跟特徵數值無關。** 所以 spconv 故意分兩步：

1. **`GetIndicePairsImplicitGemm`** — 只用幾何，算出 rulebook（索引表）。
2. **`ImplicitGemm`** — 拿 rulebook + 特徵，做真正的 gather-GEMM。

這個「幾何與特徵分離」正是後面 DDS 優化能成立的根本原因（§6）。

---

## 3. `GetIndicePairsImplicitGemm` 詳解

> 原始碼：
> - Python 包裝 / ONNX symbolic：`projects/SparseConvolution/sparse_functional.py`（class `GetIndicePairsImplicitGemm`）
> - C++ 實作：`spconv_cpp/.../SpconvOps/SpconvOps_get_indice_pairs_implicit_gemm.cc`
> - kernel：`.../SparseConvIndicesKernel/generate_conv_inds_*` 與 `unique_hash`

### 3.1 輸入（input）

| 輸入 | 意思 |
|------|------|
| `indices` | 這一層的活躍體素座標 `[num_act_in, 4]`，每列 `[batch, x, y, z]`。 |
| `batch_size`, `spatial_shape` | 批次大小與這層的空間尺寸（如 `[1440,1440,41]`）。 |
| `ksize, stride, padding, dilation, out_padding` | kernel 幾何參數。 |
| `subm` | 是否 submanifold（true=不改座標集）。 |
| `algo` | =1（`kMaskImplicitGemm`）。 |

注意：**沒有特徵（features）**。它只吃座標。

### 3.2 輸出（output，5 個）

Python `forward` 回傳 tuple `(out_inds, pair_fwd, pair_mask_fwd, mask_argsort_fwd, num_act_out)`；
ONNX 裡就是 `…GetIndicePairsImplicitGemm_output_{0,1,2,3,4}`：

| # | 名稱 | 形狀 | 意思 |
|---|------|------|------|
| 0 | **out_indices** | `[N, 4]` | 輸出活躍體素座標 `[batch,x,y,z]`，即下採樣後的輸出幾何。是**下一層的輸入座標**。 |
| 1 | **pair_fwd** | `[KV, N]` | rulebook 本體。`pair_fwd[k, j]` = 透過 kernel 位置 `k` 餵給輸出體素 `j` 的**輸入列索引**（`-1`=空）。 |
| 2 | **pair_mask** | `[N, 1]`(uint32) | 每個輸出體素的 bitmask：第 `k` bit=1 ⇔ `pair_fwd[k,j]≠-1`。（`mask_int_count = ceil(KV/32) = 1`，因 KV≤32。） |
| 3 | **mask_argsort** | `[N]` | 把輸出體素依 mask bit-pattern 排序的順序表，給 GEMM 排程用。 |
| 4 | **num_act_out** | scalar | `= N`，活躍輸出數。**這是 DDS 量**（見 §3.4）。優化後被丟棄（見 §6.3）。 |

### 3.3 內部在做什麼（三步驟 + 排序）

以下採樣（`subm=0`, `direct_table` 路徑）為例，對照 `SpconvOps_get_indice_pairs_implicit_gemm.cc`：

**Stage 1 — 產生候選輸出座標**（`generate_conv_inds_mask_stage1_direct_table`）
- 啟動 `num_act_in × KV` 個 thread；對每個（輸入體素, kernel 位置 k），用 `ConvOutLocIter` 算出它會落到哪個**輸出座標**。
- 用一張 **Murmur3 雜湊表**（`LinearHashTableSplit`）登記這些候選輸出座標，同時：
  - 把候選座標寫進 `indice_pairs_uniq`（之後要去重），
  - 把反向配對寫進 `pair_bwd`（推論用不到但 API 需要），
  - 計每個 kernel 位置的命中數 `indice_num_per_loc`。
- 這一步的工作量是「輸入數 × KV」，是幾何展開，**還不知道有幾個不重複的輸出**。

**Stage 1.5 — unique（去重）→ 得到 `num_act_out`**（`unique_hash` 或 thrust unique）
- 對候選輸出座標做去重，得到真正不重複的輸出體素數 `num_act_out = N`。
- **這就是 DDS 的根源**：`N` 要等這個去重在 GPU 上跑完才知道。
- `num_out_act_bound` 會把 `N` 夾在上限內（部署用的 256000 上限）。

**Stage 2 — 填出最終 rulebook**（`assign_output_direct_hash` + `generate_conv_inds_stage2_mask_direct_table`）
- 現在 `N` 已知，配置 `out_indices [N,4]`、`pair_fwd [KV,N]`、`pair_mask [N,1]`。
- `assign_output_direct_hash`：把去重後的輸出座標寫進 `out_indices`，並在雜湊表給每個輸出座標一個列號。
- stage2 kernel：再跑一次（輸入, k），用雜湊查出輸出列號 `j`，把輸入列號填進 `pair_fwd[k][j]`，並把第 `k` bit 設進 `pair_mask[j]`。

**排序**（`sort_1d_by_key_allocator_v2`）
- 對 `pair_mask` 做 key-sort 得到 `mask_argsort`（`do_sort=true`）。讓 active-tap pattern 相同的輸出體素排在一起 → GEMM tile 規整。

> subm 路徑更簡單：輸出座標 = 輸入座標（`out_inds = indices`、`num_act_out = num_act_in`），用 `generate_subm_conv_inds` 直接建 rulebook，**沒有 unique、沒有 DDS**。這就是為什麼 17 個 subm 節點不造成 trainStation。

### 3.4 為什麼它是 DDS／trainStation 的根源

只有「下採樣」會做 stage1.5 的 unique，而 `num_act_out` 必須等 unique 算完。TensorRT 為了知道後續 tensor 的形狀，必須在這裡把 `num_act_out` 從 GPU 拷回 CPU（`DeviceToShapeHostCopy`），於是把 engine 切成一段一段（trainStation），打斷 pipeline。4 個下採樣 = 4 個 `DeviceToShapeHostCopy` = 6 個 trainStation（見 §5）。

---

## 4. `ImplicitGemm` 詳解

> 原始碼：
> - Python / ONNX symbolic：`sparse_functional.py`（class `ImplicitGemm`）
> - C++：`spconv_cpp/.../ConvGemmOps/ConvGemmOps_implicit_gemm.cc`

### 4.1 輸入（input）

ONNX op 的輸入正是 `[features, filters, pair_fwd, pair_mask_fwd, mask_argsort_fwd]`（+ 可選 bias）：

| 輸入 | 形狀 | 意思 |
|------|------|------|
| `features` | `[num_act_in, C_in]` | 這一層輸入體素的特徵。 |
| `filters` | `[C_out, KV, C_in]` | 卷積權重（內部 view 成這個形狀）。 |
| `pair_fwd` | `[KV, N]` | 來自 §3 的 rulebook。 |
| `pair_mask_fwd` | `[N,1]` uint32 | 來自 §3。 |
| `mask_argsort_fwd` | `[N]` | 來自 §3。 |
| `num_activate_out` | scalar | = `N`。**注意：在 ONNX 裡這不是一個 input**，而是由 `pair_mask` 的 shape 推導（見 §4.4）。 |
| `masks` | — | mask filter（哪些 bit 屬於這個 split；非 split 時 = `0xffffffff`）。 |

### 4.2 輸出（output）

| 輸出 | 形狀 | 意思 |
|------|------|------|
| `out_features` | `[N, C_out]` | 這一層輸出體素的特徵。下採樣時用 `zeros` 初始化（沒被填到的列保持 0）。 |

`out_features` 的列數 = `N`，列對應 `out_indices` 的體素；它接著當作下一層的 `features`。

### 4.3 內部在做什麼

對照 `ConvGemmOps_implicit_gemm.cc`：

1. **auto-tune**：`conv_tuner.get_tuned_algo(...)` 依 (輸入/權重/輸出 dtype, `C_out`, `C_in`, GPU arch) 查快取；沒查到就 `tune_and_cache` 實測挑一個最快的 CUTLASS 風格 implicit-gemm kernel（決定 tile 形狀等）。
2. **配置輸出**：`out_features = zeros([N, C_out])`（非 subm）。
3. **跑 masked implicit GEMM**：`run_with_tuned_result(...)`，把這層卷積當成一個大矩陣乘法在 GPU 上跑，但用三個 rulebook tensor 把「稀疏」織進 GEMM：
   - **`pair_fwd`**：在 GEMM 取輸入列時，依 `pair_fwd[k][j]` gather 對應的輸入特徵列（`-1` 就補 0）。
   - **`pair_mask`**：每個輸出列的 active-tap bitmask，讓 kernel 跳過空的 kernel 位置（`mask_filter = masks[0]`）。
   - **`mask_argsort`**：把輸出列依 mask pattern 重排，使每個 `mask_width`（= tile_shape[0]）列的 tile 內 pattern 一致 → 密實 tile，避免分支與補零浪費。
   - 支援 `fp32_accum`、融合 bias / activation（本專案還有 INT8 與 ReLU fuse 變體）。

一句話：**`ImplicitGemm` = 用 rulebook 把稀疏卷積偽裝成一個高效的 dense GEMM 來跑。**

### 4.4 `N` 怎麼進到 `ImplicitGemm`（關鍵）

ONNX 裡 `num_activate_out` 不是 op 的 input。`ImplicitGemm` plugin 的輸出形狀由**輸入維度**推導：

```
outputs[0].d[0] = inputs[3].d[0]   // = pair_mask 的 dim0 = N
outputs[0].d[1] = inputs[1].d[0]   // = C_out
```

也就是說：**只要 `pair_mask`（與其他 rulebook）的 shape 被定成 `N`，輸出 `[N, C_out]` 就完全確定，不需要另外傳 `num_act_out` 這個 scalar。** 這是 §6.3「`out[4]` 可以丟掉」的關鍵。

---

## 5. trainStation / DDS 是什麼、為什麼慢

- TensorRT 遇到 DDS（形狀要執行時才知道）時，必須 `DeviceToShapeHostCopy` 把該形狀（這裡是 `num_act_out`）從 GPU 拷回 host，才能配置/啟動下一段 → 把 engine 切成多個執行段（**trainStation**）。
- 每個邊界都打斷 GPU pipeline、讓 GPU 閒置，也無法把整個 sparse encoder 收成一個 CUDA Graph。
- 板上 profile（見 [`34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md`](34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md) §2）：單次推論 34.5ms，其中 **GPU idle ≈ 10.3ms ≈ 29.7%**；6 個 trainStation 對應 4 個下採樣的 `DeviceToShapeHostCopy`。

---

## 6. DDS 優化（Route A）做了什麼、為什麼可以

### 6.1 可行的根本前提

§2.3 已說明：**rulebook 只依賴體素幾何，不依賴特徵。** 而體素座標在 voxelization 之後（preprocessing）就已經知道。所以——4 個下採樣的整段 rulebook 與輸出座標，**可以在 engine 跑之前就先算好**，當成 engine 的輸入餵進去。

### 6.2 Export 端的圖手術

`export/onnx_remove_trainstation_dds.py::remove_trainstation_dds`：

1. **刪掉** 4 個下採樣的 `GetIndicePairsImplicitGemm` 節點。
2. 把它們的 `out[0..3]` **提升成 graph input**，命名為乾淨的 `rulebook/<tag>/<slot>`（`tag∈{l1,l2,l3,out}`、`slot∈{out_indices,pair_fwd,pair_mask,mask_argsort}`）。`out[4]`（num_act_out）沒有 consumer，丟棄。
3. **改寫 consumer edge**：原本接到節點輸出的 12 條 `ImplicitGemm` 邊改接到新 input。graph input 從 3 → 19（+16）。
4. `ImplicitGemm` **完全不動**（它的輸出形狀本來就從 input dim 推導）。

結果：size tensor 消失 → 4 個 `DeviceToShapeHostCopy` 消失 → 6 個 trainStation 全部歸 0。

### 6.3 為什麼 `out[4]`（num_act_out）可以丟

見 §4.4：engine 內 `ImplicitGemm` 透過 `pair_mask` 的 dim0 取得 `N`。只要 runtime 在 `enqueueV3` 前用 `setInputShape` 把 rulebook 的 shape 設成 `N`，整個圖的形狀就解析得出來，不需要那個 scalar。

### 6.4 為什麼數值等價

precompute 用的就是 baseline 圖內用的**同一支** `SpconvOps::get_indice_pairs_implicit_gemm`，輸入同樣是體素座標，所以算出的 rulebook 與 baseline 圖內算的逐 byte 對得起來。AWML 已在 `_ts_tmp/validate_equiv.py` 用 40k 隨機 voxel 驗證：baseline vs modified 的 `lidar_bev` `max abs diff = 0.0088`（fp16 等級）→ MATCH。

---

## 7. `pilot-auto.x2` / `autoware_bevfusion` 的實作

> 檔案：`autoware.universe/perception/autoware_bevfusion/`
> （build 目標在 `pilot-auto.x2/src/autoware/universe/...`）

### 7.1 preprocess 流程（`bevfusion_trt.cpp::preProcess`）

```
validatePointCloud → enqueuePointCloud → generateSweepPoints
   → cudaStreamSynchronize
   → processPointCloudVoxelization  ─►  voxel_coords_d_（座標）
   → 若 sparse_remove_trainstation：
         sparse_rulebook_ptr_->compute(voxel_coords_d_, num_voxels, ...)   ← 新增：算 rulebook
   → configureTensorRTInputs（內含 setSparseRulebookInputShapes）
```

只有開了 `sparse_remove_trainstation` 才會跑 rulebook precompute；關掉就是原本 baseline 行為（no-op）。

### 7.2 `SparseRulebookPrecompute` 內部（`preprocess/sparse_rulebook_precompute.cu`）

**建構時** `allocateStageBuffers()`：
- 為 4 個 stage 各配置穩定的 device buffer：`out_indices [N,4]`、`pair_fwd [KV,N]`、`pair_mask [N,1]`、`mask_argsort [N]`（`N = out_indices_num_limit_ = 256000` 上限）。這些 buffer 同時就是 engine 的 input tensor。
- 配置一塊共用的 spconv workspace（內含 index-gen 工作區、`indices_kernel_num`、`pair_bwd` 等，以及 8MB thrust 暫存），大小取所有 stage 的最壞情況。

**每幀** `compute(coors_d, num_in, ...)`：
1. `buildBatchedCoordsKernel`：把 voxelization 的 `coors [z,y,x]` 轉成 spconv 要的 `[batch,x,y,z]`。
2. **cascade 跑 4 個下採樣 stage**（`computeStage`）：每個 stage 呼叫圖內 plugin 用的同一支 `SpconvOps::get_indice_pairs_implicit_gemm`，輸出寫進該 stage 的穩定 buffer，回傳 `num_act_out`。
3. **把 `out_indices` 往前串**：stage i 的 `out_indices` 當 stage i+1 的輸入座標（中間的 submanifold 層不改座標集，留在 engine 圖內）。空間尺度 `1440→720→360→180`。
4. 每個 stage 的 `num_act_out` 存進 `stage_counts_[i]`，供後面 `setInputShape`。

### 7.3 誠實說明：仍有 4 個循序的 D2H

`computeStage` 回傳的 `num_act_out` 是**host int**——`get_indice_pairs_implicit_gemm` 內部在 unique 那步做了一次 device→host 讀回。由於 cascade 有資料相依（stage i+1 的輸入＝stage i 的輸出座標＋計數），這 4 個讀回是**循序的：4 個 sync，每 stage 一個**，數量和 baseline 一樣。

**所以 DDS 優化省的不是「sync 數量」，而是 sync 的「位置」：**

| | baseline | Route A（本優化） |
|---|---|---|
| rulebook 在哪算 | engine 圖**內**（4 個下採樣節點） | engine **之前**的 preprocessing |
| host sync | 4 次，卡在 TRT 圖中間 → trainStation、整個 engine 被切碎 | 4 次，在 preprocessing；TRT engine 變成不被切斷、可 CUDA-graph 的一整段 |
| engine | 6 trainStation、~30% GPU idle | 0 trainStation |

（無法把 4 次合併成 1 次：幾何 cascade 的相依性不允許。設計文件 §4.1 曾寫「single sync」是設計意圖，不是實作的樣子。）

### 7.4 inference（`bevfusion_trt.cpp`）

engine 建立時（`initTrt`）：
- `addSparseRulebookNetworkIO`：宣告 16 個 rulebook input（名稱 `rulebook/<tag>/<slot>`）。
- `addSparseRulebookProfileDims`：給每個 input 設 `[min,opt,max]`（max = 256000 上限）。

每幀（`preProcess` 之後、`enqueueV3` 之前）：
- `bindSparseRulebookAddresses`：`setTensorAddress` 把 16 個 input 綁到 §7.2 的穩定 buffer（位址固定，只需綁一次的概念）。
- `setSparseRulebookInputShapes`：`setInputShape(rulebook/<tag>/<slot>, N_i)` 用 §7.2 算出的 `stage_counts_` 設形狀。

然後 `inference()` 跑 `network_trt_ptr_->enqueueV3(stream_)`。engine 內的 `ImplicitGemm` 就用綁進來的 rulebook，按 §4 的方式做 gather-GEMM，並從 `pair_mask` 的 `N` 維決定輸出大小。

### 7.5 已做的小優化（靜態 workspace layout）

`computeStage` 原本每幀呼叫 `get_handcrafted_max_act_out(num_in,...)` 來切 workspace，導致每幀 layout 略有變動。已改為使用建構時算好的**最壞情況常數** `max_act_out_theory_worst_`（max over stages at `N`）。因為 `num_in ≤ N` 且該函式對輸入數非遞減，最壞值永遠 ≥ 每幀需求，buffer 在建構時也用同一最壞值配置 → 一定 fit。效果：移除每幀一次 host 端呼叫、layout/offset 變成 frame-invariant。屬 host-side 清理，不影響 GPU 工作量。

> 啟用方式（在實際載入的 ml-package param，預設 `~/autoware_data/bevfusion/ml_package_bevfusion_lidar.param.yaml`）：
> ```yaml
> sparse_remove_trainstation: true
> ```

---

## 8. 加速最終從哪裡來（總結）

1. **不是減少計算量**：rulebook 那 4 次建構（含 unique/sort）baseline 本來就要算，只是算在 engine 圖內。Route A 把它搬到 preprocessing，總工作量幾乎不變。
2. **真正省的是「去碎片化」**：4 個 mid-graph 的 `DeviceToShapeHostCopy` 不再切斷 TRT engine → engine 變成一整段連續執行、可整段 CUDA Graph 化 → 消掉 profile 上 ~30% 的 GPU idle 氣泡。
3. **量測結果**（見 [`34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md`](34_README_SPCONV_DDS_TRAINSTATION_REMOVAL.md) §8）：
   - 結構面（與硬體無關、最具決定性）：**trainStation 6 → 0**，mAP 不變（fp16 噪音內）。
   - 強 dGPU 上：Sparse Encoder `9.37 → 8.00 ms`（約 −15%）。
   - 板上目標（原 profile 有 ~30% idle 來自 6 個 trainStation）→ 相對效益預期比 15% 更大。

### 誠實的 caveats

- Python 原型量到的 −15% 沒有把 rulebook precompute 的時間算進「Sparse Encoder」那一格（那格只算 TRT enqueue）。完全公平的端到端數字要把 precompute 成本算進 preprocessing。最具決定性、與硬體無關的結論是結構面：**trainStation 6→0 且 mAP 不變**。
- preprocessing 仍有 4 個循序 D2H（§7.3）；要再壓需要 spconv 提供「計數留在 device、下一 stage 用 device-side bound 啟動」的 API，屬 library 級改動，不是 runtime 小調。

---

## 9. 檔案對照表

| 角色 | 檔案 |
|------|------|
| Export 圖手術 | `deployment/projects/bevfusion_l/export/onnx_remove_trainstation_dds.py`（`remove_trainstation_dds`、`rulebook_input_name`） |
| Export pipeline 串接 / merge | `export/component_builder.py`（sparse `post_transforms`）、`export/transforms.py`（`merge_split_sparse_dense_onnx`） |
| Deploy 設定（`spconv_remove_trainstation` 旗標） | `config/deploy_config_split_fp16_remove_trainstation.py` |
| 16 個 rulebook input 的 TRT profile（由 flag 推導） | `config/component_layout.py`（`add_rulebook_input_profiles`） |
| AWML eval runtime（Python 參考實作） | `io/sparse_rulebook_inputs.py`、`inference/tensorrt_inference_pipeline.py` |
| 車上 runtime（C++/CUDA） | `autoware_bevfusion/preprocess/sparse_rulebook_precompute.{hpp,cu}`、`lib/bevfusion_trt.cpp` |
| spconv rulebook 實作 | `spconv_cpp/.../SpconvOps/SpconvOps_get_indice_pairs_implicit_gemm.cc` + `SparseConvIndicesKernel/*` |
| spconv GEMM 實作 | `spconv_cpp/.../ConvGemmOps/ConvGemmOps_implicit_gemm.cc` |
| spconv op 的 PyTorch / ONNX 包裝 | `projects/SparseConvolution/sparse_functional.py`、`sparse_conv.py` |

---

## 10. 一頁回顧

- BEVFusion(LiDAR)：點雲 → voxelize → **sparse encoder** → dense backbone → head。
- 稀疏卷積 = gather-GEMM；spconv 拆成 **幾何(`GetIndicePairs`) + 特徵(`ImplicitGemm`)**。
- `GetIndicePairsImplicitGemm`：吃座標，吐 `out_indices / pair_fwd / pair_mask / mask_argsort / num_act_out`；內部 stage1(候選)→unique(得 N，**DDS 來源**)→stage2(填 rulebook)→sort。
- `ImplicitGemm`：吃 `features+filters+rulebook`，用 pair_fwd gather、pair_mask 跳 tap、mask_argsort 排 tile，做 masked GEMM；`N` 由 `pair_mask` 的 shape 帶入（所以 num_act_out 可省）。
- DDS 優化：因為 rulebook 只依賴幾何 → 匯出時刪 4 個下採樣節點、把 rulebook 變 graph input → trainStation 6→0。
- autoware_bevfusion：preprocess 用 `SparseRulebookPrecompute` cascade 算好 4 個 rulebook（仍 4 次循序 D2H，但移出 engine）→ inference 綁 input + `setInputShape` + `enqueueV3`。
- 加速來源：**不是少算，而是讓 engine 不再被 DDS 切碎** → 消掉 ~30% GPU idle、可 CUDA-graph 化。
