# AWML spconv INT8 加速新計畫（重新檢視 New3D 與 CUDA-BEVFusion 後）

本文件是在重新比對以下三條路線後，為 **AWML BEVFusion sparse encoder INT8** 制定的新計畫：

- **AWML 現況**：`TensorRT + ImplicitGemmInt8 plugin`
- **開源 New3D**：`libraries/New3DSparseConvolution`
- **closed-source CUDA-BEVFusion**：`Lidar_AI_Solution/CUDA-BEVFusion`

重點不是再重述「哪裡理論上可以更快」，而是要回答：

1. 為什麼先前第 8.1 節的方向 **技術上合理**，但在 AWML 上 **實測沒有加速**？
2. New3D / CUDA-BEVFusion 真正吃到的 sparse INT8 加速，核心來自哪些地方？
3. 以 AWML 現有架構來看，**接下來應該優先做什麼**，以及 **不該再花時間在什麼**？

---

## 1. 新結論摘要

### 1.1 第 8.1 節不是錯，但優先級被高估

AWML 先前把 `launch_compute_w_scales`、`launch_quantize_weights_per_channel`、`launch_fuse_output_scale_into_gemm_scale_bias` 從每幀搬到 cache / constant-only path，**方向本身正確**，而且已成功實作。

但實測 **sparse encoder 幾乎沒有加速**，代表：

- 這三段 kernel 在 AWML 的 steady-state 成本 **不是主要瓶頸**
- 真正的瓶頸更可能在：
  - **`implicit_gemm` 本體**
  - **indice pair generation / sort**
  - **跨 layer 的 FP16 邊界 / activation 重複量化**
  - **多個 plugin / op 邊界造成的 launch 與 memory traffic**

因此：  
**第 8.1 應保留為「已驗證但收益有限」的優化，而不是後續主軸。**

### 1.2 AWML 和 New3D / CUDA-BEVFusion 最大差異，不是只有權重量化時機

真正關鍵差異是：

- **New3D / CUDA-BEVFusion 的 sparse encoder 是一個自訂 spconv engine**
- **AWML 的 sparse encoder 是 TensorRT graph + custom plugin 邊界**

這個架構差異直接影響：

- 能不能把 **activation 持續維持 INT8**
- 能不能在 builder 階段做 **graph-level fusion**
- 能不能在 sparse runtime 內做 **sort / pair / scatter / add / relu** 的整體協調

### 1.3 目前 AWML 不應再把「插 Q / QDQ」當主路線

文件與實驗都指出：

- `custom plugin` **不會自動吃到 Q/DQ fusion**
- AWML 的 `ImplicitGemmInt8` 目前 **輸出是 FP16**
- 因此在 ONNX 裡額外插 `QuantizeLinear`，多半只是把量化位置搬動，**不等於建立真正的 INT8 activation chain**

所以：

**「在現有 Path B 上靠 ONNX 插 Q 來拿明顯加速」不是值得繼續投入的主方向。**

---

## 2. 三條路線的本質差異

### 2.1 AWML：TensorRT + `ImplicitGemmInt8`

目前 AWML Path B 的特徵如下：

- sparse conv 仍是 **TensorRT custom plugin**
- plugin 內部呼叫 `ConvGemmOps::implicit_gemm`
- plugin **I/O 邊界是 FP16**
- INT8 主要存在於：
  - plugin 內部的 feature / weight buffer
  - `implicit_gemm` 的 A / B tensor

這代表 AWML 雖然「有跑到 INT8 GEMM」，但 **沒有天然得到整段 sparse graph 的 INT8 chain**。

### 2.1.1 為什麼可判定 AWML 目前是「層間 FP16，層內再量成 INT8」

這件事不是推測，而是可直接從目前 `ImplicitGemmInt8` plugin 實作讀出來：

- `supportsFormatCombination()` 對 `IN_FEATURES` 要求 **`kHALF`**
- `getOutputDataTypes()` 回傳 **跟 input 相同型別**，而目前 input 是 `FP16`
- `enqueue()` 一進來就呼叫 `launch_quantize_features(...)`
- `implicit_gemm` 的 `output_dtype` 明確設為 **`tv::float16`**

這代表目前 AWML Path B 的實際資料流是：

```text
上一層輸出 FP16
  -> 中間 Relu / Cast 仍是 FP16
  -> 下一層 ImplicitGemmInt8 進 plugin 後再做 FP16 -> INT8
  -> implicit_gemm 內部以 INT8 A/B 計算
  -> epilogue / output buffer 回寫 FP16
```

因此，在目前 AWML 這條 sparse 路徑上，**大多數 `ImplicitGemmInt8` 都會重複做一次 feature 的 `FP16 -> INT8` 量化**，而不是 layer-to-layer 直接走 **`INT8 -> INT8`**。

這也解釋了為什麼：

- 只把權重量化從每幀搬到初始化
- 但不處理 activation 邊界

通常拿不到明顯 sparse encoder 加速。

### 2.2 開源 New3D：自訂 `spconv::Engine`

`libraries/New3DSparseConvolution` 的關鍵不是只有「configure 一次量好權重」，還包括：

- **INT8 / FP16 precision 是每個 node 的屬性**
- activation 在 layer 間可保持 **INT8**
- **Add / Relu / ScatterDense / Transpose** 能在 engine build 階段做結構性處理
- `SparseConvolution::forward` 內明確有：
  - **若 input 已是 INT8 就直接用**
  - **INT8 mode 下跳過某些排序成本**（`do_sort = !int8_inference_`）

這些加總起來，才是它真正的 sparse INT8 優勢。

### 2.2.1 New3D 裡 `INT8 -> INT8` 是怎麼成立的

New3D 與 AWML 最大的差別之一，是 **每個 node 都有 `input_precision` / `output_precision` 概念**，而不是被 TensorRT custom plugin 的固定 I/O 型別綁住。

以 `SparseConvolution` 為例：

- `configure()` 會依 `output_precision_` 決定 `out_features_` 的 dtype  
  - `output_precision_ == Int8` -> `out_features_` 配成 `tv::int8`
  - 否則才是 `tv::float16`
- `forward()` 會依 `input_precision_` 檢查輸入型別  
  - 若輸入已是 `Int8`，就**直接用**
  - 只有 `input_precision_ == Float16` 時，才做 `FP16 -> INT8`

也就是說，New3D 的卷積層不是固定：

```text
FP16 in -> plugin內量化 -> INT8 GEMM -> FP16 out
```

而是可以是：

```text
INT8 in -> INT8 GEMM -> INT8 out
```

只要該層的 `input_precision` / `output_precision` 被設成 `Int8`，中間 activation 就可以在 engine 內真的保持 INT8。

更重要的是，其他 sparse op 也有對應支援：

- `SparseRelu`
  - `int8_inference_ == true` 時直接跑 `relu_kernel<int8_t>`
  - output dtype 繼承 input dtype，所以可維持 INT8
- `SparseAdd`
  - 支援 `Int8 + Int8 -> Int8`
  - 也支援 `FP16 + FP16 -> Int8`

這表示在 New3D 裡，真正的 INT8 chain 不只是 conv 本身，而是：

```text
SparseConv(INT8)
  -> ReLU(INT8)
  -> Add(INT8)
  -> 下一層 SparseConv(INT8)
```

這正是 AWML 目前 sparse Path B 還做不到的事。

### 2.3 closed-source CUDA-BEVFusion：不是「TRT sparse plugin」思路

`Lidar_AI_Solution/CUDA-BEVFusion` 的 sparse encoder 不是把 sparse conv 交給 TensorRT plugin，而是：

- 匯出專用 ONNX
- 交給 **自訂 spconv engine / `libspconv.so`**
- camera / dense / head 才走 TensorRT

而且它在 export 時明確做：

- 大多數 sparse layer 設為 `precision="int8"` / `output_precision="int8"`
- 只在某些邊界（例如 first / last conv）保留 FP16

這表示 closed-source 真正的思路是：

**讓 sparse encoder 自成一個可控的 INT8 engine，而不是把每個 sparse layer 當成 TRT plugin。**

### 2.4 什麼叫「獨立 sparse engine」

這裡的「獨立 sparse engine」不是指：

- 換一套完全不同的 GEMM kernel
- 或不再使用 `spconv` / `cumm`

而是指：

- **整段 sparse encoder 不再交給 TensorRT 逐層排程**
- 而是由 **`spconv::Engine` / 自訂 sparse runtime** 自己接管 graph、tensor、dtype 與 execution order

在 closed-source CUDA-BEVFusion 中，`lidar-scn.cpp` 直接做的是：

- 用 `spconv::load_engine_from_onnx(...)` 載入 sparse ONNX
- 將 voxelization 輸出直接餵給 `native_scn_`
- 呼叫 `native_scn_->forward(stream)`

也就是說，對 sparse encoder 而言：

- TensorRT 不是總調度器
- sparse graph 是由 `spconv::Engine` 自己執行

### 2.4.1 它怎麼運作

以開源 New3D 為例：

- `onnx-parser.cpp` 會把 ONNX node 解析成 engine builder 的 node
- builder 為每個 node 建立：
  - op type
  - input / output tensor
  - precision / output_precision
  - attribute（例如 kernel size、stride、dynamic range）
- `EngineImplement::forward()` 會沿著 node list 依序呼叫各 operation 的 `forward()`

也就是：

```text
ONNX
  -> 自訂 parser
  -> 自訂 sparse graph / engine
  -> runtime forward
```

而不是：

```text
ONNX
  -> TensorRT network
  -> 每層交給 plugin enqueue
```

### 2.4.2 為什麼它不走 TensorRT plugin

主因不是「TRT 完全不能做 sparse」，而是 sparse encoder 這類 graph 對以下能力非常敏感：

- layer 間 dtype chain（特別是 `INT8 -> INT8`）
- sparse add / relu / pair / scatter 的協調
- graph fusion
- 對每層 precision / output_precision 的控制

若放在 TRT plugin 模式下，常會遇到：

- plugin I/O dtype 受限
- custom plugin 不吃 Q/DQ fusion
- graph 級 sparse fusion 不容易跨 plugin 邊界做

這些限制對 dense op 不是致命，但對 sparse INT8 chain 影響很大。

### 2.4.3 底層還是不是 `ConvGemmOps::implicit_gemm`

**是。**

以開源 New3D 為例，`SparseConvolution::forward()` 最後仍然呼叫：

- `ConvGemmOps::implicit_gemm(...)`

所以差異不在「底層算子換掉了」，而在：

- 誰掌控 graph
- 誰決定每層 input / output precision
- 誰決定 activation 是否可以保持 INT8
- 誰可以做 Add / ReLU / Scatter / layout 等 graph-level 優化

一句話說：

**獨立 sparse engine 不是不用 `implicit_gemm`，而是不用 TensorRT 當 sparse graph 的總調度器。**

### 2.4.4 對 AWML 的實際意義

這個 distinction 很重要，因為它代表：

- 若 AWML 只留在「TRT + per-layer sparse plugin」架構
- 即使底層也調到同一個 `ConvGemmOps::implicit_gemm`

仍然**不等於**可以自然複製 New3D / CUDA-BEVFusion 的整體加速效果。

真正差距常常來自：

- layer 間真正的 `INT8 -> INT8`
- 更少 plugin / op 邊界
- sparse runtime 自己掌握 pair / sort / add / relu / scatter

而不只是單一 conv kernel 的 math precision。

---

## 3. 對第八章的重新判定

### 3.1 8.1 權重量化前置

**判定：保留，但降級為「必要清理，不是主要加速來源」**

原因：

- 技術上與 New3D 一致，沒有錯
- 已實作後仍無明顯加速，說明它不是主要 hotspot

應修改原先表述：

- 不再描述成「最大收益切入點」
- 改為「低風險、可先完成的 housekeeping」

### 3.2 8.2 圖級融合

**判定：仍然正確，而且重要性上升**

New3D / CUDA-BEVFusion 真正吃到的一部分收益，來自 builder / engine 內能處理：

- Add + Relu
- ScatterDense + layout 處理
- 更少 plugin / op 邊界

AWML 在 TRT graph 上天然吃虧，因此：

- 若不換架構，就只能靠 **更大的 fused sparse plugin**
- 或者直接改為 **獨立 sparse engine**

### 3.3 8.4 層間 dequant / requant

**判定：這一節應升級成後續主軸之一**

但要修正關鍵說法：

- 問題不只是「中間回到 FP16」
- 更重要的是：**只靠 ONNX Q/DQ 無法解決**

要真的減掉 repeated feature quant，至少要滿足其中一條：

1. plugin 支援 **INT8 output**，建立真實 activation chain  
2. 將多層 sparse op 合進更大的 plugin / engine  
3. 整段 sparse encoder 改為 New3D / CUDA-BEVFusion 式 engine

### 3.4 8.6 是否改為整段 sparse engine

**判定：重要性明顯上升**

在目前實驗結果下，這條路不再只是「架構變動太大所以先不碰」，而是：

- 如果目標是 **顯著** 拉開 sparse INT8 與 FP16 差距
- 而不是只撿 0.x ms

那麼這條路很可能才是最符合 New3D / CUDA-BEVFusion 事實路徑的方案。

---

## 4. 目前已驗證的事實

### 4.1 已確認有效，但收益小

- `ImplicitGemmInt8` constant-only weight cache 已生效
- log 可看到 `constant-only cache mode active`
- 但 sparse encoder latency 幾乎不變

### 4.2 已確認不值得再沿用的方向

- 只靠 ONNX 插 `QuantizeLinear` / QDQ，**不會自然變成 plugin 加速**
- 就算 plugin 現在支援 `INT8` input，若上游仍是 `FP16` output + graph 中間插 Q，收益也很有限
- 這條路目前不應當作主線

### 4.3 尚未驗證但高機率重要

- **INT8 mode 下的 pair-gen / sort 行為**
- sparse op 邊界的 launch / memory traffic
- `Add + Relu` 等 elementwise 鏈的融合收益

---

## 5. 新的優先級計畫

## Priority A：先確認真正瓶頸

### A1. 做 sparse encoder 專項 profile

目的：

- 分離 `implicit_gemm`、pair-gen、sort、elementwise、scatter 的時間
- 不再只看總 latency

必做項目：

- steady-state 測速（warmup 與 measured 分離）
- TensorRT layer timing
- Nsight Systems
- 若可行，對核心 sparse kernel 做 Nsight Compute

成功標準：

- 能明確回答「時間前 3 名在哪裡」

### A2. 驗證目前是否有 sort 成本

New3D 在 INT8 路徑會做：

- `do_sort = !int8_inference_`

AWML 目前的 TensorRT + plugin 路線未必有對應優化。  
若這段在 AWML 還存在，可能比搬權重量化更重要。

---

## Priority B：若留在 TensorRT 架構，優先做的不是 QDQ，而是更大的融合

### B1. 融合 Add + Relu / 小型 sparse elementwise 鏈

目標：

- 減少 sparse 段中的 plugin / layer 邊界
- 降低 launch 與中間張量讀寫

優先觀察：

- residual block 的 Add / Relu
- 是否存在可以與 `ImplicitGemmInt8` 合併的固定 pattern

建議作法：

- 先從最常見的一個 block pattern 做單一 fused plugin
- 不一次重寫整個 sparse graph

### B2. 只有在 plugin 能輸出 INT8 時，才再談 activation chain

若仍維持：

- input 可以是 INT8
- output 仍固定 FP16

則「每層省 feature quant」的上限仍然受限。  
所以除非 plugin 往 **INT8 output** 前進，否則不建議再在 ONNX transform 上投入大量 Q/DQ chaining 工作。

### B3. 若要硬做 `ImplicitGemmInt8` 的 `INT8 in/out`，需要面對什麼

理論上，AWML 的 `ImplicitGemmInt8` 並不是完全不可能改成 `INT8 in/out`。  
但這不是小改，而是要同時解決 **TensorRT plugin 限制** 與 **sparse graph chain** 兩個層面的問題。

#### 必改項

- 讓 plugin 支援 **`INT8` output**
  - `supportsFormatCombination()`
  - `getOutputDataTypes()`
  - `enqueue()` 的 `out_features` dtype / `output_dtype`
- 讓 plugin 的 feature tensor 不再綁死為目前的 **2D `[N, C]` FP16 I/O** 模式
- 若要遵守 `spconv/docs/TENSORRT_INT8_GUIDE.md` 的限制，需處理：
  - **INT8 plugin tensor 維度需 >= 3**
  - 因此可能要把 sparse feature 包裝成例如 **`[1, N, C]`**
  - 或重做 symbolic / parser / plugin 介面，讓 TRT 能接受你需要的 sparse feature 形式

#### 高風險項

- 不能只改 conv plugin，還要一起處理：
  - `Relu`
  - `Add`
  - `Cast`
  - residual path
  - block 邊界
  - `conv_out`
- 若這些中間節點仍是 FP16，conv 之間仍會掉回 FP16，INT8 chain 就斷掉
- ONNX transform 也要能表達：
  - 哪些 layer 是 `INT8 -> INT8`
  - 哪些邊界必須回 FP16 / FP32

#### 不建議項

- 只靠插 `Q` / `QDQ` 想讓 custom plugin 自動吃到 explicit INT8 fusion
- 只改 `ImplicitGemmInt8` 單層，卻不處理中間的 sparse elementwise op

#### 實務上的二選一

若要真正建立可觀的 sparse INT8 chain，通常是二選一：

1. **方案 A：維持 TRT 架構，但做完整 sparse INT8 chain**
   - `ImplicitGemmInt8(INT8 out)`
   - `SparseReluInt8`
   - `SparseAddInt8`
   - 必要邊界才回 FP16
2. **方案 B：不要再用 TRT plugin graph 拼 sparse 段**
   - 直接改走 **New3D / CUDA-BEVFusion** 那種 **獨立 sparse engine**

我的判斷是：

- **理論上可做**
- 但在 AWML 現有 TRT Path B 上，**成本高且風險中高**
- 若目標只是再撿一點 latency，優先級仍應低於：
  - `sort / pair-gen profiling`
  - `sparse fused add / relu`
- 若目標是複製 New3D 那種真正的 INT8 chain 效果，  
  **改成獨立 sparse engine 很可能比硬拗 TRT plugin 更合理**

---

## Priority C：中長期高收益方案 = sparse encoder 脫離 TRT plugin 邊界

### C1. 評估「整段 sparse encoder = 專用 engine」

這是最接近 New3D / CUDA-BEVFusion 成功模式的方案。

目標：

- sparse encoder 使用獨立 spconv engine
- dense / camera / head 繼續交給 TensorRT

潛在收益來源：

- 真正的 INT8 activation chain
- builder 階段圖級融合
- sort / pair / add / relu / scatter 的整體優化
- 避免每層 custom plugin 邊界

代價：

- 整合與維護成本高
- 要重做 sparse / dense 交界設計
- 需要重新驗證部署鏈

但若目標是明顯超過目前 Path B 的收益，這條路線最符合現有對照結果。

### C2. 若把 Lidar CUDA-BEVFusion / New3D 當 baseline，AWML 要套進去有多難

若把：

- `CUDA-BEVFusion` 的 `lidar-scn` 路線
- 或 `New3DSparseConvolution` 的 `spconv::Engine`

當作 AWML sparse encoder 的 baseline，**技術上是合理的比較基準**，但導入 AWML **並不輕鬆**。

#### 為什麼它是好的 baseline

因為這兩條路線都已經證明：

- sparse encoder 可以不是 TRT plugin graph
- layer 間可以維持真正的 `INT8 -> INT8`
- graph-level sparse fusion 與 runtime 協調可以由 engine 自己掌控

所以如果 AWML 想驗證「TRT plugin 架構是不是天花板」，這是很好的 baseline。

#### 為什麼導入 AWML 困難

AWML 現在的部署鏈是：

- PTQ / export / ONNX transform
- split sparse / dense
- sparse 走 TRT plugin
- dense / head 繼續 TRT

若改成 Lidar CUDA-BEVFusion / New3D 風格，至少要重做或重接以下部分：

- sparse ONNX 的 node type / attribute 契約
  - AWML 現在是 `autoware::ImplicitGemm`
  - New3D / CUDA-BEVFusion 用的是 `SparseConvolution` / `Add` / `Relu` / `ScatterDense` 等自訂 parser 契約
- sparse encoder runtime
  - 需要新增或接入 `spconv::Engine`
- sparse / dense 交界
  - `lidar_bev` 的 shape、dtype、layout、memory ownership
- export pipeline
  - 不能再只靠現在的 `ImplicitGemm -> ImplicitGemmInt8` transform
- evaluate / deployment orchestration
  - 需要讓 sparse 與 dense 走兩套不同 runtime，但仍共用整體 pipeline

#### 導入難度判定

- **做為研究 baseline / proof-of-concept**：可做，且很有價值
- **直接替換進 AWML 正式部署鏈**：中高難度
- **短期內想靠這條路立刻撿到 1~2ms**：不現實

#### 建議做法

若要使用這條 baseline，建議不要一開始就完整替換 AWML，而是：

1. 先做 **獨立 sparse encoder micro-benchmark**
   - 同一份 voxel input
   - 比較：
     - AWML Path B
     - New3D / CUDA-BEVFusion sparse engine
2. 先只比 sparse encoder latency / output consistency
3. 只有在確定 gain 顯著時，才評估整合回 AWML split deployment

這樣能先回答最關鍵的問題：

**AWML 的瓶頸到底是 plugin 架構本身，還是目前實作細節。**

---

## 6. 不建議再投入的方向

以下方向目前不建議繼續投入太多時間：

- 再進一步優化「權重量化從每幀搬到初始化」  
  原因：已驗證收益很小

- 單純在 ONNX graph 上插更多 `QuantizeLinear` / QDQ  
  原因：custom plugin 不吃 Q/DQ fusion，且目前 plugin output 仍是 FP16

- 把第 8.1 當成 sparse INT8 的主要突破口  
  原因：實測已否定其高收益假設

---

## 7. 建議執行順序

### 短期（1）

- 完成 sparse encoder 專項 profile
- 明確確認：
  - `implicit_gemm` 是否絕對主導
  - pair-gen / sort 是否顯著
  - Add / Relu / scatter 是否可觀

### 短期（2）

- 若 profile 顯示 elementwise / 邊界成本不小：
  - 先做一個 **Add+Relu fused sparse plugin** 原型

### 中期（3）

- 若 profile 顯示 sparse 段仍主要被 plugin 邊界與 FP16 邊界限制：
  - 開始做「**獨立 sparse engine**」可行性驗證

### 中期（4）

- 只有在確定要繼續留在 TRT sparse plugin 路線時：
  - 再評估 plugin 支援 **INT8 output** 的試點設計

---

## 8. 建議文件關係

建議把原本 `README_NEW3D_LIDAR_OPEN_SPCONV.md` 第八章視為：

- 「第一次分析與假設」

而把本文件視為：

- 「**根據實作與實測回饋後的修正版計畫**」

兩者應並存，避免遺失已驗證過的脈絡。

---

## 9. 核心判斷一句話

**AWML 已證明：把權重量化前置化不等於 sparse INT8 就會快。**  
若要真正複製 New3D / CUDA-BEVFusion 的收益，主戰場不是再調 `8.1`，而是：

- **sort / pair / graph fusion**
- **減少 plugin 邊界**
- 或直接走 **獨立 sparse engine**

---

## 10. Priority A 實測結果（`bevfusion_sparse.engine`）

執行條件：
- Engine: `work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine`
- Input: `info/t4dataset_j6gen2_base_infos_test.pkl` sample idx=0 → `voxels=(70747, 10, 5)`, `coors=(70747, 3)`
- Warmup 20 / iterations 200、per-layer `IProfiler` 打開
- 工具：`deployment/projects/bevfusion/benchmark/profile_sparse_encoder.py`（**v2 版本**：修掉 `ImplicitGemm_int8` 被誤歸類為 `implicit_gemm_fp` 的 regex bug）
- 詳細使用見 `16_PRIORITY_A_PROFILING_USAGE.md`

### 10.1 Steady-state GPU latency（event-based）

| metric | value |
|---|---|
| mean | **15.320 ms** |
| std  | ± 0.569 ms |
| median | 15.289 ms |
| min / max | 14.333 / 18.542 ms |
| n | 200 |

這是 sparse engine 自身 `execute_async_v3` 從 start → end 的完整 GPU 時間（不含 pre/post H2D / D2H）。  
跑到跑之間有 ~0.9 ms 的漂移（前次同 engine 同輸入量到 14.447 ± 0.246 ms），屬於系統/排程雜訊；以下比例分析用目前這輪。

### 10.2 Op-bucket breakdown（mean per-iteration layer sum）

| bucket | count | sum_ms | % of layer-sum |
|---|---:|---:|---:|
| **pair_gen**（`GetIndicePairsImplicitGemm`） | 25 | **7.355** | **51.70%** |
| **implicit_gemm_int8**（真正的 INT8 conv） | 20 | **4.322** | **30.38%** |
| **implicit_gemm_fp**（⚠️ 仍為 FP 的 conv） | **1** | **0.062** | 0.44% |
| relu | 20 | 0.817 | 5.74% |
| cast | 8 | 0.196 | 1.38% |
| layout | 2 | 0.005 | 0.04% |
| other | 101 | 1.469 | 10.32% |
| **layer-sum 合計** |  | **14.226** | 100% |

Sanity check：`layer-sum 14.226 ms` vs. `event total 15.320 ms`，delta = **+1.094 ms**（約 7.1%，屬正常的 plugin launch / workspace memcpy / TRT runtime overhead；驗證兩個度量彼此 consistent）。

#### 重要觀察：sparse engine 並非 100% INT8

Classifier 修正後浮現一個之前被隱藏的事實：**有 1 個 conv layer 仍然是 FP 版本的 `ImplicitGemm`**（不是 `ImplicitGemm_int8`），貢獻 0.062 ms / 0.44%。

- 時間佔比不高（<0.5%），不是當前熱點
- 但這代表 ONNX → TRT transform 的 `ImplicitGemm → ImplicitGemmInt8` 替換**沒有 100% 覆蓋**；應該另外用 Nsight / `trtexec --dumpLayerInfo` 找出是哪一層（很可能是 `conv_out`、residual branch、或某個 stride conv），確認是「刻意保留 FP」還是「transform 漏掉」
- 若屬後者，修好後這條會進一步驗證 INT8 活化鏈是否「全程 INT8」；若屬前者（如 New3D 就會把最後一層 `conv_out` 留給 dense 側去消化），就要在文檔上標註清楚

→ 追蹤建議：`trtexec --loadEngine=bevfusion_sparse.engine --dumpLayerInfo --profilingVerbosity=detailed` grep 出唯一一個沒有 `_int8` 後綴的 `ImplicitGemm` layer name。

### 10.3 Block roll-up

| block | count | sum_ms | % |
|---|---:|---:|---:|
| conv_input | 5 | 0.805 | 5.66% |
| conv_out | 4 | 0.497 | 3.49% |
| other（encoder_layer1~4 + 元件） | 168 | 12.925 | 90.85% |

encoder layer 主體（~91%）集中在 `encoder_layer{1..4}`，對應五個解析度階段；入/出口 block（conv_input + conv_out < 10%）對應的是 shape transition 而非運算熱點。後續要優化要瞄準 encoder_layer 內部。

### 10.4 Top layers（前 15）

| # | mean_ms | bucket | layer |
|---:|---:|---|---|
| 1 | 0.862 | other | `[trainStation1]`（Myelin 融合 region，非單一 op）|
| 2 | 0.847 | pair_gen | `encoder_layer1.2.0 / GetIndicePairsImplicitGemm`（降採樣 stride conv）|
| 3 | 0.643 | pair_gen | `encoder_layer2.2.0 / GetIndicePairsImplicitGemm`（降採樣 stride conv）|
| 4 | 0.521 | pair_gen | `encoder_layer3.2.0 / GetIndicePairsImplicitGemm`（降採樣 stride conv）|
| 5 | 0.424 | pair_gen | `conv_out.0 / GetIndicePairsImplicitGemm`（出口）|
| 6 | 0.405 | pair_gen | `encoder_layer1.0.conv1 / GetIndicePairsImplicitGemm` |
| 7 | 0.405 | pair_gen | `conv_input.0 / GetIndicePairsImplicitGemm`（入口）|
| 8–13, 15 | 0.35–0.39 | pair_gen | 其餘 `encoder_layer{1,2}` 的 `conv1 / conv2` |
| 14 | 0.366 | **implicit_gemm_int8** | `conv_input.0 / ImplicitGemm_int8`（唯一擠進 Top 15 的實際 conv）|

**關鍵觀察**：Top 15 裡 **13 個是 `pair_gen`**，1 個 Myelin fused region，只有 1 個 `implicit_gemm_int8`。INT8 conv 已經被 plugin 做得非常快（平均每個節點 ~0.22 ms），瓶頸已完全移到 pair-gen / sort 上。

### 10.5 A2 成本驗證（pair-gen / sort）

**假設檢驗結論：✅ 假設成立。**

| 指標 | 閾值（`README_PLAN` §2 設定） | 實測 | 結論 |
|---|---|---|---|
| pair-gen / sort 佔 sparse engine 比例 | ≥ 20%（才值得優化） | **51.70%** | 遠超門檻 |
| pair-gen / sort 絕對時間 | — | **7.355 ms/frame** | — |
| Top 熱點主成分 | pair_gen 應進入 Top 5 | Top 5 全部是 `pair_gen` 變體（第 1 名是 Myelin region） | 一致 |

對照 New3D 開源版 `do_sort = !int8_inference_` 的策略：INT8 推論時直接**關閉 argsort**（只做排序才需要的穩定性保證，在 INT8 inference 階段被認為可以省略）。假設 AWML 可做相同處理，理論上整個 `pair_gen` bucket 應該能壓縮到原本的 ~40–60%（因為 argsort 佔 pair-gen 時間很大一部分），也就是：

- 樂觀（argsort 佔 pair-gen 60%）：省 **~4.4 ms/frame** → 15.32 ms → **~10.9 ms**（-29%）
- 保守（argsort 佔 pair-gen 30%）：省 **~2.2 ms/frame** → 15.32 ms → **~13.1 ms**（-14%）

需用 Nsight Compute 確認 argsort 在 `GetIndicePairsImplicitGemm` plugin 內部的實際佔比，命令見 `16_PRIORITY_A_PROFILING_USAGE.md` 的 Nsight 一節。

補充：已完成一輪 `nsys stats --report cuda_kern_exec_sum` 的 recurring-kernel 粗分桶（排除一次性初始化 kernel）：

| recurring kernel bucket | share |
|---|---|
| `conv` | **43.56%** |
| `pair_gen_non_sort` | **33.11%** |
| `sort` (`DeviceMergeSort*`) | **8.94%** |
| `other` (Myelin / reformat / reduce) | **9.59%** |
| `quant_misc` | **4.81%** |

其中 `pair_gen_non_sort + sort = 42.05%`，說明 **A2 方向依然成立**: pair-gen / sort 整條鏈非常重，值得持續投入；但也顯示 **`argsort`/merge-sort 本身不是唯一主戰場**。目前可保守解讀為：`sort` 單獨約佔 recurring sparse kernels 的 **8.94%**，約佔 `pair_gen + sort` bucket 的 **21.3%**，折算約 **1.25 ms/frame**（若含 warmup 共 40 次）到 **1.66 ms/frame**（若只以 30 次量測迭代估算）。因此 `do_sort = !int8_inference_` 仍可能帶來可見收益，但不應直接把整個 `pair_gen` bucket 都視為 `argsort` 可省時間；若要精準回答 `argsort / scan / unique` 各佔多少，仍需 `ncu` 進一步拆解。

### 10.6 INT8 活化鏈（A1 連帶觀察）

- **`quant_dquant` bucket 空**（count=0）：engine 裡沒有獨立 Q/DQ layer，說明 transform 已把 INT8 邊界**摺進** `ImplicitGemm_int8` plugin；活化不會在 layer 之間反覆 QDQ
- **`cast` bucket 極小**（0.196 ms, 1.38%）：表示 INT8 ↔ FP16 邊界已壓得很乾淨
- **`relu` 不多但不是零**（0.817 ms, 5.74%）：目前 ReLU 還是獨立 layer 而不是 fuse 進 conv epilogue；是 Priority B（TRT 層級 fusion）可以優化的點
- **`add` bucket 空**：residual `Add` 可能被 `other`（Myelin fusion 區塊 `[trainStation1]`）吞掉了；需要 Nsight 展開 region 才能確認
- **仍有 1 個 FP conv** (§10.2)：INT8 鏈**未完全閉合**，需確認是 by design 還是 transform bug

### 10.7 與 `README_PLAN` §2 成功判準的對應

| Priority A 成功判準 | 判定 |
|---|---|
| A1: 能分別量到 pair-gen / implicit_gemm / add / relu / scatter / cast 的時間佔比 | ✅ 完成（§10.2）|
| A2: 明確判定 sort/pair-gen 是否值得投 ≥ 20% 資源 | ✅ **值得投入**（51.70%）|
| Steady-state latency 可 reproduce（±5%） | ✅（std ≤ 3.7% of mean；跨 run mean 漂移 ~6%，標記為待調 perf locking）|
| 能在真實 eval 流程內互相驗證 | 可（`_TRT_SPARSE_PROFILE=1` in-situ overlay，見 `tensorrt.py`）|

### 10.8 下一步建議（基於實測，按 ROI 排序）

1. **Nsight Compute 拆解 pair-gen 內核** ← 最高 ROI（**§10.10 後請調整焦點**）  
   **已關 `do_sort`（§10.9–10.10）後**，`DeviceMergeSort*` 應大幅下降；此時應優先拆解 **剩餘 pair-gen** 裡的 **hash / stage1–2 / unique / scatter** 等非 sort kernel，並對照開源同一套 `SpconvOps` 是否還能換 algo 或減少 launch。若需對照舊 baseline，在 deploy_config 把 `spconv_do_sort = True` 後重 export ONNX → 重 build engine，再跑 `argsort|scan`。  
   → `benchmark/nsys_profile_sparse.sh` + `ncu --set full`（kernel filter 依當前 Top kernel 名單調整）

2. **查出唯一那個 `implicit_gemm_fp` 節點**  
   0.062 ms 雖小但是 INT8 覆蓋率的洞；先確認是哪層，再判斷是有意保留（如 New3D 把 `conv_out` 留到 dense 側）還是 transform 遺漏  
   → `trtexec --loadEngine=... --dumpLayerInfo | jq '.[] | select(.Name | test("ImplicitGemm($|[^_])"))'`

3. **Myelin region `[trainStation1]` 展開**  
   0.862 ms / iter 的 fused region（Top 1）目前看不到內部，不知道裡面是否混著 residual add、QDQ、layout；若展開發現含有 add / layout shuffle，那就是 Priority B (TRT fusion) 的具體切入點  
   → 用 `trtexec --profilingVerbosity=detailed --dumpLayerInfo` + `nsys` NVTX range 比對

4. **跨 encoder_layer 比較**  
   目前 `encoder_layer1.2.0` (0.847 ms) 明顯比 `encoder_layer2.2.0` (0.643 ms) / `encoder_layer3.2.0` (0.521 ms) 大；是因為 encoder_layer1 的 active voxel 數量最多。建議在 optimization 策略上優先處理**解析度最高的兩層**（能覆蓋 ~60% pair-gen 總時間）

5. **Priority B TRT fusion 預估收益（中等 ROI）**  
   relu (0.817 ms) + cast (0.196 ms) + 可能被 `other` 吃掉的 add ≈ 1.0–1.5 ms  
   若能 fuse 進 conv epilogue，大約是 7–10% 的收益，屬於中等回報；但需要改 plugin，工作量中偏高

綜合來說，**pair-gen（含 sort）曾是最大桶**；關閉 sort 後實際端到端約 **−11%**（§10.10），下一刀應轉向 **pair-gen 非 sort 段 + graph 邊界**，見 **§10.11**。

### 10.9 A2 實作：把 `do_sort` 作為 plugin 屬性（FP16/INT8 分開控制）

對照 [traveller59/spconv INT8 guide](https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md#performance-guide) 與 New3D 的 `bool do_sort = !int8_inference_;`（`sparseConvImplicit.cu:368`），在 INT8 inference 階段 pair-mask 的 argsort 可以省略，但 FP16 不應一起跳過。  
AWML 的 INT8 sparse encoder 走 `autoware_tensorrt_plugins::GetIndicesPairsImplicitGemmPlugin`（不是 New3D 的 `libspconv.so`），而**同一個 `.so` 會同時服務 FP16/INT8 engine**，因此不能只用 runtime env 做全域開關（FP16 engine 會被誤關 sort）。

**設計：把 `do_sort` 固化到 ONNX → TRT engine 的 plugin 屬性，由 deploy_config 唯一控制。**  
每個 engine 在 build 時把 `do_sort_i=0/1` 寫進 graph；deploy 時 plugin 自己讀 `params_.do_sort`，不再依賴任何 env var。這樣：

- **同一個 `.so`** 可以服務 FP16（預設 `do_sort=1`）與 INT8（`spconv_do_sort=False` → `0`）。
- 舊 ONNX graph（沒帶 `do_sort` 屬性）自動 fallback 為 `1`，**100% 向後相容**。
- 單一 source of truth：`deploy_config.spconv_do_sort`；不設即 `True`（sort on）。

**交付方式：把 `autoware_tensorrt_plugins` 的改動直接 commit 到 AWML fork；build script 改 clone fork，不再 build-time patch**（先前 patcher-based 版本已移除）。這樣 source of truth 只在 fork，CI / Docker build 完全可重現。

**實作（三個 layer）：**

1. **C++ plugin（固化在 fork `vividf/autoware.universe` 的 `feat/spconv-do-sort-attribute` branch）** — `perception/autoware_tensorrt_plugins` 內三份檔：
   - `include/autoware/tensorrt_plugins/get_indices_pairs_implicit_gemm_plugin.hpp`：`GetIndicesPairsImplicitGemmParameters` 新增 `std::int32_t do_sort{1};`（**預設 1 = sort**）。
   - `src/get_indices_pairs_implicit_gemm_plugin.cpp`：
     - `initFieldsToSerialize()` 多一筆 `("do_sort", &params_.do_sort, kINT32, 1)`。
     - 兩處 `SpconvOps::get_indice_pairs_implicit_gemm(...)` 改為 `..., use_direct_table, static_cast<bool>(params_.do_sort));`。
   - `src/get_indices_pairs_implicit_gemm_plugin_creator.cpp`：
     - 建構子多一筆 `plugin_attributes_.emplace_back("do_sort", nullptr, kINT32, 1)`。
     - `PLUGIN_VALIDATE(num_fields == 11)` 放寬為 `11 || 12`（舊 ONNX 不帶 `do_sort` 時仍可 build）。
     - ONNX 解析迴圈多一個 `attr_name == "do_sort"` 分支；若沒帶就使用結構預設值 `1`。
2. **Python symbolic** `projects/SparseConvolution/sparse_functional.py`：
   - `GetIndicePairsImplicitGemm.symbolic()` 的 `g.op(...)` 寫 `do_sort_i=int(_resolve_do_sort())`，`forward()` 的 `do_sort = _resolve_do_sort()` 亦走同一來源。
   - 模組級 `_do_sort: bool = True` 與 `set_do_sort(value)` 兩個 API；**沒有 env-var fallback**，deploy_config 是唯一入口。不呼叫即維持預設 `True`。
3. **Deploy CLI** `deployment/projects/bevfusion/entrypoint.py`：讀 `deploy_cfg.get("spconv_do_sort", True)`，呼叫 `set_do_sort(value)` 並 log。
4. **Deploy configs**：`deploy_config_split_int8.py` 寫 `spconv_do_sort = False`（INT8 不 sort）；FP16 config 不寫或顯式設 `True`（保留 sort）。
5. **Build 腳本** `projects/BEVFusion/plugins/build_plugin_inside_container.sh`：
   - 新增 env `AUTOWARE_UNIVERSE_REPO`（預設 `https://github.com/vividf/autoware.universe.git`）與既有的 `AUTOWARE_UNIVERSE_REF`（預設改為 `feat/spconv-do-sort-attribute`）；clone 時一併使用。
   - **已移除** 以前的 `patches/add_do_sort_attribute.py` 呼叫與 `AWML_DISABLE_DO_SORT_PATCH` 開關 — fork 本身即為 source of truth。
   - Clone 完會檢查 `get_indices_pairs_implicit_gemm_plugin_creator.cpp` 是否含 `"do_sort"` 欄位；缺失時僅印 WARNING（方便 A/B 對比 stock upstream），不中斷 build。
6. **Dockerfile** `projects/BEVFusion/Dockerfile`：`ARG AUTOWARE_UNIVERSE_REPO` 與 `ARG AUTOWARE_UNIVERSE_REF` 一同透傳給 build script；`COPY plugins/patches` 已移除。

**A/B against stock upstream**（需要對照未改動版本時）：

```bash
AUTOWARE_UNIVERSE_REPO=https://github.com/autowarefoundation/autoware.universe.git \
AUTOWARE_UNIVERSE_REF=main \
bash projects/BEVFusion/plugins/build_plugin_inside_container.sh
```

Build log 會出現 `WARNING: cloned source does NOT expose the do_sort attribute.`，提醒這不是 production INT8 path。

**從 deploy_config 到 runtime 的串接**：

```text
[config]   deploy_config.spconv_do_sort = False
[CLI]      entrypoint.run() → set_do_sort(False)
[export]   GetIndicePairsImplicitGemm.symbolic() 寫 do_sort_i=0 到 ONNX
[build]    TRT builder → plugin creator 解析 "do_sort" → params_.do_sort = 0
[serialize] plugin fields 多一項 "do_sort" → 寫進 .engine
[runtime]  enqueue() 用 params_.do_sort → SpconvOps::get_indice_pairs_implicit_gemm(..., do_sort=false)
```

> **為什麼不用 env var？** Env var 是進程級全域，容易污染其他 precision 的 export，而且 CLI invocation 之間沒有持久紀錄。Deploy_config 是 per-engine 設定，天生就跟 `precision_policy` / `plugin_libraries` 等並列，source of truth 只有一個，git 可追蹤。

**對應情境**：

| 情境 | Deploy config | ONNX `do_sort_i` | Plugin `params_.do_sort` | 是否 sort |
|---|---|---:|---:|---|
| FP16 deploy（預設） | 不設 或 `spconv_do_sort = True` | `1` | `1` | ✅ sort |
| INT8 deploy | `spconv_do_sort = False` | `0` | `0` | ❌ skip |
| **舊 ONNX**（未帶屬性）+ 新 `.so` | — | （缺） | `1`（struct 預設） | ✅ sort（與舊行為一致） |
| 新 ONNX（帶屬性）+ **舊 `.so`** | — | `0/1` | N/A（舊 creator 只收 11 欄） | ⚠️ build 失敗，必須配對新 `.so` |

> **注意**：`GetIndicesPairsImplicitGemmParameters` 新增欄位後，struct 大小改變；**已存在的 `.engine`（用舊 `.so` serialize 的）無法用新 `.so` 直接 load**，必須重 build engine。這是 attribute-based 方案的固有代價，換來的是「每個 engine 自己帶 `do_sort`」的語意乾淨性。

**驗證步驟：**

1. 重新 build image（或直接在 container 內跑 `bash projects/BEVFusion/plugins/build_plugin_inside_container.sh`）。  
   log 應出現 `[build_plugin] Plugin repo: https://github.com/vividf/autoware.universe.git` / `[build_plugin] Plugin ref:  feat/spconv-do-sort-attribute` 以及 `[build_plugin] OK: cloned source exposes do_sort plugin attribute`；若 clone 指向 stock upstream 會印 `WARNING: cloned source does NOT expose the do_sort attribute.`。  
2. 重新 export ONNX：INT8 path 用含 `spconv_do_sort = False` 的 deploy_config（如 `deploy_config_split_int8.py`）；FP16 config 不寫或設 `True`。  
   entrypoint log 會印 `spconv_do_sort: False (baked into ...)`。在 ONNX 上用 `onnx.helper` 掃 `GetIndicePairsImplicitGemm` node 的 `do_sort` 屬性應能看到對應值（見下方 sanity-check snippet）。  
3. 重新 build engine（**必要**）。跑 `benchmark/nsys_profile_sparse.sh`：  
   - INT8：`nsys stats --report cuda_kern_exec_sum` 裡 `DeviceMergeSort*` 應近乎消失、端到端 −1.2~1.7 ms/frame。  
   - FP16：`DeviceMergeSort*` 應與 stock plugin 表現一致（sort 仍然發生）。  
4. 跑完整 eval，確認 INT8 的 mAP / mAPH 沒有漂移（argsort 僅是 locality 優化，不改變 pair 數學正確性；與 New3D 的 `do_sort = !int8_inference_` 精神一致）。  
5. A/B：如要把 INT8 engine 的 sort 重新打開比較 baseline，把 deploy_config 的 `spconv_do_sort` 設 `True`（或暫時 comment 掉讓它走 env fallback），重跑 Step 2 → 5；**runtime 完全不用動**。

**Sanity-check ONNX 屬性**：

```python
import onnx
m = onnx.load("work_dirs/bevfusion_split_int8_deployment/onnx/bevfusion_sparse.onnx")
vals = [(n.name, next((a.i for a in n.attribute if a.name == "do_sort"), None))
        for n in m.graph.node if n.op_type == "GetIndicePairsImplicitGemm"]
print(f"{len(vals)} pair-gen nodes; do_sort values: {sorted({v for _, v in vals})}")
```

- `deploy_config_split_int8.py`（`spconv_do_sort = False`）→ `{0}`。
- FP16 config（未設或 `True`）→ `{1}`。
- `None` → ONNX 是舊版 `sparse_functional.py` export 的；重跑 Step 2 即可。新 plugin 會以 struct 預設值 `1` fallback，不會崩。

### 10.10 實測：`do_sort` 關閉後（layer profiler / CUDA-event）

**前提**：已套用 §10.9 的 pair-gen plugin patch（`SpconvOps::get_indice_pairs_implicit_gemm` 傳入 `do_sort=false` 預設）；其餘量測條件與 §10 開頭一致（`profile_sparse_encoder`、warmup/iter 可比）。以下路徑為 container 內 `/workspace` 視角。

```
Engine  : /workspace/work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine
Inputs  : real:info/t4dataset_j6gen2_base_infos_test.pkl#0 (num_voxels=70747)
Total GPU latency (CUDA-event, steady-state):
  mean=13.652 ± 0.536 ms  median=13.507  min=13.005  max=15.933  n=200

------------------------------------------------------------------------------
Layer-sum breakdown by op-bucket (mean per-iteration sum):
------------------------------------------------------------------------------
  bucket                 count      sum_ms   % of layers
  pair_gen                  25       5.300        43.54%
  implicit_gemm_int8        20       4.318        35.47%
  implicit_gemm_fp           1       0.060         0.49%
  relu                      20       0.832         6.84%
  cast                       8       0.207         1.70%
  layout                     2       0.005         0.04%
  other                    101       1.452        11.93%

[A2] pair_gen (GetIndicePairsImplicitGemm) = 5.300 ms/iter (43.54% of layer-sum)
     → 已關 sort 後 pair_gen 仍為最大 bucket；後續若要再削時間，應拆解剩餘 pair-gen（hash / unique / scatter 等非 sort）或改 plugin fusion。

------------------------------------------------------------------------------
Block roll-up (encoder_layer granularity):
------------------------------------------------------------------------------
  block                  count      sum_ms   % of layers
  other                    168      10.930        89.79%
  conv_input                 5       0.890         7.31%
  conv_out                   4       0.353         2.90%

------------------------------------------------------------------------------
Top 15 layers by mean time:
------------------------------------------------------------------------------
    #     mean_ms  bucket                layer_name
    1       0.863  other                 [trainStation1]
    2       0.737  pair_gen              /pts_middle_encoder/encoder_layer1/encoder_layer1.2/encoder_layer1.2.0/GetIndicePairsImplicitGemm
    3       0.587  implicit_gemm_int8    /pts_middle_encoder/conv_input/conv_input.0/ImplicitGemm_int8
    4       0.566  pair_gen              /pts_middle_encoder/encoder_layer2/encoder_layer2.2/encoder_layer2.2.0/GetIndicePairsImplicitGemm
    5       0.404  pair_gen              /pts_middle_encoder/encoder_layer3/encoder_layer3.2/encoder_layer3.2.0/GetIndicePairsImplicitGemm
    6       0.283  pair_gen              /pts_middle_encoder/conv_out/conv_out.0/GetIndicePairsImplicitGemm
    7       0.273  pair_gen              /pts_middle_encoder/encoder_layer2/encoder_layer2.0/conv1/GetIndicePairsImplicitGemm
    8       0.270  pair_gen              /pts_middle_encoder/conv_input/conv_input.0/GetIndicePairsImplicitGemm
    9       0.266  pair_gen              /pts_middle_encoder/encoder_layer1/encoder_layer1.0/conv1/GetIndicePairsImplicitGemm
   10       0.266  pair_gen              /pts_middle_encoder/encoder_layer1/encoder_layer1.1/conv1/GetIndicePairsImplicitGemm
   11       0.264  pair_gen              /pts_middle_encoder/encoder_layer1/encoder_layer1.1/conv2/GetIndicePairsImplicitGemm
   12       0.264  pair_gen              /pts_middle_encoder/encoder_layer1/encoder_layer1.0/conv2/GetIndicePairsImplicitGemm
   13       0.258  pair_gen              /pts_middle_encoder/encoder_layer2/encoder_layer2.0/conv2/GetIndicePairsImplicitGemm
   14       0.257  pair_gen              /pts_middle_encoder/encoder_layer2/encoder_layer2.1/conv1/GetIndicePairsImplicitGemm
   15       0.256  pair_gen              /pts_middle_encoder/encoder_layer2/encoder_layer2.1/conv2/GetIndicePairsImplicitGemm
==============================================================================
```

**與 §10.1 / §10.2（有 sort／未關 `do_sort`）對照**

| 指標 | §10.1–10.2 baseline | §10.10（disable sort） | Δ |
|---|---:|---:|---|
| CUDA-event **mean** | 15.320 ms | **13.652 ms** | **−1.668 ms**（約 **−10.9%**） |
| CUDA-event std | ±0.569 | ±0.536 | 相近 |
| **pair_gen sum_ms** | 7.355（51.70%） | **5.300**（43.54%） | **−2.055 ms**；佔 layer-sum **−8.16 pp** |
| implicit_gemm_int8 sum_ms | 4.322（30.38%） | 4.318（35.47%） | 合計時間幾乎相同；佔比上升是因為 pair_gen 縮小後分母效應 |
| implicit_gemm_fp | 1×0.062 ms | 1×0.060 ms | 仍為單一 FP 節點，量級一致 |
| relu / cast | 0.817 / 0.196 | 0.832 / 0.207 | 小幅波動（profiler 與 fused region 邊界定義 noise） |

**解讀**：關閉 pair-mask sort 後，**端到端 sparse engine GPU 時間下降約 1.67 ms**，與 §10.5 用 `nsys` 粗估的 sort bucket（約 1.25–1.66 ms/frame）同一量級；**pair_gen layer-sum 仍約 5.3 ms**，說明 pair-gen 裡仍有大量 **非 sort** 成本（hash / stage1–2 / mask 生成等），後續仍可做 Nsight Compute 拆解或演算法級優化。Top 15 觀察：`ImplicitGemm_int8`（`conv_input.0`）升到第 3 名，反映 sort 成本拿掉後 **INT8 conv** 與 **剩餘 pair_gen** 的相對排序更接近真實比例。

### 10.11 disable sort 後 vs 開源路線：仍存在的差距與可 improvement 方向

以下對照 **§2**（AWML / New3D / CUDA-BEVFusion 架構差異），並以 **§10.10** 的量測為準：關 sort 之後 **pair_gen 仍 ~5.3 ms（~44% layer-sum）**、**implicit_gemm_int8 仍 ~4.3 ms（~35%）**，瓶頸從「sort 一枝獨秀」變成 **「pair-gen 非 sort 段 + INT8 GEMM + TRT graph 開銷」** 並重。

#### 已與開源敘事對齊的部分

- **INT8 inference 下略過 pair-mask sort**：對應 New3D `do_sort = !int8_inference_`、traveller59 [INT8 performance guide](https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md#performance-guide) 的精神；實測端到端約 **−11%** sparse engine latency（§10.10）。
- **INT8 implicit GEMM 本身不慢**：§10.10 Top 15 裡 `ImplicitGemm_int8` 已能與單層 pair_gen 同量級競爭，與「主戰場在 pair / graph 而非純 GEMM kernel」的判斷一致。

#### 仍與開源有結構性差異、因而仍可能帶來收益的項目

| 方向 | 開源 / 參考作法 | AWML 現況與缺口 | 可能收益與備註 |
|---|---|---|---|
| **層間 INT8 activation chain** | New3D：`input_precision` / `output_precision` 可維持 **INT8→INT8**，`SparseRelu` / `SparseAdd` 可走 INT8（§2.2） | Path B：`ImplicitGemmInt8` **I/O 仍 FP16**，每層進 plugin 再 `FP16→INT8`（§2.1.1） | 減少 **per-layer 量化與記憶體 traffic**；收益需用 Nsight 看 `launch_quantize_features` / cast 是否仍顯著；屬 **中高工作量**（改 plugin I/O 或 fusion） |
| **稀疏段改由獨立 engine 執行** | CUDA-BEVFusion：sparse 走 **`spconv::Engine` / libspconv**，非逐層 TRT plugin（§2.3–2.4） | AWML：整段仍是 **TensorRT + 多個 custom plugin**，Myelin fused region（如 `[trainStation1]`）與 layer 邊界開銷仍在 §10.10 Top 1 | **graph 級** 才有可能吃掉 launch / D2D / 調度開銷；屬 **架構級**，長期 ROI 高、短期成本高 |
| **pair-gen 與 conv 的靜態拆分／reuse** | traveller59 / New3D 註解：靜態 inference 可把 **pair-gen 與 conv 拆層** 以重用 pair（減少重複 indice work） | TRT 圖上每層仍各自 **GetIndicePairsImplicitGemm**，§10.10 顯示 **25 個 pair_gen 節點加總仍是大頭** | 若幾何與 tensor 形狀在 deploy 時固定，理論上可評估 **cache / fuse pair**；需改 ONNX / builder 策略，**中長期** |
| **pair-gen 演算法段（非 sort）** | 同一套 `SpconvOps::get_indice_pairs_implicit_gemm`，開源亦受 hash / direct_table / mask stage 成本影響 | 關 sort 後 **pair_gen 仍 5.3 ms**：剩餘為 **hash、unique、stage1/2、mask** 等 | **Nsight Compute** 應改瞄準 **非 `DeviceMergeSort*`** 的 top kernels；必要時評估 **algo、`direct_table`、subm stride 路徑** 與 spconv 版本對齊 |
| **Residual / Epilogue fusion** | New3D 在 engine 內可把 add/relu 與 conv 協調；CUDA-BEVFusion 偏向單一 sparse runtime | §10.10：`relu` + `cast` + `other` 仍佔可見比例；`[trainStation1]` 仍為 Top 1 | **TRT Priority B**：relu epilogue、fuse cast、減少獨立 kernel；與 §10.8 第 3、5 點一致 |
| **INT8 覆蓋率** | 開源可在 export 明確設多層 `precision=int8`，僅邊界留 FP16 | §10.10 仍見 **1 個 `implicit_gemm_fp`** | 補齊 transform 或接受「最後一層 FP」的 product 決策；屬 **正確性/覆蓋率** 而非純 perf |
| **deploy 語意：`do_sort` 只對 INT8 engine 關** | 開源在 C++ 內用 **`int8_inference_`** 綁定 | ✅ 已解決：§10.9 把 `do_sort` 做成 ONNX/TRT **序列化屬性**，FP16 export 維持 1、INT8 export 寫 0，舊 ONNX 回退為 1；同一個 `.so` 可服務兩種精度。 | — |

#### 建議優先順序（在 §10.8 基礎上，針對「已關 sort」後的調整）

1. **Profiler + Nsight**：以 §10.10 為 baseline，對 **pair_gen（非 sort）** 與 **`ImplicitGemm_int8`** 分別做 kernel 級拆解；確認下一刀是 **hash/stage** 還是 **quantize/cast**。  
2. **TRT 層級**：優先展開 **`[trainStation1]`** 與 **relu/cast** 是否可 fusion（與開源「減少邊界」方向一致）。  
3. **中長期**：若 sparse 仍佔端到端大头，再評估 **獨立 sparse engine** 或 **INT8 I/O plugin**，向 §2.2 / §2.3 靠攏。

**一句話**：關 sort 後，AWML 與開源的差距從「沒做 INT8 pair 優化」縮小到 **「TRT plugin 邊界 + 層間仍 FP16 + 整段 pair/conv 無法像 `spconv::Engine` 一樣全域編排」**；下一階段應優化 **剩餘 pair-gen 與 activation 邊界**，而非再盯 merge-sort。
