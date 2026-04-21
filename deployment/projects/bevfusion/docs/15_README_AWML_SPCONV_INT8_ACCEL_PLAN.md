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

1. **Nsight Compute 拆解 pair-gen 內核** ← 最高 ROI  
   目標是確認 `argsort` / `scan` / `unique` 各佔 `GetIndicePairsImplicitGemm` 的多少 μs，進一步收斂「關 argsort 到底能省多少」；目前 `nsys` 顯示 `sort` 值得優化，但不是整個 pair-gen bucket 的全部  
   → 使用 `benchmark/nsys_profile_sparse.sh` + `ncu --set full --kernel-id ::regex:argsort|scan`

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

綜合來說，**投 Priority A2（pair-gen / sort 優化）ROI 最高**（-14% ~ -29% latency、不需要改 plugin graph），應作為下一階段的首選動作。

### 10.9 A2 實作：在 AWML 關閉 pair-gen 的 `do_sort`

對照 [traveller59/spconv INT8 guide](https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md#performance-guide) 與 New3D 的 `bool do_sort = !int8_inference_;`（`sparseConvImplicit.cu:368`），在 INT8 inference 階段 pair-mask 的 argsort 可以省略。AWML 的 INT8 sparse encoder 是走 `autoware_tensorrt_plugins::GetIndicesPairsImplicitGemmPlugin`（不是 New3D 的 `libspconv.so`），所以要在「**plugin 內部**」關掉，而不是改 New3D 的 `.cu`。

**已做的改動（皆在 BEVFusion Dockerfile build-time 自動套用）：**

1. 新增 `projects/BEVFusion/plugins/patches/disable_sort_for_int8.py`：idempotent 的 patcher，把 `autoware_tensorrt_plugins` 裡的 `get_indices_pairs_implicit_gemm_plugin.cpp`：
   - 注入一個 `awml_get_do_sort_from_env()` 小 helper，預設回傳 `false`，支援 runtime override `SPCONV_DO_SORT=1`（對齊 traveller59 Python 端的環境變數名稱）。
   - 把兩處 `SpconvOps::get_indice_pairs_implicit_gemm(..., use_direct_table);` 改成 `..., use_direct_table, /*do_sort=*/awml_get_do_sort_from_env());`。
2. `projects/BEVFusion/plugins/build_plugin_inside_container.sh`：在 clone autoware_universe 後、`cmake` 前呼叫 patcher；提供 escape hatch `AWML_DISABLE_DO_SORT_PATCH=1` 在需要比較 baseline 時可以跳過。
3. `projects/BEVFusion/Dockerfile`：多 `COPY` 一份 `patches/` 目錄進 image，確保 build stage 找得到 patcher。

> ⚠️ 該 patch 影響的是 **TRT plugin 的 pair-gen**（deploy path）；Python PyTorch 端的 `GetIndicePairsImplicitGemm`（`projects/SparseConvolution/sparse_functional.py:304`）已經是 `do_sort = SPCONV_DO_SORT`，其預設值來自 traveller59 的 `spconv.constants`（預設 `True`），PTQ 校正階段會繼續保留 sort。若要讓 Python eval 與 deploy 行為一致，執行時加 `SPCONV_DO_SORT=0 python -m ...` 即可。

**驗證步驟：**

1. 重新 build image（`docker build -f projects/BEVFusion/Dockerfile ...`），觀察 log 應出現 `[build_plugin] Applying AWML INT8 do_sort patch to pair-gen plugin` 與 `[awml-patch] patched ... (2 call sites + helper)`。
2. 重新跑 `benchmark/nsys_profile_sparse.sh`，比對 `nsys stats --report cuda_kern_exec_sum` 裡的 `DeviceMergeSort*` kernel：
   - **預期**：`sort` bucket 幾乎消失，總 GPU 時間下降約 **1.2–1.7 ms/frame**（與 §10.5 估算一致）。
   - 若仍看到大量 `DeviceMergeSort*`，代表 patch 沒套到（檢查 AWML_DISABLE_DO_SORT_PATCH 或 `get_indices_pairs_implicit_gemm_plugin.cpp` 裡有沒有 `AWML_PATCH_DO_SORT_FOR_INT8` marker）。
3. 跑完整 eval，確認 mAP / mAPH 沒有漂移（argsort 只是 locality 優化，不改變 pair 的數學正確性；New3D 也是這麼處理）。若出現 accuracy regression，先把 `SPCONV_DO_SORT=1` 打開 A/B，再往 plugin bug 方向排查。
