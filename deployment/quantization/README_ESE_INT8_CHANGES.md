# eSE 模組 INT8 量化：變更摘要

本文件整理為讓 **VoVNet eSE 模組** 在 TensorRT 下達成 **全 INT8** 且 **只出現一次 FP32 reformat** 的程式變更與設定。

---

## 1. 正確作法總覽：為什麼要「單一 Q」、怎麼改

### 1.1 改之前的結構（會出現兩次 FP32 reformat）

上游有一個 **conv/Conv**，輸出是 **FP32**，從這個 FP32 輸出分岔到 eSE 的兩條路。  
若在 eSE 一開始放**兩個**「輸入量化」：

- 左：`/ese/pool_input_quantizer/QuantizeLinear`（給 GAP 路）
- 右：`/ese/_input_quantizer/QuantizeLinear`（給 bypass / mul 那路）

則 ONNX / TRT 實際看到的是：

```
(conv_out FP32)
   ├── Reformat(FP32) → Q1: /ese/pool_input_quantizer/QuantizeLinear → (GAP 路)
   └── Reformat(FP32) → Q2: /ese/_input_quantizer/QuantizeLinear     → (bypass 路)
```

**兩次 FP32 Reformat 的來源**：同一個 FP32 tensor（conv_out）被量化了**兩次**，每一次 QuantizeLinear 前 TRT 都會插一個 FP32 Reformat。

### 1.2 為什麼 QuantizeLinear 前面會 Reformat（而且會出現兩次）

QuantizeLinear 的輸入是 FP32，但 TRT 會為每個 layer 選它偏好的 **FP32 memory layout / alignment** 以優化效能；更重要的是：

- **Q1** 後面的 consumer（GAP / conv1x1 那條）會讓 TRT 偏好某一種 INT8 format（例如 NC/4HW4、CHW32…）
- **Q2** 後面的 consumer（bypass → elementwise/mul 那條）會讓 TRT 偏好另一種 INT8 format（例如 NHWC16…）

因此 TRT 會在 **Q 前的 FP32** 先把資料整理成「最適合該分支」的 layout。  
有兩個 QuantizeLinear、兩條分支偏好不同 → TRT 就會做**兩次** FP32 reformat。

**結論**：兩次 reformat 不是因為 QDQ 插錯位置，而是因為**同一個 conv_out 被當成兩次「獨立的量化起點」**。

### 1.3 目標：不要把同一個 conv_out FP32 量化兩次

只留**一個** QuantizeLinear 作為 eSE 的入口量化（唯一 Qx），然後從它的輸出**分岔（fan-out）**到兩條路。

**改完後應該長這樣**：

```
(conv_out FP32)
   └── Reformat(FP32) → Qx: QuantizeLinear   （只做一次量化）
            ├── DQ → GlobalAveragePool → ... → gate ...
            └── DQ → bypass → (之後進 Mul)
```

- 仍然可以有**兩個** DequantizeLinear（DQ），分別餵 GAP 路、bypass 路。
- **QuantizeLinear 只能有一個**（Qx）。

### 1.4 從「現在」改成「應該」的具體操作（Step A/B/C）

| 步驟 | 動作 |
|------|------|
| **Step A** | 保留 **Q1**（`/ese/pool_input_quantizer/QuantizeLinear`），因為它已經接 GAP 那路。 |
| **Step B** | 把 **bypass 那路的量化入口**改成吃 **Q1 的輸出**，而不是 conv_out。亦即：<br>• 原本：`conv_out → (Reformat FP32) → Q2 → ...`<br>• 改後：`Q1_output (int8) → (直接分支) → (bypass 需要的 DQ) → ...` |
| **Step C** | **刪掉 Q2**（以及它前面的那個 FP32 reformat 就會自然消失）。Q2 不存在後，TRT 就不需要為它做「第二個 FP32 reformat」。 |

### 1.5 為什麼這樣改就能消掉兩次 FP32 reformat

- **改前**：`conv_out (FP32) → Q1` 與 `conv_out (FP32) → Q2` → 兩次 FP32→INT8 入口 → 兩次 reformat。
- **改後**：`conv_out (FP32) → Qx（唯一）`，`Qx 輸出 → 分岔給兩條路`。  
  對 conv_out 而言只剩**一次** FP32 → (reformat) → Quantize；另一條路不再「從 FP32 重新量化一次」，自然就少掉一組「FP32 reformat + Q」。

---

## 2. 單一 Q 在 eSE 輸入（實作對應）

**問題**：若 conv_out (FP32) 分岔到兩個 QuantizeLinear（Q1=pool_input_quantizer、Q2=mul_identity_quantizer），TRT 會為每個 Q 做一次 FP32 Reformat → 出現兩次 reformat。

**作法**：eSE 入口**只保留一個 Q**（保留 `pool_input_quantizer` 作為 Qx），bypass 路**改吃 Qx 的輸出**（不再用 `mul_identity_quantizer` 對 conv_out 再量化一次）。

- **attach 順序**：先 `attach_ese_pool_input_quantizer`，再 `attach_ese_mul_identity_quantizer`。
- **當 `pool_input_quantizer` 存在時**：`attach_ese_mul_identity_quantizer` **不掛** `mul_identity_quantizer`，只掛 `mul_gate_quantizer`；forward 裡 bypass 用 `qx = pool_input_quantizer(x)` 的輸出進 Mul。
- **當沒有 `pool_input_quantizer` 時**：仍掛 `mul_identity_quantizer` + `mul_gate_quantizer`（legacy 兩路各一 Q）。

Deploy config 建議同時開 `quant_ese_pool_input=True` 與 `quant_ese_mul_identity=True`，這樣會走「單一 Qx + gate Q」路徑，只有一次 FP32→INT8。

---

## 3. 讓 eSE 變 INT8 的關鍵變更（staged changes）

### 3.1 問題起點

- eSE 的 Mul 對應 **Mul–HardSigmoid**：Mul 的兩個輸入為 identity 與 gate（HardSigmoid 輸出）。
- Mul 兩輸入都要有 Q-DQ；且**同一個 conv_out 不要被兩個 Q 吃**，否則 TRT 會做兩次 FP32 reformat。

### 3.2 程式變更摘要

| 檔案 | 變更內容 |
|------|----------|
| **replace.py** | 1) **eSEModuleForwardHook**：若有 `pool_input_quantizer`，則 `qx = pool_input_quantizer(x)`，gate = GAP(qx)→fc→hsigmoid→mul_gate_quantizer，**identity = qx**（bypass 吃 Qx 輸出），`return qx * gate`；不再對 identity 用 `mul_identity_quantizer`。<br>2) **attach 順序**：先 `attach_ese_pool_input_quantizer`，再 `attach_ese_mul_identity_quantizer`。<br>3) **attach_ese_mul_identity_quantizer**：若已有 `pool_input_quantizer`，只補 `mul_gate_quantizer` 與 forward hook，**不掛** `mul_identity_quantizer`；否則才掛 `mul_identity_quantizer` + `mul_gate_quantizer`。 |

### 3.3 目標結構（engine 圖）

```
(conv_out FP32)
   └── Reformat(FP32) → Qx: pool_input_quantizer/QuantizeLinear   （只做一次量化）
            ├── DQ → GlobalAveragePool → fc → hsigmoid → mul_gate_quantizer → Mul
            └── DQ → bypass ──────────────────────────────────────────────→ Mul
```

- **QuantizeLinear 只有一個**（Qx）。
- 可以有兩個 DequantizeLinear（或 DQ 後分岔），分別餵 GAP 路與 bypass 路。

### 3.4 Deploy config

```python
quantization = dict(
    ...
    quant_ese_mul_identity=True,   # 確保 gate 有 mul_gate_quantizer；若已有 pool_input 則 bypass 用 Qx
    quant_ese_pool_input=True,     # 單一 Q 在 eSE 輸入（Qx），bypass 吃 Qx 輸出
    ...
)
```

---

## 4. 簡短結論

- **單一 Q 在 eSE 輸入**：當 `quant_ese_pool_input=True` 時，`pool_input_quantizer` 是唯一的 Qx；bypass 路使用 Qx 輸出，不再掛 `mul_identity_quantizer`，TRT 只會做一次 FP32 reformat。
- **Mul 兩輸入仍有 Q-DQ**：identity 側來自 Qx 的 DQ 輸出，gate 側來自 `mul_gate_quantizer` 的 DQ 輸出。
- **quant_ese_trt_friendly** 已移除；行為由 `quant_ese_pool_input` + `quant_ese_mul_identity` 與上述順序決定。
