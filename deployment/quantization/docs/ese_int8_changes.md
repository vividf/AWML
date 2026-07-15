# eSE 模組 INT8 量化：單一 Q 設計說明

本文件說明 **VoVNet eSE 模組** 在 TensorRT 下達成 **全 INT8** 且 **只出現一次 FP32 reformat** 的設計原理與現行實作。

> 現行實作（單一入口）：`recipes/attach.py` 的 **`attach_ese_quantizers`**（每個 eSEModule 掛
> `pool_input_quantizer` + `mul_gate_quantizer` 並安裝 `eSEModuleForwardHook`，一次呼叫、無順序契約）。
> eSE recipe 是 **always-on、class-gated**：CenterPoint scheme 一律 attach，模型沒有 eSEModule 時為
> no-op；要關掉時在 deploy config 用 `disable_recipes=["ese"]`。
> 舊的兩段式 attach（`attach_ese_pool_input_quantizer` → `attach_ese_mul_identity_quantizer`）、
> legacy 兩-Q fallback（`mul_identity_quantizer`）、以及 `quant_ese_*` config keys 都已刪除。

---

## 1. 為什麼要「單一 Q」

### 1.1 兩個 Q 的結構（會出現兩次 FP32 reformat）

上游有一個 **conv/Conv**，輸出是 **FP32**，從這個 FP32 輸出分岔到 eSE 的兩條路。
若在 eSE 一開始放**兩個**「輸入量化」：

```
(conv_out FP32)
   ├── Reformat(FP32) → Q1: pool_input_quantizer/QuantizeLinear → (GAP 路)
   └── Reformat(FP32) → Q2: (bypass 的第二個 QuantizeLinear)     → (bypass 路)
```

**兩次 FP32 Reformat 的來源**：同一個 FP32 tensor（conv_out）被量化了**兩次**，每一次 QuantizeLinear 前 TRT 都會插一個 FP32 Reformat。

### 1.2 為什麼 QuantizeLinear 前面會 Reformat（而且會出現兩次）

QuantizeLinear 的輸入是 FP32，但 TRT 會為每個 layer 選它偏好的 **FP32 memory layout / alignment**：

- **Q1** 後面的 consumer（GAP / conv1x1 那條）會讓 TRT 偏好某一種 INT8 format（例如 NC/4HW4、CHW32…）
- **Q2** 後面的 consumer（bypass → elementwise/mul 那條）會讓 TRT 偏好另一種 INT8 format（例如 NHWC16…）

有兩個 QuantizeLinear、兩條分支偏好不同 → TRT 就會做**兩次** FP32 reformat。

**結論**：兩次 reformat 不是因為 QDQ 插錯位置，而是因為**同一個 conv_out 被當成兩次「獨立的量化起點」**。

### 1.3 單一 Q + fan-out（現行結構）

只留**一個** QuantizeLinear 作為 eSE 的入口量化（唯一 Qx），從它的輸出**分岔（fan-out）**到兩條路：

```
(conv_out FP32)
   └── Reformat(FP32) → Qx: pool_input_quantizer/QuantizeLinear   （只做一次量化）
            ├── DQ → GlobalAveragePool → fc → hsigmoid → mul_gate_quantizer → Mul
            └── DQ → bypass ──────────────────────────────────────────────→ Mul
```

- **QuantizeLinear 只有一個**（Qx）；可以有兩個 DequantizeLinear，分別餵 GAP 路與 bypass 路。
- `Mul` 兩輸入仍有 Q-DQ：identity 側來自 Qx 的 DQ 輸出，gate 側來自 `mul_gate_quantizer`。

## 2. 現行實作對應

| 元件 | 行為 |
|------|------|
| `attach_ese_quantizers`（`recipes/attach.py`） | 對每個 class name 為 `eSEModule` 的模組：掛 `pool_input_quantizer`（唯一 Qx）與 `mul_gate_quantizer`，安裝 forward hook。冪等；一次呼叫完成。 |
| `eSEModuleForwardHook`（`recipes/forward_hooks.py`） | `qx = pool_input_quantizer(x)`；gate = GAP(qx) → fc → hsigmoid → `mul_gate_quantizer`；**identity = qx**（bypass 吃 Qx 輸出）；`return qx * gate`。沒有 quantizer 時走純 FP 路（class-gate 的 no-op 行為）。 |
| Deploy config | 不需要任何 eSE key —— recipe always-on。要整組關掉：`disable_recipes=["ese"]`。 |

結構保證由 `deployment/tests/test_ese_single_q_recipe.py` 鎖定（單一 Q fan-out、無 legacy `mul_identity_quantizer`）；數值由 Docker e2e mAP 把關。
