# 04 — 各 backbone 的量化特殊處理:ResNet add / VoV eSE / ConvNeXt

> 前置:[01 — Q/DQ 基礎](01_qdq_basics.md)(尤其 §6 的 7 條官方擺放建議)。
> 本篇解釋 `deployment/quantization/recipes/` 的三套 architecture recipe:
> 它們全部在解同一個問題 — **讓 INT8 區段不斷裂、不產生多餘的 reformat**。
> (程式碼位置以 branch `feat/quantization_framework` 為準。)
>
> 每個 recipe 底下都標了它對應 NVIDIA
> [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
> 的哪一條規則;§6 集中列出**官方有講但我們做法不同**的地方。

## 0. 框架的分工(先建立地圖)

```
deployment/quantization/
├── core/          # 引擎:只認識 nn.Conv2d / ConvTranspose2d / Linear、BN 融合、校準
├── recipes/       # 架構知識:BasicBlock / eSEModule / _OSA_module / ConvNeXtBlock / MaxPool2d
├── schemes/       # 接縫:QuantizationScheme.prepare() / QuantizationPlan
└── producer.py    # PTQ/QAT producer 共用件

deployment/projects/centerpoint/quantization/
├── plan.py        # build_centerpoint_plan(config) — 唯一入口
├── quant_model.py # CenterPoint 的塔級組合(哪個塔換 Conv、哪個塔換 Linear)
└── schemes.py     # CenterPointDenseScheme(fuse BN → expand keep_fp16 → quant_model)
```

**神聖不可侵犯的 invariant**:PTQ producer、QAT hook、deploy loader 三個消費者
都呼叫同一個 `build_centerpoint_plan(config).prepare(model)`,所以校準出來的
state_dict 和部署時 load 的 module tree **由構造保證對齊**
(有 unit test 鎖著:`deployment/tests/test_qat_tree_parity.py`)。

`prepare()` 的順序(`schemes.py:50`):

1. `fuse_model_bn(model)` — **無視 keep_fp16,整個模型都融合**(因為融合改變
   state_dict key,兩邊必須融合同一組)。
2. `expand_keep_fp16(model, patterns)` — fnmatch 展開成具體 module 名 + 整棵子樹;
   **pattern 沒 match 到任何東西會 WARNING**(防打錯字)。
3. `quant_model(model, skip_names, disable_recipes)` — 換模組,然後掛 recipes:

```python
# quant_model.py — 塔級知識寫死在程式碼,不是 config
pts_backbone      → Conv 換 QuantConv2d/QuantConvTranspose2d + Linear 換 QuantLinear
pts_neck          → 只換 Conv
pts_bbox_head     → 只換 Conv
pts_voxel_encoder → 只換 Linear
# recipes:always-on、class-gated(模型裡沒有該類別就是 no-op)
attach_quant_add(model)              # disable_recipes=["add"] 可關
attach_ese_quantizers(model)         # disable_recipes=["ese"] 可關
attach_maxpool_input_quantizer(...)  # disable_recipes=["maxpool"] 可關
```

> **舊 flag 對照**(如果你看過舊的 log `backbone=True, ..., quant_ese_mul_identity=False`):
> 那套 13 個 boolean 已經整組移除。`quant_voxel_encoder=False` → `keep_fp16=["pts_voxel_encoder"]`;
> `skip_backbone_stages=[0]` → `keep_fp16=["pts_backbone.blocks.0"]`;`quant_add` →
> `disable_recipes=["add"]`;`quant_ese_pool_input` / `quant_ese_mul_identity` → 併入
> always-on 的單 Q eSE 設計(下述);`quant_linear_backbone` → 移除,backbone Linear 一律量化。

### Quantizer 的預設(core/descriptors.py — 全框架唯一的來源)

| tensor | granularity | calibrator |
|---|---|---|
| Conv2d / Linear 的 input | per-tensor | **histogram**(→ MSE) |
| Conv2d weight | **per-output-channel** (axis=0) | max |
| **ConvTranspose2d weight** | **per-tensor**(特例!) | max |
| Linear weight | per-row (axis=0) | max |
| recipe 掛的 quantizer(add/eSE/concat/maxpool) | per-tensor | histogram(與 conv input 共用同一個 descriptor 物件) |

這張表就是官方第 7 條建議的實作:
> "Use per-tensor quantization for activations and per-channel quantization for weights.
> This configuration has been demonstrated empirically to lead to the best quantization accuracy."

**Linear 為什麼是 axis=0**,官方文件有一段專門的 warning 解釋(這是很容易搞錯的坑):
PyTorch 的 `nn.Linear` 會 export 成 ONNX GEMM,weight layout 是 `(K, C)` 且帶
`transB` 屬性;TensorFlow 則在 export 前就先轉置成 `(C, K)`。所以
> "GEMM layers originating from ONNX QAT models that were exported from PyTorch use dimension
> `0` for per-channel quantization (axis `K = 0`), while models originating from TensorFlow use
> dimension `1`."

我們是 PyTorch 路線 → axis=0,和表格一致。

**ConvTranspose2d 是唯一的偏離**:官方把 Transposed Convolution 明確列在
「weighted operations」裡(意即照理也該 per-channel weight),但實測上 TensorRT 的
INT8 deconv 對 per-channel scale 很脆弱(engine build 會炸 `vol == 1` /
`Could not find any implementation`),所以我們退成 per-tensor。
每個 int8 config 裡也都留了一行註解掉的逃生門
`# "pts_neck.deblocks.*.0"`(整顆 deconv 保持 FP16)。

---

## 1. SECOND(本 tutorial 的模型):最單純的情況

SECOND 的 2D backbone 是純 Conv-BN-ReLU 堆疊 — **沒有 residual、沒有 concat、
沒有 pooling**,所以三個 recipe 全是 no-op。config 仍寫 `disable_recipes=["add"]`
只是把「不量化 add」的意圖寫成文件(行為上等價)。

release recipe 的兩個 keep_fp16 是純精度考量:

- `pts_voxel_encoder`:所有 int8 config 一致保留 FP16。它是模型的數值大門
  (原始點座標 + pillar offset,動態範圍大),而且它的 BatchNorm1d **本來就不會被融合**
  (mmdet3d 的 PFNLayer 把 norm 註冊在 linear 前面,BN 融合只認 Conv→BN 的相鄰順序)。
- `pts_backbone.blocks.0`:第一個 stage 在全解析度 (1020²) 上跑、量化誤差會傳遍全網 —
  實驗上不穩就整段保持 FP16(release 名稱 skip_stage_0 的由來)。

## 2. ResNet:residual add 的「只量 identity branch」

**問題**:BasicBlock 的 `out = conv_path + identity`。如果兩條 branch 都插 Q/DQ,
TensorRT 沒辦法把 Add fuse 進前面的 conv kernel,會多出獨立的 Add 層和 reformat。

**官方怎麼說**:這正是 *Q/DQ Layer-Placement Recommendations* 的第 4 條 ——
**"Quantize the residual input in skip connections."** 文件給的理由和我們遇到的現象
一字不差:如果 residual 那一路(文件寫作 `x_f^2`)維持高精度,
> "the precision of `x_f^2` is high precision, so the output of the fused convolution is
> limited to high precision, and the **trailing Q-layer cannot be fused** with the convolution."

反之把它量化成 INT8:
> "the output of the fused convolution is also INT8, and the trailing Quantize layer is fused
> with the convolution."

也就是說 **conv 輸出的精度是由「Add 的另一個 operand」決定的** —— 這就是為什麼
一定要量 identity branch,而不是量 conv path。文件同時提到這個 fusion 的適用範圍:
> "TensorRT can fuse element-wise addition following weighted layers, which is useful for
> models with skip connections like ResNet and EfficientNet."

**解法**(和 NVIDIA lidar-ai-solution / CUDA-BEVFusion 相同):**只量化 identity branch,
conv path 的輸出完全不碰**(後者也正好是官方第 2 條「預設不要量化 weighted op 的輸出」)。
TensorRT 的 INT8 conv kernel 支援把「+residual → ReLU」做成 conv 的 epilogue,
條件是 conv 自己的輸出不被 Q/DQ 打斷、而 Add 的另一個 operand 是有已知 scale 的
INT8 tensor:

```
x ── Q/DQ ── Conv1 ── ReLU ── Q/DQ ── Conv2 ──(FP,不插 Q)──┐
 └── Q/DQ(residual_quantizer)── identity ────────────────────┴─ Add ─ ReLU ─ ...
                                    ↑ TensorRT 把 Add+ReLU 吸進 Conv2 的 kernel
```

實作是換掉 `BasicBlock.forward`(`recipes/forward_hooks.py`),
在 `out = out + identity` 前插一句 `identity = self.residual_quantizer(identity)`。

**quantizer 從哪來**(`recipes/attach.py`)— 一個重要的省 reformat 細節:

- block **有 downsample** → identity 是另一個 tensor(分佈不同)→ **開新的 quantizer**。
- block **沒 downsample** → identity 就是 block 輸入 `x` 本身,而 `conv1._input_quantizer`
  已經在看同一個 tensor → **直接重用 conv1 的 input quantizer**(共享校準統計,
  TensorRT 看到同一個 scale,不會為同一個 tensor 建兩個獨立的量化起點)。

Class gating:`{"BasicBlock", "SparseBasicBlock", "ConvNeXtBlock", "_OSA_module"}`
的類名 exact-or-substring match。**注意 `Bottleneck` 不在名單裡** —
換 ResNet-50 的話 residual recipe 不會生效(目前 CenterPoint ResNet 都是 R34/BasicBlock)。

> 歷史註:早期版本有一個 `QuantAdd` module + 自訂 ONNX symbolic(把兩個 operand
> 用同一個 quantizer 量化)。現行設計已刪除它 — identity-branch-only 的 forward hook
> 不需要任何自訂 op:`TensorQuantizer` 在 `use_fb_fake_quant` 下自己 trace 成標準
> Q/DQ,`+` trace 成標準 `Add`。

## 3. VoVNet:eSE、OSA concat、MaxPool — reformat 三連戰

VoV 的 OSA block 結構:

```
x ─┬─ layer1 ─ layer2 ─ ... ─ layerN ─┐
   │    └──────┴──── (每層輸出) ───────┼── Concat ── 1×1 conv ── eSE ──(+x if identity)
   └──────────────────────────────────┘
eSE: x ── AdaptiveAvgPool(1) ── 1×1 conv(fc) ── hsigmoid ──┐
      └────────────────────────────────────────────────────┴── Mul
```

### 3a. eSE:單一 Q + fan-out(這就是舊 `quant_ese_*` flag 的最終形態)

**問題**:eSE 的輸入 `x` 有兩個消費者(gate path 的 avg_pool、Mul 的 bypass)。
如果各給一個獨立的 input quantizer,TensorRT 會視為**兩個獨立的量化起點**,
因為兩個消費者偏好不同的 INT8 layout(`NC/4HW4` vs `NHWC16`),
於是在**每個** QuantizeLinear 前面各插一個 FP32 Reformat — 兩倍的 reformat 開銷。

**解法**(`docs/ese_int8_changes.md`、`forward_hooks.py` 的 `eSEModuleForwardHook`):
**輸入只量化一次**,量化後的 `qx` 同時餵給兩條路;gate 另外配一個 quantizer
讓 Mul 的兩個 operand 都是 INT8:

```python
qx   = pool_input_quantizer(x)          # 唯一的一次輸入量化
gate = hsigmoid(fc(avg_pool(qx)))
gate = mul_gate_quantizer(gate)         # Mul 的另一個 operand 也是 INT8
return qx * gate
```

```
(conv_out FP32)
   └─ Reformat(FP32) ─ QuantizeLinear ──┬─ DQ ─ AvgPool ─ fc ─ hsigmoid ─ Q ─┐
        (只有這一個!)                   └─ DQ ─ bypass ────────────────────┴─ Mul (INT8×INT8)
```

一個 QuantizeLinear、兩個 DequantizeLinear、一個 reformat。

**對照官方**:文件用 **AveragePool** 當 quantizable layer 的示範圖 ——
"A quantizable `AveragePool` layer (in blue) is fused with the surrounding Dequantize and
Quantize layers. All three layers are replaced by a single quantized `AveragePool` layer."
關鍵是 average pooling **不像 max pooling 那樣可交換**(平均會產生新數值),所以它
只能靠 `DQ → AvgPool → Q` 被夾住才會變 INT8。

我們的 eSE hook 剛好給出了這個 pattern:`avg_pool` 的輸入是 `qx`(帶 DQ),
輸出餵給 `fc`(1×1 QuantConv2d,自帶 input quantizer → 提供 Q)。
所以 gate path 上是 `DQ → AvgPool → Q → Conv`,符合官方的 fusible pattern。
(caveat:VoV 的 `avg_pool` 是 `AdaptiveAvgPool2d(1)`,export 成
GlobalAveragePool/ReduceMean 而非 AveragePool;是否真的 fuse 成 INT8 pooling
要看 engine 的 layer information,不能只靠讀文件斷定。)

Mul 的兩個 operand 都給 INT8 則是官方第 5 條的實例:
**非 weighted 的 op(這裡是 ElementWise Mul)一旦輸入是 INT8,輸出也必須是 INT8**,
所以兩個 operand 的 scale 都必須是已知的。

### 3b. OSA concat:skip branch 全部量化、主路徑留 FP

Concat 的輸入如果一半 FP 一半 INT8(或 scale 各自為政),TensorRT 只能先全部
dequant 回 FP 再 concat — INT8 區段直接斷掉。

**對照官方**:Concatenation 沒有被官方文件單獨點名(該頁只詳細證明了 Max Pooling
可交換,並沒有給 concat / slice / reshape / transpose 的清單),但第 5 條建議的括號
那句直接適用:
> "Try quantizing layers that do not commute with Q/DQ. Currently, non-weighted layers with
> INT8 inputs also require **INT8 outputs**, so quantize both inputs and outputs."

Concat 是 non-weighted op,所以「輸入全部 INT8」和「輸出也是 INT8」要同時成立;
我們的 recipe 負責前者,後者由下游 1×1 conv 的 input quantizer 提供。

Recipe 的做法:

- 為 block 輸入 + 每層輸出(除了**最後一層**)各配一個 `concat_input_quantizers[i]`,
  在 `torch.cat` 之前把每個 skip branch 都量化 → Concat 的輸入有一致、已知的 scale。
- **最後一層的輸出不量化** — 跟 ResNet add 同一個原理:留 FP 讓 TensorRT
  把它 fuse 進該層 conv 的 epilogue。
- `identity=True` 的 block,輸入 `x` 有**三個**消費者(第一層 conv、concat、post-eSE add)。
  量三次 = 三個 reformat → hook 重用 `concat_input_quantizers[0]` 作為唯一的 Q,
  三個消費者共享同一個量化結果。

### 3c. MaxPool:QuantBeforePool wrapper

`_OSA_stage` 每個 stage(除 stage2)開頭有一個 `nn.MaxPool2d`。沒有 Q/DQ 的話,
pool 會落在 INT8 區段中間變成 FP 島(前後各一次 reformat)。Recipe 把它換成:

```python
class QuantBeforePool(nn.Module):
    def forward(self, x):
        return self.pool(self.quantizer(x))   # ONNX: Q → DQ → MaxPool
```

**對照官方**:max pooling 是該頁**唯一**給出完整數學證明的 commuting layer ——
因為 `Q` 是單調的,取 max 不會改變大小關係:
```
max({Q(x_j, scale), Q(x_k, scale)}) = Q(max({x_j, x_k}), scale)
```
所以它與 quantization、dequantization **都**可交換。

這也解釋了為什麼 `QuantBeforePool` 只量化輸入、不用管輸出(表面上看起來違反第 5 條
的「non-weighted op 要連輸出一起量化」):正因為 MaxPool 可交換,DQ 可以被
propagation **往後推過** pool,pool 自然就變成 INT8-in / INT8-out。
不可交換的 op(AvgPool、Concat、Mul)才必須自己把兩端都標好。

### 3d. VoV 的 keep_fp16:stem + stage2

`deploy_config_int8_vov57/99.py`:`keep_fp16=["pts_voxel_encoder", "pts_backbone.stem", "pts_backbone.stage2"]`。
理由(`docs/ptq_accuracy_vov99.md`):早期層在全解析度上跑、amax 最難估、誤差會
一路放大,而 concat+eSE 結構對 scale 特別敏感。不夠穩的話,加寬順序是
`stage3` → `pts_bbox_head`(config 裡留有註解)。

## 4. ConvNeXt:Linear、LayerNorm、permute

ConvNeXt block(mmpretrain,`linear_pw_conv=True`):

```
dwconv 7×7 (groups=C) → LayerNorm2d → permute(0,2,3,1) → Linear → GELU → Linear → permute(0,3,1,2) → ×γ → +shortcut
```

框架對它的處理,重點是**做了什麼**跟**沒做什麼**都要知道:

| 元件 | 處理 |
|---|---|
| depthwise conv | → `QuantConv2d`(`groups` 保留,per-channel weight) |
| 兩個 pointwise `nn.Linear` | → `QuantLinear`(backbone 的 Linear **一律**量化 — 這就是舊 `quant_linear_backbone` flag 的取代) |
| LayerNorm / LayerNorm2d | **不融合、不量化** — BN 融合只認 BatchNorm;LN 留在 graph 裡當 FP op |
| GELU | 不量化(Q/DQ 邊界在兩側 Linear 的 input/weight quantizer 上) |
| permute ×2 | **原樣保留** — 沒有任何 Linear→1×1 conv 的改寫 |
| residual add | ConvNeXtBlockForwardHook,只量 shortcut,重用 `depthwise_conv._input_quantizer`(shortcut 和 dwconv 輸入是同一個 tensor) |

「沒有各種 reshape」的正確理解:框架**不是**把 permute/reshape 從 graph 裡消掉,
而是保證 permute 兩側的 MatMul/Gemm 都帶著 Q/DQ — TensorRT 於是能用 INT8 跑
pointwise MLP,permute 只是廉價的 layout 操作,不會成為「把 INT8 區段切成 FP 島」
的斷點。真正會斷的是「**沒有 scale 資訊的 tensor**」,不是 shape 操作本身。

其他 ConvNeXt 專屬細節:

- BN 融合可能把某些 norm 換成 `nn.Identity`,而 mmpretrain 的 LN2d forward 需要
  `data_format` kwarg — hook 裡有 `_safe_call` 依型別決定要不要傳(不然會 crash)。
- `with_cp=True`(gradient checkpointing)在 export 時被強制關掉(`model_loader.py`)。
- config:`keep_fp16=["pts_voxel_encoder"]` **而已** — ConvNeXt 不需要像 VoV 那樣
  保留早期 stage。opset **20**(LayerNormalization 需要 ≥17)。

## 5. opset 版本速查(per-backbone)

| backbone | opset | 原因 |
|---|---|---|
| ResNet | 16 | 純 conv,最保守 |
| SECOND | 17 | 同上(release config) |
| ConvNeXt | 20 | LayerNormalization |
| VoVNet | 22 | 最新 op 集(hsigmoid 等 pattern) |

## 6. 與 NVIDIA 官方文件的分歧(以及為什麼)

官方文件 = [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)。
上面所有 recipe 都是照它的建議做的,以下三處是**明知故違**,理由記在這裡以免後人誤修:

### 6a. `fuse_bn=True` vs 官方「不要在訓練框架裡模擬 BN 融合」

官方第 3 條建議:
> "Do not simulate batch normalization and ReLU fusions in the training framework because
> TensorRT optimizations guarantee the preservation of these operations' arithmetic semantics.
> BatchNorm is fused with convolution and ReLU while keeping the same execution order defined
> in the pre-fusion network."

我們的 `prepare()` 第一步就是 `fuse_model_bn(model)`。ReLU 我們確實沒碰(從來不插
Q/DQ 在 ReLU 兩側),但 BN 是先融合再插 weight quantizer。三個理由:

1. **weight amax 必須量在「TensorRT 真正會用的那組權重」上**。BN 融合是逐 output
   channel 的縮放(`γ/σ`),融合前後 per-channel `max|w|` 差一個 channel-wise 係數。
   先融合再取 amax,scale 和 kernel 完全對齊,不依賴 TensorRT 事後怎麼重算。
2. **與 CUDA-CenterPoint / lidar-ai-solution 的參考實作對齊**(release recipe 的數值
   基準就是它)。
3. **工程上的硬約束**:融合會改變 state_dict key,而 PTQ producer / QAT hook /
   deploy loader 三個消費者必須 load 同一組 key,所以融合**無視 `keep_fp16`、
   整個模型一起做**(見 §0)。這條和精度無關,是 invariant。

代價要誠實講:這條路等於放棄了「讓 TensorRT 自己決定 BN 融合順序」的彈性。
如果哪天 QAT 精度在某個 backbone 上詭異地掉,`fuse_bn` 是值得回頭質疑的變數之一。

### 6b. ConvTranspose2d weight 用 per-tensor

官方把 Transposed Convolution 列入 weighted operations(第 1、7 條 → 該 per-channel)。
我們退成 per-tensor,純粹因為 TensorRT 的 INT8 deconv kernel 實測會 build 失敗。
細節見 §0 的表。

### 6c. 工具鏈:`pytorch-quantization` vs TensorRT Model Optimizer

官方該頁現在只提 **TensorRT Model Optimizer**(ModelOpt)做 PTQ/QAT + export,
`pytorch-quantization` 已經不在文件裡。我們仍在 `pytorch-quantization` 上
(它產生的 Q/DQ 是標準 ONNX node,TensorRT 端沒有差別)。
遷移的接觸面在 `core/descriptors.py`(quantizer 預設)與 `core/replace.py`(module 置換),
recipes/ 的架構知識可以整批沿用。

### 順帶:官方有講、我們目前用不到的東西

| 官方主題 | 我們的狀況 |
|---|---|
| **Weight-Only Quantization**(INT4 block quant + GEMM,memory-bound 才有意義) | 用不到 —— 我們是 conv-bound 的 BEV 網路 |
| INT4 / FP4 packing(兩個 element 塞一個 byte)、ONNX opset **21**(INT4 + block quant)、opset **23**(FP4E2M1) | §5 的 opset 表最高只到 22,因為我們只用 INT8 |
| **Q/DQ Interaction with Plugins**:plugin 若吃 INT8 就必須把 input DQ / output Q **收進 plugin 內部**並從 network 移除,還要 `setOutputType(kINT8)` | CenterPoint 沒有 plugin;BEVFusion 的 spconv plugin 有踩到這條(見該專案的 deploy config) |
| refit 會關閉 scale-equality 相關的優化 | 我們沒開 refit(`grep -rn refit deployment/` 是空的)—— **別開**,單 Q fan-out 的效益依賴 scale 相等,見 `01` §6 |

## 7. 一句話總結三個 recipe

> **residual add / eSE / concat / maxpool 全都在回答同一個問題:
> 「這個非 conv 的 op,它的每個輸入有沒有已知的 INT8 scale?」**
> 有 → TensorRT 讓它留在 INT8 區段(甚至 fuse 進 conv kernel);
> 沒有 → dequant 回 FP、插 reformat、時間精度雙輸。
> 而「量幾次」的答案永遠是:**同一個 tensor 只量一次,大家共享**(單 Q fan-out)。

## 參考資料

- **NVIDIA TensorRT — Explicit Quantization**:
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html>
  (Q/DQ Layer-Placement Recommendations、commutation 定義與 Max Pooling 證明、
  residual add 的 fusion 條件、per-channel axis 的 PyTorch/TF 差異、Q/DQ Limitations、
  Q/DQ Interaction with Plugins)
- [01 — Q/DQ 基礎](01_qdq_basics.md) §6:同一份規則的入門版 + 7 條建議與本框架的逐條對照。
- 框架內文件:`docs/ese_int8_changes.md`(eSE 單 Q 設計)、`docs/ptq_accuracy_vov99.md`
  (VoV keep_fp16 的實驗依據)。
