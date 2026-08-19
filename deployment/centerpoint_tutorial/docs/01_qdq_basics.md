# 01 — Q/DQ 基礎:INT8 量化到底在做什麼

> 對象:第一次接觸模型量化的新人。讀完你應該能回答:
> scale 是怎麼來的、校準時每一層是獨立觀察還是接力推進、
> Q/DQ node 在 ONNX graph 裡長什麼樣、TensorRT 看到 Q/DQ 之後做了什麼。
>
> **權威參考**:本篇 §6 的規則全部出自 NVIDIA TensorRT 官方文件
> [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
> — 有疑問時以那頁為準,這裡只做「對照我們的 recipe」的翻譯。

## 1. 為什麼要 INT8

推論時的卷積主要是乘加運算。FP16 → INT8 的好處:

- **吞吐量**:NVIDIA GPU 的 INT8 Tensor Core 理論吞吐是 FP16 的 2 倍。
- **記憶體/頻寬**:activation 與 weight 都縮小一半,對 memory-bound 的層(大 feature map 的 conv)常常是主要收益來源。
- **代價**:INT8 只有 256 個可表示的值,表示範圍與解析度需要用「校準」來決定 — 這正是 PTQ 的核心工作。

## 2. 對稱線性量化(我們用的方案)

pytorch-quantization / TensorRT 的 INT8 是 **對稱(symmetric)線性量化**,沒有 zero-point:

```
q = clamp( round( x / scale ), -128, 127 )     # Quantize
x̂ = q * scale                                  # Dequantize
```

只有一個自由參數 `scale`,由 **amax**(要覆蓋的最大絕對值)決定:

```
scale = amax / 127
```

兩種誤差在拉扯,amax 的選擇就是在兩者間找平衡:

- **clipping error**:|x| > amax 的值被截斷。amax 太小 → 大值全被砍。
- **rounding error**:相鄰可表示值的間距是 scale = amax/127。amax 太大 → 間距太粗,
  大多數(小的)值的量化誤差變大。

### 範例:一個真實的 scale 到底在做什麼

拿本 tutorial 校準出來的第一個量化 conv(`pts_backbone.blocks.1.0`)的 activation
quantizer:amax = 6.0219 → `scale = 6.0219 / 127 = 0.047417`。代幾個值進去:

| x (FP) | x / scale | q (INT8) | x̂ = q·scale | 誤差 |
|---|---|---|---|---|
| 0.0123 | 0.26 | 0 | 0.0000 | −0.0123 |
| 0.5000 | 10.54 | 11 | 0.5216 | +0.0216 |
| 1.3000 | 27.42 | 27 | 1.2802 | −0.0198 |
| −2.7000 | −56.94 | −57 | −2.7027 | −0.0027 |
| 6.0219 | 127.00 | 127 | 6.0219 | 0 |
| 8.5000 | 179.26 | **127**(clamp) | 6.0219 | **−2.4781** |

前五列是 rounding error(上限固定是 ±scale/2 = ±0.0237),最後一列是 clipping error —
量級差了 100 倍。amax 的選擇就是在「多付一點 rounding」和「少數值付爆量 clipping」
之間下注。

這就是為什麼「直接拿觀察到的 max 當 amax」(method=max)通常不好:
activation 的分佈幾乎都是長尾,一兩個 outlier 就把 scale 撐大,傷害 99.9% 的值。

用**這一層真實記錄下來的 histogram**(60 筆校準資料 × 64ch × 1020×1020 ≈ 4.0e9 個值,
`calib_trace/hist_trace.pkl`)可以直接把四種 amax 的總誤差算出來:

| amax 來源 | amax | scale | 平均量化誤差 (MSE) | RMSE | 被 clip 掉的值 |
|---|---|---|---|---|---|
| 觀察到的 max | 16.8703 | 0.1328 | 8.39e−04 | 0.0290 | 0% |
| **mse(我們用的)** | **6.0219** | **0.0474** | **1.01e−04** | **0.0101** | 0.0011% |
| percentile 99.9 | 2.7558 | 0.0217 | 1.11e−03 | 0.0334 | 0.1006% |
| entropy | 0.9534 | 0.0075 | 2.91e−02 | 0.1706 | 2.7569% |

`max` 的誤差是 `mse` 的 **8.3 倍** — 不是因為它砍錯東西(它 0% clipping),
而是為了容納那個 16.87 的 outlier,scale 被撐粗 2.8 倍,而這一層 **97.3% 的值
其實都在 |x| ≤ 1.0**。反方向也會輸:`percentile 99.9` 只砍掉 0.1% 的值,
誤差卻比 `mse` 大 11 倍。`02_ptq_calibration_histogram.md` 會展示這些數字是怎麼
從 histogram 一步步長出來的。

### per-tensor vs per-channel

| | 誰在用 | amax 形狀 | 為什麼 |
|---|---|---|---|
| **per-tensor** | activation(input quantizer) | scalar | activation 每次推論都不同,只能用一組 scale;TensorRT 的 tensor 也只有一組 dynamic range |
| **per-channel** | weight(weight quantizer) | `[out_channels]` | weight 是常數,可以逐 output channel 精確取 max(axis=0),沒有理由共用 scale |

Weight 的 amax 用 `MaxCalibrator`(直接取每個 channel 的 max|w|,不需要資料集);
activation 的 amax 用 `HistogramCalibrator` + MSE 準則(需要校準資料)— 見下一篇。

**範例:per-channel 值多少錢。** 同一個 `blocks.1.0` 的 weight 形狀是 `[128, 64, 3, 3]`,
128 個 output channel 的 `max|w|`:

```
min     0.0567      # 最「小聲」的 channel
median  0.1547
max     0.4273      # max / min = 7.53×
```

若強制 per-tensor(128 個 channel 共用 amax = 0.4273),那個 amax = 0.0567 的 channel
只用得到 `127 × 0.0567 / 0.4273 ≈ 17` 個量化階 — 等於白丟將近 3 bit 的解析度。
per-channel 讓每個 channel 都吃滿 127 階,而且是免費的:weight scale 是常數,
TensorRT 直接把這 128 個 scale 折進 kernel。

## 3. 校準是「一次算完整張圖」,推論才是流水式

最常見的誤解:以為 PTQ 是「先定第一層的 amax → 量化它 → 用量化後的輸出重新算第二層的
輸入 → 再定第二層的 amax」這樣一層層接力推進。**不是。**

校準期間整個網路是**純 FP 前向**,所有 quantizer 只旁觀、不量化。`CalibrationManager`
在資料迴圈**之前**就把 56 個 quantizer 一次全部切到 collect 模式
(`deployment/quantization/core/calibration.py`):

```python
def _enable_calibration_mode(self):          # L99,在迴圈之前呼叫一次
    for name, module in self.model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable_quant()   # fake quant 關掉 → forward 完全是 FP 行為
            module.enable_calib()    # 只把流過的 tensor 統計進自己的 histogram
```

時序長這樣:

```
第 1..60 筆 ── 純 FP forward ──┬──> blocks.1.0 的 histogram  (看到 FP 的 stage0 輸出)
                              ├──> blocks.1.3 的 histogram  (看到 FP 的 blocks.1.0 輸出)
                              └──> ... 56 個 quantizer 各自累積,互不相干
迴圈結束 ──── compute_amax("mse") × 56 ──> 彼此完全獨立,沒有先後依賴
```

換句話說:**amax 是在乾淨的 FP 分佈上校準的**,不含上游量化誤差。

### 這個近似為什麼安全

上游量化帶來的擾動,相對於分佈本身很小。以 `blocks.1.0` 的輸入為例(從 §2 那個真實
histogram 算出來):

| | 值 |
|---|---|
| 訊號 RMS(\|x\|) | 0.4203 |
| 量化噪聲 RMSE | 0.0101 |
| **噪聲 / 訊號** | **2.4%** |
| 噪聲 / amax | 0.17% |

2.4% 的擾動幾乎不改變 histogram 的形狀,MSE 搜出來的最佳 amax 也就幾乎不動 —
所以「用 FP 分佈校準、拿去量化後的 pipeline 用」是安全的近似。

(真正做流水式校準的是 AdaRound / BRECQ 這類**逐層/逐 block 重建**方法:用已量化的
前綴產生輸入、每層各跑一次優化。精度略好,成本高一個數量級。
pytorch-quantization 與 TensorRT 的 PTQ 都不走這條路。)

### 但推論確實是流水式的

| 階段 | 流水式? | 第 N 層實際吃到的輸入 |
|---|---|---|
| **校準**(collect + compute_amax) | ❌ 一次性、平行觀察 | FP 的第 N−1 層輸出 |
| **推論**(fake-quant eval / TRT engine) | ✅ | 第 N−1 層量化→dequant 後的輸出,誤差會累積 |

誤差累積是真實存在的,校準只是不去建模它 — 它反映在最終指標上:
PyTorch fake-quant eval mAP 0.4973 → 0.4857。要讓**權重**去適應這份累積誤差,
就得靠 **QAT**:所有 fake quant 開著繼續 fine-tune,第 N 層在訓練時就看得到
量化過的輸入。

### 一個「大家觀察的是同一張 FP 圖」的實例

head 六個分支的 input quantizer,amax **完全相同**,因為它們量化的是同一個
`shared_conv` 輸出張量:

```
pts_bbox_head.task_heads.0.{heatmap,reg,height,dim,rot,vel}.0.conv._input_quantizer
    → amax = 6.6205  (六個一模一樣)
```

quantizer 是掛在「tensor 的**消費端**」上的:同一個 tensor 被 N 個 conv 消費,就有
N 個 quantizer 統計到完全一樣的分佈。TensorRT 之後會把這些重複的 Q/DQ 合併掉
(見 §6 的 Q/DQ propagation)。

## 4. Fake quantization(QAT/PTQ 在 PyTorch 裡的樣子)

訓練框架裡沒有真正的 INT8 kernel。`pytorch-quantization` 的 `TensorQuantizer`
做的是 **fake quant**:數值上執行 quantize→dequantize,但 tensor 仍然是 FP:

```
x̂ = clamp(round(x/scale), -128, 127) * scale    # 仍是 float,但已經「長得像」INT8
```

`QuantConv2d` = 在 Conv2d 的輸入和 weight 前各插一個 `TensorQuantizer`:

```
x ──> [input_quantizer (per-tensor)] ──┐
                                       ├──> Conv2d ──> y
w ──> [weight_quantizer (per-channel)]─┘
```

所以 PTQ 之後在 PyTorch 裡 evaluate 到的 mAP,就是「INT8 數值行為」的預覽
(我們 tutorial 中 pytorch backend 的 eval 就是這個)。

### 範例:PTQ checkpoint 比 FP checkpoint 多了什麼

只多了 `_amax` buffer — 權重一個 bit 都沒動(校準不改權重):

```python
>>> sd = torch.load("checkpoints/epoch_29_ptq_tutorial.pth")["state_dict"]
>>> len(sd), len([k for k in sd if "quantizer" in k])
(132, 56)                       # 56 = 28 個量化 conv × (input + weight)

>>> sd["pts_backbone.blocks.1.0._input_quantizer._amax"]
tensor(6.0219)                  # per-tensor → scalar

>>> sd["pts_backbone.blocks.1.0._weight_quantizer._amax"].shape
torch.Size([128, 1, 1, 1])      # per-channel → 每個 output channel 一個
```

同一組 amax 也另外存成 `checkpoints/epoch_29_ptq_tutorial.calib`(純 amax dict,
方便單獨檢視或和 release 對照 — `03_compare_amax_table.py` 讀的就是它)。

## 5. ONNX 裡的 Q/DQ node

Export 時開啟 `use_fb_fake_quant` 後,每個 `TensorQuantizer` 會被 export 成一對
ONNX node:

```
QuantizeLinear(x, scale)  ->  int8 tensor
DequantizeLinear(q, scale)  ->  float tensor
```

graph 變成(以一個 conv 為例):

```
input ─ QuantizeLinear ─ DequantizeLinear ─ Conv ─ ...
weight ─ QuantizeLinear ─ DequantizeLinear ─┘
```

注意:ONNX graph 裡 Conv 本身仍然是 float op。Q/DQ 對只是把
「這個 tensor 應該用這個 scale 量成 INT8」的資訊寫進 graph。

### 範例:export 出來的真實節點(`int8/onnx/pts_backbone_neck_head.onnx`)

`blocks.1.0` 這個 conv 在 graph 裡實際是 10 個節點(node index 8–17):

```
 8 Constant          .../blocks.1.0/_input_quantizer/Constant       → zero_point: int8 scalar 0
 9 Constant          .../blocks.1.0/_input_quantizer/Constant_1     → scale: float32 0.04741656
10 QuantizeLinear    .../blocks.1.0/_input_quantizer/QuantizeLinear
       in: /backbone/blocks.0/blocks.0.11/Relu_output_0, scale, zero_point
11 DequantizeLinear  .../blocks.1.0/_input_quantizer/DequantizeLinear
12 Constant          .../blocks.1.0/_weight_quantizer/Constant       → scale: float32[128]  ← per-channel
13 Constant          .../blocks.1.0/_weight_quantizer/Constant_1     → zero_point: int8[128] 全 0
14 QuantizeLinear    .../blocks.1.0/_weight_quantizer/QuantizeLinear
       in: model.backbone.blocks.1.0.weight, scale[128], zero_point[128]
15 DequantizeLinear  .../blocks.1.0/_weight_quantizer/DequantizeLinear
16 Conv              .../blocks.1.0/Conv
       in: _input_quantizer/DequantizeLinear_output_0,
           _weight_quantizer/DequantizeLinear_output_0,
           model.backbone.blocks.1.0.bias          ← bias 沒有被量化,維持 FP
17 Relu              .../blocks.1.2/Relu
```

三件可以直接對回前面章節的事:

- `scale = 0.04741656`,乘 127 = 6.0219 = checkpoint 裡的 `_input_quantizer._amax`。
  **ONNX 存的是 scale,checkpoint 存的是 amax**,兩者差一個 127。
- input 的 scale 是 scalar、weight 的 scale 是 `[128]` — per-tensor / per-channel
  的差別直接寫在 graph 裡,肉眼可辨。
- zero_point 全是 0 → 對稱量化;bias 沒有 Q/DQ(TensorRT 在 INT32 accumulator 裡加 bias)。

### 範例:PTQ 前 / PTQ 後同一張 graph 的 op 統計

```python
Counter(n.op_type for n in onnx.load(...).graph.node)
```

| op | `fp16/onnx` | `int8/onnx` |
|---|---|---|
| Conv | 31 | 31 |
| ConvTranspose | 1 | 1 |
| Relu | 26 | 26 |
| Concat | 1 | 1 |
| QuantizeLinear | 0 | **56** |
| DequantizeLinear | 0 | **56** |
| Constant | 0 | 112(56 組 scale + zero_point) |

計算網路本體(59 個 op)完全一樣 — INT8 export 純粹是「在上面貼標註」。
56 對 Q/DQ = 28 個量化 conv × 2(input + weight),和 checkpoint 裡的 56 個 amax 對得上。

## 6. TensorRT 怎麼消化 Q/DQ(explicit quantization)

TensorRT 看到 Q/DQ node 走的是 **explicit quantization** 模式:

1. **Layer fusion**:`DQ → Conv → ReLU → Q` 這種 pattern 被 fuse 成一個
   **INT8-in / INT8-out 的 conv kernel**,scale 直接烙進 kernel。
   官方把「能這樣吃掉 Q/DQ 的 layer」叫 **quantizable layer**
   (文件舉的例子:Convolution、GEMM、**AveragePool**)。
2. **Q/DQ propagation**:移動 Q/DQ 以「讓低精度區段的比例最大化」。方向是明確的 ——
   > "TensorRT propagates **Quantize nodes backward** (so quantization happens as early as
   > possible) and **Dequantize nodes forward** (so dequantization happens as late as possible)."

   能不能移動,取決於該 op 是否**可交換(commute)**。官方給的是形式定義:

   ```
   Op 與 quantization   可交換  ⇔  Q(Op(x)) == Op(Q(x))
   Op 與 dequantization 可交換  ⇔  Op(DQ(x)) == DQ(Op(x))
   ```

   文件唯一給出完整證明的例子是 **Max Pooling**(取 max 不改變 Q 的大小順序,
   所以和 Q、DQ 都可交換)。**AveragePool 則不可交換** —— 平均會產生新的數值,
   所以它只能靠「被 Q/DQ 夾住然後 fuse」變成 INT8,這正是 quantizable 與 commuting
   兩個詞的差別。
3. **沒被 Q/DQ 包住的區段**維持 FP16/FP32 執行。

幾個直接影響我們 recipe 設計的推論:

- **精度的控制權在我們手上**:哪裡插了 Q/DQ,哪裡就是 INT8;沒插的地方 TensorRT
  不會自作主張(這和 implicit/calibration-cache 模式相反)。`keep_fp16` 清單
  (例如 voxel encoder、backbone stage 0)就是靠「不插 Q/DQ + disable quantizer」實現的。
- **斷點很貴**:INT8 區段中間如果有一個沒被量化的 op(或是一個 Reshape/Transpose
  擋住 fusion),TensorRT 就得插入 reformat(INT8→FP16→INT8 的轉換層),
  時間和精度雙輸。`04_backbone_recipes.md` 裡 ResNet 的 residual add、VoV 的 eSE、
  ConvNeXt 的各種 permute,全都是在解「怎麼讓 INT8 區段不斷裂」這個問題。
- **Conv+Add fusion 有形狀要求**:TensorRT 想把 residual add fuse 進 conv kernel,
  前提是 add 的另一個輸入(identity branch)也有正確的 Q/DQ 標註 — 這是
  residual-add recipe 只量化 identity branch 的原因。

### 範例:`keep_fp16` 在 graph 裡長什麼樣

tutorial 用的 recipe([configs/deploy_config_int8_tutorial.py](../configs/deploy_config_int8_tutorial.py)):

```python
keep_fp16=[
    "pts_voxel_encoder",       # PillarFeatureNet 太小,量化只有損失
    "pts_backbone.blocks.0",   # release recipe:skip stage 0
],
disable_recipes=["add"],       # SECOND 沒有 residual add,recipe 明確關掉
```

對應到 ONNX:backbone+neck+head 一共 32 個 conv-like op,其中 **恰好 4 個沒有 Q/DQ**,
就是 stage 0 的 `blocks.0.0 / 0.3 / 0.6 / 0.9`;graph 開頭是一段乾淨的 float:

```
spatial_features
  → Conv(blocks.0.0) → Relu → Conv(0.3) → Relu → Conv(0.6) → Relu → Conv(0.9) → Relu
                                                                                  │  ← FP16 / INT8 邊界
                        ┌─────────────────────────────────────────────────────────┤
                        ├→ QuantizeLinear → DequantizeLinear → Conv(blocks.1.0) → ...
                        └→ QuantizeLinear → DequantizeLinear → ConvT(deblocks.0.0) → ...
```

stage 0 的輸出被兩個 `QuantizeLinear` 消費(backbone stage 1 和 neck 的 deblocks.0),
這兩處就是 TensorRT 會放 reformat 的地方;之後的 28 個 conv 全在同一個 INT8 區段裡,
所以整條網路只付這一次進場成本。實測([README](../README.md)):
backbone+neck+head **5.92 ms → 3.47 ms(1.71×)**,TensorRT mAP 0.4996 → 0.4938(−0.006)。

想自己驗證「哪些 conv 沒被量化」:

```python
import onnx
g = onnx.load("int8/onnx/pts_backbone_neck_head.onnx").graph
dq = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
print([n.name for n in g.node
       if n.op_type in ("Conv", "ConvTranspose") and n.input[0] not in dq])
```

### 官方的 7 條「Q/DQ Layer-Placement Recommendations」對照我們的框架

這是整份官方文件最該背下來的一節。每一條都能直接對到 `deployment/quantization/`
裡的一個設計決定:

| # | 官方建議 | 我們的做法 | 在哪裡 |
|---|---|---|---|
| 1 | Quantize **all inputs of weighted operations**(Convolution / Transposed Convolution / GEMM) | Conv2d / ConvTranspose2d / Linear 的 input + weight 都插 quantizer | `core/descriptors.py`、`quant_model.py` |
| 2 | **By default, do not quantize the outputs** of weighted operations | 從來不量化 conv 輸出 —— 只有 input 與 weight quantizer | 同上 |
| 3 | **Do not simulate BN and ReLU fusions** in the training framework | ⚠️ ReLU 確實沒碰,但我們**有** `fuse_bn=True` —— 見 `04` §7 的分歧說明 | `schemes.py` |
| 4 | **Quantize the residual input in skip connections** | `add` recipe:只量化 identity branch | `recipes/attach.py`(SECOND 用不到,`disable_recipes=["add"]`) |
| 5 | Try quantizing layers that **do not commute** with Q/DQ;注意「non-weighted layers with INT8 inputs also require INT8 outputs」 | concat / eSE 的 Mul / maxpool recipe 就是在補這些 op 的 scale | `recipes/` |
| 6 | **Be conservative** when adding Q/DQ;同時看精度和效能 | `keep_fp16` + `disable_recipes` 兩個開關,預設保守(voxel encoder、stage 0 留 FP16) | 各 int8 config |
| 7 | **per-tensor for activations、per-channel for weights** | activation per-tensor + histogram/MSE;Conv weight per-channel(axis=0)+ max | `core/descriptors.py`(見 §2 的表) |

第 1 條的「weighted operations」名單(Conv / Transposed Conv / GEMM)正好就是我們
框架 `core/` 唯一認識的三種 module —— 不是巧合,是照著這條寫的。

第 5 條的括號那句最容易被忽略:**非 weighted op 一旦輸入是 INT8,輸出也必須是 INT8**。
所以「量化 concat 的輸入」不是量一半就好,它的輸出也得有下游的 Q(在我們的 graph 裡
是後面 1×1 conv 的 input quantizer 提供的)。

### 兩個官方限制,踩到會默默失去 fusion

- **可 refit 的 engine 會關掉「比較 scale 是否相等」的優化。** 官方 *Q/DQ Limitations*:
  > "TensorRT will not apply these scale-dependent rewrites in cases where refitting Q/DQ
  > scales could result in two scales changing from equal to not equal."

  我們有好幾個 recipe 的效益**正是**建立在「兩個 Q 的 scale 相等」上(單 Q fan-out、
  no-downsample 的 residual 重用 `conv1._input_quantizer`)。所以:
  **INT8 engine 不要開 refit**,不然會白白多出 reformat。目前我們的 exporter 沒有開,
  這是一條「別去改」的注意事項。
- **官方建議 explicit quantization 搭配 strongly typed network**,而且
  > "Precision-control build flags are not required and should not be specified."

  但我們的 int8 config 用的是 `precision_policy="fp16"`(即 `BuilderFlag.FP16`,
  weakly typed):
  ```python
  # deployment/projects/centerpoint/config/_deploy_config_int8_base.py:93
  precision_policy="fp16",
  ```
  這是**刻意的**:strongly typed 會嚴格照 ONNX 的 dtype 決定精度,而我們的 ONNX
  是 FP32 型別 —— 那麼 `keep_fp16` 的區段(voxel encoder、backbone stage 0)就會跑成
  FP32,而不是我們想要的 FP16。Q/DQ 存在時 TensorRT 一樣走 explicit quantization,
  FP16 flag 只影響「沒被量化的那些層可以用什麼精度」。

## 7. 名詞對照表

| 名詞 | 意義 |
|---|---|
| amax | 量化覆蓋的最大絕對值;`scale = amax/127` |
| dynamic range | TensorRT 用語,= ±amax |
| calibration | 用少量資料統計 activation 分佈、決定 amax 的過程 |
| fake quant | float 域中模擬 quantize→dequantize 的運算 |
| Q/DQ | ONNX 的 QuantizeLinear / DequantizeLinear node 對 |
| explicit quantization | TensorRT 由 graph 中的 Q/DQ 決定精度配置的模式 |
| quantizable layer | 能把前後的 DQ/Q 吃進自己 kernel 的 layer(Conv、GEMM、AveragePool…) |
| commuting layer | 與 Q/DQ 可交換的 layer(`Q(Op(x))==Op(Q(x))`),官方證明的例子是 Max Pooling |
| Q/DQ propagation | TensorRT 把 Q 往前推、DQ 往後推,以擴大低精度區段的過程 |
| strongly typed | 精度完全由 network 的 dtype 決定、不接受 precision flag 的 build 模式 |
| WoQ (weight-only quant) | 只有 weight 量化(INT4 block),GEMM 的輸入與計算維持高精度 |
| PTQ | Post-Training Quantization:只校準、不重訓 |
| QAT | Quantization-Aware Training:插著 fake quant 繼續 fine-tune,讓權重適應量化誤差 |

## 參考資料

- **NVIDIA TensorRT — Explicit Quantization**(§6 的所有規則、7 條擺放建議、commutation
  定義、Q/DQ Limitations 的出處):
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html>
- 該頁的章節結構(方便查原文):Explicit Quantization / Quantized Weights / ONNX Support /
  TensorRT Processing of Q/DQ Networks / Weight-Only Quantization /
  **Q/DQ Layer-Placement Recommendations** / **Q/DQ Limitations** /
  Q/DQ Interaction with Plugins / QAT Networks Using TensorFlow / QAT Networks Using PyTorch
- 一個時代註記:該頁現在推薦用 **TensorRT Model Optimizer**(ModelOpt)做 PTQ/QAT 與 export,
  已經不再提 `pytorch-quantization`。我們的框架仍建立在 `pytorch-quantization` 上
  (與 CUDA-CenterPoint / lidar-ai-solution 對齊),數值語意相同,但如果未來要跟上官方
  工具鏈,遷移點會在 `core/descriptors.py` 與 `core/replace.py`。

→ 下一篇:[02 — PTQ 校準:histogram 怎麼變成 amax](02_ptq_calibration_histogram.md)
