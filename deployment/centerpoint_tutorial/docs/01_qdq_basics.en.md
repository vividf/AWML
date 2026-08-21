# 01 — Q/DQ Basics: What INT8 Quantization Actually Does

*English version — [中文版 / Chinese](01_qdq_basics.md)*

> Audience: newcomers seeing model quantization for the first time. After reading you should
> be able to answer: where does the scale come from, does each layer observe independently
> during calibration or is it a relay, what does a Q/DQ node look like inside an ONNX graph,
> and what does TensorRT do once it sees Q/DQ.
>
> **Authoritative reference**: every rule in §6 comes from the official NVIDIA TensorRT document
> [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
> — when in doubt that page wins; this is only a "mapped onto our recipe" translation of it.

## 1. Why INT8

Convolution at inference time is mostly multiply-accumulate. Going FP16 → INT8 buys you:

- **Throughput**: the theoretical throughput of INT8 Tensor Cores on NVIDIA GPUs is 2× FP16.
- **Memory / bandwidth**: both activations and weights shrink by half, which is often the
  dominant win for memory-bound layers (convs on large feature maps).
- **The price**: INT8 has only 256 representable values, so the representable range and
  resolution have to be decided by "calibration" — which is exactly the core job of PTQ.

## 2. Symmetric linear quantization (the scheme we use)

INT8 in pytorch-quantization / TensorRT is **symmetric linear quantization**, with no zero-point:

```
q = clamp( round( x / scale ), -128, 127 )     # Quantize
x̂ = q * scale                                  # Dequantize
```

There is exactly one free parameter, `scale`, determined by **amax** (the largest absolute
value you want to cover):

```
scale = amax / 127
```

Two error terms pull against each other, and choosing amax is finding the balance between them:

- **clipping error**: values with |x| > amax get truncated. amax too small → all the large
  values get chopped off.
- **rounding error**: the spacing between adjacent representable values is scale = amax/127.
  amax too large → the spacing gets too coarse, and the quantization error of the majority
  (the small values) grows.

### Example: what a real scale actually does

Take the activation quantizer of the first quantized conv this tutorial calibrated
(`pts_backbone.blocks.1.0`): amax = 6.0219 → `scale = 6.0219 / 127 = 0.047417`.
Push a few values through it:

| x (FP) | x / scale | q (INT8) | x̂ = q·scale | error |
|---|---|---|---|---|
| 0.0123 | 0.26 | 0 | 0.0000 | −0.0123 |
| 0.5000 | 10.54 | 11 | 0.5216 | +0.0216 |
| 1.3000 | 27.42 | 27 | 1.2802 | −0.0198 |
| −2.7000 | −56.94 | −57 | −2.7027 | −0.0027 |
| 6.0219 | 127.00 | 127 | 6.0219 | 0 |
| 8.5000 | 179.26 | **127** (clamp) | 6.0219 | **−2.4781** |

The first five rows are rounding error (bounded at a fixed ±scale/2 = ±0.0237); the last row
is clipping error — two orders of magnitude larger. Choosing amax is betting between
"pay a bit more rounding everywhere" and "let a few values pay a blow-up clipping cost".

This is exactly why "just take the observed max as amax" (method=max) is usually a bad idea:
activation distributions are nearly always long-tailed, and one or two outliers blow up the
scale, hurting 99.9% of the values.

Using the **histogram actually recorded for this layer** (60 calibration samples × 64ch ×
1020×1020 ≈ 4.0e9 values, `calib_trace/hist_trace.pkl`) we can compute the total error of all
four amax choices directly:

| amax source | amax | scale | mean quantization error (MSE) | RMSE | values clipped |
|---|---|---|---|---|---|
| observed max | 16.8703 | 0.1328 | 8.39e−04 | 0.0290 | 0% |
| **mse (what we use)** | **6.0219** | **0.0474** | **1.01e−04** | **0.0101** | 0.0011% |
| percentile 99.9 | 2.7558 | 0.0217 | 1.11e−03 | 0.0334 | 0.1006% |
| entropy | 0.9534 | 0.0075 | 2.91e−02 | 0.1706 | 2.7569% |

The error of `max` is **8.3× that of `mse`** — not because it chops off the wrong things
(it clips 0%), but because accommodating that single 16.87 outlier makes the scale 2.8×
coarser, while **97.3% of the values in this layer actually live at |x| ≤ 1.0**. The other
direction loses too: `percentile 99.9` only clips 0.1% of the values, yet its error is 11×
that of `mse`. `02_ptq_calibration_histogram.en.md` shows how these numbers grow out of the
histogram step by step.

### per-tensor vs per-channel

| | who uses it | amax shape | why |
|---|---|---|---|
| **per-tensor** | activations (input quantizer) | scalar | activations differ on every inference, so only one scale is possible; a TensorRT tensor also has only one dynamic range |
| **per-channel** | weights (weight quantizer) | `[out_channels]` | weights are constants, so you can take an exact max per output channel (axis=0); there is no reason to share a scale |

Weight amax uses `MaxCalibrator` (just take each channel's max|w|, no dataset needed);
activation amax uses `HistogramCalibrator` + the MSE criterion (needs calibration data) —
see the next document.

**Example: what per-channel is worth.** The weight of that same `blocks.1.0` has shape
`[128, 64, 3, 3]`; the `max|w|` across its 128 output channels:

```
min     0.0567      # the "quietest" channel
median  0.1547
max     0.4273      # max / min = 7.53×
```

If we forced per-tensor (all 128 channels sharing amax = 0.4273), the channel whose
amax is 0.0567 would only get `127 × 0.0567 / 0.4273 ≈ 17` quantization levels — throwing
away nearly 3 bits of resolution. per-channel lets every channel use all 127 levels, and it
is free: weight scales are constants, and TensorRT folds those 128 scales straight into the
kernel.

## 3. Calibration computes the whole graph at once; only inference is pipelined

The most common misconception: that PTQ goes "fix the first layer's amax → quantize it →
recompute the second layer's input from the quantized output → then fix the second layer's
amax", relaying layer by layer. **It does not.**

During calibration the entire network runs a **pure FP forward**; all quantizers only watch,
they do not quantize. `CalibrationManager` flips all 56 quantizers into collect mode in one
go, **before** the data loop (`deployment/quantization/core/calibration.py`):

```python
def _enable_calibration_mode(self):          # L99, called once before the loop
    for name, module in self.model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable_quant()   # fake quant off → the forward is pure FP behavior
            module.enable_calib()    # only accumulate the tensors flowing through into its own histogram
```

The timeline looks like this:

```
samples 1..60 ── pure FP forward ──┬──> histogram of blocks.1.0  (sees the FP stage0 output)
                                   ├──> histogram of blocks.1.3  (sees the FP blocks.1.0 output)
                                   └──> ... 56 quantizers each accumulate, mutually independent
end of loop ──── compute_amax("mse") × 56 ──> fully independent, no ordering dependency
```

In other words: **amax is calibrated on clean FP distributions**, free of upstream
quantization error.

### Why this approximation is safe

The perturbation introduced by upstream quantization is small relative to the distribution
itself. Take the input of `blocks.1.0` (computed from that same real histogram in §2):

| | value |
|---|---|
| signal RMS (\|x\|) | 0.4203 |
| quantization noise RMSE | 0.0101 |
| **noise / signal** | **2.4%** |
| noise / amax | 0.17% |

A 2.4% perturbation barely changes the shape of the histogram, so the optimal amax found by
the MSE search barely moves either — which is why "calibrate on the FP distribution, then use
it in the quantized pipeline" is a safe approximation.

(The methods that genuinely calibrate in a pipelined fashion are **layer-wise / block-wise
reconstruction** methods like AdaRound / BRECQ: generate the input from the already-quantized
prefix, then run one optimization per layer. Slightly better accuracy, an order of magnitude
more cost. Neither pytorch-quantization's nor TensorRT's PTQ takes that road.)

### But inference really is pipelined

| stage | pipelined? | what layer N actually receives |
|---|---|---|
| **calibration** (collect + compute_amax) | ❌ one-shot, parallel observation | the FP output of layer N−1 |
| **inference** (fake-quant eval / TRT engine) | ✅ | the quantize→dequantize output of layer N−1; error accumulates |

Error accumulation is real; calibration simply does not model it — it shows up in the final
metrics: PyTorch fake-quant eval mAP 0.4973 → 0.4857. To make the **weights** adapt to that
accumulated error you need **QAT**: keep all the fake quants on and continue fine-tuning, so
that at training time layer N already sees quantized inputs.

### A concrete case of "everyone is observing the same FP graph"

The input quantizers of the head's six branches have **exactly identical** amax, because they
all quantize the same `shared_conv` output tensor:

```
pts_bbox_head.task_heads.0.{heatmap,reg,height,dim,rot,vel}.0.conv._input_quantizer
    → amax = 6.6205  (all six the same)
```

A quantizer hangs off the **consumer side** of a tensor: if the same tensor is consumed by N
convs, then N quantizers collect exactly the same distribution. TensorRT will merge these
duplicate Q/DQ pairs later (see Q/DQ propagation in §6).

## 4. Fake quantization (what QAT/PTQ looks like inside PyTorch)

Training frameworks have no real INT8 kernels. What `pytorch-quantization`'s `TensorQuantizer`
does is **fake quant**: numerically it performs quantize→dequantize, but the tensor stays FP:

```
x̂ = clamp(round(x/scale), -128, 127) * scale    # still float, but already "looks like" INT8
```

`QuantConv2d` = a `TensorQuantizer` inserted in front of both the input and the weight of a
Conv2d:

```
x ──> [input_quantizer (per-tensor)] ──┐
                                       ├──> Conv2d ──> y
w ──> [weight_quantizer (per-channel)]─┘
```

So the mAP you evaluate in PyTorch after PTQ is a preview of the "INT8 numerical behavior"
(the pytorch-backend eval in this tutorial is exactly that).

### Example: what a PTQ checkpoint has that an FP checkpoint doesn't

Only the `_amax` buffers — not a single bit of the weights moved (calibration does not touch
weights):

```python
>>> sd = torch.load("checkpoints/epoch_29_ptq_tutorial.pth")["state_dict"]
>>> len(sd), len([k for k in sd if "quantizer" in k])
(132, 56)                       # 56 = 28 quantized convs × (input + weight)

>>> sd["pts_backbone.blocks.1.0._input_quantizer._amax"]
tensor(6.0219)                  # per-tensor → scalar

>>> sd["pts_backbone.blocks.1.0._weight_quantizer._amax"].shape
torch.Size([128, 1, 1, 1])      # per-channel → one per output channel
```

The same set of amax values is also stored separately as
`checkpoints/epoch_29_ptq_tutorial.calib` (a pure amax dict, convenient for inspecting on its
own or diffing against the release — this is what `03_compare_amax_table.py` reads).

## 5. Q/DQ nodes in ONNX

Once `use_fb_fake_quant` is enabled at export time, every `TensorQuantizer` is exported as a
pair of ONNX nodes:

```
QuantizeLinear(x, scale)  ->  int8 tensor
DequantizeLinear(q, scale)  ->  float tensor
```

The graph becomes (for one conv):

```
input ─ QuantizeLinear ─ DequantizeLinear ─ Conv ─ ...
weight ─ QuantizeLinear ─ DequantizeLinear ─┘
```

Note: in the ONNX graph the Conv itself is still a float op. The Q/DQ pair merely writes the
information "this tensor should be quantized to INT8 with this scale" into the graph.

### Example: the real exported nodes (`int8/onnx/pts_backbone_neck_head.onnx`)

The `blocks.1.0` conv is actually 10 nodes in the graph (node index 8–17):

```
 8 Constant          .../blocks.1.0/_input_quantizer/Constant       → zero_point: int8 scalar 0
 9 Constant          .../blocks.1.0/_input_quantizer/Constant_1     → scale: float32 0.04741656
10 QuantizeLinear    .../blocks.1.0/_input_quantizer/QuantizeLinear
       in: /backbone/blocks.0/blocks.0.11/Relu_output_0, scale, zero_point
11 DequantizeLinear  .../blocks.1.0/_input_quantizer/DequantizeLinear
12 Constant          .../blocks.1.0/_weight_quantizer/Constant       → scale: float32[128]  ← per-channel
13 Constant          .../blocks.1.0/_weight_quantizer/Constant_1     → zero_point: int8[128] all zeros
14 QuantizeLinear    .../blocks.1.0/_weight_quantizer/QuantizeLinear
       in: model.backbone.blocks.1.0.weight, scale[128], zero_point[128]
15 DequantizeLinear  .../blocks.1.0/_weight_quantizer/DequantizeLinear
16 Conv              .../blocks.1.0/Conv
       in: _input_quantizer/DequantizeLinear_output_0,
           _weight_quantizer/DequantizeLinear_output_0,
           model.backbone.blocks.1.0.bias          ← bias is not quantized, it stays FP
17 Relu              .../blocks.1.2/Relu
```

Three things here map straight back to earlier sections:

- `scale = 0.04741656`, times 127 = 6.0219 = the `_input_quantizer._amax` in the checkpoint.
  **ONNX stores the scale, the checkpoint stores the amax**; they differ by a factor of 127.
- The input scale is a scalar and the weight scale is `[128]` — the per-tensor / per-channel
  distinction is written right into the graph, visible to the naked eye.
- All zero_points are 0 → symmetric quantization; bias has no Q/DQ (TensorRT adds the bias
  inside the INT32 accumulator).

### Example: op statistics of the same graph before / after PTQ

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
| Constant | 0 | 112 (56 pairs of scale + zero_point) |

The compute network itself (59 ops) is completely unchanged — INT8 export is purely
"annotating on top of it". 56 Q/DQ pairs = 28 quantized convs × 2 (input + weight), which
matches the 56 amax values in the checkpoint.

## 6. How TensorRT digests Q/DQ (explicit quantization)

When TensorRT sees Q/DQ nodes it runs in **explicit quantization** mode:

1. **Layer fusion**: a pattern like `DQ → Conv → ReLU → Q` is fused into a single
   **INT8-in / INT8-out conv kernel**, with the scales burned into the kernel. The official
   term for "a layer that can swallow Q/DQ this way" is a **quantizable layer**
   (the examples the doc gives: Convolution, GEMM, **AveragePool**).
2. **Q/DQ propagation**: Q/DQ nodes are moved around to "maximize the proportion of the
   low-precision region". The direction is explicit:
   > "TensorRT propagates **Quantize nodes backward** (so quantization happens as early as
   > possible) and **Dequantize nodes forward** (so dequantization happens as late as possible)."

   Whether a node can move depends on whether the op **commutes**. The doc gives a formal
   definition:

   ```
   Op commutes with quantization    ⇔  Q(Op(x)) == Op(Q(x))
   Op commutes with dequantization  ⇔  Op(DQ(x)) == DQ(Op(x))
   ```

   The only example the doc proves in full is **Max Pooling** (taking a max does not change
   the ordering that Q preserves, so it commutes with both Q and DQ). **AveragePool does not
   commute** — averaging produces new values — so it can only become INT8 by being sandwiched
   between Q/DQ and fused. That is precisely the difference between the words *quantizable*
   and *commuting*.
3. **Regions not wrapped in Q/DQ** keep executing in FP16/FP32.

A few consequences that directly shape our recipe design:

- **We hold the control over precision**: wherever we insert Q/DQ is INT8; where we don't,
  TensorRT will not decide on its own (the opposite of implicit / calibration-cache mode).
  The `keep_fp16` list (e.g. the voxel encoder, backbone stage 0) is implemented exactly by
  "insert no Q/DQ + disable the quantizers".
- **Breaks are expensive**: if there is one unquantized op in the middle of an INT8 region
  (or a Reshape/Transpose blocking fusion), TensorRT has to insert a reformat
  (an INT8→FP16→INT8 conversion layer) — losing both time and accuracy. In
  `04_backbone_recipes.en.md`, ResNet's residual add, VoV's eSE and ConvNeXt's various
  permutes are all solving the same problem: "how do I keep the INT8 region from breaking".
- **Conv+Add fusion has shape requirements**: for TensorRT to fuse a residual add into the
  conv kernel, the other input of the add (the identity branch) must also carry the correct
  Q/DQ annotation — which is why the residual-add recipe quantizes only the identity branch.

### Example: what `keep_fp16` looks like in the graph

The recipe this tutorial uses ([configs/deploy_config_int8_tutorial.py](../configs/deploy_config_int8_tutorial.py)):

```python
keep_fp16=[
    "pts_voxel_encoder",       # PillarFeatureNet is too small; quantizing it is pure loss
    "pts_backbone.blocks.0",   # release recipe: skip stage 0
],
disable_recipes=["add"],       # SECOND has no residual add; the recipe is explicitly off
```

Mapped onto ONNX: the backbone+neck+head has 32 conv-like ops in total, of which
**exactly 4 have no Q/DQ** — stage 0's `blocks.0.0 / 0.3 / 0.6 / 0.9`; the start of the graph
is a clean stretch of float:

```
spatial_features
  → Conv(blocks.0.0) → Relu → Conv(0.3) → Relu → Conv(0.6) → Relu → Conv(0.9) → Relu
                                                                                  │  ← FP16 / INT8 boundary
                        ┌─────────────────────────────────────────────────────────┤
                        ├→ QuantizeLinear → DequantizeLinear → Conv(blocks.1.0) → ...
                        └→ QuantizeLinear → DequantizeLinear → ConvT(deblocks.0.0) → ...
```

The stage 0 output is consumed by two `QuantizeLinear` nodes (backbone stage 1 and the neck's
deblocks.0); those two spots are where TensorRT will put reformats. Everything after that —
28 convs — lives in one single INT8 region, so the whole network pays this entry cost only
once. Measured ([README](../README.en.md)): backbone+neck+head
**5.92 ms → 3.47 ms (1.71×)**, TensorRT mAP 0.4996 → 0.4938 (−0.006).

To verify "which convs are not quantized" yourself:

```python
import onnx
g = onnx.load("int8/onnx/pts_backbone_neck_head.onnx").graph
dq = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
print([n.name for n in g.node
       if n.op_type in ("Conv", "ConvTranspose") and n.input[0] not in dq])
```

### The official 7 "Q/DQ Layer-Placement Recommendations" mapped to our framework

This is the one section of the official document most worth memorizing. Each item maps
directly to one design decision inside `deployment/quantization/`:

| # | Official recommendation | What we do | Where |
|---|---|---|---|
| 1 | Quantize **all inputs of weighted operations** (Convolution / Transposed Convolution / GEMM) | Both the input and the weight of Conv2d / ConvTranspose2d / Linear get a quantizer | `core/descriptors.py`, `quant_model.py` |
| 2 | **By default, do not quantize the outputs** of weighted operations | We never quantize a conv output — only input and weight quantizers exist | same as above |
| 3 | **Do not simulate BN and ReLU fusions** in the training framework | ⚠️ We do leave ReLU alone, but we **do** use `fuse_bn=True` — see the divergence note in `04` §6a | `schemes.py` |
| 4 | **Quantize the residual input in skip connections** | the `add` recipe: quantize only the identity branch | `recipes/attach.py` (unused by SECOND, `disable_recipes=["add"]`) |
| 5 | Try quantizing layers that **do not commute** with Q/DQ; note that "non-weighted layers with INT8 inputs also require INT8 outputs" | The concat / eSE Mul / maxpool recipes exist exactly to supply scales for these ops | `recipes/` |
| 6 | **Be conservative** when adding Q/DQ; watch both accuracy and performance | Two switches, `keep_fp16` + `disable_recipes`, conservative by default (voxel encoder and stage 0 stay FP16) | every int8 config |
| 7 | **per-tensor for activations, per-channel for weights** | activations per-tensor + histogram/MSE; conv weights per-channel (axis=0) + max | `core/descriptors.py` (see the table in §2) |

The list of "weighted operations" in item 1 (Conv / Transposed Conv / GEMM) happens to be
exactly the three module types our framework's `core/` knows about — not a coincidence, it
was written to follow this item.

The parenthetical in item 5 is the easiest one to overlook: **once a non-weighted op has INT8
inputs, its output must be INT8 too**. So "quantize the inputs of a concat" is not half the
job — its output needs a downstream Q as well (in our graph that is provided by the input
quantizer of the following 1×1 conv).

### Two official limitations that silently cost you fusion if you hit them

- **A refittable engine disables the "are these two scales equal" optimization.** From the
  official *Q/DQ Limitations*:
  > "TensorRT will not apply these scale-dependent rewrites in cases where refitting Q/DQ
  > scales could result in two scales changing from equal to not equal."

  Several of our recipes derive their benefit **precisely** from "two Q nodes having equal
  scales" (single-Q fan-out; the no-downsample residual reusing `conv1._input_quantizer`).
  Therefore: **do not enable refit on the INT8 engine**, or you gain extra reformats for
  nothing. Our exporter does not enable it today; treat this as a "don't touch it" note.
- **The official guidance is to pair explicit quantization with a strongly typed network**, and
  > "Precision-control build flags are not required and should not be specified."

  But our int8 config uses `precision_policy="fp16"` (i.e. `BuilderFlag.FP16`, weakly typed):
  ```python
  # deployment/projects/centerpoint/config/_deploy_config_int8_base.py:93
  precision_policy="fp16",
  ```
  This is **deliberate**: strongly typed would decide precision strictly from the ONNX dtypes,
  and our ONNX is FP32-typed — which would make the `keep_fp16` regions (voxel encoder,
  backbone stage 0) run in FP32 instead of the FP16 we want. As long as Q/DQ is present
  TensorRT still runs explicit quantization; the FP16 flag only affects "what precision the
  unquantized layers are allowed to use".

## 7. Glossary

| Term | Meaning |
|---|---|
| amax | the maximum absolute value quantization covers; `scale = amax/127` |
| dynamic range | TensorRT's term, = ±amax |
| calibration | the process of collecting activation distributions on a small amount of data to decide amax |
| fake quant | simulating quantize→dequantize arithmetic in the float domain |
| Q/DQ | a pair of ONNX QuantizeLinear / DequantizeLinear nodes |
| explicit quantization | the TensorRT mode where the precision layout is decided by the Q/DQ in the graph |
| quantizable layer | a layer that can swallow the surrounding DQ/Q into its own kernel (Conv, GEMM, AveragePool…) |
| commuting layer | a layer that commutes with Q/DQ (`Q(Op(x))==Op(Q(x))`); the officially proven example is Max Pooling |
| Q/DQ propagation | TensorRT pushing Q backward and DQ forward to widen the low-precision region |
| strongly typed | a build mode where precision is decided entirely by the network's dtypes and precision flags are not accepted |
| WoQ (weight-only quant) | only weights are quantized (INT4 block); GEMM inputs and compute stay high precision |
| PTQ | Post-Training Quantization: calibrate only, no retraining |
| QAT | Quantization-Aware Training: keep fake quant inserted and continue fine-tuning so the weights adapt to quantization error |

## References

- **NVIDIA TensorRT — Explicit Quantization** (the source of every rule in §6: the 7 placement
  recommendations, the commutation definition, the Q/DQ Limitations):
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html>
- That page's section structure (for looking up the original): Explicit Quantization /
  Quantized Weights / ONNX Support / TensorRT Processing of Q/DQ Networks /
  Weight-Only Quantization / **Q/DQ Layer-Placement Recommendations** / **Q/DQ Limitations** /
  Q/DQ Interaction with Plugins / QAT Networks Using TensorFlow / QAT Networks Using PyTorch
- A note on the era: that page now recommends **TensorRT Model Optimizer** (ModelOpt) for
  PTQ/QAT and export, and no longer mentions `pytorch-quantization` at all. Our framework is
  still built on `pytorch-quantization` (aligned with CUDA-CenterPoint / lidar-ai-solution);
  the numerical semantics are the same, but if we ever want to follow the official toolchain
  the migration points are `core/descriptors.py` and `core/replace.py`.

→ Next: [02 — PTQ calibration: how a histogram becomes an amax](02_ptq_calibration_histogram.en.md)
