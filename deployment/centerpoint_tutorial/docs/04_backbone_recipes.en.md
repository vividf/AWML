# 04 — Per-Backbone Quantization Special Handling: ResNet add / VoV eSE / ConvNeXt

*English version — [中文版 / Chinese](04_backbone_recipes.md)*

> Prerequisite: [01 — Q/DQ Basics](01_qdq_basics.en.md) (especially the 7 official placement
> recommendations in §6).
> This document explains the three architecture recipes in
> `deployment/quantization/recipes/`: all of them solve the same problem —
> **keep the INT8 region unbroken and avoid superfluous reformats**.
> (Code locations refer to branch `feat/quantization_framework`.)
>
> Each recipe is annotated with which NVIDIA
> [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
> rule it corresponds to; §6 collects the places where **the official doc says something and we
> do it differently**.

## 0. How the framework divides responsibility (build the map first)

```
deployment/quantization/
├── core/          # the engine: knows only nn.Conv2d / ConvTranspose2d / Linear, BN fusion, calibration
├── recipes/       # architecture knowledge: BasicBlock / eSEModule / _OSA_module / ConvNeXtBlock / MaxPool2d
├── schemes/       # the seam: QuantizationScheme.prepare() / QuantizationPlan
└── producer.py    # shared pieces of the PTQ/QAT producers

deployment/projects/centerpoint/quantization/
├── plan.py        # build_centerpoint_plan(config) — the single entry point
├── quant_model.py # CenterPoint's tower-level composition (which tower swaps Conv, which swaps Linear)
└── schemes.py     # CenterPointDenseScheme (fuse BN → expand keep_fp16 → quant_model)
```

**The sacred, inviolable invariant**: the PTQ producer, the QAT hook and the deploy loader —
all three consumers — call the same `build_centerpoint_plan(config).prepare(model)`, so the
state_dict produced by calibration and the module tree loaded at deployment time are
**aligned by construction** (locked down by a unit test:
`deployment/tests/test_qat_tree_parity.py`).

The order inside `prepare()` (`schemes.py:50`):

1. `fuse_model_bn(model)` — **ignores keep_fp16, fuses the entire model** (because fusion
   changes state_dict keys, and both sides must fuse the same set).
2. `expand_keep_fp16(model, patterns)` — fnmatch-expanded into concrete module names plus their
   whole subtrees; **a pattern that matches nothing raises a WARNING** (typo guard).
3. `quant_model(model, skip_names, disable_recipes)` — swap the modules, then attach recipes:

```python
# quant_model.py — tower-level knowledge is hardcoded, not config
pts_backbone      → swap Conv for QuantConv2d/QuantConvTranspose2d + Linear for QuantLinear
pts_neck          → swap Conv only
pts_bbox_head     → swap Conv only
pts_voxel_encoder → swap Linear only
# recipes: always-on, class-gated (a no-op if the model has no such class)
attach_quant_add(model)              # can be turned off with disable_recipes=["add"]
attach_ese_quantizers(model)         # can be turned off with disable_recipes=["ese"]
attach_maxpool_input_quantizer(...)  # can be turned off with disable_recipes=["maxpool"]
```

> **Mapping from the old flags** (in case you have seen the old log line
> `backbone=True, ..., quant_ese_mul_identity=False`): that set of 13 booleans has been removed
> wholesale. `quant_voxel_encoder=False` → `keep_fp16=["pts_voxel_encoder"]`;
> `skip_backbone_stages=[0]` → `keep_fp16=["pts_backbone.blocks.0"]`; `quant_add` →
> `disable_recipes=["add"]`; `quant_ese_pool_input` / `quant_ese_mul_identity` → merged into the
> always-on single-Q eSE design (described below); `quant_linear_backbone` → removed, backbone
> Linears are always quantized.

### Quantizer defaults (core/descriptors.py — the framework's single source)

| tensor | granularity | calibrator |
|---|---|---|
| input of Conv2d / Linear | per-tensor | **histogram** (→ MSE) |
| Conv2d weight | **per-output-channel** (axis=0) | max |
| **ConvTranspose2d weight** | **per-tensor** (special case!) | max |
| Linear weight | per-row (axis=0) | max |
| quantizers attached by recipes (add/eSE/concat/maxpool) | per-tensor | histogram (sharing the same descriptor object as conv inputs) |

This table is the implementation of the official recommendation #7:
> "Use per-tensor quantization for activations and per-channel quantization for weights.
> This configuration has been demonstrated empirically to lead to the best quantization accuracy."

**Why Linear uses axis=0** is explained by a dedicated warning in the official document (an
easy trap to fall into): PyTorch's `nn.Linear` exports to an ONNX GEMM whose weight layout is
`(K, C)` with a `transB` attribute; TensorFlow transposes to `(C, K)` before export. Hence
> "GEMM layers originating from ONNX QAT models that were exported from PyTorch use dimension
> `0` for per-channel quantization (axis `K = 0`), while models originating from TensorFlow use
> dimension `1`."

We are on the PyTorch route → axis=0, consistent with the table.

**ConvTranspose2d is the one deviation**: the official doc explicitly lists Transposed
Convolution among the "weighted operations" (meaning its weight should also be per-channel), but
in practice TensorRT's INT8 deconv is fragile with per-channel scales (the engine build blows up
with `vol == 1` / `Could not find any implementation`), so we fall back to per-tensor.
Every int8 config also keeps a commented-out escape hatch,
`# "pts_neck.deblocks.*.0"` (keeps the entire deconv in FP16).

---

## 1. SECOND (this tutorial's model): the simplest case

SECOND's 2D backbone is a pure stack of Conv-BN-ReLU — **no residual, no concat, no pooling** —
so all three recipes are no-ops. The config still writes `disable_recipes=["add"]` purely to
document the intent of "do not quantize add" (behaviorally equivalent).

The release recipe's two keep_fp16 entries are pure accuracy considerations:

- `pts_voxel_encoder`: consistently kept in FP16 across all int8 configs. It is the model's
  numerical front door (raw point coordinates + pillar offsets, a wide dynamic range), and its
  BatchNorm1d **would not be fused anyway** (mmdet3d's PFNLayer registers the norm before the
  linear, and BN fusion only recognizes the adjacent Conv→BN order).
- `pts_backbone.blocks.0`: the first stage runs at full resolution (1020²), so its quantization
  error propagates through the whole network — experimentally unstable, so the whole segment
  stays FP16 (the origin of the release name skip_stage_0).

## 2. ResNet: "quantize the identity branch only" for the residual add

**The problem**: a BasicBlock computes `out = conv_path + identity`. If Q/DQ is inserted on both
branches, TensorRT cannot fuse the Add into the preceding conv kernel, and you get a separate
Add layer plus reformats.

**What the official doc says**: this is exactly item 4 of the
*Q/DQ Layer-Placement Recommendations* — **"Quantize the residual input in skip connections."**
The reasoning the doc gives matches the phenomenon we hit word for word: if the residual path
(written `x_f^2` in the doc) stays in high precision, then
> "the precision of `x_f^2` is high precision, so the output of the fused convolution is
> limited to high precision, and the **trailing Q-layer cannot be fused** with the convolution."

Conversely, quantizing it to INT8:
> "the output of the fused convolution is also INT8, and the trailing Quantize layer is fused
> with the convolution."

In other words, **the precision of the conv output is determined by "the other operand of the
Add"** — which is why you must quantize the identity branch rather than the conv path. The doc
also states the scope of this fusion:
> "TensorRT can fuse element-wise addition following weighted layers, which is useful for
> models with skip connections like ResNet and EfficientNet."

**The solution** (identical to NVIDIA lidar-ai-solution / CUDA-BEVFusion):
**quantize only the identity branch and leave the conv path's output completely alone** (the
latter also happens to be official item 2, "by default do not quantize the output of a weighted
op"). TensorRT's INT8 conv kernel supports doing "+residual → ReLU" as a conv epilogue, provided
that the conv's own output is not interrupted by Q/DQ and that the Add's other operand is an
INT8 tensor with a known scale:

```
x ── Q/DQ ── Conv1 ── ReLU ── Q/DQ ── Conv2 ──(FP, no Q inserted)──┐
 └── Q/DQ (residual_quantizer)── identity ─────────────────────────┴─ Add ─ ReLU ─ ...
                                    ↑ TensorRT absorbs Add+ReLU into Conv2's kernel
```

The implementation replaces `BasicBlock.forward` (`recipes/forward_hooks.py`), inserting one
line `identity = self.residual_quantizer(identity)` before `out = out + identity`.

**Where the quantizer comes from** (`recipes/attach.py`) — an important reformat-saving detail:

- block **has downsample** → identity is a different tensor (different distribution) →
  **allocate a new quantizer**.
- block **has no downsample** → identity is the block input `x` itself, and
  `conv1._input_quantizer` is already observing that same tensor → **reuse conv1's input
  quantizer directly** (shared calibration statistics; TensorRT sees a single scale and does not
  create two independent quantization origins for the same tensor).

Class gating: exact-or-substring match on the class names
`{"BasicBlock", "SparseBasicBlock", "ConvNeXtBlock", "_OSA_module"}`.
**Note that `Bottleneck` is not on the list** — switch to ResNet-50 and the residual recipe
will not fire (all current CenterPoint ResNets are R34/BasicBlock).

> Historical note: an early version had a `QuantAdd` module plus a custom ONNX symbolic (which
> quantized both operands with the same quantizer). The current design deleted it — the
> identity-branch-only forward hook needs no custom op at all: under `use_fb_fake_quant` a
> `TensorQuantizer` traces itself into standard Q/DQ, and `+` traces into a standard `Add`.

## 3. VoVNet: eSE, OSA concat, MaxPool — the reformat triple bill

The structure of VoV's OSA block:

```
x ─┬─ layer1 ─ layer2 ─ ... ─ layerN ─┐
   │    └──────┴──── (each output) ───┼── Concat ── 1×1 conv ── eSE ──(+x if identity)
   └──────────────────────────────────┘
eSE: x ── AdaptiveAvgPool(1) ── 1×1 conv(fc) ── hsigmoid ──┐
      └────────────────────────────────────────────────────┴── Mul
```

### 3a. eSE: a single Q + fan-out (the final form of the old `quant_ese_*` flags)

**The problem**: eSE's input `x` has two consumers (the gate path's avg_pool, and the Mul's
bypass). If each gets its own independent input quantizer, TensorRT treats them as **two
independent quantization origins**, because the two consumers prefer different INT8 layouts
(`NC/4HW4` vs. `NHWC16`), and therefore inserts an FP32 Reformat in front of **each**
QuantizeLinear — double the reformat overhead.

**The solution** (`docs/ese_int8_changes.md`, `eSEModuleForwardHook` in `forward_hooks.py`):
**quantize the input exactly once** and feed the quantized `qx` to both paths; give the gate a
separate quantizer so that both operands of the Mul are INT8:

```python
qx   = pool_input_quantizer(x)          # the one and only input quantization
gate = hsigmoid(fc(avg_pool(qx)))
gate = mul_gate_quantizer(gate)         # the Mul's other operand is INT8 too
return qx * gate
```

```
(conv_out FP32)
   └─ Reformat(FP32) ─ QuantizeLinear ──┬─ DQ ─ AvgPool ─ fc ─ hsigmoid ─ Q ─┐
        (only this one!)                └─ DQ ─ bypass ──────────────────────┴─ Mul (INT8×INT8)
```

One QuantizeLinear, two DequantizeLinears, one reformat.

**Compared with the official doc**: the doc uses **AveragePool** as its illustration of a
quantizable layer — "A quantizable `AveragePool` layer (in blue) is fused with the surrounding
Dequantize and Quantize layers. All three layers are replaced by a single quantized
`AveragePool` layer." The key point is that average pooling **does not commute the way max
pooling does** (averaging produces new values), so it can only become INT8 by being sandwiched
as `DQ → AvgPool → Q`.

Our eSE hook happens to produce exactly that pattern: `avg_pool`'s input is `qx` (carrying a DQ)
and its output feeds `fc` (a 1×1 QuantConv2d that brings its own input quantizer → providing the
Q). So the gate path is `DQ → AvgPool → Q → Conv`, matching the official fusible pattern.
(caveat: VoV's `avg_pool` is `AdaptiveAvgPool2d(1)`, which exports as
GlobalAveragePool/ReduceMean rather than AveragePool; whether it really fuses into an INT8
pooling has to be checked in the engine's layer information — you cannot conclude it from
reading the doc alone.)

Giving both operands of the Mul INT8 is an instance of official item 5:
**once a non-weighted op (here an ElementWise Mul) has INT8 inputs, its output must be INT8
too**, so both operands' scales must be known.

### 3b. OSA concat: quantize all the skip branches, leave the main path FP

If half of a Concat's inputs are FP and half INT8 (or the scales are all over the place),
TensorRT can only dequantize everything back to FP before concatenating — the INT8 region breaks
outright.

**Compared with the official doc**: Concatenation is not called out individually in the official
document (that page only proves Max Pooling's commutation in detail; it gives no list for
concat / slice / reshape / transpose), but the parenthetical of recommendation #5 applies
directly:
> "Try quantizing layers that do not commute with Q/DQ. Currently, non-weighted layers with
> INT8 inputs also require **INT8 outputs**, so quantize both inputs and outputs."

Concat is a non-weighted op, so "all inputs INT8" and "output also INT8" must hold
simultaneously; our recipe handles the former, and the latter is provided by the input quantizer
of the downstream 1×1 conv.

What the recipe does:

- Allocate one `concat_input_quantizers[i]` for the block input and for every layer output
  **except the last**, quantizing each skip branch before `torch.cat` → the Concat's inputs have
  consistent, known scales.
- **The last layer's output is not quantized** — the same principle as the ResNet add: leave it
  FP so TensorRT can fuse it into that layer's conv epilogue.
- For a block with `identity=True`, the input `x` has **three** consumers (the first layer's
  conv, the concat, and the post-eSE add). Quantizing it three times = three reformats → the hook
  reuses `concat_input_quantizers[0]` as the single Q, and all three consumers share the same
  quantization result.

### 3c. MaxPool: the QuantBeforePool wrapper

Every `_OSA_stage` (except stage2) starts with an `nn.MaxPool2d`. Without Q/DQ, the pool lands
in the middle of the INT8 region as an FP island (one reformat on each side). The recipe swaps
it for:

```python
class QuantBeforePool(nn.Module):
    def forward(self, x):
        return self.pool(self.quantizer(x))   # ONNX: Q → DQ → MaxPool
```

**Compared with the official doc**: max pooling is the **only** commuting layer that page proves
mathematically in full — because `Q` is monotonic, taking a max does not change the ordering:
```
max({Q(x_j, scale), Q(x_k, scale)}) = Q(max({x_j, x_k}), scale)
```
so it commutes with **both** quantization and dequantization.

This also explains why `QuantBeforePool` only quantizes the input and can ignore the output
(which superficially looks like a violation of item 5's "a non-weighted op must have its output
quantized too"): precisely because MaxPool commutes, propagation can push the DQ **backward past**
the pool, and the pool naturally becomes INT8-in / INT8-out. Only non-commuting ops (AvgPool,
Concat, Mul) are forced to annotate both ends themselves.

### 3d. VoV's keep_fp16: stem + stage2

`deploy_config_int8_vov57/99.py`:
`keep_fp16=["pts_voxel_encoder", "pts_backbone.stem", "pts_backbone.stage2"]`.
The reasoning (`docs/ptq_accuracy_vov99.md`): the early layers run at full resolution, their
amax is the hardest to estimate, and the error is amplified all the way down; on top of that the
concat+eSE structure is especially scale-sensitive. If that is not stable enough, the order for
widening the list is `stage3` → `pts_bbox_head` (left as comments in the config).

## 4. ConvNeXt: Linear, LayerNorm, permute

A ConvNeXt block (mmpretrain, `linear_pw_conv=True`):

```
dwconv 7×7 (groups=C) → LayerNorm2d → permute(0,2,3,1) → Linear → GELU → Linear → permute(0,3,1,2) → ×γ → +shortcut
```

For how the framework handles it, you need to know both **what it does** and **what it does not**:

| Component | Handling |
|---|---|
| depthwise conv | → `QuantConv2d` (`groups` preserved, per-channel weight) |
| the two pointwise `nn.Linear` | → `QuantLinear` (backbone Linears are **always** quantized — this is what replaced the old `quant_linear_backbone` flag) |
| LayerNorm / LayerNorm2d | **not fused, not quantized** — BN fusion only recognizes BatchNorm; LN stays in the graph as an FP op |
| GELU | not quantized (the Q/DQ boundaries sit on the input/weight quantizers of the two surrounding Linears) |
| permute ×2 | **preserved as-is** — there is no Linear→1×1-conv rewriting whatsoever |
| residual add | ConvNeXtBlockForwardHook, quantizes only the shortcut and reuses `depthwise_conv._input_quantizer` (the shortcut and the dwconv input are the same tensor) |

The correct reading of "there are no reshapes to worry about": the framework does **not** remove
permute/reshape from the graph; it guarantees that the MatMul/Gemm on both sides of the permute
carry Q/DQ — so TensorRT can run the pointwise MLP in INT8, and the permute is just a cheap
layout operation that never becomes the break that "cuts the INT8 region into FP islands". What
actually breaks a region is a **tensor with no scale information**, not the shape operation
itself.

Other ConvNeXt-specific details:

- BN fusion may replace some norms with `nn.Identity`, while mmpretrain's LN2d forward needs a
  `data_format` kwarg — the hook has a `_safe_call` that decides by type whether to pass it
  (otherwise it crashes).
- `with_cp=True` (gradient checkpointing) is forcibly disabled at export time
  (`model_loader.py`).
- config: `keep_fp16=["pts_voxel_encoder"]` **and nothing else** — unlike VoV, ConvNeXt does not
  need the early stages kept. opset **20** (LayerNormalization requires ≥17).

## 5. opset version quick reference (per backbone)

| backbone | opset | reason |
|---|---|---|
| ResNet | 16 | pure conv, the most conservative choice |
| SECOND | 17 | same as above (release config) |
| ConvNeXt | 20 | LayerNormalization |
| VoVNet | 22 | the newest op set (patterns like hsigmoid) |

## 6. Divergences from the official NVIDIA document (and why)

The official document = [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html).
Every recipe above follows its recommendations; the following three are **knowing violations**,
and the reasons are recorded here so that nobody "fixes" them by mistake later:

### 6a. `fuse_bn=True` vs. the official "do not simulate BN fusion in the training framework"

Official recommendation #3:
> "Do not simulate batch normalization and ReLU fusions in the training framework because
> TensorRT optimizations guarantee the preservation of these operations' arithmetic semantics.
> BatchNorm is fused with convolution and ReLU while keeping the same execution order defined
> in the pre-fusion network."

The very first step of our `prepare()` is `fuse_model_bn(model)`. We really do leave ReLU alone
(we never insert Q/DQ around a ReLU), but BN is fused before the weight quantizer is inserted.
Three reasons:

1. **The weight amax must be measured on "the set of weights TensorRT will actually use".** BN
   fusion is a per-output-channel scaling (`γ/σ`), so the per-channel `max|w|` differs by a
   channel-wise factor before and after fusion. Fusing first and then taking the amax makes the
   scale and the kernel line up exactly, with no dependence on how TensorRT recomputes things
   afterwards.
2. **Alignment with the CUDA-CenterPoint / lidar-ai-solution reference implementation** (which is
   the numerical baseline for the release recipe).
3. **A hard engineering constraint**: fusion changes state_dict keys, and the three consumers
   (PTQ producer / QAT hook / deploy loader) must load the same set of keys — so fusion
   **ignores `keep_fp16` and is applied to the whole model at once** (see §0). This one has
   nothing to do with accuracy; it is an invariant.

The price should be stated honestly: this road gives up the flexibility of "letting TensorRT
decide the BN fusion order itself". If QAT accuracy ever drops mysteriously on some backbone,
`fuse_bn` is one of the variables worth going back and questioning.

### 6b. ConvTranspose2d weight uses per-tensor

The official doc lists Transposed Convolution among the weighted operations (items 1 and 7 →
should be per-channel). We fall back to per-tensor purely because TensorRT's INT8 deconv kernel
fails to build in practice. Details in the table in §0.

### 6c. Toolchain: `pytorch-quantization` vs. TensorRT Model Optimizer

That official page now mentions only **TensorRT Model Optimizer** (ModelOpt) for PTQ/QAT +
export; `pytorch-quantization` is no longer in the documentation. We are still on
`pytorch-quantization` (the Q/DQ it produces are standard ONNX nodes, so there is no difference
on the TensorRT side). The migration surface is `core/descriptors.py` (quantizer defaults) and
`core/replace.py` (module replacement); the architecture knowledge in recipes/ can be carried
over wholesale.

### Incidentally: things the official doc covers that we currently do not need

| Official topic | Our situation |
|---|---|
| **Weight-Only Quantization** (INT4 block quant + GEMM; only meaningful when memory-bound) | Not needed — ours is a conv-bound BEV network |
| INT4 / FP4 packing (two elements per byte), ONNX opset **21** (INT4 + block quant), opset **23** (FP4E2M1) | The opset table in §5 tops out at 22, because we only use INT8 |
| **Q/DQ Interaction with Plugins**: a plugin that consumes INT8 must **absorb the input DQ / output Q into the plugin itself** and remove them from the network, and also call `setOutputType(kINT8)` | CenterPoint has no plugins; BEVFusion's spconv plugin did hit this one (see that project's deploy config) |
| refit disables the scale-equality-related optimizations | We do not enable refit (`grep -rn refit deployment/` comes back empty) — **do not enable it**: the benefit of single-Q fan-out depends on scale equality, see `01` §6 |

## 7. All three recipes in one sentence

> **residual add / eSE / concat / maxpool are all answering the same question:
> "for this non-conv op, does every one of its inputs have a known INT8 scale?"**
> Yes → TensorRT keeps it inside the INT8 region (or even fuses it into a conv kernel);
> No → dequantize back to FP, insert a reformat, and lose on both time and accuracy.
> And the answer to "how many times do we quantize" is always:
> **quantize a given tensor once and let everyone share it** (single-Q fan-out).

## References

- **NVIDIA TensorRT — Explicit Quantization**:
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html>
  (Q/DQ Layer-Placement Recommendations, the commutation definition and the Max Pooling proof,
  the fusion conditions for a residual add, the PyTorch/TF difference in per-channel axis,
  Q/DQ Limitations, Q/DQ Interaction with Plugins)
- [01 — Q/DQ Basics](01_qdq_basics.en.md) §6: the beginner's version of the same rules, plus the
  item-by-item mapping of the 7 recommendations onto this framework.
- In-framework documents: `docs/ese_int8_changes.md` (the single-Q eSE design) and
  `docs/ptq_accuracy_vov99.md` (the experimental basis for VoV's keep_fp16).
