# BatchNorm Fusion for Quantization

This module (`bn_fusion.py`) implements **Conv–BN fusion** to eliminate standalone BatchNorm layers before quantization. Removing BN before inserting Q/DQ nodes is critical because:

1. It reduces op count and memory traffic at inference.
2. It removes a linear transform sitting between fake-quant boundaries, which would otherwise introduce extra quantization error.
3. TensorRT / ONNX Runtime expect fused graphs; leaving BN unfused can prevent kernel fusion.

## Supported Fusion Patterns

### Case 1 — Conv → BN (standard)

```text
x ──► Conv ──► BN ──► y
```

This is the most common pattern. The BN affine transform is folded **into the Conv** by scaling along the **output channel** dimension.

#### Math

Given:

- Conv: `z_o = Σ_i W_{o,i} * x_i + b_o`
- BN:   `y_o = α_o · z_o + β_o`

where:

```
α_o = γ_o / √(var_o + ε)
β_o = β_o^bn − α_o · mean_o
```

Substituting Conv into BN:

```
y_o = α_o · (Σ_i W_{o,i} * x_i + b_o) + β_o
    = Σ_i (α_o · W_{o,i}) * x_i + (α_o · b_o + β_o)
```

**Fused parameters:**

| Parameter | Formula | Dimension |
|-----------|---------|-----------|
| `W'_{o,i}` | `α_o · W_{o,i}` | scale along **output** channel (dim 0) |
| `b'_o` | `α_o · b_o + β_o` | per output channel |

#### Implementation

- `fuse_bn_weights()` — low-level weight computation
- `fuse_conv_bn()` — in-place module-level fusion
- `find_conv_bn_pairs()` — automatic pair detection

Supported module combinations:

| Conv type | BN type | Match condition |
|-----------|---------|-----------------|
| `Conv1d` | `BatchNorm1d` | `conv.out_channels == bn.num_features` |
| `Conv2d` | `BatchNorm2d` | `conv.out_channels == bn.num_features` |
| `ConvTranspose2d` | `BatchNorm2d` | `conv.out_channels == bn.num_features` |
| `Linear` | `BatchNorm1d` | `linear.out_features == bn.num_features` |

---

### Case 2 — BN → Conv (reverse / pre-norm)

```text
x ──► BN ──► Conv ──► y
```

This pattern appears in architectures where the downsample path uses `BN → Conv` ordering (e.g. some ConvNeXt downsampling layers). The BN affine transform is folded **into the Conv** by scaling along the **input channel** dimension.

#### Math

Given:

- BN:   `s_i = α_i · x_i + β_i`
- Conv: `y_o = Σ_i W_{o,i} * s_i + b_o`

where:

```
α_i = γ_i / √(var_i + ε)
β_i = β_i^bn − α_i · mean_i
```

Substituting BN into Conv:

```
y_o = Σ_i W_{o,i} * (α_i · x_i + β_i) + b_o
    = Σ_i (W_{o,i} · α_i) · x_i + (Σ_i W_{o,i} · β_i + b_o)
```

**Fused parameters:**

| Parameter | Formula | Dimension |
|-----------|---------|-----------|
| `W'_{o,i}` | `W_{o,i} · α_i` | scale along **input** channel (dim 1 for Conv2d) |
| `b'_o` | `b_o + Σ_i W_{o,i} · β_i` | per output channel (requires weighted sum over input channels) |

> **Key differences from Case 1:**
>
> - The scale `α` is indexed by `i` (input channel) instead of `o` (output channel).
> - The bias requires an additional **weighted summation** `Σ_i W_{o,i} · β_i` across the input channel and spatial dimensions.
> - For **grouped convolutions** (including depthwise), the scaling and summation are performed independently per group.

#### Implementation

- `fuse_bn_conv_weights()` — low-level weight computation (group-aware)
- `fuse_bn_conv()` — in-place module-level fusion
- `find_bn_conv_pairs()` — automatic pair detection

Supported module combinations:

| BN type | Conv type | Match condition |
|---------|-----------|-----------------|
| `BatchNorm1d` | `Conv1d` | `bn.num_features == conv.in_channels` |
| `BatchNorm1d` | `Linear` | `bn.num_features == linear.in_features` |
| `BatchNorm2d` | `Conv2d` | `bn.num_features == conv.in_channels` |
| `BatchNorm2d` | `ConvTranspose2d` | `bn.num_features == conv.in_channels` |

#### Grouped / Depthwise Convolution Handling

When `groups > 1`, the BN features are partitioned into groups and each group's `α_i`, `β_i` are applied only to the corresponding slice of the weight tensor:

```text
For group g (g = 0 .. G-1):
    input channels in group:  [g * I/G  ..  (g+1) * I/G - 1]
    output channels in group: [g * O/G  ..  (g+1) * O/G - 1]

    W'[o, i_local, ...] = W[o, i_local, ...] · α[g * I/G + i_local]
    b'[o] = b[o] + Σ_{i_local} W[o, i_local, ...] · β[g * I/G + i_local]
```

This is handled automatically by `fuse_bn_conv_weights()`.

---

## Top-Level Entry Point: `fuse_model_bn()`

`fuse_model_bn(model)` is the single entry point used by PTQ, QAT, and deployment pipelines. It performs both fusion patterns in one call:

```text
fuse_model_bn(model)
    │
    ├── 1. find_conv_bn_pairs(model)        # detect Conv → BN
    │       └── fuse_conv_bn(conv, bn)      # fold BN into Conv
    │           └── BN replaced with Identity
    │
    ├── 2. find_bn_conv_pairs(model)        # detect BN → Conv
    │       └── (skip BNs already claimed by step 1)
    │       └── fuse_bn_conv(bn, conv)      # fold BN into Conv
    │           └── BN replaced with Identity
    │
    └── print summary
```

**Conflict avoidance:** If a BN layer was already fused as part of a `Conv → BN` pair in step 1, it is excluded from `BN → Conv` matching in step 2. This prevents double-fusion.

### When It Runs

| Pipeline | Where | Code |
|----------|-------|------|
| **PTQ** | Before inserting Q/DQ nodes | `centerpoint_quantization.py` → `fuse_model_bn(model)` |
| **QAT** | `QATHook.before_train()` when `freeze_bn=True` | `qat_hook.py` → `fuse_model_bn(model)` |
| **Sensitivity** | Before quantized model evaluation | `centerpoint_quantization.py` → `fuse_model_bn(model)` |
| **Deploy / Load PTQ** | Before loading quantized checkpoint | `ptq.py` / `deploy/utils.py` → `fuse_model_bn(model)` |

The model **must be in eval mode** before fusion (the function calls `model.eval()` internally).

---

## API Reference

### Low-Level Weight Functions

```python
fuse_bn_weights(conv_weight, conv_bias, bn_mean, bn_var, bn_eps, bn_weight, bn_bias,
                is_transposed=False) -> (fused_weight, fused_bias)
```

Fuse a **following** BN into Conv weights (Case 1). Scale along output channel.

```python
fuse_bn_conv_weights(conv_weight, conv_bias, bn_mean, bn_var, bn_eps, bn_weight, bn_bias,
                     is_transposed=False, groups=1) -> (fused_weight, fused_bias)
```

Fuse a **preceding** BN into Conv weights (Case 2). Scale along input channel, group-aware.

### Module-Level Functions

```python
fuse_conv_bn(conv, bn)       # Case 1: in-place, modifies conv.weight/bias
fuse_bn_conv(bn, conv)       # Case 2: in-place, modifies conv.weight/bias
```

### Pair Detection

```python
find_conv_bn_pairs(model) -> List[(conv_name, bn_name)]   # Case 1
find_bn_conv_pairs(model) -> List[(bn_name, conv_name)]   # Case 2
```

### Top-Level

```python
fuse_model_bn(model, inplace=True) -> model
```

Fuses all Conv–BN and BN–Conv pairs, replaces fused BNs with `nn.Identity`.

---

## File Layout

```text
projects/CenterPoint/quantization/fusion/
├── __init__.py       # Public exports
├── bn_fusion.py      # All fusion logic
└── README.md         # This file
```
