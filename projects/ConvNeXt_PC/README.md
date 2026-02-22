# AWML ConvNeXt_PC Architecture README

This document explains the **AWML ConvNeXt** architecture used in this repository, with a focus on `ConvNeXt_PC` and its integration in CenterPoint.

## 1. What AWML ConvNeXt Means

In AWML, ConvNeXt refers to the ConvNeXt-family backbone adapted for BEV detection pipelines (CenterPoint), not just the vanilla image-classification usage.  
It keeps the CNN structure (Conv + normalization + activation), while using AWML-specific compatibility handling for training, quantization, and deployment.

AWML goals:

- Keep convolution-friendly deployment properties for ONNX/TensorRT
- Improve BEV feature quality with modern ConvNeXt blocks
- Maintain compatibility with AWML configs and quantization flow

## 2. ConvNeXt Block (Core Building Unit)

A ConvNeXt block contains:

1. **Depthwise Convolution (7x7)**  
   Captures spatial context with low compute cost.
2. **Normalization**
3. **Pointwise expansion (1x1 or Linear)**  
   Expands channels by `mlp_ratio` (commonly 4x).
4. **Activation (GELU)**
5. **Pointwise projection back to input channels**
6. **Layer Scale (`gamma`)** (optional but common)
7. **DropPath** (stochastic depth, optional)
8. **Residual Add**

High-level formula:

```text
y = x + DropPath( gamma * PW2( Act( PW1( Norm( DWConv(x) ) ) ) ) )
```

AWML ConvNeXt block diagram:

```text
Input x (N,C,H,W)
   |
   v
Depthwise Conv (k=7, groups=C)
   |
   v
+-------------------------------+
| if linear_pw_conv=True        |
|   Permute NCHW -> NHWC        |
|   Norm(data_format=last)      |
|   Linear C -> 4C              |
|   GELU                        |
|   Linear 4C -> C              |
|   Permute NHWC -> NCHW        |
| else                          |
|   Norm(data_format=first)     |
|   Conv1x1 C -> 4C             |
|   GELU                        |
|   Conv1x1 4C -> C             |
+-------------------------------+
   |
   v
Layer Scale gamma (optional)
   |
   v
DropPath (optional)
   |
   v
Residual Add with shortcut
   |
   v
Output y
```

### Why normalization appears after Add

If you inspect the full network graph, it is normal to see `Add -> Normalization`.
In AWML ConvNeXt, this usually comes from one of these two places:

1. **Next block pre-norm**
   - A block ends with residual add: `x = shortcut + self.drop_path(x)`.
   - The **next** block starts and runs its own `Norm(...)` path.
   - So graph viewers often show this as if norm is right after add.
2. **Stage output norm (`norm{i}`)**
   - In backbone forward, when `i in out_indices`, AWML applies `norm{i}` before appending output.
   - This is another valid source of normalization after stage/block outputs.

Quick view:

```text
Block k: ... -> Add --------------------> output
                                     |
                                     v
Block k+1: DWConv -> Norm -> ... -> Add

or

Stage output x -> norm{i} -> neck/head
```

## 2.1 Corresponding AWML Code

`ConvNeXtBlockPC.forward` (block-level add):

```python
def forward(self, x):
    def _inner_forward(x):
        shortcut = x
        x = self.depthwise_conv(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)
            x = self._apply_norm(x, data_format="channel_last")
            x = self.pointwise_conv1(x)
            x = self.act(x)
            if self.grn is not None:
                x = self.grn(x, data_format="channel_last")
            x = self.pointwise_conv2(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self._apply_norm(x, data_format="channel_first")
            x = self.pointwise_conv1(x)
            x = self.act(x)
            if self.grn is not None:
                x = self.grn(x, data_format="channel_first")
            x = self.pointwise_conv2(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)  # residual add
        return x
```

`ConvNeXt_PC.forward` (stage output norm):

```python
def forward(self, x):
    outs = []
    for i, stage in enumerate(self.stages):
        if i >= self.first_downsample:
            x = self.downsample_layers[i](x)
        x = stage(x)
        if i in self.out_indices:
            norm_layer = getattr(self, f"norm{i}")
            if self.gap_before_final_norm:
                gap = x.mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))
            else:
                outs.append(norm_layer(x).contiguous())
    return tuple(outs)
```

## 3. Two Equivalent Pointwise Implementations

ConvNeXt supports two equivalent ways for pointwise operations:

- **Linear path (`linear_pw_conv=True`)**
  - Tensor permutation: `NCHW -> NHWC`
  - `Norm + Linear + GELU + Linear`
  - Permute back: `NHWC -> NCHW`
- **Conv path (`linear_pw_conv=False`)**
  - Stay in `NCHW`
  - `Norm + Conv1x1 + GELU + Conv1x1`

The repository primarily uses the linear path to match upstream ConvNeXt behavior.

## 4. Stage-Level Backbone Structure

A ConvNeXt backbone is built from multiple stages:

- **Stem/downsample layers** to reduce spatial resolution
- **Stage blocks** with repeated ConvNeXt blocks
- **Channel scaling** across stages (e.g., 96 -> 192 -> 384 -> 768 style)
- **Depth scaling** by architecture variant (small/base/etc.)

In detection tasks, each stage output can be forwarded to a neck (e.g., FPN-like modules).

AWML ConvNeXt backbone diagram (CenterPoint context):

```text
BEV Feature Map Input
        |
        v
  [Stem / First Downsample]
        |
        v
  Stage 0: ConvNeXt Blocks x d0, channels c0
        |
        v
  Stage 1: Downsample -> ConvNeXt Blocks x d1, channels c1
        |
        v
  Stage 2: Downsample -> ConvNeXt Blocks x d2, channels c2
        |
        v
  Stage 3: Downsample -> ConvNeXt Blocks x d3, channels c3
        |
        +------------------------------+
        | Multi-scale outputs (out_indices)
        v
      Neck (SECFPN)
        |
        v
    Detection Head
```

## 5. What `ConvNeXt_PC` Changes in AWML

`projects/ConvNeXt_PC/backbones/convnext_pc.py` extends upstream ConvNeXt to match AWML point-cloud BEV workflows and deployment constraints.

Main adjustments:

- **Norm compatibility helper (`_apply_norm`)**
  - Handles both norm types that require `data_format` and those that do not.
  - Prevents interface mismatch when using BatchNorm-family modules.
- **Custom block class (`ConvNeXtBlockPC`)**
  - Preserves ConvNeXt forward logic while adding robust norm invocation.
- **Backbone wrapper (`ConvNeXt_PC`)**
  - Supports AWML-specific options such as:
    - `first_downsample`
    - `large_arch`
    - `use_bn_relu`
  - Keeps stage output behavior aligned with CenterPoint integration.

## 6. AWML CenterPoint Integration (Conceptual Flow)

Typical AWML flow in CenterPoint with ConvNeXt backbone:

```text
Voxel features -> scatter to BEV map -> ConvNeXt_PC backbone -> neck (SECFPN) -> detection head
```

Where ConvNeXt contributes:

- Strong multi-scale feature extraction on BEV feature maps
- Better backbone capacity than classic shallow CNN alternatives

## 7. AWML Quantization and Deployment Notes

For AWML PTQ/QAT and deployment:

- Some norm layers may be fused/replaced during quantization (for example becoming `Identity`).
- Forward hooks or custom wrappers must avoid assuming every norm accepts `data_format`.
- Residual branch quantization should be handled carefully to keep graph export stable.

This is especially important for AWML ConvNeXt blocks using custom forward hooks in INT8 pipelines.

## 8. Practical Config Knobs

Common knobs that affect architecture behavior:

- `depths`: number of blocks per stage
- `out_channels`: channels per stage
- `linear_pw_conv`: use Linear path or Conv1x1 path
- `drop_path_rate`: regularization strength
- `layer_scale_init_value`: stability for deep networks
- `with_cp`: checkpointing for memory saving

## 9. When to Use AWML ConvNeXt Here

Use AWML ConvNeXt-based backbones when you want:

- Better feature quality on complex scenes
- A CNN architecture with strong modern baseline performance
- Compatibility with your existing convolution-focused deployment toolchain

Trade-offs:

- Higher compute and memory than lightweight backbones
- More care needed for quantization/export edge cases

## 10. Reference Entry Points

- Backbone implementation: `projects/ConvNeXt_PC/backbones/convnext_pc.py`
- CenterPoint project root: `projects/CenterPoint/`
- CenterPoint ConvNeXt-related docs: `projects/CenterPoint/docs/`
