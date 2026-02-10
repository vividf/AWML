# SECOND Backbone 中的 BatchNorm 融合详解

## 概述

在 SECOND backbone 中，**训练时** Conv 和 BatchNorm 是**分开的模块**，但在**推理/部署时**，可以通过 BN 融合（BN Fusion）将 BatchNorm 的参数融合到卷积层中，从而：
1. 减少计算量（少一次 BN 操作）
2. 提高推理速度
3. 减少量化误差（在量化场景中很重要）

## SECOND 的结构

### 训练时的结构

SECOND backbone 的每个 block 结构如下：

```python
# 每个 stage 的第一个 block（带下采样）
Conv2d(3×3, stride=2) → BatchNorm2d → ReLU

# 后续 blocks（无下采样）
Conv2d(3×3, stride=1) → BatchNorm2d → ReLU
Conv2d(3×3, stride=1) → BatchNorm2d → ReLU
...
```

**代码实现**（来自 `mmdet3d/models/backbones/second.py`）：

```python
for i, layer_num in enumerate(layer_nums):
    block = [
        # 第一个 block：下采样
        build_conv_layer(conv_cfg, in_filters[i], out_channels[i], 3,
                        stride=layer_strides[i], padding=1),
        build_norm_layer(norm_cfg, out_channels[i])[1],  # BatchNorm
        nn.ReLU(inplace=True),
    ]
    # 后续 blocks
    for j in range(layer_num):
        block.append(
            build_conv_layer(conv_cfg, out_channels[i], out_channels[i], 3, padding=1)
        )
        block.append(build_norm_layer(norm_cfg, out_channels[i])[1])  # BatchNorm
        block.append(nn.ReLU(inplace=True))
```

**关键点：**
- Conv 和 BN 是**独立的模块**，按顺序执行
- Conv 通常设置 `bias=False`（因为 BN 会添加 bias）
- BN 使用 `eps=1e-3, momentum=0.01`（与标准 BN 不同）

## BatchNorm 融合原理

### 数学推导

**融合前：**
```
y = BN(Conv(x))
  = BN(W * x + b)
  = (W * x + b - mean) * (gamma / sqrt(var + eps)) + beta
```

**融合后：**
```
y = Conv_fused(x)
  = W_fused * x + b_fused
```

**融合公式：**

将 BN 的变换合并到 Conv 的权重和偏置中：

```
scale = gamma / sqrt(var + eps)

W_fused = W * scale
b_fused = (b - mean) * scale + beta
```

**代码实现**（来自 `quantization/fusion/bn_fusion.py`）：

```python
def fuse_bn_weights(conv_weight, conv_bias, bn_mean, bn_var, bn_eps,
                    bn_weight, bn_bias):
    # 计算 scale 因子
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
    scale = bn_weight * bn_var_rsqrt  # gamma / sqrt(var + eps)

    # 融合权重：W_fused = W * scale
    shape = [-1] + [1] * (conv_weight.ndim - 1)  # [out_channels, 1, 1, 1]
    fused_weight = conv_weight * scale.reshape(shape)

    # 融合偏置：b_fused = (b - mean) * scale + beta
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias

    return fused_weight, fused_bias
```

### 融合后的结构

**融合前：**
```
Input → Conv → BN → ReLU → Output
```

**融合后：**
```
Input → Conv_fused → ReLU → Output
       (BN 参数已融合到 Conv 中)
```

BN 层被替换为 `nn.Identity()`（或直接删除），不再执行任何计算。

## 融合的时机

### 1. 量化时（PTQ/QAT）

在量化流程中，BN 融合是**必须的**步骤：

**PTQ (Post-Training Quantization)：**
```python
from projects.CenterPoint.quantization import fuse_model_bn

model.eval()
fuse_model_bn(model)  # 融合所有 Conv-BN 对
# 然后进行量化
```

**QAT (Quantization-Aware Training)：**
```python
# 在 QATHook 中，训练开始前融合 BN
if self.freeze_bn:
    model.eval()
    fuse_model_bn(model)
    model.train()  # 继续训练，但 BN 已被融合
```

### 2. 部署时

在部署配置中，如果 checkpoint 是量化模型（PTQ/QAT），需要确保 BN 已融合：

```python
# deploy_config_int8.py
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,  # 表示 checkpoint 中的 BN 已融合
)
```

部署时会再次检查并融合（如果还未融合）：

```python
# deploy/utils.py
if fuse_bn:
    from projects.CenterPoint.quantization import fuse_model_bn
    fuse_model_bn(model)
```

## 融合过程详解

### 步骤 1: 查找 Conv-BN 对

```python
def find_conv_bn_pairs(model):
    """查找所有 Conv-BN 对"""
    pairs = []
    prev_name = None
    prev_module = None

    for name, module in model.named_modules():
        # 检查是否是 BN
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # 检查前一个模块是否是 Conv
            if isinstance(prev_module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                # 验证通道数匹配
                if _get_conv_out_channels(prev_module) == _get_bn_num_features(module):
                    pairs.append((prev_name, name))

        prev_name = name
        prev_module = module

    return pairs
```

**匹配规则：**
- Conv 和 BN 在模块遍历中**相邻**
- Conv 的输出通道数 = BN 的 `num_features`
- 支持 Conv1d/Conv2d/ConvTranspose2d/Linear

### 步骤 2: 融合参数

```python
def fuse_conv_bn(conv, bn):
    """将 BN 参数融合到 Conv 中"""
    assert not conv.training and not bn.training, "必须在 eval 模式"

    # 融合权重和偏置
    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,    # BN 的 running mean
        bn.running_var,     # BN 的 running variance
        bn.eps,
        bn.weight,          # BN 的 gamma
        bn.bias,            # BN 的 beta
    )
```

### 步骤 3: 替换 BN 层

```python
# 将 BN 替换为 Identity（不执行任何操作）
parent, attr = _get_parent_module(model, bn_name)
setattr(parent, attr, nn.Identity())
```

## 融合前后的对比

### 计算量对比

**融合前：**
```
Conv: O(C_out * C_in * K * K * H * W)
BN:   O(C_out * H * W)  # 逐通道归一化
总计: O(C_out * C_in * K * K * H * W) + O(C_out * H * W)
```

**融合后：**
```
Conv_fused: O(C_out * C_in * K * K * H * W)
总计: O(C_out * C_in * K * K * H * W)
```

**节省：** 减少了 BN 的计算（虽然相对较小，但在量化场景中很重要）

### 内存对比

**融合前：**
- Conv 权重: `[C_out, C_in, K, K]`
- Conv 偏置: `[C_out]`（通常为 None）
- BN 参数: `running_mean [C_out]`, `running_var [C_out]`, `weight [C_out]`, `bias [C_out]`

**融合后：**
- Conv_fused 权重: `[C_out, C_in, K, K]`（已包含 BN 的 scale）
- Conv_fused 偏置: `[C_out]`（已包含 BN 的 shift）
- BN 层: 被 Identity 替换（无参数）

**节省：** 减少了 BN 的 4 个参数张量

## 在量化中的重要性

### 为什么必须融合？

1. **减少量化误差**
   - 如果 Conv 和 BN 分开量化，BN 的 scale 会引入额外的量化误差
   - 融合后，整个 Conv+BN 作为一个整体量化，误差更小

2. **TensorRT 优化**
   - TensorRT 可以更好地优化融合后的 Conv+ReLU
   - 减少 kernel 启动次数

3. **精度一致性**
   - 融合后的模型在 FP32/FP16/INT8 下行为一致
   - 避免 BN 在不同精度下的数值差异

## 注意事项

### 1. 必须在 eval 模式下融合

```python
model.eval()  # 必须！
fuse_model_bn(model)
```

**原因：** BN 在训练和推理时的行为不同：
- 训练时：使用 batch 统计（mean, var）
- 推理时：使用 running 统计（running_mean, running_var）

融合使用的是 `running_mean` 和 `running_var`，所以必须在 eval 模式。

### 2. Conv 的 bias 处理

SECOND 的 Conv 通常设置 `bias=False`，融合时会自动创建零偏置：

```python
if conv_bias is None:
    conv_bias = torch.zeros_like(bn_mean)
```

### 3. ConvTranspose2d 的特殊处理

转置卷积的权重形状不同，需要特殊处理：

```python
# Conv2d: [out_channels, in_channels, H, W] → scale 应用到 dim 0
# ConvTranspose2d: [in_channels, out_channels, H, W] → scale 应用到 dim 1
if is_transposed:
    shape = [1, -1] + [1] * (conv_weight.ndim - 2)
else:
    shape = [-1] + [1] * (conv_weight.ndim - 1)
```

### 4. 融合后的模型结构

融合后，BN 层被替换为 `nn.Identity()`，但模型结构保持不变：

```python
# 融合前
Sequential(
    Conv2d(...),
    BatchNorm2d(...),
    ReLU(...)
)

# 融合后
Sequential(
    Conv2d(...),  # 权重和偏置已更新
    Identity(),   # 原 BatchNorm2d 的位置
    ReLU(...)
)
```

## 实际使用示例

### PTQ 流程

```python
from projects.CenterPoint.quantization import quantize_ptq, fuse_model_bn

# 1. 加载模型
model = init_model(cfg, checkpoint)
model.eval()

# 2. 融合 BN（在量化前）
fuse_model_bn(model)

# 3. 量化
quantized_model = quantize_ptq(
    model,
    dataloader,
    fuse_bn=False,  # 已经融合过了
    ...
)

# 4. 保存
torch.save({'state_dict': quantized_model.state_dict()}, 'ptq.pth')
```

### 部署流程

```python
# deploy_config_int8.py
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,  # checkpoint 中的 BN 已融合
)

# 部署时会再次检查
# 如果 checkpoint 已融合，再次融合是安全的（BN 已是 Identity）
```

## 总结

1. **训练时**：Conv 和 BN 是**分开的模块**，按顺序执行
2. **推理/部署时**：通过 BN 融合，将 BN 的参数合并到 Conv 中
3. **融合公式**：
   - `W_fused = W * (gamma / sqrt(var + eps))`
   - `b_fused = (b - mean) * scale + beta`
4. **融合后**：BN 层被替换为 `Identity()`，不再执行计算
5. **量化场景**：BN 融合是**必须的**，可以减少量化误差并提高性能

这样既保持了训练时的灵活性，又优化了推理时的效率！
