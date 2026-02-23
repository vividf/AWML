# ResNet Residual Connection Quantization: 实现说明

本文档说明 AWML CenterPoint 在处理 ResNet residual connection quantization 时的实现策略，与 lidar-ai-solution 对齐。

## 问题背景

对于 ResNet 的 residual block，正确的 QDQ 节点插入策略是：
- **只在 skip connection (identity branch) 上添加 QDQ**
- **不在 main branch (conv path) 上添加 QDQ**

这样才能让 TensorRT 融合 Conv+Add 操作，减少 reformat 操作。

## AWML CenterPoint 实现（与 Lidar_AI_Solution 对齐）

AWML CenterPoint 采用与 Lidar_AI_Solution 相同的 PyTorch-only 策略。

### 参考实现: Lidar_AI_Solution (CUDA-BEVFusion)

**实现位置**: `Lidar_AI_Solution/CUDA-BEVFusion/qat/lean/quantize.py`

**策略**: **纯 PyTorch 层面实现**

**关键代码**:
```python
class hook_bottleneck_forward:
    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        # ... conv2, conv3 ...

        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)  # 只量化 identity

        out += identity  # conv path 不量化
        return out

def quantize_camera_backbone(model_camera_backbone):
    # 创建 residual_quantizer
    bottleneck.residual_quantizer = quant_nn.TensorQuantizer(
        quant_nn.QuantConv2d.default_quant_desc_input
    )
    # 替换 forward 方法
    bottleneck.forward = hook_bottleneck_forward(bottleneck)
```

**特点**:
- ✅ 在 PyTorch forward hook 中只量化 identity branch
- ✅ 简单直接，不依赖 ONNX 后处理
- ⚠️ 依赖 TensorQuantizer 的 ONNX 导出行为
- ⚠️ 如果 ONNX 导出时出现问题，无法修正

**工作流程**:
```
PyTorch Model (residual_quantizer)
  → ONNX Export (TensorQuantizer 导出 QDQ)
  → ONNX Model (理论上只有 skip connection 有 QDQ)
```

---

### AWML CenterPoint 实现

**实现位置**: `projects/CenterPoint/quantization/replace.py`

**策略**: **PyTorch-only（与 lidar-ai-solution 对齐）**

**关键代码**:
```python
class BasicBlockForwardHook:
    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        # ... conv path ...

        # 只量化 identity branch
        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)

        out = out + identity  # conv path 不量化
        return out
```

**特点**:
- ✅ **简单直接**: 只在 PyTorch 层面实现，代码清晰
- ✅ **与 lidar-ai-solution 对齐**: 使用相同的策略和实现方式
- ✅ **依赖 TensorQuantizer**: 依赖 TensorQuantizer 的 ONNX 导出行为正确
- ✅ **维护成本低**: 只需要维护一套代码

**工作流程**:
```
PyTorch Model (residual_quantizer 只量化 identity)
  → ONNX Export (TensorQuantizer 导出 QDQ)
  → ONNX Model (skip connection 有 QDQ，main branch 无 QDQ)
```

---

## 实现说明

AWML CenterPoint 采用与 Lidar_AI_Solution 相同的 PyTorch-only 策略：

1. **在 PyTorch forward hook 中只量化 identity branch**
2. **依赖 TensorQuantizer 的 ONNX 导出行为**
3. **简单直接，维护成本低**

最终生成的 ONNX 模型结构：
```
Conv -> BN -> ReLU -> Conv -> BN -> (无 QDQ) -> Add <- (有 QDQ) Identity
```
