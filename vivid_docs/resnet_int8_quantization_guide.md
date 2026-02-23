# ResNet INT8 量化实现指南

## 概述

本文档说明如何将 CenterPoint 模型中的 ResNet backbone 成功转换为 INT8 量化模型，特别是处理 ResNet 的 skip connection (residual connection) 的量化策略。

## 核心设计原则

### TensorRT 最佳实践

对于 ResNet 的 residual block，TensorRT 推荐以下 QDQ 节点放置策略：

```
Input
  ↓
Conv1 → Norm1 → ReLU → Conv2 → Norm2
  ↓                                    ↓
  └────────────────────────────────────┼
                                       ↓
                                    [QDQ] ← 只量化 identity branch
                                       ↓
                                    Add
                                       ↓
                                    ReLU
                                       ↓
                                    Output
```

**关键点**：
- ✅ **只量化 identity branch**（skip connection），不量化 main branch 的输出
- ✅ 这样可以让 TensorRT 将 `Conv + Add` 融合成单个 kernel，减少 reformat 操作
- ❌ **不要**在 main branch 的 `norm2` 输出后添加 QDQ 节点

## 实现架构

### 1. Forward Hook 机制

使用 Forward Hook 替换 `BasicBlock` 的 `forward` 方法，在 forward 过程中插入量化逻辑。

**文件**: `projects/CenterPoint/quantization/replace.py`

```python
class BasicBlockForwardHook:
    """Forward hook for BasicBlock to use residual_quantizer for residual connections."""

    def __call__(self, x):
        self = self.obj
        identity = x

        # Main branch (conv path) - 不量化
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # ✅ 关键：只量化 identity branch
        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)

        out = out + identity
        out = self.relu(out)
        return out
```

### 2. Residual Quantizer 附加

**文件**: `projects/CenterPoint/quantization/replace.py`

`attach_quant_add` 函数负责为每个 `BasicBlock` 附加 `residual_quantizer`：

```python
def attach_quant_add(model: nn.Module, target_class_names: Optional[Set[str]] = None):
    """
    为 BasicBlock 附加 residual_quantizer 并替换 forward 方法。

    策略：
    - 如果有 downsample: 创建新的 TensorQuantizer
    - 如果没有 downsample: 复用 conv1._input_quantizer（共享校准数据）
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ in {"BasicBlock", "SparseBasicBlock"}:
            if not hasattr(module, "residual_quantizer"):
                if hasattr(module, "downsample") and module.downsample is not None:
                    # 有 downsample: 创建新的 TensorQuantizer
                    residual_quantizer = TensorQuantizer(quant_desc)
                    module.add_module("residual_quantizer", residual_quantizer)
                elif hasattr(module, "conv1") and hasattr(module.conv1, "_input_quantizer"):
                    # 无 downsample: 复用 conv1._input_quantizer
                    module.residual_quantizer = module.conv1._input_quantizer
                else:
                    # Fallback: 创建新的 TensorQuantizer
                    residual_quantizer = TensorQuantizer(quant_desc)
                    module.add_module("residual_quantizer", residual_quantizer)

            # 替换 forward 方法
            module.forward = BasicBlockForwardHook(module)
```

### 3. ONNX 导出配置

**文件**: `deployment/exporters/common/onnx_exporter.py`

在 ONNX 导出时，需要设置以下关键标志：

```python
def _do_onnx_export(self, model, sample_input, output_path, export_cfg):
    from pytorch_quantization.nn import TensorQuantizer

    # 1. 启用 use_fb_fake_quant 以生成标准 ONNX QDQ 节点
    TensorQuantizer.use_fb_fake_quant = True

    # 2. 设置 _enable_onnx_export 标志（类变量和实例变量）
    TensorQuantizer._enable_onnx_export = True

    # 3. 为所有 TensorQuantizer 实例设置实例变量
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module._enable_onnx_export = True

    # 4. 执行 ONNX 导出
    torch.onnx.export(model, sample_input, output_path, ...)
```

### 4. 部署配置

**文件**: `projects/CenterPoint/deploy/utils.py`

在加载量化 checkpoint 时，必须传递 `quant_add` 参数：

```python
def _load_quantized_checkpoint(model, checkpoint_path, device, quantization):
    # ...
    quant_model(
        model,
        quant_backbone=bool(quantization.get("quant_backbone", True)),
        quant_neck=bool(quantization.get("quant_neck", True)),
        quant_head=bool(quantization.get("quant_head", True)),
        quant_voxel_encoder=bool(quantization.get("quant_voxel_encoder", True)),
        quant_add=bool(quantization.get("quant_add", False)),  # ✅ 关键参数
        skip_names=skip_layers,
    )
```

**配置文件**: `projects/CenterPoint/deploy/configs/deploy_config_int8_resnet.py`

```python
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=True,  # ✅ 启用 residual quantization
    # ...
)
```

## 完整流程

### PTQ (Post-Training Quantization) 流程

1. **量化模型构建**
   ```bash
   python tools/detection3d/centerpoint_quantization.py ptq \
       --config projects/CenterPoint/configs/.../resnet34_...py \
       --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config_int8_resnet.py \
       --checkpoint work_dirs/centerpoint_resnet_bone/epoch_2.pth \
       --calibrate-batches 100 \
       --output work_dirs/centerpoint_resnet34_exp4_ptq.pth
   ```

2. **ONNX 导出**
   ```bash
   python projects/CenterPoint/deploy/main.py \
       projects/CenterPoint/deploy/configs/deploy_config_int8_resnet.py \
       projects/CenterPoint/configs/.../resnet34_...py
   ```

### 关键步骤说明

1. **PTQ 阶段**：
   - `attach_quant_add` 被调用，为所有 `BasicBlock` 附加 `residual_quantizer`
   - Forward hook 替换 `forward` 方法
   - 校准数据收集时，`residual_quantizer` 也会被校准

2. **Checkpoint 保存**：
   - `residual_quantizer` 的状态（包括 `_amax`）被保存到 checkpoint

3. **部署加载**：
   - `_load_quantized_checkpoint` 调用 `quant_model(..., quant_add=True)`
   - `attach_quant_add` 再次被调用，恢复 `residual_quantizer`
   - Forward hook 重新附加

4. **ONNX 导出**：
   - `TensorQuantizer.use_fb_fake_quant = True` 启用标准 ONNX QDQ 节点
   - `TensorQuantizer._enable_onnx_export = True` 启用 ONNX 导出模式
   - `residual_quantizer` 在 forward hook 中被调用，生成 QDQ 节点

## 关键修改总结

### 必需的核心修改

1. **`projects/CenterPoint/quantization/replace.py`**
   - `BasicBlockForwardHook` 类：实现只量化 identity branch 的逻辑
   - `attach_quant_add` 函数：附加 `residual_quantizer` 并替换 forward 方法

2. **`projects/CenterPoint/deploy/utils.py`**
   - `_load_quantized_checkpoint`：添加 `quant_add` 参数到 `quant_model` 调用

3. **`deployment/exporters/common/onnx_exporter.py`**
   - `_do_onnx_export`：设置 `use_fb_fake_quant` 和 `_enable_onnx_export` 标志

### 配置文件

4. **`projects/CenterPoint/deploy/configs/deploy_config_int8_resnet.py`**
   - `quantization.quant_add = True`：启用 residual quantization

## 验证方法

1. **检查日志**：
   - PTQ 阶段：应该看到 "Attached residual_quantizer to X residual blocks"
   - ONNX 导出：应该看到 "Found X residual_quantizer instances in model"

2. **检查 ONNX 模型**：
   - 使用 Netron 打开导出的 ONNX 模型
   - 找到 `Add` 节点，检查其输入之一应该有 `QuantizeLinear` → `DequantizeLinear` 节点

3. **TensorRT 引擎构建**：
   - 应该看到 `Conv + Add` 被融合，reformat 操作减少

## 注意事项

1. **复用 `conv1._input_quantizer` 的情况**：
   - 当没有 `downsample` 时，`residual_quantizer` 是 `conv1._input_quantizer` 的引用
   - 这是属性引用，不是 submodule，但 ONNX 导出器仍能正确追踪

2. **`_enable_onnx_export` 实例变量**：
   - `TensorQuantizer.forward` 检查的是实例变量 `self._enable_onnx_export`
   - 必须为所有 `TensorQuantizer` 实例设置实例变量，而不仅仅是类变量

3. **校准数据共享**：
   - 复用 `conv1._input_quantizer` 时，`residual_quantizer` 共享相同的校准数据（`_amax`）
   - 这确保了量化尺度的一致性

## 参考实现

本实现参考了以下项目的设计：
- **Lidar_AI_Solution/CUDA-BEVFusion**: `hook_bottleneck_forward` 和 `residual_quantizer` 的实现
- **TensorRT Model Optimizer**: ResNet QDQ 放置的最佳实践
