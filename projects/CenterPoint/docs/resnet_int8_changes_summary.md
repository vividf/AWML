# ResNet INT8 量化实现 - 修改总结

## 核心修改文件

### 1. `projects/CenterPoint/quantization/replace.py`

**新增内容**：
- `BasicBlockForwardHook` 类：实现只量化 identity branch 的 forward hook
- `SparseBasicBlockForwardHook` 类：SparseBasicBlock 的 forward hook
- `attach_quant_add` 函数：为 BasicBlock 附加 `residual_quantizer` 并替换 forward 方法

**关键逻辑**：
- 如果有 `downsample`：创建新的 `TensorQuantizer` 作为 `residual_quantizer`
- 如果没有 `downsample`：复用 `conv1._input_quantizer` 作为 `residual_quantizer`（共享校准数据）

### 2. `projects/CenterPoint/deploy/utils.py`

**修改位置**：`_load_quantized_checkpoint` 函数

**修改内容**：
```python
# 修改前
quant_model(
    model,
    quant_backbone=bool(quantization.get("quant_backbone", True)),
    quant_neck=bool(quantization.get("quant_neck", True)),
    quant_head=bool(quantization.get("quant_head", True)),
    quant_voxel_encoder=bool(quantization.get("quant_voxel_encoder", True)),
    skip_names=skip_layers,
)

# 修改后
quant_model(
    model,
    quant_backbone=bool(quantization.get("quant_backbone", True)),
    quant_neck=bool(quantization.get("quant_neck", True)),
    quant_head=bool(quantization.get("quant_head", True)),
    quant_voxel_encoder=bool(quantization.get("quant_voxel_encoder", True)),
    quant_add=bool(quantization.get("quant_add", False)),  # ✅ 新增
    skip_names=skip_layers,
)
```

**原因**：确保在部署时 `attach_quant_add` 被调用，`residual_quantizer` 被正确附加。

### 3. `deployment/exporters/common/onnx_exporter.py`

**修改位置**：`_do_onnx_export` 方法

**修改内容**：
1. 设置 `TensorQuantizer.use_fb_fake_quant = True`：启用标准 ONNX QDQ 节点导出
2. 设置 `TensorQuantizer._enable_onnx_export = True`（类变量和所有实例变量）：启用 ONNX 导出模式
3. 简化了调试日志（移除了详细的 `residual_quantizer` 状态检查）

**关键代码**：
```python
# 设置 use_fb_fake_quant
TensorQuantizer.use_fb_fake_quant = True

# 设置类变量
TensorQuantizer._enable_onnx_export = True

# 设置所有实例变量（包括 residual_quantizer）
for name, module in model.named_modules():
    if isinstance(module, TensorQuantizer):
        module._enable_onnx_export = True
```

### 4. `projects/CenterPoint/deploy/configs/deploy_config_int8_resnet.py`

**修改内容**：
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

## 已清理的代码

### 移除的调试代码

1. **`deployment/exporters/common/onnx_exporter.py`**：
   - 移除了详细的 `residual_quantizer` 状态检查日志
   - 移除了 `residual_details` 和 `is_reference` 检查
   - 简化了日志输出，只保留关键信息

2. **移除了调试用的 `print` 语句**

## 设计要点

### 1. Forward Hook 机制

使用 Forward Hook 而不是修改原始 `forward` 方法，这样可以：
- 保持原始模型结构不变
- 在运行时动态插入量化逻辑
- 便于 ONNX 导出器追踪

### 2. Residual Quantizer 策略

- **有 downsample**：创建独立的 `TensorQuantizer`，因为 downsample 改变了特征图尺寸
- **无 downsample**：复用 `conv1._input_quantizer`，因为输入和 identity 的量化尺度应该一致

### 3. ONNX 导出配置

- `use_fb_fake_quant = True`：使用 PyTorch 的 `FakeQuantize` 操作，导出为标准 ONNX `QuantizeLinear`/`DequantizeLinear` 节点
- `_enable_onnx_export = True`：启用 ONNX 导出模式，`TensorQuantizer.forward` 会检查此标志

### 4. 实例变量 vs 类变量

`TensorQuantizer.forward` 检查的是实例变量 `self._enable_onnx_export`，因此必须为所有实例设置，而不仅仅是类变量。

## 验证清单

- [x] PTQ 阶段：`attach_quant_add` 被调用，`residual_quantizer` 被附加
- [x] Checkpoint 保存：`residual_quantizer` 状态被保存
- [x] 部署加载：`quant_add=True` 被传递，`residual_quantizer` 被恢复
- [x] ONNX 导出：`use_fb_fake_quant` 和 `_enable_onnx_export` 被设置
- [x] ONNX 模型：Add 节点前有 QDQ 节点
- [x] TensorRT 引擎：Conv + Add 被融合，reformat 操作减少

## 参考文档

- `projects/CenterPoint/docs/resnet_int8_quantization_guide.md`：完整的实现指南
- `projects/CenterPoint/docs/resnet_quantization_explanation.md`：设计原理说明
- `projects/CenterPoint/docs/resnet_quantization_code_comparison.md`：与其他实现的对比
