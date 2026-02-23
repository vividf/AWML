# Residual Quantizer ONNX Export 问题诊断

## 问题描述

在 ONNX 导出时，`residual_quantizer` 在 `add` 操作之前没有生成 QDQ 节点。

## 已实施的关键修复

### 1. `replace.py` - Forward Hook 实现

**文件**: `projects/CenterPoint/quantization/replace.py`

#### BasicBlockForwardHook
```python
class BasicBlockForwardHook:
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

#### attach_quant_add 函数
```python
def attach_quant_add(model: nn.Module, target_class_names: Optional[Set[str]] = None):
    # ...
    for name, module in model.named_modules():
        if cls_name in target_class_names:
            if not hasattr(module, "residual_quantizer"):
                if hasattr(module, "downsample") and module.downsample is not None:
                    # 有 downsample: 创建新的 TensorQuantizer
                    residual_quantizer = TensorQuantizer(quant_desc)
                    module.add_module("residual_quantizer", residual_quantizer)  # ✅ 注册为 submodule
                elif hasattr(module, "conv1") and hasattr(module.conv1, "_input_quantizer"):
                    # 无 downsample: 复用 conv1._input_quantizer
                    # ⚠️ 注意：不能使用 add_module()，因为已经是 conv1 的 submodule
                    module.residual_quantizer = module.conv1._input_quantizer  # 作为属性引用
```

### 2. `onnx_exporter.py` - ONNX 导出配置

**文件**: `deployment/exporters/common/onnx_exporter.py`

```python
def _do_onnx_export(self, ...):
    try:
        from pytorch_quantization.nn import TensorQuantizer

        # ✅ 设置 use_fb_fake_quant
        TensorQuantizer.use_fb_fake_quant = True
        self.logger.info("Enabled use_fb_fake_quant for ONNX export of quantized model")

        # ✅ 检查 residual_quantizer 实例
        residual_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "residual_quantizer"):
                residual_count += 1
                # 验证状态...
        if residual_count > 0:
            self.logger.info(f"Found {residual_count} residual_quantizer instances in model")

        # ✅ 设置 _enable_onnx_export
        try:
            from pytorch_quantization import enable_onnx_export
            with enable_onnx_export():  # 设置 TensorQuantizer._enable_onnx_export = True
                torch.onnx.export(...)
        except ImportError:
            # 手动设置
            TensorQuantizer._enable_onnx_export = True
            try:
                torch.onnx.export(...)
            finally:
                TensorQuantizer._enable_onnx_export = False
```

## 可能的问题点

### 问题 1: `residual_quantizer` 作为属性引用时可能无法被追踪

**位置**: `replace.py` 第378行
```python
module.residual_quantizer = module.conv1._input_quantizer  # 只是属性引用
```

**原因**: 当复用 `conv1._input_quantizer` 时，`residual_quantizer` 只是一个属性引用，不是 submodule。PyTorch ONNX 导出器可能无法正确追踪这种调用。

**解决方案**:
- 确保在 forward hook 中直接调用 `self.residual_quantizer(identity)`
- ONNX 导出器应该能够追踪模块调用，即使它是属性引用

### 问题 2: `enable_onnx_export` 导入失败

**位置**: `onnx_exporter.py` 第175行

**症状**: 日志显示 "pytorch-quantization not available, skipping quantization export settings"

**解决方案**:
- 已添加 fallback：手动设置 `TensorQuantizer._enable_onnx_export = True`
- 已添加调试日志以确认状态

### 问题 3: `TensorQuantizer._quant_forward` 不使用 `use_fb_fake_quant`

**位置**: `TensorRT/tools/pytorch-quantization/pytorch_quantization/nn/modules/tensor_quantizer.py` 第307-320行

**当前实现**:
```python
def _quant_forward(self, inputs):
    amax = self._get_amax(inputs)
    if self._fake_quant:
        outputs = fake_tensor_quant(inputs, amax, ...)  # 总是使用 fake_tensor_quant
```

**问题**: `_quant_forward` 方法没有检查 `TensorQuantizer.use_fb_fake_quant` 标志。

**解决方案**:
- `FakeTensorQuantFunction.symbolic` 方法已经会生成 `QuantizeLinear`/`DequantizeLinear` 节点
- 只要 `_enable_onnx_export` 被正确设置，`FakeTensorQuantFunction.symbolic` 就会被调用

## 验证步骤

### 1. 检查 `residual_quantizer` 是否正确附加

在 `centerpoint_quantization.py` 的 PTQ 流程中，已添加了 `residual_quantizer` 状态检查：
```python
if quant_flags["quant_add"]:
    print("\nResidual Quantizer Status:")
    for name, module in model.named_modules():
        if hasattr(module, "residual_quantizer"):
            rq = module.residual_quantizer
            print(f"  {name}.residual_quantizer:")
            print(f"    - Has calibrator: {hasattr(rq, '_calibrator') and rq._calibrator is not None}")
            print(f"    - Has amax: {hasattr(rq, '_amax') and rq._amax is not None}")
            print(f"    - Disabled: {getattr(rq, '_disabled', False)}")
```

### 2. 检查 ONNX 导出时的状态

在 `onnx_exporter.py` 中已添加：
- 检查 `residual_quantizer` 实例数量
- 验证每个 `residual_quantizer` 的状态
- 确认 `_enable_onnx_export` 标志的值

### 3. 检查 ONNX 模型

导出后，检查 ONNX 模型：
```python
import onnx

model = onnx.load("pts_backbone_neck_head.onnx")
# 查找 Add 节点之前的 QuantizeLinear/DequantizeLinear 节点
# 应该看到：... -> QuantizeLinear -> DequantizeLinear -> Add
```

## 下一步调试建议

1. **检查 ONNX 模型**: 使用 `netron` 或 `onnx` 库检查导出的 ONNX 模型，确认 `residual_quantizer` 是否生成了 QDQ 节点
2. **启用详细日志**: 在 ONNX 导出时设置 `verbose=True`，查看详细的节点信息
3. **对比 lidar-ai-solution**: 检查 lidar-ai-solution 的 ONNX 导出结果，确认他们的 `residual_quantizer` 是否成功生成 QDQ 节点
4. **检查 forward hook**: 确认 forward hook 在 ONNX 导出时是否被正确调用

## 关键代码位置

1. **Forward Hook**: `projects/CenterPoint/quantization/replace.py` 第226-268行 (BasicBlockForwardHook)
2. **附加 residual_quantizer**: `projects/CenterPoint/quantization/replace.py` 第312-404行 (attach_quant_add)
3. **ONNX 导出配置**: `deployment/exporters/common/onnx_exporter.py` 第164-213行 (_do_onnx_export)
4. **PTQ 调用**: `tools/detection3d/centerpoint_quantization.py` 第288-301行 (run_ptq)
