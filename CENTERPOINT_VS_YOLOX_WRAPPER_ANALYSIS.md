# CenterPoint vs YOLOX Wrapper 分析

## 核心问题

**是否可以用 CenterPoint ONNX wrapper 的方式，像 YOLOX 一样移除 `export_onnx` 和 `export_tensorrt` 的 override？**

## 关键差异分析

### 1. YOLOX Wrapper 的本质

**YOLOX wrapper (`YOLOXONNXWrapper`)** 的作用：
- ✅ **修改输出格式**：将模型的输出转换为 Tier4 兼容的格式
- ✅ **单文件导出**：整个模型导出为一个 ONNX 文件
- ✅ **结构不变**：模型结构完全不变，只是包装 forward 方法
- ✅ **简单转换**：输入 → 模型 → wrapper 转换 → 输出

```python
# YOLOX wrapper 的工作方式
class YOLOXONNXWrapper:
    def forward(self, x):
        feat = self.model.extract_feat(x)
        cls_scores, bbox_preds, objectnesses = self.bbox_head(feat)
        # 只是重新组织输出格式
        return concatenated_outputs
```

### 2. CenterPoint 的特殊需求

**CenterPoint 需要**：
- ❌ **模型拆分**：必须将模型拆分为两个独立的 ONNX 文件
  - `pts_voxel_encoder.onnx` - voxel encoder 部分
  - `pts_backbone_neck_head.onnx` - backbone + neck + head 部分
- ❌ **中间步骤处理**：需要运行 middle encoder (PyTorch) 来连接两个部分
- ❌ **多文件导出**：不能导出为单个文件
- ❌ **特殊导出逻辑**：需要创建临时的 `CenterPointHeadONNX` wrapper

**为什么需要拆分？**
- Middle encoder 使用 sparse convolution，**无法转换为 ONNX**
- 必须在 voxel encoder 和 backbone 之间保留 PyTorch 中间层

```python
# CenterPoint 的导出流程
1. 导出 voxel encoder → pts_voxel_encoder.onnx
2. 运行 voxel encoder 得到 voxel_features
3. 运行 middle encoder (PyTorch) 得到 spatial_features  ← 关键步骤
4. 创建 CenterPointHeadONNX wrapper
5. 导出 backbone+neck+head → pts_backbone_neck_head.onnx
```

### 3. 概念差异对比

| 特性 | YOLOX Wrapper | CenterPoint Exporter |
|------|---------------|---------------------|
| **目的** | 输出格式转换 | 模型结构拆分 |
| **文件数量** | 1 个 ONNX 文件 | 2 个 ONNX 文件 |
| **模型结构** | 不变 | 需要拆分 |
| **中间步骤** | 无 | 需要运行 middle encoder |
| **Wrapper 类型** | 输出格式 wrapper | 结构拆分 wrapper |
| **适用场景** | 单文件导出 + 格式转换 | 多文件导出 + 结构拆分 |

## 结论

### ❌ **不能完全用 wrapper 替代**

**原因：**

1. **Wrapper 的限制**：
   - Wrapper 只能修改 forward 的输出格式
   - 不能拆分模型为多个部分
   - 不能处理中间步骤（middle encoder）
   - 不能导出多个文件

2. **CenterPoint 需要的是**：
   - 模型结构拆分（不是格式转换）
   - 多文件导出逻辑
   - 中间步骤处理
   - 特殊的导出流程

3. **本质不同**：
   - YOLOX wrapper = **输出格式转换器**
   - CenterPoint exporter = **模型拆分器 + 多文件导出器**

## 改进方案

虽然不能完全移除 override 函数，但可以**改进架构**，让代码更简洁：

### 方案 1：配置驱动的 Exporter 选择（推荐）

**在配置文件中指定 exporter 类型**，让 `DeploymentRunner` 自动选择：

```python
# deploy_config.py
onnx_config = dict(
    opset_version=16,
    # 指定使用 CenterPoint 专用的 exporter
    exporter_type='centerpoint',  # 或 'standard', 'yolox', 'centerpoint'
    # ... 其他配置
)
```

**改进 `DeploymentRunner.export_onnx`**：
```python
def export_onnx(self, pytorch_model, **kwargs):
    if self._export_onnx_fn:
        return self._export_onnx_fn(...)
    
    # 从配置中获取 exporter 类型
    onnx_settings = self.config.get_onnx_settings()
    exporter_type = onnx_settings.get('exporter_type', 'standard')
    
    if exporter_type == 'centerpoint':
        from autoware_ml.deployment.exporters import CenterPointONNXExporter
        exporter = CenterPointONNXExporter(onnx_settings, self.logger)
        return exporter.export(...)
    elif exporter_type == 'standard':
        exporter = ONNXExporter(onnx_settings, self.logger)
        return exporter.export(...)
```

**优势**：
- ✅ 不需要在 main.py 中 override `export_onnx`
- ✅ 配置驱动，更灵活
- ✅ 代码更简洁

### 方案 2：基于模型类型的自动检测

**根据模型类型自动选择 exporter**：

```python
def export_onnx(self, pytorch_model, **kwargs):
    # 检测模型类型
    model_type = type(pytorch_model).__name__
    
    if 'CenterPointONNX' in model_type:
        # 自动使用 CenterPoint exporter
        exporter = CenterPointONNXExporter(...)
    else:
        exporter = ONNXExporter(...)
```

**优势**：
- ✅ 完全自动化
- ✅ 不需要配置

**劣势**：
- ❌ 依赖模型类型名称，可能不够稳定

### 方案 3：改进 TensorRT 导出

**对于 TensorRT 多文件导出，可以创建 `CenterPointTensorRTExporter`**：

```python
class CenterPointTensorRTExporter:
    """处理 CenterPoint 多文件 TensorRT 导出"""
    
    def export(self, onnx_dir, config, ...):
        # 处理两个 ONNX 文件的转换
        ...
```

**然后在配置中指定**：
```python
backend_config = dict(
    exporter_type='centerpoint',  # 多文件 TensorRT 导出
    ...
)
```

## 推荐实现

### 1. 改进 `BaseDeploymentConfig.get_onnx_settings()`

添加 exporter 类型支持：

```python
def get_onnx_settings(self) -> Dict[str, Any]:
    settings = {...}
    
    # 支持指定 exporter 类型
    if 'exporter_type' in self.onnx_config:
        settings['exporter_type'] = self.onnx_config['exporter_type']
    
    return settings
```

### 2. 改进 `DeploymentRunner.export_onnx()`

支持自动选择 exporter：

```python
def export_onnx(self, pytorch_model, **kwargs):
    if self._export_onnx_fn:
        return self._export_onnx_fn(...)
    
    onnx_settings = self.config.get_onnx_settings()
    exporter_type = onnx_settings.get('exporter_type', 'standard')
    
    if exporter_type == 'centerpoint':
        from autoware_ml.deployment.exporters import CenterPointONNXExporter
        exporter = CenterPointONNXExporter(onnx_settings, self.logger)
        # CenterPoint 需要特殊的导出逻辑
        return exporter.export(
            model=pytorch_model,
            data_loader=self.data_loader,
            output_dir=self.config.export_config.work_dir,
            sample_idx=0
        )
    else:
        # 标准导出流程
        ...
```

### 3. 简化 `main.py`

移除 override 函数，使用配置驱动：

```python
# main.py - 简化后
def main():
    # ... 配置加载 ...
    
    # 创建 runner，不需要 override export_onnx
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=onnx_model_cfg,
        logger=logger,
        load_model_fn=load_model_fn,
        # 不再需要 export_onnx_fn 和 export_tensorrt_fn
    )
    
    runner.run(checkpoint_path=args.checkpoint)
```

## 总结

1. **不能完全用 wrapper 替代**：CenterPoint 需要的是模型拆分和多文件导出，不是输出格式转换
2. **可以改进架构**：通过配置驱动的 exporter 选择，减少 main.py 中的代码
3. **推荐方案**：在配置文件中指定 `exporter_type`，让 `DeploymentRunner` 自动选择对应的 exporter
4. **TensorRT 同样**：可以创建 `CenterPointTensorRTExporter` 来处理多文件导出

这样可以在保持功能完整性的同时，让代码更简洁、更易维护。


