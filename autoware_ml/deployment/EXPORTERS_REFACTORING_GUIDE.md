# Exporters 架构重构指南

## 概述

本次重构移除了注册机制，改为依赖注入（Dependency Injection）模式。每个项目在 `main.py` 中显式创建并传入所需的组件（exporter、wrapper），使架构更加清晰和灵活。

## 主要变更

### 1. 移除注册机制

**之前**（使用注册机制）：
```python
# 在 config 中配置
onnx_config = dict(
    model_wrapper='yolox',  # 通过字符串名称注册
    ...
)

# BaseExporter 内部通过注册表查找
wrapper_class = get_model_wrapper('yolox')
```

**现在**（依赖注入）：
```python
# 在 main.py 中直接创建并传入
from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
from autoware_ml.deployment.exporters.yolox.onnx_exporter import YOLOXONNXExporter

onnx_exporter = YOLOXONNXExporter(onnx_settings, logger, model_wrapper=YOLOXONNXWrapper)
```

### 2. BaseExporter 变更

**之前**：
- 通过 `config.get('model_wrapper')` 从配置读取
- 使用 `get_model_wrapper()` 从注册表查找

**现在**：
- 通过 `model_wrapper` 参数直接传入
- 支持传入类（class）或可调用对象（callable）

### 3. DeploymentRunner 变更

**新增参数**：
- `model_wrapper`: 可选的 model wrapper 类
- 如果 exporters 没有 wrapper，会自动注入

## 使用示例

### 示例 1: YOLOX（需要自定义 wrapper）

```python
from autoware_ml.deployment.runners import DeploymentRunner
from autoware_ml.deployment.exporters.yolox.onnx_exporter import YOLOXONNXExporter
from autoware_ml.deployment.exporters.yolox.tensorrt_exporter import YOLOXTensorRTExporter
from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper

def main():
    # ... 创建 data_loader, evaluator, config, logger ...
    
    # 创建 exporters（wrapper 会自动应用）
    onnx_settings = config.get_onnx_settings()
    trt_settings = config.get_tensorrt_settings()
    
    onnx_exporter = YOLOXONNXExporter(
        onnx_settings, 
        logger,
        model_wrapper=YOLOXONNXWrapper  # 可选：默认就是 YOLOXONNXWrapper
    )
    
    tensorrt_exporter = YOLOXTensorRTExporter(trt_settings, logger)
    
    # 创建 runner 并传入 exporters
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=load_pytorch_model,
        onnx_exporter=onnx_exporter,
        tensorrt_exporter=tensorrt_exporter,
        # model_wrapper=YOLOXONNXWrapper,  # 可选：如果 exporter 已有 wrapper，不需要
    )
    
    runner.run(checkpoint_path=args.checkpoint)
```

### 示例 2: CenterPoint（使用 IdentityWrapper）

```python
from autoware_ml.deployment.runners import DeploymentRunner
from autoware_ml.deployment.exporters.centerpoint.onnx_exporter import CenterPointONNXExporter
from autoware_ml.deployment.exporters.centerpoint.tensorrt_exporter import CenterPointTensorRTExporter
from autoware_ml.deployment.exporters.base.model_wrappers import IdentityWrapper

def main():
    # ... 创建 data_loader, evaluator, config, logger ...
    
    # 创建 exporters
    onnx_settings = config.get_onnx_settings()
    trt_settings = config.get_tensorrt_settings()
    
    onnx_exporter = CenterPointONNXExporter(
        onnx_settings, 
        logger,
        model_wrapper=IdentityWrapper  # 可选：默认就是 IdentityWrapper
    )
    
    tensorrt_exporter = CenterPointTensorRTExporter(trt_settings, logger)
    
    # 创建 runner
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=load_pytorch_model,
        onnx_exporter=onnx_exporter,
        tensorrt_exporter=tensorrt_exporter,
    )
    
    runner.run(checkpoint_path=args.checkpoint)
```

### 示例 3: Calibration（简单模型，使用默认）

```python
from autoware_ml.deployment.runners import DeploymentRunner
from autoware_ml.deployment.exporters.calibration.onnx_exporter import CalibrationONNXExporter
from autoware_ml.deployment.exporters.calibration.tensorrt_exporter import CalibrationTensorRTExporter

def main():
    # ... 创建 data_loader, evaluator, config, logger ...
    
    # 创建 exporters（使用默认 IdentityWrapper）
    onnx_settings = config.get_onnx_settings()
    trt_settings = config.get_tensorrt_settings()
    
    onnx_exporter = CalibrationONNXExporter(onnx_settings, logger)
    tensorrt_exporter = CalibrationTensorRTExporter(trt_settings, logger)
    
    # 创建 runner（不传入 exporters，使用默认）
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=load_pytorch_model,
        onnx_exporter=onnx_exporter,  # 可选：不传则使用默认 ONNXExporter
        tensorrt_exporter=tensorrt_exporter,  # 可选：不传则使用默认 TensorRTExporter
    )
    
    runner.run(checkpoint_path=args.checkpoint)
```

### 示例 4: 使用默认 exporter（最简单）

```python
from autoware_ml.deployment.runners import DeploymentRunner
from autoware_ml.deployment.exporters.base.onnx_exporter import ONNXExporter
from autoware_ml.deployment.exporters.base.tensorrt_exporter import TensorRTExporter
from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper

def main():
    # ... 创建 data_loader, evaluator, config, logger ...
    
    # 方式 1: 在 runner 中传入 model_wrapper
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=load_pytorch_model,
        model_wrapper=YOLOXONNXWrapper,  # runner 会自动传给默认 exporter
    )
    
    # 方式 2: 显式创建 exporter 并传入 wrapper
    onnx_settings = config.get_onnx_settings()
    onnx_exporter = ONNXExporter(
        onnx_settings, 
        logger,
        model_wrapper=YOLOXONNXWrapper
    )
    
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=load_pytorch_model,
        onnx_exporter=onnx_exporter,
    )
    
    runner.run(checkpoint_path=args.checkpoint)
```

## 架构优势

### 1. 显式依赖
- 所有依赖都在 `main.py` 中清晰可见
- 不需要查找配置文件或注册表
- 更容易理解和调试

### 2. 灵活性
- 可以为不同的 exporter 使用不同的 wrapper
- 可以轻松替换或扩展组件
- 支持测试时注入 mock 对象

### 3. 类型安全
- IDE 可以提供更好的类型提示
- 编译时就能发现错误
- 不需要字符串查找

### 4. 简化配置
- 不再需要在 config 中配置 `model_wrapper`
- 配置专注于导出参数（opset_version, batch_size 等）
- 代码逻辑更清晰

## 迁移指南

### 从旧架构迁移

1. **移除 config 中的 model_wrapper 配置**
   ```python
   # 旧代码
   onnx_config = dict(
       model_wrapper='yolox',  # 删除这行
       ...
   )
   ```

2. **在 main.py 中导入并创建 wrapper**
   ```python
   # 新代码
   from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
   ```

3. **创建 exporter 时传入 wrapper**
   ```python
   # 新代码
   onnx_exporter = YOLOXONNXExporter(
       onnx_settings, 
       logger,
       model_wrapper=YOLOXONNXWrapper
   )
   ```

4. **传入 DeploymentRunner**
   ```python
   runner = DeploymentRunner(
       ...,
       onnx_exporter=onnx_exporter,
       ...
   )
   ```

## 总结

新的架构通过依赖注入模式，使代码更加：
- **清晰**：所有依赖显式声明
- **灵活**：易于替换和扩展
- **可测试**：易于注入 mock 对象
- **类型安全**：IDE 支持更好

每个项目现在完全控制自己的导出流程，不再依赖全局注册表。

