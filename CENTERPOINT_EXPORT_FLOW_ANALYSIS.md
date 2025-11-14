# CenterPoint Export 流程分析

本文档详细说明 CenterPoint 在不同 device 设置下的 export、verify 和 evaluate 流程。

## 配置位置

主要配置文件：`projects/CenterPoint/deploy/configs/deploy_config.py`

```python
export = dict(
    mode="both",      # Export both ONNX and TensorRT
    verify=True,      # Enable verification
    device="cpu",     # 或 "cuda:0"
    work_dir="work_dirs/centerpoint_deployment",
)
```

## 流程概览

整个流程由 `DeploymentRunner.run()` 方法控制，按以下顺序执行：

1. **Load PyTorch Model** - 加载 PyTorch 模型到指定 device
2. **Export ONNX** - 导出 ONNX 模型（如果 mode 包含 "onnx" 或 "both"）
3. **Export TensorRT** - 导出 TensorRT 引擎（如果 mode 包含 "trt" 或 "both"）
4. **Verify** - 验证导出模型（如果 verify=True）
5. **Evaluate** - 评估模型性能（如果 evaluation.enabled=True）

---

## 情况 1: device="cpu"

### 1.1 ONNX Export

**代码路径**: 
- `autoware_ml/deployment/runners/deployment_runner.py::export_onnx()`
- `autoware_ml/deployment/exporters/centerpoint_exporter.py::export()`

**流程**:

1. **模型加载** (`main.py::build_model_from_cfg()`)
   ```python
   model = MODELS.build(model_config)
   model.to("cpu")  # 模型加载到 CPU
   load_checkpoint(model, checkpoint_path, map_location="cpu")
   ```

2. **数据准备** (`centerpoint_exporter.py::export()`)
   ```python
   # 从 data_loader 提取真实数据
   input_features, voxel_dict = model._extract_features(data_loader, sample_idx)
   # input_features 在 CPU 上（因为 model 在 CPU 上）
   ```

3. **导出两个 ONNX 文件**:
   - **Voxel Encoder** (`pts_voxel_encoder.onnx`):
     ```python
     # 使用 torch.onnx.export()
     torch.onnx.export(
         model.pts_voxel_encoder,
         input_features,  # CPU tensor
         "pts_voxel_encoder.onnx",
         ...
     )
     ```
   - **Backbone+Neck+Head** (`pts_backbone_neck_head.onnx`):
     ```python
     # 先运行 voxel encoder 获取中间特征
     voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)  # CPU
     x = model.pts_middle_encoder(voxel_features, coors, batch_size)  # CPU
     
     # 导出 backbone+neck+head
     torch.onnx.export(
         backbone_neck_head,
         x,  # CPU tensor
         "pts_backbone_neck_head.onnx",
         ...
     )
     ```

**结果**: ✅ 成功导出两个 ONNX 文件到 `work_dirs/centerpoint_deployment/`

---

### 1.2 TensorRT Export

**代码路径**: 
- `autoware_ml/deployment/runners/deployment_runner.py::export_tensorrt()`
- `autoware_ml/deployment/exporters/centerpoint_tensorrt_exporter.py::export()`

**流程**:

1. **检查 ONNX 文件存在**
   ```python
   onnx_files = [
       ("pts_voxel_encoder.onnx", "pts_voxel_encoder.engine"),
       ("pts_backbone_neck_head.onnx", "pts_backbone_neck_head.engine")
   ]
   ```

2. **创建 sample_input** (`centerpoint_tensorrt_exporter.py:108-113`)
   ```python
   device = "cpu"  # 从 config.export_config.device 获取
   
   if "voxel_encoder" in onnx_file:
       sample_input = torch.randn(10000, 32, 11, device=device)  # CPU tensor
   else:
       sample_input = torch.randn(1, 32, 200, 200, device=device)  # CPU tensor
   ```

3. **调用父类 TensorRTExporter** (`tensorrt_exporter.py::export()`)
   ```python
   # TensorRT 需要 CUDA，但 sample_input 在 CPU 上
   # TensorRT builder 会尝试在 CUDA 上构建引擎
   ```

**问题**: ⚠️ **TensorRT 需要 CUDA 设备**，即使 sample_input 在 CPU 上创建，TensorRT builder 仍然需要 CUDA 来构建引擎。如果系统没有 CUDA，TensorRT export 会失败。

**结果**: 
- 如果有 CUDA 可用：✅ 可能成功（TensorRT 会在 CUDA 上构建）
- 如果没有 CUDA：❌ 失败

---

### 1.3 Verify

**代码路径**: 
- `autoware_ml/deployment/runners/deployment_runner.py::run_verification()`
- `projects/CenterPoint/deploy/evaluator.py::verify()`

**流程**:

1. **创建 Pipeline** (`evaluator.py::_create_pipeline()`)
   ```python
   device = "cpu"  # 从 config.export_config.device 获取
   
   # ONNX Pipeline
   onnx_pipeline = CenterPointONNXPipeline(
       pytorch_model,  # 加载到 CPU
       onnx_dir=onnx_model_path,
       device="cpu"
   )
   
   # TensorRT Pipeline
   if not device.startswith("cuda"):
       logger.warning("TensorRT requires CUDA device, skipping TensorRT verification")
       tensorrt_pipeline = None  # ⚠️ 跳过 TensorRT verify
   ```

2. **运行验证** (`evaluator.py::verify()`)
   ```python
   # 对每个 sample:
   # 1. 运行 PyTorch reference (CPU)
   pytorch_outputs, pytorch_latency, _ = pytorch_pipeline.infer(...)
   
   # 2. 运行 ONNX (CPU)
   if onnx_pipeline:
       onnx_outputs, onnx_latency, _ = onnx_pipeline.infer(...)
       # 比较 onnx_outputs 和 pytorch_outputs
   
   # 3. 运行 TensorRT (跳过，因为 device="cpu")
   if tensorrt_pipeline:  # None
       # 不会执行
   ```

**结果**: 
- ✅ ONNX verify: 可以执行（在 CPU 上）
- ⚠️ TensorRT verify: **自动跳过**（因为 device="cpu"）

---

### 1.4 Evaluate

**代码路径**: 
- `autoware_ml/deployment/runners/deployment_runner.py::run_evaluation()`
- `projects/CenterPoint/deploy/evaluator.py::evaluate()`

**流程**:

1. **创建 Pipeline** (`evaluator.py::_create_pipeline()`)
   ```python
   device = "cpu"
   
   if backend == "onnx":
       pipeline = CenterPointONNXPipeline(
           pytorch_model,  # CPU
           onnx_dir=model_path,
           device="cpu"
       )
   
   elif backend == "tensorrt":
       if not str(device).startswith("cuda"):
           logger.warning("TensorRT requires CUDA device, skipping TensorRT evaluation")
           return None  # ⚠️ 跳过 TensorRT evaluate
   ```

2. **运行评估**
   ```python
   # 对每个 sample:
   predictions, latency, latency_breakdown = pipeline.infer(points, sample_meta)
   # 计算 mAP, NDS 等指标
   ```

**结果**: 
- ✅ ONNX evaluate: 可以执行（在 CPU 上，但可能较慢）
- ⚠️ TensorRT evaluate: **自动跳过**（因为 device="cpu"）

---

## 情况 2: device="cuda:0"

### 2.1 ONNX Export

**流程**:

1. **模型加载**
   ```python
   model = MODELS.build(model_config)
   model.to("cuda:0")  # 模型加载到 CUDA
   load_checkpoint(model, checkpoint_path, map_location="cuda:0")
   ```

2. **数据准备**
   ```python
   input_features, voxel_dict = model._extract_features(data_loader, sample_idx)
   # input_features 在 CUDA:0 上
   ```

3. **导出 ONNX**
   ```python
   # torch.onnx.export() 会自动处理 CUDA tensor
   # 导出过程中，tensor 会在 CUDA 上，但 ONNX 模型本身是 device-agnostic
   torch.onnx.export(
       model.pts_voxel_encoder,
       input_features,  # CUDA tensor
       "pts_voxel_encoder.onnx",
       ...
   )
   ```

**结果**: ✅ 成功导出两个 ONNX 文件（与 CPU 相同，但导出速度可能更快）

---

### 2.2 TensorRT Export

**流程**:

1. **创建 sample_input** (`centerpoint_tensorrt_exporter.py:108-113`)
   ```python
   device = "cuda:0"
   
   if "voxel_encoder" in onnx_file:
       sample_input = torch.randn(10000, 32, 11, device="cuda:0")  # CUDA tensor
   else:
       sample_input = torch.randn(1, 32, 200, 200, device="cuda:0")  # CUDA tensor
   ```

2. **TensorRT Builder** (`tensorrt_exporter.py::export()`)
   ```python
   # TensorRT builder 在 CUDA 上构建引擎
   builder = trt.Builder(trt_logger)
   builder_config = builder.create_builder_config()
   
   # 配置 optimization profile
   profile.set_shape(input_name, min_shape, opt_shape, max_shape)
   
   # 构建引擎（在 CUDA 上）
   serialized_engine = builder.build_serialized_network(network, builder_config)
   ```

**结果**: ✅ 成功导出两个 TensorRT 引擎到 `work_dirs/centerpoint_deployment/tensorrt/`

---

### 2.3 Verify

**流程**:

1. **创建 Pipeline**
   ```python
   device = "cuda:0"
   
   # ONNX Pipeline
   onnx_pipeline = CenterPointONNXPipeline(
       pytorch_model,  # CUDA:0
       onnx_dir=onnx_model_path,
       device="cuda:0"
   )
   
   # TensorRT Pipeline
   tensorrt_pipeline = CenterPointTensorRTPipeline(
       pytorch_model,  # CUDA:0
       tensorrt_dir=tensorrt_model_path,
       device="cuda:0"
   )
   ```

2. **运行验证**
   ```python
   # 对每个 sample:
   # 1. PyTorch reference (CUDA:0)
   pytorch_outputs, pytorch_latency, _ = pytorch_pipeline.infer(...)
   
   # 2. ONNX (CUDA:0，使用 ONNX Runtime GPU provider)
   onnx_outputs, onnx_latency, _ = onnx_pipeline.infer(...)
   
   # 3. TensorRT (CUDA:0)
   tensorrt_outputs, tensorrt_latency, _ = tensorrt_pipeline.infer(...)
   
   # 比较所有输出
   ```

**结果**: 
- ✅ ONNX verify: 可以执行（在 CUDA:0 上）
- ✅ TensorRT verify: 可以执行（在 CUDA:0 上）

---

### 2.4 Evaluate

**流程**:

1. **创建 Pipeline**
   ```python
   device = "cuda:0"
   
   # 所有 backend 都可以创建 pipeline
   pytorch_pipeline = CenterPointPyTorchPipeline(pytorch_model, device="cuda:0")
   onnx_pipeline = CenterPointONNXPipeline(pytorch_model, onnx_dir=..., device="cuda:0")
   tensorrt_pipeline = CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir=..., device="cuda:0")
   ```

2. **运行评估**
   ```python
   # 对每个 sample:
   predictions, latency, latency_breakdown = pipeline.infer(points, sample_meta)
   # 计算 mAP, NDS 等指标
   ```

**结果**: 
- ✅ ONNX evaluate: 可以执行（在 CUDA:0 上，速度快）
- ✅ TensorRT evaluate: 可以执行（在 CUDA:0 上，速度最快）

---

## 关键代码位置总结

### Export 相关

1. **ONNX Export**:
   - `autoware_ml/deployment/exporters/centerpoint_exporter.py::export()`
   - `autoware_ml/deployment/exporters/onnx_exporter.py::export()`
   - Device 影响: 模型和输入 tensor 的 device，但不影响 ONNX 文件本身（ONNX 是 device-agnostic）

2. **TensorRT Export**:
   - `autoware_ml/deployment/exporters/centerpoint_tensorrt_exporter.py::export()`
   - `autoware_ml/deployment/exporters/tensorrt_exporter.py::export()`
   - Device 影响: sample_input 的 device，但 TensorRT builder **必须**在 CUDA 上运行

### Verify 相关

1. **Verify 入口**:
   - `autoware_ml/deployment/runners/deployment_runner.py::run_verification()`
   - `projects/CenterPoint/deploy/evaluator.py::verify()`

2. **Device 检查**:
   - `evaluator.py:129-131`: TensorRT verify 在 CPU 时跳过
   - `evaluator.py:499-501`: TensorRT evaluate 在 CPU 时跳过

### Evaluate 相关

1. **Evaluate 入口**:
   - `autoware_ml/deployment/runners/deployment_runner.py::run_evaluation()`
   - `projects/CenterPoint/deploy/evaluator.py::evaluate()`

2. **Pipeline 创建**:
   - `evaluator.py::_create_pipeline()`: 根据 device 创建相应的 pipeline

---

## 总结对比表

| 操作 | device="cpu" | device="cuda:0" |
|------|-------------|-----------------|
| **ONNX Export** | ✅ 成功 | ✅ 成功（更快） |
| **TensorRT Export** | ⚠️ 需要 CUDA（可能失败） | ✅ 成功 |
| **ONNX Verify** | ✅ 可以执行 | ✅ 可以执行（更快） |
| **TensorRT Verify** | ⚠️ **自动跳过** | ✅ 可以执行 |
| **ONNX Evaluate** | ✅ 可以执行（较慢） | ✅ 可以执行（快） |
| **TensorRT Evaluate** | ⚠️ **自动跳过** | ✅ 可以执行（最快） |

---

## 注意事项

1. **TensorRT 需要 CUDA**: 
   - TensorRT export、verify 和 evaluate 都需要 CUDA 设备
   - 如果 device="cpu"，这些操作会被自动跳过

2. **ONNX 可以在 CPU 或 CUDA 上运行**:
   - ONNX export 可以在任何 device 上执行
   - ONNX verify/evaluate 可以在 CPU 或 CUDA 上执行，但 CUDA 会更快

3. **Device 参数传递路径**:
   ```
   deploy_config.py::export.device
   → BaseDeploymentConfig.export_config.device
   → DeploymentRunner.config.export_config.device
   → 传递给 exporter、evaluator、pipeline
   ```

4. **模型加载**:
   - 所有 backend（PyTorch、ONNX、TensorRT）都需要加载 PyTorch 模型
   - PyTorch 模型会根据 device 参数加载到相应的 device 上

