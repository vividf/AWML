# QDQ ONNX 模型 INT8 优化方法对比分析

## 问题背景

对于已经包含 QDQ (Quantize-Dequantize) 节点的 ONNX 模型，有两种不同的方法来构建 TensorRT INT8 引擎：

1. **AWML deployment with precision --fp16**
2. **autoware_lidar_centerpoint build engine**

## 核心区别

### 1. AWML Deployment 方法 (`precision_policy="int8"`)

**代码位置**: `deployment/core/config/base_config.py:78`

```python
# INT8 explicit quantization: Use FP16 as fallback, don't set INT8 builder flag.
# INT8 builder flag is for implicit quantization with calibrator.
# For explicit quantization (Q/DQ nodes from PTQ/QAT), TensorRT reads precision
# from QuantizeLinear/DequantizeLinear ops in ONNX automatically.
PrecisionPolicy.INT8.value: {"FP16": True},
```

**关键特点**:
- ✅ **不设置 `kINT8` builder flag**
- ✅ **设置 `kFP16` 作为 fallback**
- ✅ **TensorRT 自动从 ONNX 的 Q/DQ 节点读取精度信息**
- ✅ **适用于显式量化（Explicit Quantization）**

**工作原理**:
1. TensorRT 解析 ONNX 时，会自动识别 `QuantizeLinear` 和 `DequantizeLinear` 节点
2. 这些节点已经包含了量化信息（scale, zero_point）
3. TensorRT 会根据这些 QDQ 节点自动选择 INT8 精度
4. FP16 作为 fallback，用于 QDQ 节点之间的操作或无法量化的层

**实现代码**: `deployment/exporters/common/tensorrt_exporter.py:143-149`
```python
# Apply precision flags to builder config
for flag_name, enabled in policy_flags.items():
    if flag_name == "STRONGLY_TYPED":
        continue
    if enabled and hasattr(trt.BuilderFlag, flag_name):
        builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))
        self.logger.info(f"BuilderFlag.{flag_name} enabled")
```

### 2. autoware_lidar_centerpoint 方法 (`trt_precision="fp16"`)

**代码位置**: `autoware_tensorrt_common/src/tensorrt_common.cpp:491-494`

```cpp
if (trt_config_->precision == "fp16") {
  builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
} else if (trt_config_->precision == "int8") {
  builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);
}
```

**实际配置**: `centerpoint_tiny.param.yaml:8`
```yaml
trt_precision: fp16
```

**关键特点**:
- ✅ **设置 `kFP16` builder flag**（与 AWML 相同）
- ✅ **TensorRT 会自动识别 QDQ 节点并使用 INT8**
- ✅ **FP16 作为 fallback，用于无法量化的层**
- ⚠️ **如果使用 `trt_precision="int8"`，会设置 `kINT8` flag，可能导致冲突**

**工作原理**:
1. 当 `trt_precision="fp16"` 时，设置 `kFP16` flag
2. TensorRT 解析 ONNX 时，**会自动检测 QDQ 节点**
3. 如果检测到 QDQ 节点，TensorRT 会在这些层使用 INT8 精度
4. FP16 用于 QDQ 节点之间的操作或无法量化的层
5. **如果使用 `trt_precision="int8"`，会设置 `kINT8` flag，这适用于隐式量化（需要 calibrator）**

## 详细对比表

| 特性 | AWML Deployment (`precision_policy="int8"`) | autoware_lidar_centerpoint (`trt_precision="fp16"`) |
|------|----------------|---------------------------|
| **Builder Flag** | `kFP16` | `kFP16` |
| **QDQ 节点支持** | ✅ 自动识别 | ✅ 自动识别 |
| **量化类型** | 显式量化 (Explicit) | 显式量化 (Explicit) |
| **Calibrator 需求** | ❌ 不需要 | ❌ 不需要 |
| **适用场景** | QAT/PTQ 导出的 QDQ ONNX | QAT/PTQ 导出的 QDQ ONNX |
| **TensorRT 行为** | 从 QDQ 节点读取精度 | 从 QDQ 节点读取精度 |
| **设计意图** | 明确支持 QDQ 显式量化 | 通用 FP16 优化（自动识别 QDQ） |

**注意**: 如果 `autoware_lidar_centerpoint` 使用 `trt_precision="int8"`，则会设置 `kINT8` flag，这适用于隐式量化（需要 calibrator），对于已包含 QDQ 的 ONNX 可能产生冲突。

## 技术细节

### QDQ 节点的工作原理

QDQ (Quantize-Dequantize) 节点是 ONNX 中表示量化的标准方式：

```
Input (FP32) → QuantizeLinear → INT8 → DequantizeLinear → Output (FP32)
```

- `QuantizeLinear`: 将 FP32 转换为 INT8
- `DequantizeLinear`: 将 INT8 转换回 FP32

### TensorRT 的两种量化模式

1. **显式量化 (Explicit Quantization)**:
   - ONNX 模型已经包含 Q/DQ 节点
   - TensorRT **自动检测**这些节点并使用量化信息
   - **不需要设置 `kINT8` flag**
   - **可以设置 `kFP16` 作为 fallback**
   - TensorRT 的 ONNX parser 会自动识别 `QuantizeLinear` 和 `DequantizeLinear` 操作符
   - 当检测到 QDQ 节点时，TensorRT 会在这些层使用 INT8 精度

2. **隐式量化 (Implicit Quantization)**:
   - ONNX 模型是 FP32/FP16（不包含 QDQ 节点）
   - TensorRT 需要 calibrator 来生成量化参数
   - **需要设置 `kINT8` flag**
   - **需要提供 `IInt8Calibrator`**
   - TensorRT 会使用校准数据来确定每层的量化参数

### TensorRT 自动检测 QDQ 的机制

**重要**: TensorRT 的 ONNX parser 在解析模型时会自动检测 `QuantizeLinear` 和 `DequantizeLinear` 操作符。当检测到这些节点时：

1. **自动使用 INT8**: TensorRT 会在 QDQ 节点之间的层使用 INT8 精度
2. **不需要 `kINT8` flag**: 显式量化不需要设置 `kINT8` builder flag
3. **FP16 fallback**: 如果设置了 `kFP16` flag，无法量化的层会使用 FP16
4. **精度信息**: TensorRT 从 QDQ 节点的 `scale` 和 `zero_point` 参数获取量化信息

这就是为什么两种方法（AWML 和 centerpoint）在设置 `kFP16` flag 时都能正确处理 QDQ ONNX 模型。

## 关键发现

### 两种方法实际上非常相似

**当处理包含 QDQ 节点的 ONNX 模型时**：

1. **AWML Deployment** (`precision_policy="int8"`):
   - 设置 `kFP16` flag
   - TensorRT 自动识别 QDQ 节点并使用 INT8
   - FP16 作为 fallback

2. **autoware_lidar_centerpoint** (`trt_precision="fp16"`):
   - 也设置 `kFP16` flag
   - TensorRT 同样会自动识别 QDQ 节点并使用 INT8
   - FP16 作为 fallback

**结论**: 两种方法在处理 QDQ ONNX 模型时的行为是**一致的**，都依赖 TensorRT 的自动 QDQ 检测机制。

### 潜在问题

**如果 autoware_lidar_centerpoint 使用 `trt_precision="int8"`**：

1. **双重量化风险**: 设置 `kINT8` flag 可能导致 TensorRT 对已经量化的层再次进行量化
2. **精度损失**: 可能导致额外的精度损失
3. **构建失败**: 某些情况下可能导致引擎构建失败
4. **需要 Calibrator**: 隐式量化需要提供 `IInt8Calibrator`

### AWML Deployment 的设计优势

1. **明确的语义**: `precision_policy="int8"` 明确表示处理量化模型
2. **文档清晰**: 代码注释明确说明 QDQ 节点的处理方式
3. **避免混淆**: 不会让用户误以为需要设置 `kINT8` flag

## 建议

### 对于包含 QDQ 的 ONNX 模型

**两种方法都可以使用**：

1. **AWML deployment**:
   ```python
   precision_policy="int8"  # 实际设置 FP16 fallback，让 TensorRT 自动识别 QDQ
   ```

2. **autoware_lidar_centerpoint**:
   ```yaml
   trt_precision: fp16  # TensorRT 会自动识别 QDQ 节点并使用 INT8
   ```

**关键**: 两种方法都设置 `kFP16` flag，TensorRT 会自动检测 QDQ 节点并在这些层使用 INT8。

### 对于未量化的 ONNX 模型

如果需要在 autoware_lidar_centerpoint 中进行量化：

1. **选项 1**: 先使用 AWML 进行 QAT/PTQ 量化，导出包含 QDQ 的 ONNX，然后使用 `trt_precision="fp16"`
2. **选项 2**: 使用 `trt_precision="int8"` 进行隐式量化（需要提供 calibrator）
3. **选项 3**: 使用 `trt_precision="fp16"` 进行 FP16 优化（不进行 INT8 量化）

## 代码修改建议

### 修改 autoware_tensorrt_common 以支持 QDQ

可以在 `tensorrt_common.cpp` 中添加检测逻辑：

```cpp
// 检测 ONNX 是否包含 QDQ 节点
bool has_qdq_nodes = checkForQDQNodes(network_);

if (trt_config_->precision == "int8") {
  if (has_qdq_nodes) {
    // 显式量化：不设置 kINT8，只设置 FP16 fallback
    builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    logger_->log(nvinfer1::ILogger::Severity::kINFO,
                 "Detected QDQ nodes, using explicit quantization with FP16 fallback");
  } else {
    // 隐式量化：需要 calibrator
    builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (!calibrator_) {
      logger_->log(nvinfer1::ILogger::Severity::kERROR,
                   "INT8 precision requires calibrator for implicit quantization");
      return false;
    }
  }
}
```

## 总结

### 核心发现

1. **两种方法在处理 QDQ ONNX 模型时行为一致**:
   - 都设置 `kFP16` builder flag
   - TensorRT 都会自动检测 QDQ 节点并在这些层使用 INT8
   - FP16 作为 fallback 用于无法量化的层

2. **关键区别**:
   - **AWML**: `precision_policy="int8"` 明确表示处理量化模型，语义更清晰
   - **centerpoint**: `trt_precision="fp16"` 是通用 FP16 优化，但也能自动识别 QDQ

3. **重要警告**:
   - 如果 centerpoint 使用 `trt_precision="int8"`，会设置 `kINT8` flag
   - 这适用于隐式量化（需要 calibrator）
   - 对于已包含 QDQ 的 ONNX，可能导致双重量化或冲突

### 最佳实践

- **对于包含 QDQ 的 ONNX**: 两种方法都可以，都使用 `kFP16` flag
- **对于未量化的 ONNX**:
  - 使用 `trt_precision="fp16"` 进行 FP16 优化
  - 或使用 `trt_precision="int8"` 进行隐式量化（需要 calibrator）
