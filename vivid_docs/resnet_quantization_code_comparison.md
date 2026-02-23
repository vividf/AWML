# ResNet Residual Connection Quantization: 三种方法代码实现对比

本文档详细整理并对比了三种方法（ModelOpt、Lidar_AI_Solution、AWML CenterPoint）处理 ResNet residual connection quantization 的完整代码实现。

## 核心原则

**所有三种方法都遵循相同的核心原则：**
- ✅ **只在 skip connection (identity branch) 上添加 QDQ**
- ❌ **不在 main branch (conv path) 上添加 QDQ**

这样才能让 TensorRT 融合 Conv+Add 操作，减少 reformat 操作。

---

## 方法 1: Lidar_AI_Solution (PyTorch 层面实现)

### 实现位置
- **文件**: `Lidar_AI_Solution/CUDA-BEVFusion/qat/lean/quantize.py`
- **策略**: 纯 PyTorch 层面实现，使用 forward hook

### 完整代码实现

#### 1. Forward Hook 类定义

```python
class hook_bottleneck_forward:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        self = self.obj
        identity = x

        # Main branch (conv path)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        # Handle downsample if exists
        if self.downsample is not None:
            identity = self.downsample(x)

        # ✅ 关键：只量化 identity branch (skip connection)
        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)

        # ❌ conv path (out) 不量化，直接相加
        out += identity
        out = self.relu(out)
        return out
```

#### 2. 创建和附加 residual_quantizer

```python
def quantize_camera_backbone(model_camera_backbone):
    replace_to_quantization_module(model_camera_backbone)

    for name, bottleneck in model_camera_backbone.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            print(f"Add QuantAdd to {name}")

            # 创建或复用 residual_quantizer
            if bottleneck.downsample is not None:
                # 有 downsample: 创建新的 TensorQuantizer
                bottleneck.downsample[0]._input_quantizer = bottleneck.conv1._input_quantizer
                bottleneck.residual_quantizer = quant_nn.TensorQuantizer(
                    quant_nn.QuantConv2d.default_quant_desc_input
                )
            else:
                # 无 downsample: 复用 conv1._input_quantizer (共享校准数据)
                bottleneck.residual_quantizer = bottleneck.conv1._input_quantizer

            # 替换 forward 方法
            bottleneck.forward = hook_bottleneck_forward(bottleneck)
```

### 关键特点

- ✅ **简单直接**: 在 PyTorch forward hook 中只量化 identity branch
- ✅ **不依赖 ONNX 后处理**: 纯 PyTorch 实现
- ⚠️ **依赖 TensorQuantizer 的 ONNX 导出行为**: 如果导出时出现问题，无法修正
- ✅ **共享校准数据**: 无 downsample 时复用 `conv1._input_quantizer`

### 工作流程

```
PyTorch Model (residual_quantizer)
  → ONNX Export (TensorQuantizer 导出 QDQ)
  → ONNX Model (理论上只有 skip connection 有 QDQ)
```

---

## 方法 2: ModelOpt (ONNX 层面后处理)

### 实现位置
- **文件**:
  - `TensorRT-Model-Optimizer/modelopt/onnx/quantization/int8.py`
  - `TensorRT-Model-Optimizer/modelopt/onnx/quantization/graph_utils.py`
- **策略**: ONNX 图分析后移除 main branch 的 QDQ

### 完整代码实现

#### 1. 主量化流程

```python
def quantize(
    onnx_path: str,
    quantize_mode: str = "int8",
    calibration_data: CalibrationDataType = None,
    ...
):
    # Step 1: 先量化所有节点（包括两个分支都有 QDQ）
    quantize_static(onnx_path, tmp_onnx_path, ...)

    # Step 2: 加载量化后的 ONNX 模型
    graph = Graph.from_onnx(onnx.load(tmp_onnx_path))

    # Step 3: 识别 residual Add 操作并构建非 residual 输入映射
    non_residual_inputs, no_quantize_inputs = build_non_residual_input_map(graph)

    # Step 4: 移除 main branch 的 QDQ
    remove_partial_input_qdq(graph, no_quantize_inputs)

    # Step 5: 保存最终模型
    onnx.save(gs.export_onnx(graph), output_path)
```

#### 2. 识别 residual Add 操作

```python
def build_non_residual_input_map(
    graph: Graph,
) -> tuple[dict[str, str], list[tuple[Node, Node, str]]]:
    """
    识别 residual Add 操作，并标记哪些输入不应该量化。

    返回:
        - non_residual_inputs: 非 residual 输入的映射
        - no_quantize_inputs: 不应该量化的输入列表 (source, target, input_name)
    """
    non_residual_inputs = {}
    no_quantize_inputs = []

    # 遍历所有 Add 节点
    for node in graph.nodes:
        if node.op != "Add":
            continue

        # 检查是否是 residual connection
        # 识别模式: Conv -> BN -> ReLU -> Conv -> BN -> Add
        # 其中 main branch 不应该有 QDQ

        # 找到 main branch (conv path) 的输入
        for inp in node.inputs:
            # 检查输入是否来自 Conv/BN/ReLU 模式
            if _is_conv_bn_relu_pattern(inp):
                # 标记这个输入不应该有 QDQ
                no_quantize_inputs.append((inp, node, inp.name))

    return non_residual_inputs, no_quantize_inputs
```

#### 3. 移除 main branch 的 QDQ

```python
def remove_partial_input_qdq(
    graph: Graph,
    no_quantize_inputs: list[tuple[Node, Node, str]],
) -> None:
    """
    从标记的输入中移除 QDQ 节点。

    修改 ONNX 图，移除 main branch 的 QDQ，保留 skip connection 的 QDQ。
    """
    logger.info("Deleting QDQ nodes from marked inputs to make certain operations fusible")
    graph_nodes = {node.name: node for node in graph.nodes}

    for source, target, non_qdq_input_name in no_quantize_inputs:
        # 找到 source node 在量化图中的对应节点
        source_node = graph_nodes[source.name]

        try:
            # 找到 Q -> DQ 节点
            # source_node -> Q -> DQ -> target_node
            dq_node = source_node.o().o()
        except Exception:
            # 到达图末尾
            continue

        if dq_node.op == "DequantizeLinear":
            dq_node = dq_node.outputs[0]

            # 移除 DQ 节点，直接连接 source 到 target
            while len(dq_node.outputs):
                # 找到 target 中连接到 dq_node 的输入索引
                target_input_idx_arr = [
                    idx
                    for idx, inp in enumerate(dq_node.outputs[0].inputs)
                    if inp.name == dq_node.name
                ]
                target_input_idx = target_input_idx_arr[0] if target_input_idx_arr else 0

                # 直接连接 source_node 的输出到 target 的输入
                # 跳过 Q -> DQ 节点
                dq_node.outputs[0].inputs[target_input_idx] = source_node.outputs[0]

    # 清理图并重新排序
    graph.cleanup()
    graph.toposort()
```

### 关键特点

- ✅ **灵活**: 可以处理任意 ONNX 模型，不依赖 PyTorch 实现
- ✅ **可修正**: 即使 PyTorch 导出时出错，也可以在 ONNX 层面修正
- ⚠️ **复杂度高**: 需要图分析和节点操作
- ✅ **通用性强**: 适用于任何框架导出的 ONNX 模型

### 工作流程

```
PyTorch/其他框架 Model
  → ONNX Export (所有节点都有 QDQ)
  → ONNX Model (两个分支都有 QDQ)
  → Graph Analysis (识别 residual Add)
  → Remove QDQ (移除 main branch QDQ)
  → Final ONNX Model (只有 skip connection 有 QDQ)
```

---

## 方法 3: AWML CenterPoint (PyTorch 层面实现，与 Lidar_AI_Solution 对齐)

### 实现位置
- **文件**:
  - `AWML/projects/CenterPoint/quantization/replace.py`
  - `AWML/deployment/exporters/common/onnx_exporter.py`
- **策略**: PyTorch 层面实现，与 Lidar_AI_Solution 对齐

### 完整代码实现

#### 1. Forward Hook 类定义

##### BasicBlockForwardHook

```python
class BasicBlockForwardHook:
    """
    Forward hook for BasicBlock to use residual_quantizer for residual connections.

    This hook replaces the forward method of BasicBlock to quantize only the identity
    branch (residual connection), not the conv path output. This enables TensorRT to
    fuse Conv+Add operations, reducing reformat operations.
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection."""
        self = self.obj

        identity = x

        # Main branch (conv path)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Handle downsample if exists
        if self.downsample is not None:
            identity = self.downsample(x)

        # ✅ 关键：只量化 identity branch (skip connection)
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)

        # ❌ conv path (out) 不量化，直接相加
        out = out + identity
        out = self.relu(out)
        return out
```

##### SparseBasicBlockForwardHook

```python
class SparseBasicBlockForwardHook:
    """
    Forward hook for SparseBasicBlock to use residual_quantizer for residual connections.

    SparseBasicBlock works with SparseConvTensor which requires replace_feature.
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection for sparse tensors."""
        self = self.obj

        identity = x
        out = self.conv1(x)

        # Handle ReLU (may be fused in conv1)
        if hasattr(self, "relu") and not getattr(self.conv1, "act_type", None):
            out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # ✅ 关键：只量化 identity branch (skip connection)
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            identity = identity.replace_feature(
                self.residual_quantizer(identity.features)
            )

        # ❌ conv path (out) 不量化，直接相加
        out = out.replace_feature(out.features + identity.features)
        return out
```

#### 2. 创建和附加 residual_quantizer

```python
def attach_quant_add(model: nn.Module, target_class_names: Optional[Set[str]] = None):
    """
    Attach residual_quantizer to modules that perform residual add and replace their forward methods.

    This follows the same approach as lidar-ai-solution (CUDA-BEVFusion):
    - Only quantize the identity branch (residual connection), not the conv path output
    - This enables TensorRT to fuse Conv+Add operations, reducing reformat operations
    """
    try:
        from pytorch_quantization import tensor_quant
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError(
            "pytorch-quantization is required for residual quantization. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )

    # Ensure quantization descriptors are initialized
    _ensure_quant_descriptors_initialized()

    target_class_names = target_class_names or {"BasicBlock", "SparseBasicBlock"}

    attached_count = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in target_class_names or any(name in cls_name for name in target_class_names):
            # Attach residual_quantizer if not already present
            # Aligned with lidar-ai-solution:
            # - If downsample exists: create new TensorQuantizer
            # - If no downsample: reuse conv1._input_quantizer (shares calibration data)
            if not hasattr(module, "residual_quantizer"):
                if hasattr(module, "downsample") and module.downsample is not None:
                    # Has downsample: create new quantizer
                    quant_desc = QuantConv2d.default_quant_desc_input
                    if quant_desc is None:
                        quant_desc = tensor_quant.QuantDescriptor(
                            num_bits=8, calib_method="histogram"
                        )
                    else:
                        # Ensure calib_method is set for calibration
                        if not hasattr(quant_desc, "calib_method") or quant_desc.calib_method is None:
                            quant_desc.calib_method = "histogram"
                    residual_quantizer = TensorQuantizer(quant_desc)
                    # ✅ 关键：注册为 submodule，确保 ONNX 导出时可以追踪
                    module.add_module("residual_quantizer", residual_quantizer)
                    attached_count += 1
                elif hasattr(module, "conv1") and hasattr(module.conv1, "_input_quantizer"):
                    # No downsample: reuse conv1._input_quantizer (same as lidar-ai-solution)
                    # Note: This is a reference, not a copy, so it shares the same quantizer instance
                    module.residual_quantizer = module.conv1._input_quantizer
                    attached_count += 1
                else:
                    # Fallback: create new quantizer
                    quant_desc = QuantConv2d.default_quant_desc_input
                    if quant_desc is None:
                        quant_desc = tensor_quant.QuantDescriptor(
                            num_bits=8, calib_method="histogram"
                        )
                    else:
                        if not hasattr(quant_desc, "calib_method") or quant_desc.calib_method is None:
                            quant_desc.calib_method = "histogram"
                    residual_quantizer = TensorQuantizer(quant_desc)
                    # ✅ 关键：注册为 submodule
                    module.add_module("residual_quantizer", residual_quantizer)
                    attached_count += 1

            # Replace forward method with hook that uses residual_quantizer
            is_sparse = "Sparse" in cls_name

            if is_sparse:
                # SparseBasicBlock: use SparseBasicBlockForwardHook
                if not isinstance(module.forward, SparseBasicBlockForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = SparseBasicBlockForwardHook(module)
            else:
                # BasicBlock: use BasicBlockForwardHook
                if not isinstance(module.forward, BasicBlockForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = BasicBlockForwardHook(module)

    if attached_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Attached residual_quantizer to {attached_count} residual blocks")
```

#### 3. ONNX 导出配置

```python
# 文件: AWML/deployment/exporters/common/onnx_exporter.py

def _do_onnx_export(
    self,
    model: torch.nn.Module,
    sample_input: Any,
    output_path: str,
    export_cfg: ONNXExportConfig,
) -> None:
    # ... 其他代码 ...

    # ✅ 关键：启用量化导出设置，确保 TensorQuantizer 导出为 QDQ 节点
    try:
        from pytorch_quantization import enable_onnx_export
        from pytorch_quantization.nn import TensorQuantizer

        # 设置 use_fb_fake_quant 以正确导出 QDQ
        TensorQuantizer.use_fb_fake_quant = True
        self.logger.debug("Enabled use_fb_fake_quant for ONNX export")

        # ✅ 关键：使用 enable_onnx_export context manager
        # 这会设置 TensorQuantizer._enable_onnx_export = True
        with enable_onnx_export():
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=export_cfg.export_params,
                    keep_initializers_as_inputs=export_cfg.keep_initializers_as_inputs,
                    opset_version=export_cfg.opset_version,
                    do_constant_folding=export_cfg.do_constant_folding,
                    input_names=list(export_cfg.input_names),
                    output_names=list(export_cfg.output_names),
                    dynamic_axes=export_cfg.dynamic_axes,
                    verbose=export_cfg.verbose,
                )
    except ImportError:
        # pytorch-quantization not available, skip quantization settings
        with torch.no_grad():
            torch.onnx.export(...)
```

#### 4. 调用入口

```python
# 文件: AWML/projects/CenterPoint/quantization/replace.py

def quant_model(
    model: nn.Module,
    quant_backbone: bool = True,
    quant_neck: bool = True,
    quant_head: bool = True,
    quant_voxel_encoder: bool = True,
    quant_add: bool = False,  # ✅ 控制是否启用 residual quantization
    skip_names: Optional[Set[str]] = None,
):
    # ... 其他量化设置 ...

    # ✅ 关键：如果启用 quant_add，则附加 residual_quantizer
    if quant_add:
        attach_quant_add(model)
```

### 关键特点

- ✅ **与 Lidar_AI_Solution 对齐**: 使用相同的 PyTorch-only 策略
- ✅ **注册为 submodule**: 使用 `add_module()` 确保 ONNX 导出时可以追踪
- ✅ **共享校准数据**: 无 downsample 时复用 `conv1._input_quantizer`
- ✅ **ONNX 导出配置**: 正确设置 `use_fb_fake_quant` 和 `enable_onnx_export()`
- ✅ **支持 SparseBasicBlock**: 处理稀疏卷积的特殊情况

### 工作流程

```
PyTorch Model (residual_quantizer registered as submodule)
  → attach_quant_add() (创建/复用 residual_quantizer)
  → Forward Hook (只量化 identity branch)
  → ONNX Export (TensorQuantizer.use_fb_fake_quant = True)
  → enable_onnx_export() context (TensorQuantizer._enable_onnx_export = True)
  → ONNX Model (理论上只有 skip connection 有 QDQ)
```

---

## 三种方法对比总结

| 特性 | Lidar_AI_Solution | ModelOpt | AWML CenterPoint |
|------|------------------|----------|------------------|
| **实现层面** | PyTorch | ONNX | PyTorch |
| **核心方法** | Forward hook | Graph analysis + QDQ removal | Forward hook |
| **Main branch QDQ** | ❌ 无（不添加） | ❌ 无（移除） | ❌ 无（不添加） |
| **Skip connection QDQ** | ✅ 有 | ✅ 有 | ✅ 有 |
| **复杂度** | 低 | 高 | 低 |
| **灵活性** | 中 | 高 | 中 |
| **依赖** | PyTorch + pytorch-quantization | ONNX + graph analysis | PyTorch + pytorch-quantization |
| **适用场景** | PyTorch 模型 | 任意 ONNX 模型 | PyTorch 模型 |

## 核心概念一致性 ✅

**三种方法的核心概念完全一致：**
- ✅ **只在 skip connection 上添加 QDQ**
- ❌ **不在 main branch 上添加 QDQ**

虽然实现方式不同，但最终效果完全一致，都能让 TensorRT 正确融合 Conv+Add 操作。

---

## 为什么 ResNet 需要这样处理？

### TensorRT 融合规则

TensorRT 可以将以下模式融合成单个 kernel：
```
Conv -> BN -> ReLU -> Add
```

但融合的前提是：
- **Conv 的输出必须是 INT8**（有 QDQ）
- **Add 的一个输入是 INT8**（有 QDQ）
- **Add 的另一个输入（main branch）不能有 QDQ**（必须是 FP32）

### 错误做法的影响

如果两个分支都量化：
```
Conv -> BN -> ReLU -> Conv -> BN -> QDQ -> Add <- QDQ Identity
                              ↑                    ↑
                         (main branch)      (skip connection)
```

**问题**:
- Main branch 有 QDQ → TensorRT 无法识别 Conv+Add 融合模式
- 会产生大量 reformat 操作（INT8 ↔ FP32 转换）
- 性能下降，内存开销增加

### 正确做法的影响

只量化 skip connection：
```
Conv -> BN -> ReLU -> Conv -> BN -> (无 QDQ) -> Add <- QDQ Identity
                              ↑                    ↑
                         (main branch)      (skip connection)
```

**优势**:
- Main branch 无 QDQ → TensorRT 可以识别 Conv+Add 融合模式
- 融合成单个 INT8 kernel
- 减少 reformat 操作
- 性能提升，内存效率提高

---

## 参考文档

- [ResNet Quantization Explanation](./resnet_quantization_explanation.md)
- [ResNet QDQ Concept Verification](./resnet_qdq_concept_verification.md)
- [Quantization Comparison](./quantization_comparison.md)
