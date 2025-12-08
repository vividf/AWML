# CenterPoint Quantization Integration Changelog

本文档记录了将 PTQ (Post-Training Quantization) 和 QAT (Quantization-Aware Training) 集成到 AWML CenterPoint 部署框架中的所有改动。

## 概述

本次集成基于 `Lidar_AI_Solution/CUDA-CenterPoint` 的量化实现，将量化功能适配到 AWML CenterPoint 的 pillar-based 架构（使用 2D 卷积而非 3D 稀疏卷积）。

## 主要改动

### 1. BatchNorm Fusion 修复 (`quantization/fusion/bn_fusion.py`)

#### 问题 1: Conv-BN 配对错误
**错误信息:**
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (128) at non-singleton dimension 0
```

**原因:** `find_conv_bn_pairs()` 函数仅基于模块遍历顺序配对 Conv 和 BN，未验证通道维度是否匹配。

**修复:**
- 添加 `_get_conv_out_channels()` 和 `_get_bn_num_features()` 辅助函数
- 在 `find_conv_bn_pairs()` 中添加通道维度验证
- 仅当 Conv 输出通道数等于 BN num_features 时才配对

**代码位置:** `projects/CenterPoint/quantization/fusion/bn_fusion.py:105-169`

#### 问题 2: ConvTranspose2d 权重形状处理错误
**错误信息:**
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (128) at non-singleton dimension 0
```

**原因:** `ConvTranspose2d` 的权重形状为 `[in_channels, out_channels, H, W]`，而 `Conv2d` 为 `[out_channels, in_channels, H, W]`。BN 融合时 scale 应应用到维度 1（out_channels），而非维度 0。

**修复:**
- 在 `fuse_bn_weights()` 中添加 `is_transposed` 参数
- 当 `is_transposed=True` 时，将 scale reshape 为 `[1, -1, 1, 1]` 以应用到维度 1
- 在 `fuse_conv_bn()` 中自动检测 `ConvTranspose*` 类型并传递标志

**代码位置:** `projects/CenterPoint/quantization/fusion/bn_fusion.py:16-118`

---

### 2. 量化模块初始化修复 (`quantization/replace.py`)

#### 问题: 量化描述符未初始化
**错误信息:**
```
AttributeError: 'NoneType' object has no attribute 'num_bits'
```

**原因:** `transfer_to_quantization()` 使用 `__new__()` 创建实例（跳过 `__init__`），导致 `default_quant_desc_input` 和 `default_quant_desc_weight` 类属性保持为 `None`。

**修复:**
- 添加 `_ensure_quant_descriptors_initialized()` 函数，显式初始化所有量化模块类的默认描述符
- 在 `transfer_to_quantization()` 开始时调用此函数
- 使用全局标志 `_quant_descriptors_initialized` 避免重复初始化

**代码位置:** `projects/CenterPoint/quantization/replace.py:10-60`

---

### 3. 量化器状态打印修复 (`quantization/utils.py`)

#### 问题: 多元素 amax 张量无法转换为标量
**错误信息:**
```
RuntimeError: a Tensor with 16 elements cannot be converted to Scalar
```

**原因:** 对于 per-channel 量化，`_amax` 是多元素张量，但 `print_quantizer_status()` 尝试使用 `.item()` 转换为标量。

**修复:**
- 检查 `amax.numel()` 判断是否为标量
- 标量：直接打印值
- 多元素：打印元素数量、最小值、最大值

**代码位置:** `projects/CenterPoint/quantization/utils.py:117-147`

---

### 4. 量化模型部署支持 (`deploy/utils.py`)

#### 新增功能: 量化检查点加载
**功能:** 支持加载 PTQ/QAT 量化检查点，应用与量化时相同的变换（BN 融合、Q/DQ 节点插入）。

**实现:**
1. **`_load_quantized_checkpoint()`** - 加载量化检查点的主函数
   - 应用 BN 融合（如果启用）
   - 插入 Q/DQ 节点
   - 加载检查点状态字典
   - 移动量化器 amax 到目标设备
   - 禁用敏感层的量化
   - 配置 ONNX 导出模式

2. **`_move_quantizer_amax_to_device()`** - 将量化器 amax 张量移动到正确设备
   - 解决校准在 GPU 但部署在 CPU 的设备不匹配问题

3. **`_disable_quantization_for_sensitive_layers()`** - 禁用敏感层的量化
   - 用于处理 TensorRT 不支持的层（如 ConvTranspose2d）

4. **`setup_quantization_for_onnx_export()`** - 配置 pytorch-quantization 用于 ONNX 导出
   - 设置 `TensorQuantizer.use_fb_fake_quant = True`
   - 确保 Q/DQ 节点导出为标准的 `QuantizeLinear`/`DequantizeLinear` ONNX 操作

**代码位置:** `projects/CenterPoint/deploy/utils.py:99-230`

#### 更新函数签名
- `build_model_from_cfg()` - 添加 `quantization` 参数
- `build_centerpoint_onnx_model()` - 添加 `quantization` 参数

---

### 5. 部署 Runner 更新 (`deployment/runners/projects/centerpoint_runner.py`)

#### 更新: 传递量化配置到模型加载器
**改动:** 在 `load_pytorch_model()` 中从部署配置读取 `quantization` 部分并传递给 `build_centerpoint_onnx_model()`。

**代码位置:** `deployment/runners/projects/centerpoint_runner.py:122-135`

---

### 6. TensorRT INT8 精度策略支持 (`deployment/core/config/base_config.py`)

#### 新增: INT8 精度策略
**改动:**
1. 在 `PrecisionPolicy` 枚举中添加 `INT8 = "int8"`
2. 在 `PRECISION_POLICIES` 中添加 `"int8": {"FP16": True}`

**重要说明:**
- **不设置 `INT8` builder flag** - 该标志用于隐式量化（需要校准器）
- **仅设置 `FP16` flag** - 作为回退精度
- TensorRT 会自动从 ONNX 中的 `QuantizeLinear`/`DequantizeLinear` 节点读取显式量化信息

**代码位置:** `deployment/core/config/base_config.py:34-72`

---

### 7. INT8 部署配置文件 (`deploy/configs/deploy_config_int8.py`)

#### 新增: 完整的 INT8 部署配置
**配置内容:**
- `quantization` 部分：启用量化，指定 PTQ 模式，列出敏感层
- `checkpoint_path`: 指向 PTQ 检查点
- `backend_config`: 使用 `precision_policy="int8"`
- `sensitive_layers`: 排除 ConvTranspose2d 层（TensorRT 不支持 INT8）

**敏感层排除原因:**
TensorRT 对 `ConvTranspose2d`（转置卷积）的 INT8 支持有限，会导致构建失败：
```
Error Code 10: Internal Error (Could not find any implementation for node ... ConvTranspose)
```

**代码位置:** `projects/CenterPoint/deploy/configs/deploy_config_int8.py`

---

## 文件变更清单

### 新增文件
1. `projects/CenterPoint/deploy/configs/deploy_config_int8.py` - INT8 部署配置
2. `projects/CenterPoint/quantization/CHANGELOG.md` - 本文档

### 修改文件
1. `projects/CenterPoint/quantization/fusion/bn_fusion.py`
   - 添加通道维度验证
   - 修复 ConvTranspose2d 权重融合

2. `projects/CenterPoint/quantization/replace.py`
   - 添加量化描述符初始化逻辑

3. `projects/CenterPoint/quantization/utils.py`
   - 修复量化器状态打印

4. `projects/CenterPoint/deploy/utils.py`
   - 添加量化检查点加载支持
   - 添加设备移动和敏感层禁用功能

5. `deployment/runners/projects/centerpoint_runner.py`
   - 传递量化配置到模型加载器

6. `deployment/core/config/base_config.py`
   - 添加 INT8 精度策略

## 使用指南

### 1. PTQ (Post-Training Quantization)

#### 步骤 1: 量化模型
```bash
python tools/detection3d/centerpoint_quantization.py ptq \
    --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --calibrate-batches 100 \
    --skip-layers pts_neck.deblocks.0.0 pts_neck.deblocks.1.0 pts_neck.deblocks.2.0 \
    --output work_dirs/centerpoint_ptq.pth
```

**注意:** `--skip-layers` 排除 ConvTranspose2d 层，因为 TensorRT 不支持 INT8。

#### 步骤 2: 部署量化模型
```bash
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/configs/deploy_config_int8.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py
```

### 2. 配置说明

#### deploy_config_int8.py 关键配置
```python
# 量化配置
quantization = dict(
    enabled=True,
    mode="ptq",  # 或 "qat"
    fuse_bn=True,
    sensitive_layers=[
        "pts_neck.deblocks.0.0",  # ConvTranspose2d
        "pts_neck.deblocks.1.0",  # ConvTranspose2d
        "pts_neck.deblocks.2.0",  # ConvTranspose2d
    ],
)

# TensorRT INT8 配置
backend_config = dict(
    common_config=dict(
        precision_policy="int8",  # 使用 INT8
        max_workspace_size=4 << 30,  # 4 GB
    ),
)
```

## 性能结果

基于测试结果：

| 模型 | Backbone Head 延迟 | 总延迟 | mAP |
|------|-------------------|--------|-----|
| FP16 (PyTorch) | 388.98 ms | 1241.34 ms | 0.3596 |
| INT8 (TensorRT) | 92.82 ms | 886.08 ms | 0.3573 |

**加速比:** Backbone Head 约 **4.2x**，总延迟约 **1.4x**  
**精度损失:** mAP 下降 **0.0023** (< 1%)

## 已知限制

1. **ConvTranspose2d 不支持 INT8**
   - TensorRT 对转置卷积的 INT8 支持有限
   - 解决方案：在量化时排除这些层（使用 `--skip-layers`）

2. **ONNX 导出警告**
   - pytorch-quantization 会产生 TracerWarning（关于张量到 Python 值的转换）
   - 这些警告不影响功能，可以忽略

3. **设备不匹配**
   - 如果校准在 GPU 但部署在 CPU，需要移动 amax 张量
   - 已自动处理

## 故障排除

### 问题 1: TensorRT 构建失败 - "no scaling factors detected"
**原因:** 设置了 `INT8` builder flag 但没有校准器  
**解决:** 使用 `precision_policy="int8"`（会自动从 Q/DQ 节点读取）

### 问题 2: TensorRT 构建失败 - "Could not find implementation for ConvTranspose"
**原因:** ConvTranspose2d 不支持 INT8  
**解决:** 在 `sensitive_layers` 中排除这些层

### 问题 3: 设备不匹配错误
**原因:** amax 张量在错误的设备上  
**解决:** 已自动处理，如果仍有问题，检查 `_move_quantizer_amax_to_device()` 函数

### 问题 4: 量化描述符未初始化
**原因:** `transfer_to_quantization()` 在描述符初始化前调用  
**解决:** 已修复，确保 `_ensure_quant_descriptors_initialized()` 被调用

## 参考文档

- [AWML Integration Plan](../docs/AWML_integrate_plan.md) - 详细的集成计划
- [Deployment README](../deploy/README.md) - 部署框架文档
- [CUDA-CenterPoint QAT](../../../../Lidar_AI_Solution/CUDA-CenterPoint/qat/) - 参考实现

## 版本信息

- **集成日期:** 2025-12-08
- **基于:** CUDA-CenterPoint quantization implementation
- **适配:** AWML CenterPoint (pillar-based architecture)
