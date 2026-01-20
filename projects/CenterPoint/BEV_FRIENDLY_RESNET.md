# BEV-Friendly ResNet 配置说明

## 概述

本文档说明如何将 ResNet 骨干网络配置为 BEV-friendly 模式，确保输出特征图的尺寸为：
- `out0: (H, W)` - 保持原始分辨率
- `out1: (H/2, W/2)` - 2x 下采样
- `out2: (H/4, W/4)` - 4x 下采样

## 问题分析

标准的 ResNet 在 stem 层会进行下采样：
1. **conv1**: stride=2 → 2x 下采样
2. **maxpool**: stride=2 → 2x 下采样
3. **总计**: 4x 下采样在 stem 层

这导致：
- 如果输入是 `(760, 760)`
- stem 后变成 `(190, 190)`
- stage0 (stride=1) → `(190, 190)` ❌ 不是期望的 `(760, 760)`
- stage1 (stride=2) → `(95, 95)` ❌ 不是期望的 `(380, 380)`
- stage2 (stride=2) → `(47, 47)` ❌ 不是期望的 `(190, 190)`

## 解决方案

### 1. 创建自定义 ResNet Wrapper

创建了 `/projects/CenterPoint/models/backbones/resnet.py`，继承 mmdet 的 ResNet，重写 `_make_stem_layer` 方法：

**关键修改：**
- `deep_stem=True`: 使用三个 3x3 卷积替代 7x7 卷积（推荐用于 BEV）
  - 计算更高效（3×3×3 < 7×7）
  - 边界行为更可控，避免 odd size 导致的尺寸不匹配
- `conv1_stride=1`: 第一层卷积不使用下采样
- `with_pool=False`: 禁用 maxpool
- `pool_stride`: 仅在 `with_pool=True` 时使用，默认不设置
- 使用 `nn.Identity()` 替代 maxpool，保持代码兼容性

### 2. 配置文件更新

在 `second_resnet50_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py` 中：

```python
pts_backbone=dict(
    _delete_=True,
    type="ResNet",  # 使用自定义的 BEV-friendly ResNet
    depth=50,
    num_stages=4,
    strides=(1, 2, 2, 2),  # stage strides: stage0=1, stage1=2, stage2=2, stage3=2
    out_indices=(0, 1, 2),  # 输出 stage 0, 1, 2
    # BEV-friendly stem 配置：输入不下采样
    deep_stem=True,        # 使用三个 3x3 conv 替代 7x7：更高效且边界行为更好
    conv1_stride=1,        # 第一层卷积 stride=1（不下采样）- 应用于 deep_stem 的第一个 3x3 conv
    with_pool=False,       # 禁用 maxpool（不下采样）
    # pool_stride 仅在 with_pool=True 时使用，此处省略
    frozen_stages=-1,      # 不冻结任何 stage
    base_channels=64,      # ResNet50 输出: 256, 512, 1024 通道
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    norm_eval=False,       # BN 保持训练模式
    style="pytorch",
    in_channels=32,
),
```

### 3. 特征图尺寸变化

**输入**: `(N, 32, 760, 760)` - 来自 PointPillarsScatter

**Stem 层** (BEV-friendly with deep_stem=True):
- 三个 3x3 conv (第一个 stride=1): `(N, 32, 760, 760)` → `(N, 64, 760, 760)` ✅ 尺寸不变
- maxpool (disabled): `(N, 64, 760, 760)` → `(N, 64, 760, 760)` ✅ 尺寸不变

**为什么使用 deep_stem=True？**
- ✅ **计算效率**: 三个 3×3 卷积的计算量小于一个 7×7 卷积
- ✅ **边界行为**: 3×3 卷积的边界处理更可控，避免 odd size 导致的尺寸不匹配（如 190 vs 192）
- ✅ **BEV 友好**: BEV feature map 通常尺寸较大，3×3 卷积更适合

**Stage 0** (stride=1):
- 输入: `(N, 64, 760, 760)`
- 输出: `(N, 256, 760, 760)` ✅ `(H, W)`

**Stage 1** (stride=2):
- 输入: `(N, 256, 760, 760)`
- 输出: `(N, 512, 380, 380)` ✅ `(H/2, W/2)`

**Stage 2** (stride=2):
- 输入: `(N, 512, 380, 380)`
- 输出: `(N, 1024, 190, 190)` ✅ `(H/4, W/4)`

### 4. SECONDFPN Neck 配置

```python
pts_neck=dict(
    type="SECONDFPN",
    in_channels=[256, 512, 1024],  # ResNet50 stage 0, 1, 2 的输出通道
    out_channels=[128, 128, 128],
    # BEV-friendly: 上采样到相同尺寸
    # stage0: (760, 760) -> stride=1 -> (760, 760)
    # stage1: (380, 380) -> stride=2 -> (760, 760)
    # stage2: (190, 190) -> stride=4 -> (760, 760)
    upsample_strides=[1, 2, 4],
    norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
),
```

**上采样后**:
- Deblock 0: `(N, 128, 760, 760)`
- Deblock 1: `(N, 128, 760, 760)`
- Deblock 2: `(N, 128, 760, 760)`
- **拼接后**: `(N, 384, 760, 760)` ✅ 所有特征图尺寸一致

## 验证方法

### 方法 1: 使用验证脚本

运行验证脚本检查输出尺寸：

```bash
cd /home/yihsiangfang/ml_workspace/AWML/projects/CenterPoint
python verify_resnet_outputs.py configs/t4dataset/Centerpoint/second_resnet50_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py
```

### 方法 2: 在代码中打印

在训练或推理代码中添加：

```python
# 在模型 forward 后
outputs = backbone(input_tensor)
for i, out in enumerate(outputs):
    print(f"Stage {i} output shape: {out.shape}")
```

期望输出：
```
Stage 0 output shape: torch.Size([N, 256, 760, 760])
Stage 1 output shape: torch.Size([N, 512, 380, 380])
Stage 2 output shape: torch.Size([N, 1024, 190, 190])
```

### 方法 3: 检查 ONNX 导出

如果导出 ONNX 模型，检查中间节点的输出形状：

```python
import onnx

model = onnx.load("model.onnx")
for node in model.graph.node:
    if 'backbone' in node.name.lower():
        print(f"{node.name}: {node.output[0]}")
```

## 常见问题

### Q1: 为什么会出现尺寸不匹配？

**A**: 如果看到类似 `(760, 380, 192)` 这样的尺寸，说明：
1. stem 层仍然在下采样（检查 `conv1_stride` 和 `with_pool`）
2. 或者输入尺寸本身不是 2 的幂次（如 760）

**解决方法**:
- 确保 `conv1_stride=1` 和 `with_pool=False`
- 检查输入尺寸是否可被 2 整除

### Q2: 上采样后尺寸不完全匹配怎么办？

**A**: 如果上采样后出现 `(760, 760)` vs `(768, 768)` 这样的差异：

1. **检查输入尺寸**: 确保输入是 2 的幂次或可被 2/4 整除
2. **检查 padding**: 确保所有卷积层的 padding 设置正确
3. **使用插值对齐**: 在 SECONDFPN 中添加尺寸对齐步骤

### Q3: ResNet50 比 SECOND 更好吗？

**A**: 取决于应用场景：

| 指标 | ResNet50 | SECOND |
|------|----------|--------|
| **mAP** | 可能更高（需重新调参） | 稳定 |
| **速度** | 更慢 | 更快 |
| **部署/INT8** | 更敏感 | 更稳定 |
| **工程稳定性** | 需要对齐尺寸 | 更容易对齐 |

**建议**:
1. 先确保 ResNet50 的输出尺寸严格对齐
2. 用相同设置跑一小段训练/验证
3. 比较 mAP/mAPH 和 latency（FP16/INT8）
4. 如果目标是部署+INT8，考虑 ResNet18/34 或保留 SECOND

## 文件清单

1. **自定义 ResNet**: `/projects/CenterPoint/models/backbones/resnet.py`
2. **配置文件**: `configs/t4dataset/Centerpoint/second_resnet50_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py`
3. **验证脚本**: `verify_resnet_outputs.py`
4. **本文档**: `BEV_FRIENDLY_RESNET.md`

## 下一步

1. ✅ 创建 BEV-friendly ResNet wrapper
2. ✅ 更新配置文件
3. ⏳ **运行验证脚本确认输出尺寸**
4. ⏳ 如果尺寸正确，开始训练
5. ⏳ 监控训练过程中的 loss 和 mAP
6. ⏳ 评估推理速度和部署性能

## 参考

- [MODEL_DESIGN.md](./MODEL_DESIGN.md) - 详细的模型设计说明
- mmdet ResNet 实现: `/mmdetection/mmdet/models/backbones/resnet.py`
