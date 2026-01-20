# CenterPoint 模型组件详细设计说明

本文档详细介绍 CenterPoint 3D 目标检测模型中的三个核心组件：**SECOND**、**SECONDFPN** 和 **CenterHead**。

---

## 1. SECOND 骨干网络 (Backbone)

### 1.1 概述

SECOND (Sparsely Embedded Convolutional Detection) 是用于 3D 点云目标检测的骨干网络，主要用于处理经过体素化后的点云特征图。

### 1.2 架构设计

#### 1.2.1 核心结构

SECOND 采用**多阶段卷积结构**，每个阶段包含：
- **下采样卷积层**：用于特征提取和空间维度缩减
- **多个残差卷积块**：用于深层特征提取

#### 1.2.2 关键参数

```python
# 典型配置示例
pts_backbone=dict(
    type="SECOND",
    in_channels=32,              # 输入通道数（来自 PointPillarsScatter）
    out_channels=[64, 128, 256],  # 每个阶段的输出通道数
    layer_nums=[3, 5, 5],         # 每个阶段的卷积层数量
    layer_strides=[1, 2, 2],      # 每个阶段的步长（下采样率）
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    conv_cfg=dict(type="Conv2d", bias=False),
)
```

#### 1.2.3 网络结构详解

**阶段构建过程：**

1. **输入处理**
   - 输入：`(N, C_in, H, W)` 特征图
   - `in_filters = [in_channels, *out_channels[:-1]]`
   - 例如：`in_filters = [32, 64, 128]`（对应三个阶段）

2. **每个阶段的构建**
   ```python
   for i, layer_num in enumerate(layer_nums):
       # 第一阶段：下采样卷积
       block = [
           Conv2d(in_filters[i], out_channels[i], kernel=3, stride=layer_strides[i], padding=1),
           BatchNorm2d(out_channels[i]),
           ReLU(inplace=True)
       ]

       # 后续层：特征提取卷积
       for j in range(layer_num):
           block.append(Conv2d(out_channels[i], out_channels[i], kernel=3, padding=1))
           block.append(BatchNorm2d(out_channels[i]))
           block.append(ReLU(inplace=True))

       blocks.append(nn.Sequential(*block))
   ```

3. **特征图尺寸变化**
   - Stage 0: `(N, 32, H, W)` → `(N, 64, H, W)` (stride=1，尺寸不变)
   - Stage 1: `(N, 64, H, W)` → `(N, 128, H/2, W/2)` (stride=2，下采样)
   - Stage 2: `(N, 128, H/2, W/2)` → `(N, 256, H/4, W/4)` (stride=2，下采样)

#### 1.2.4 前向传播

```python
def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
    """返回多尺度特征图"""
    outs = []
    for i in range(len(self.blocks)):
        x = self.blocks[i](x)
        outs.append(x)  # 保存每个阶段的输出
    return tuple(outs)  # 返回 (feat0, feat1, feat2)
```

**输出：**
- 多尺度特征图元组：`(feat_stage0, feat_stage1, feat_stage2)`
- 每个特征图具有不同的空间分辨率和通道数

#### 1.2.5 扩展功能（AWML 项目）

AWML 项目中的 SECOND 增加了**阶段冻结**功能：

```python
class SECOND(_SECOND):
    def __init__(self, frozen_stages: Optional[List[int]] = None, ...):
        # 可以冻结特定阶段的参数，用于迁移学习
        self._frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """冻结指定阶段的参数"""
        for i in self._frozen_stages:
            for params in self.blocks[i].parameters():
                params.requires_grad = False
```

---

## 2. SECONDFPN 特征金字塔网络 (Neck)

### 2.1 概述

SECONDFPN 是用于融合多尺度特征的颈部网络，将 SECOND 输出的不同尺度特征图上采样到相同尺寸并拼接。

### 2.2 架构设计

#### 2.2.1 核心功能

- **上采样**：将不同尺度的特征图恢复到相同尺寸
- **特征融合**：通过通道拼接融合多尺度特征
- **特征增强**：通过反卷积和归一化增强特征表示

#### 2.2.2 关键参数

```python
pts_neck=dict(
    type="SECONDFPN",
    in_channels=[64, 128, 256],      # 输入通道数（对应 SECOND 的输出）
    out_channels=[128, 128, 128],      # 输出通道数（统一通道数）
    upsample_strides=[1, 2, 4],       # 上采样步长
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),  # 使用反卷积上采样
    use_conv_for_no_stride=True,      # stride=1 时使用卷积而非反卷积
)
```

#### 2.2.3 网络结构详解

**上采样块 (Deblock) 构建：**

```python
deblocks = []
for i, out_channel in enumerate(out_channels):
    stride = upsample_strides[i]

    if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
        # 使用反卷积上采样
        upsample_layer = ConvTranspose2d(
            in_channels[i],
            out_channel,
            kernel_size=stride,
            stride=stride
        )
    else:
        # stride < 1 时使用普通卷积
        stride = round(1 / stride)
        upsample_layer = Conv2d(
            in_channels[i],
            out_channel,
            kernel_size=stride,
            stride=stride
        )

    # 每个 deblock 包含：上采样层 + BN + ReLU
    deblock = Sequential(
        upsample_layer,
        BatchNorm2d(out_channel),
        ReLU(inplace=True)
    )
    deblocks.append(deblock)
```

**特征图尺寸变化示例：**

假设输入特征图尺寸：
- Stage 0: `(N, 64, H, W)` → 上采样 stride=1 → `(N, 128, H, W)`
- Stage 1: `(N, 128, H/2, W/2)` → 上采样 stride=2 → `(N, 128, H, W)`
- Stage 2: `(N, 256, H/4, W/4)` → 上采样 stride=4 → `(N, 128, H, W)`

#### 2.2.4 前向传播

```python
def forward(self, x):
    """
    Args:
        x: List[Tensor] - 多尺度特征图列表
           [(N, 64, H, W), (N, 128, H/2, W/2), (N, 256, H/4, W/4)]

    Returns:
        [out]: List[Tensor] - 融合后的特征图
               [(N, 384, H, W)]  # 128+128+128=384
    """
    ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

    if len(ups) > 1:
        out = torch.cat(ups, dim=1)  # 通道维度拼接
    else:
        out = ups[0]

    return [out]
```

**输出特征：**
- 形状：`(N, sum(out_channels), H, W)`
- 例如：`(N, 384, H, W)` = `(N, 128+128+128, H, W)`
- 融合了多尺度信息，具有丰富的语义和细节特征

#### 2.2.5 扩展功能（AWML 项目）

同样支持阶段冻结：

```python
class SECONDFPN(_SECONDFPN):
    def __init__(self, frozen_stages: Optional[List[int]] = None, ...):
        self._frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """冻结指定 deblock 的参数"""
        for i in self._frozen_stages:
            for params in self.deblocks[i].parameters():
                params.requires_grad = False
```

---

## 3. CenterHead 检测头 (Head)

### 3.1 概述

CenterHead 是 CenterPoint 的核心检测头，采用**中心点检测**的方式预测 3D 边界框。它将目标检测问题分解为：
- **热力图预测**：预测目标中心位置和类别
- **回归预测**：预测边界框的尺寸、高度、旋转、速度等属性

### 3.2 架构设计

#### 3.2.1 核心组件

1. **共享卷积层 (Shared Conv)**：提取共享特征
2. **任务头 (Task Heads)**：每个任务（类别组）有独立的检测头
3. **分离头 (Separate Head)**：每个任务头包含多个子头（heatmap, reg, height, dim, rot, vel）

#### 3.2.2 关键参数

```python
pts_bbox_head=dict(
    type="CenterHead",
    in_channels=384,  # 输入通道数（SECONDFPN 的输出通道总和）
    tasks=[
        dict(num_class=5, class_names=["car", "truck", "bus", "bicycle", "pedestrian"]),
    ],
    bbox_coder=dict(
        voxel_size=[0.2, 0.2, 8],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        out_size_factor=1,
    ),
    separate_head=dict(
        type="CustomSeparateHead",
        init_bias=-4.595,  # 热力图初始偏置（sigmoid(-4.595) ≈ 0.01）
        final_kernel=1
    ),
    loss_cls=dict(type="mmdet.AmpGaussianFocalLoss", reduction="none", loss_weight=1.0),
    loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
    share_conv_channel=64,      # 共享卷积输出通道数
    num_heatmap_convs=2,       # 热力图卷积层数
    norm_bbox=True,            # 是否对 bbox 尺寸进行 log 归一化
)
```

#### 3.2.3 网络结构详解

**1. 共享卷积层**

```python
self.shared_conv = ConvModule(
    in_channels,           # 384 (来自 SECONDFPN)
    share_conv_channel,   # 64
    kernel_size=3,
    padding=1,
    conv_cfg=conv_cfg,
    norm_cfg=norm_cfg,
    bias=bias
)
```

**2. 任务头构建**

```python
self.task_heads = nn.ModuleList()

for num_cls in num_classes:  # 每个任务
    heads = copy.deepcopy(common_heads)  # 默认：reg, height, dim, rot, vel
    heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))  # 添加热力图头

    # 构建分离头
    separate_head.update(
        in_channels=share_conv_channel,  # 64
        heads=heads,
        num_cls=num_cls
    )
    self.task_heads.append(MODELS.build(separate_head))
```

**3. SeparateHead 结构**

每个 SeparateHead 包含多个子头：

```python
class SeparateHead:
    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1):
        """
        heads = {
            'heatmap': (num_classes, 2),      # 类别数, 卷积层数
            'reg': (2, 2),                    # 2D偏移, 卷积层数
            'height': (1, 2),                 # 高度, 卷积层数
            'dim': (3, 2),                    # 尺寸(l,w,h), 卷积层数
            'rot': (2, 2),                    # 旋转(sin,cos), 卷积层数
            'vel': (2, 2),                    # 速度(vx,vy), 卷积层数
        }
        """
        for head_name, (classes, num_conv) in heads.items():
            conv_layers = []
            c_in = in_channels  # 64

            # 中间卷积层
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(c_in, head_conv, kernel_size=final_kernel, ...)
                )
                c_in = head_conv

            # 最终输出层
            conv_layers.append(
                Conv2d(head_conv, classes, kernel_size=final_kernel, bias=True)
            )

            self.__setattr__(head_name, nn.Sequential(*conv_layers))
```

**4. 前向传播**

```python
def forward_single(self, x: Tensor) -> dict:
    """
    Args:
        x: (N, 384, H, W) - SECONDFPN 输出的特征图

    Returns:
        ret_dicts: List[dict] - 每个任务的预测结果
    """
    x = self.shared_conv(x)  # (N, 64, H, W)

    ret_dicts = []
    for task in self.task_heads:
        ret_dict = task(x)  # 返回包含 heatmap, reg, height, dim, rot, vel 的字典
        ret_dicts.append(ret_dict)

    return ret_dicts
```

**输出格式：**

```python
ret_dict = {
    'heatmap': (N, num_classes, H, W),    # 热力图：每个类别的中心点概率
    'reg': (N, 2, H, W),                   # 2D偏移：中心点的亚像素偏移
    'height': (N, 1, H, W),                # 高度：目标中心点的z坐标
    'dim': (N, 3, H, W),                   # 尺寸：目标的长宽高（log归一化）
    'rot': (N, 2, H, W),                   # 旋转：sin(θ) 和 cos(θ)
    'vel': (N, 2, H, W),                   # 速度：vx, vy（可选）
}
```

#### 3.2.4 训练目标生成

**热力图生成：**

```python
def get_targets_single(self, gt_instances_3d):
    """
    为每个目标生成高斯热力图
    """
    heatmap = zeros(num_classes, H, W)

    for obj in gt_objects:
        # 计算高斯半径
        radius = gaussian_radius((width, length), min_overlap=0.7)
        radius = max(min_radius, int(radius))

        # 计算中心点坐标
        center = (x, y)  # 在特征图坐标系中

        # 绘制高斯热力图
        draw_gaussian(heatmap[cls_id], center, radius)

    return heatmap
```

**回归目标：**

```python
anno_box = [
    center_offset_x,      # 中心点亚像素偏移
    center_offset_y,
    z,                    # 高度
    log(length),          # 长度（log归一化）
    log(width),           # 宽度（log归一化）
    log(height),          # 高度（log归一化）
    sin(rotation),        # 旋转角正弦值
    cos(rotation),        # 旋转角余弦值
    vx,                   # x方向速度
    vy,                   # y方向速度
]
```

#### 3.2.5 损失函数

**1. 分类损失（热力图）**

```python
# Gaussian Focal Loss
loss_heatmap = GaussianFocalLoss(
    pred_heatmap,      # (N, num_classes, H, W)
    gt_heatmap,        # (N, num_classes, H, W)
    avg_factor=num_pos
)
```

**2. 回归损失（边界框）**

```python
# L1 Loss（带权重）
loss_bbox = L1Loss(
    pred_anno_box,     # (N, max_objs, 10)
    gt_anno_box,       # (N, max_objs, 10)
    weight=bbox_weights,  # 每个回归项的权重
    avg_factor=num_pos
)

# code_weights 示例：[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
# 对应：[offset_x, offset_y, z, log(l), log(w), log(h), sin(θ), cos(θ), vx, vy]
```

#### 3.2.6 推理过程

**1. 热力图解码**

```python
# 从热力图中提取峰值点
batch_heatmap = preds_dict[0]['heatmap'].sigmoid()  # (N, num_classes, H, W)

# 使用 NMS 提取 top-k 中心点
topk_scores, topk_inds, topk_clses, topk_ys, topk_xs = get_topk_from_heatmap(
    batch_heatmap,
    k=max_objs
)
```

**2. 边界框解码**

```python
# 收集回归预测
pred_reg = gather_feat(preds_dict[0]['reg'], topk_inds)
pred_height = gather_feat(preds_dict[0]['height'], topk_inds)
pred_dim = exp(gather_feat(preds_dict[0]['dim'], topk_inds))  # 反log归一化
pred_rot = gather_feat(preds_dict[0]['rot'], topk_inds)

# 解码为 3D 边界框
bboxes_3d = bbox_coder.decode(
    centers=(topk_xs, topk_ys),
    heights=pred_height,
    dims=pred_dim,
    rotations=pred_rot,
    regs=pred_reg,
    ...
)
```

**3. 后处理**

```python
# Circle NMS（圆形非极大值抑制）
keep = circle_nms(
    boxes_bev,              # (N, 4) - [x, y, score, label]
    min_radius=task_radius,
    post_max_size=500
)

# 或 Rotate NMS（旋转框非极大值抑制）
keep = nms_bev(
    boxes_bev_xyxyr,        # (N, 5) - [x1, y1, x2, y2, r]
    scores,
    nms_thr=0.1,
    pre_max_size=1000,
    post_max_size=500
)
```

#### 3.2.7 扩展功能（AWML 项目）

**1. 类别级损失**

```python
class CenterHead(_CenterHead):
    def loss_by_feat(self, ...):
        if self._class_wise_loss:
            # 计算每个类别的独立损失
            loss_heatmap_cls = self.loss_cls(...)
            loss_heatmap_cls = loss_heatmap_cls.sum((0, 2, 3)) / max(num_pos, 1)

            for cls_i, class_name in enumerate(class_names):
                loss_dict[f"task{task_id}.loss_heatmap_{class_name}"] = loss_heatmap_cls[cls_i]
```

**2. 参数冻结**

```python
def _freeze_parameters(self):
    """冻结共享卷积或任务头"""
    if self.freeze_shared_conv:
        for params in self.shared_conv.parameters():
            params.requires_grad = False

    if self.freeze_task_heads:
        for task in self.task_heads:
            for params in task.parameters():
                params.requires_grad = False
```

---

## 4. 整体数据流

### 4.1 前向传播流程

```
点云数据 (N, P, 4)
    ↓
体素化 (Voxelization)
    ↓
PillarFeatureNet (N, 32, H, W)
    ↓
PointPillarsScatter (N, 32, H, W)
    ↓
SECOND Backbone
    ├─ Stage 0: (N, 64, H, W)
    ├─ Stage 1: (N, 128, H/2, W/2)
    └─ Stage 2: (N, 256, H/4, W/4)
    ↓
SECONDFPN Neck
    ├─ Deblock 0: (N, 128, H, W)
    ├─ Deblock 1: (N, 128, H, W)
    └─ Deblock 2: (N, 128, H, W)
    ↓
Concat: (N, 384, H, W)
    ↓
CenterHead
    ├─ Shared Conv: (N, 64, H, W)
    └─ Task Heads
        ├─ Heatmap: (N, num_classes, H, W)
        ├─ Reg: (N, 2, H, W)
        ├─ Height: (N, 1, H, W)
        ├─ Dim: (N, 3, H, W)
        ├─ Rot: (N, 2, H, W)
        └─ Vel: (N, 2, H, W)
    ↓
解码 + NMS → 3D 边界框
```

### 4.2 训练流程

```
1. 前向传播获取预测结果
2. 生成训练目标（热力图 + 回归目标）
3. 计算损失：
   - loss_heatmap = GaussianFocalLoss(pred_heatmap, gt_heatmap)
   - loss_bbox = L1Loss(pred_bbox, gt_bbox, weights)
4. 反向传播更新参数
```

### 4.3 推理流程

```
1. 前向传播获取预测结果
2. 从热力图中提取峰值点（top-k）
3. 收集对应的回归预测
4. 解码为 3D 边界框
5. 应用 NMS 去除重复检测
6. 返回最终检测结果
```

---

## 5. 关键设计特点

### 5.1 SECOND

- ✅ **多尺度特征提取**：通过不同 stride 的下采样获得多尺度特征
- ✅ **深层网络**：每个阶段包含多个卷积层，提取深层语义特征
- ✅ **参数可配置**：灵活配置通道数、层数、步长

### 5.2 SECONDFPN

- ✅ **特征融合**：将多尺度特征融合到同一尺寸
- ✅ **上采样策略**：使用反卷积进行上采样，保留空间信息
- ✅ **统一通道数**：将所有特征图统一到相同通道数便于融合

### 5.3 CenterHead

- ✅ **中心点检测**：无需 anchor，直接预测目标中心
- ✅ **多任务学习**：同时预测分类和回归任务
- ✅ **高斯热力图**：使用高斯分布标注目标中心，更符合检测特性
- ✅ **分离头设计**：不同任务使用独立的预测头，避免任务冲突

---

## 6. 配置示例

完整配置示例请参考：
- `/home/yihsiangfang/ml_workspace/AWML/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py`

---

## 7. 参考文献

- **SECOND**: Yan, Y., et al. "Second: Sparsely embedded convolutional detection." Sensors 18.10 (2018): 3337.
- **CenterPoint**: Yin, T., et al. "Center-based 3d object detection and tracking." CVPR 2021.
