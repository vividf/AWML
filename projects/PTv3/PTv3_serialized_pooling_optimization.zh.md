# PTv3 序列化池化元数据预计算（中文说明）

> 相关 PR：
> - [tier4/AWML#206](https://github.com/tier4/AWML/pull/206) — ONNX 导出端
> - [autowarefoundation/autoware_universe#12727](https://github.com/autowarefoundation/autoware_universe/pull/12727) — 运行时推理端

## 背景

PTv3（Point Transformer V3）是一个 LiDAR 分割模型，在其编码器中使用 **序列化池化（Serialized Pooling）** 来跨分辨率层级聚合点云特征。每个池化阶段会根据体素（voxel）序列化后的 Morton code 推导出一个「按 stride 位移的 key」，并以此对体素分组，然后为每个组计算统计量：gather 索引、CSR 行指针（row pointers）、cluster 标签、序列化顺序（serialization orders）及其逆映射。

### 原始问题：TensorRT 内部的「数据相关形状（Data-Dependent Shapes）」

在原始实现中，这个分组操作是 **在 TensorRT 图内部** 用 `Unique` 算子完成的。由于 `Unique` 的输出形状是数据相关的（输出大小取决于实际点云内容，而不仅仅是张量维度），TensorRT 无法静态推断下游张量的形状。这迫使 TensorRT 在图的中间插入 **CPU/GPU 同步屏障（synchronization barrier）**，必须先把动态大小读回 CPU 才能继续执行。

结果就是：每一次推理调用都会在 `Unique` 阶段卡住——CPU 等待 GPU 算完，然后才恢复执行。这是一笔高昂且本可避免的开销。

**优化前实测延迟：29.093 ms**

---

## 解决方案：在进入 TensorRT 之前预计算池化元数据

修复方法是把所有池化元数据的「计算/发现」过程，从 TensorRT 图中移出，挪到 **已经在每次推理前运行的 CUDA 预处理阶段**。计算出来的张量作为普通的动态输入（dynamic inputs）喂给 TensorRT。因为对 TensorRT 而言这些是外部提供的数据（而不是图内计算的结果），所有形状在一开始就已知，因此不再需要图中间的同步。

**优化后实测延迟：19.138 ms —— 减少 34%**

---

## 改动概览

这项优化拆分到两个配套 PR，必须 **一起部署**。

### PR 1：AWML#206 —— 重构 ONNX 导出

**目标：** 产出一个「接收预计算元数据作为输入」而不是「内部计算元数据」的 ONNX 模型。

导出的 ONNX 图从：

```
输入: grid_coord, feat, serialized_code
图:   Unique → argsort → segment_csr → pooled features
问题: Unique 的输出形状是数据相关的 → TRT 无法推断静态形状
```

改为：

```
输入: grid_coord, feat, serialized_code,
      serialized_pooling_0_{indices,indptr,cluster,...},
      serialized_pooling_1_{indices,indptr,cluster,...},
      ...  (每个编码器阶段一组)
图:   Gather + autoware::SegmentCSR plugin
结果: 所有形状静态已知 → 不需要 CPU/GPU 同步
```

`point_transformer_v3m1_base.py` 中的关键代码改动：

- 新增 **`SerializedPoolingMeta`** dataclass，保存每个阶段的 7 个元数据张量
- 新增 **`build_serialized_pooling_meta()`** 函数，在导出时用 Python 计算这些张量
- `SerializedPooling` 新增 `export_mode` 标志：导出模式下，它消费预先构建好的 `SerializedPoolingMeta`，而不再运行 `Unique`/`argsort`

`tools/export.py` 中的关键改动：

- 为每个编码器阶段从一个样本帧构建 `SerializedPoolingMeta`
- 把元数据张量注册为带有正确动态轴（dynamic axes）的命名 ONNX 输入
- 为 `indptr` 张量（形状 `[M+1]`）使用一个独立的符号维度 `serialized_pooling_i_out_voxels_plus_one`，以避免维度别名（dim aliasing）冲突

### PR 2：autoware_universe#12727 —— 在 CUDA 预处理中预计算元数据

**目标：** 在运行时，于调用 TensorRT 之前在 GPU 上计算池化元数据，然后把它绑定为引擎输入。

#### CUDA 内核流水线（`preprocess_kernel.cu`）

以下流程针对每个编码器池化阶段，在 `enqueueV3` 之前运行在 GPU stream 上：

| 步骤 | 内核 / API | 用途 |
|------|-------------|---------|
| 1 | `preparePoolingSortInputKernel` | 把 `serialized_code` 右移 `pooling_depth × 3` 位以得到组 key；用 `INT64_MAX` 哨兵值填充越界槽位 |
| 2 | `cub::DeviceRadixSort::SortPairs` | 完全在 GPU 上按组 key 排序体素 |
| 3 | `markPoolingRunsKernel` | 标记每个新组的第一个体素（游程编码 RLE） |
| 4 | `cub::DeviceScan::InclusiveSum` | 对标志位做前缀和，给每个体素分配连续的组 ID |
| 5 | `fillPoolingStageKernel` | 填充 `indices`、`indptr`、`head_indices`、`cluster`、`grid_coord` |
| 6 | `prepareOrderSortInputKernel` + `fillOrderAndInverseKernel` | 为每种序列化顺序（`z`、`z-trans`）计算 `serialized_order` 和 `serialized_inverse` |
| 7 | `cudaMemcpyAsync` + `cudaStreamSynchronize` | 把每个阶段的输出体素数量（几个整数）拷回 CPU —— **唯一的一次 CPU/GPU 同步** |
| 8 | `setSerializedPoolingInputShapes` | 用同步回来的计数在 TRT 引擎上设置实际运行时形状 |
| 9 | `enqueueV3` | TensorRT 推理 —— 所有形状已知，不再有同步 |

步骤 7 中唯一的那次同步只传输标量整数（每阶段一个），相比原来的图中间同步可以忽略不计。

---

## ONNX 输入契约

预计算之后，TensorRT 引擎在每个池化阶段 `i` 会额外收到 7 个输入：

| 张量名 | 形状 | 说明 |
|-------------|-------|-------------|
| `serialized_pooling_{i}_indices` | `[N_in]` | 每个体素的父组索引 |
| `serialized_pooling_{i}_indptr` | `[N_out+1]` | CSR 行指针（每个输出体素一项，再加一） |
| `serialized_pooling_{i}_cluster` | `[N_in]` | 每个体素的 cluster 标签 |
| `serialized_pooling_{i}_head_indices` | `[N_out]` | 每个组的代表（head）体素 |
| `serialized_pooling_{i}_grid_coord` | `[N_out, 4]` | 池化后体素的网格坐标 |
| `serialized_pooling_{i}_serialized_order` | `[N_in, 2]` | 序列化置换（每种顺序一列） |
| `serialized_pooling_{i}_serialized_inverse` | `[N_in, 2]` | 序列化置换的逆映射 |

`N_in` = 进入阶段 `i` 的体素数；`N_out = N_in / pooling_stride`。

`serialized_depth` 被折叠为编译期常量，**不是** 运行时输入。

---

## 必需的配置参数

必须在 `config/ml_package_ptv3.param.yaml` 中设置两个新参数，以匹配模型训练配置：

```yaml
serialization_orders: ["z", "z-trans"]   # 必须与训练配置完全一致
pooling_strides: [2, 2, 2, 2]            # 每个编码器池化阶段一项；必须是正的 2 的幂
```

这两项在启动时会被校验：`serialization_orders` 必须恰好是 `["z", "z-trans"]`，且每个 stride 都必须是正的 2 的幂。

---

## 优化前 vs 优化后

```
优化前
──────
点云
  └─ CUDA 预处理
       └─ TensorRT 图
            ├─ voxelize（体素化）
            ├─ Unique  ← 数据相关形状
            │   └─ CPU/GPU 同步  ← 卡顿
            ├─ argsort / segment_csr
            └─ PTv3 attention 块
延迟: 29.093 ms


优化后
─────
点云
  └─ CUDA 预处理
       ├─ voxelize（体素化）
       ├─ RadixSort + InclusiveSum (GPU)
       ├─ fillPoolingStageKernel (GPU)
       ├─ fillOrderAndInverseKernel (GPU)
       └─ cudaMemcpyAsync (同步: 拷回约 4 个 int)  ← 唯一同步
            └─ TensorRT 图
                 ├─ Gather + SegmentCSR  ← 形状完全静态
                 └─ PTv3 attention 块
延迟: 19.138 ms  (↓ 34%)
```

---

## 部署注意事项

- 这两个 PR 是 **不可拆分的配套（breaking pair）**：AWML#206 改变了 ONNX 输入签名，autoware_universe#12727 在运行时提供这些输入。缺一不可。
- TensorRT 引擎 **必须用 AWML#206 生成的 ONNX 模型重新构建**。
- `autoware_ptv3` 中有一个等价性测试（`serialized_pooling_metadata_test.cpp`），它会针对两个池化阶段，把全部 8 个输出张量与 CPU 参考实现做对比验证。
