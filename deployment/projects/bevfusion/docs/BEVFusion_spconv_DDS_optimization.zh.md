# BEVFusion 稀疏编码器 —— 数据相关形状（trainStation）优化（中文说明）

> Profiling 来源：`/home/yihsiangfang/bevfusion_2_7/bevfusion_profile_kambe.nsys-rep`
> 参考优化（同一类问题，已在 PTv3 上解决）：
> - [tier4/AWML#206](https://github.com/tier4/AWML/pull/206) — ONNX 导出端
> - [autowarefoundation/autoware_universe#12727](https://github.com/autowarefoundation/autoware_universe/pull/12727) — 运行时端

## 1. 问题陈述

在 BEVFusion 的 TensorRT 引擎中，稀疏中段编码器（spconv）在 Nsight Systems 时间线上的各个 sparse-conv 块之间，反复出现 `[trainStationN]` 标记。这些标记是由 **数据相关形状（DDS, Data-Dependent Shapes）** 强制产生的 TensorRT **执行段边界（execution-segment boundaries）**：下采样稀疏卷积之后的有效输出位点（active output sites）数量，在 GPU 算出来之前是未知的，因此 TensorRT 必须先把这个形状拷回主机（`DeviceToShapeHostCopy`），才能配置并启动下一段。每一个这样的边界都会打断流水线，让 GPU 空转。

## 2. Profiling 证据（61 次推理，按每次推理取平均）

| 指标 | 数值 |
|--------|-------|
| 单次推理（`ExecutionContext::enqueue`） | **34.53 ms** |
| GPU 忙碌（内核 + memcpy） | 24.26 ms |
| **GPU 空闲（气泡 bubbles）** | **10.27 ms = 29.7%** |
| 全部 6 个 `trainStation` 段 | 3.11 ms/次 |
| `trainStation2`（最大的一个） | 1.77 ms/次 |
| `GetIndicePairsImplicitGemm`（rulebook 构建） | 6.32 ms/次 |
| `DeviceToShapeHostCopy` 同步点 | **恰好 4 个** |

### 2.1 四个 DDS 同步点

`DeviceToShapeHostCopy` 恰好出现在 **四个 stride-2 下采样层**（且仅此而已）：

| 层（下采样） | 形状拷贝耗时 | 紧随其后的 GPU 空闲 |
|--------------------|--------------------:|---------------------------:|
| `encoder_layer1.2` → stage2 | 0.248 ms | 0.102 ms |
| `encoder_layer2.2` → stage3 | 0.301 ms | 0.082 ms |
| `encoder_layer3.2` → stage4 | 0.280 ms | 0.077 ms |
| `conv_out` | 0.329 ms | 0.073 ms |

**子流形卷积（Submanifold convolutions）**（`conv1`/`conv2`，它们保持有效位点集合不变）**不产生** 任何形状拷贝——这证实了 DDS 开销专属于那些会改变有效体素数量的层。

### 2.2 一个 trainStation 实际包含什么

一个 `[trainStationN]` NVTX 区间 **并不是纯粹的停顿**——它包裹了一段真实的引擎工作。在一个 `trainStation2` 窗口内（1.977 ms）：有 7 个 GPU 内核，约 1.54 ms GPU 忙碌，约 0.44 ms GPU 空闲。trainStation 是 **两个 DDS 边界之间的那一段图**；它的代价来自跨段流水线的丢失，加上边界处的主机同步，而不是段内的工作本身。

## 3. 根因及与 PTv3 的类比

| | PTv3（已优化） | BEVFusion spconv（本报告） |
|---|--------------------------|--------------------------------|
| DDS 来源 | `Unique`（池化分组） | `GetIndicePairsImplicitGemm`（rulebook + 输出坐标计数） |
| 形状为何动态 | 池化后的体素数是数据相关的 | 下采样后的有效位点数是数据相关的 |
| 图内症状 | CPU/GPU 同步屏障 | `DeviceToShapeHostCopy` + trainStation 分段 |
| 修复方法 | 在 CUDA 预处理中预计算池化元数据，作为静态输入喂入 | 在 CUDA 预处理中预计算 rulebook / 输出坐标，作为静态输入喂入 |

**让修复成为可能的关键事实：** spconv 的 **rulebook（index pairs，索引对）和每阶段输出坐标，只取决于输入体素几何（哪些格被占用），不取决于特征值。** 这正是 spconv 把 `GetIndicePairs`（几何）和 GEMM（特征）分开的原因。体素坐标在预处理的体素化之后就已知——所以每一层的有效坐标和 rulebook 的整个级联，都可以提前计算出来，并作为「形状可解析」的输入传入，从而消除图内的 DDS。

## 4. 建议的优化方案（两部分，对应 PTv3 的两个 PR）

### 4.1 运行时 / 预处理端（对应 autoware_universe#12727）

1. 体素化之后，在 GPU 上对下采样级联做一次 **仅坐标（coordinate-only）的前向传播**，为每个 sparse-conv 层导出：
   - 输出有效坐标，
   - 索引对（rulebook），
   - 每阶段有效位点计数。
2. 做 **一次** `cudaMemcpyAsync` + 单次同步，把每阶段计数取回（替代原来的 4 次图中间同步），并据此设置引擎的动态输入形状。
3. 把预计算好的 rulebook/坐标绑定到引擎输入张量。

### 4.2 ONNX 导出端（对应 AWML#206）

- 把 `GetIndicePairsImplicitGemm` 节点替换为「消费预计算 rulebook 输入」的 plugin 节点，而不是在图内计算。
- 把 rulebook/坐标张量添加为带动态轴的命名图输入；有效位点计数变成一个由预处理解析的符号维度。

### 4.3 预期收益

- 移除 4 个 `DeviceToShapeHostCopy` 同步，并合并 6 个 trainStation 段。
- 使整个稀疏编码器能被捕获为 **单个 CUDA Graph**，在消除同步气泡之外，还能消除逐内核启动的开销。
- 并非全部 10.27 ms 的空闲都可回收（部分是启动延迟 / 重排格式 reformatting），但可归因于 DDS 和分段的那部分是可观的。作为量级参考，PTv3 的同类改动带来了端到端 34% 的延迟下降（29 ms → 19 ms）。

## 5. 更轻量的替代方案（不改 ONNX 导出）

修改 spconv plugin，使其 **永不把计数拷回主机**：声明一个静态的 **上界（upper-bound）** 输出形状（最大有效位点数），并用掩码/填充（masking/padding）使下游层始终以该上界运行。这样在不改动导出图输入签名的前提下，移除 `DeviceToShapeHostCopy` 并合并 trainStation。代价：部分层会对填充部分做计算（在最大尺寸上浪费算力）。比完整预计算更容易落地，但收益回收得不那么干净。

## 6. 改动点评估（spconv_cpp + plugin）

### 6.1 DDS / D2H 实际存在于哪里

| 关注点 | 文件 | 位置 |
|---------|------|----------|
| TRT plugin (IPluginV3) | `autoware.universe/.../autoware_tensorrt_plugins/src/get_indices_pairs_implicit_gemm_plugin.cpp` | class + `enqueue()` @288 |
| **DDS 形状声明** | 同上 | `getOutputShapes()` @186–244 → 对下采样调用 `declareSizeTensor(4, min, max)` |
| **num_act_out → 写回设备**（紧跟主机读取之后的 H2D） | 同上 | @439–445 `cudaMemcpyAsync(..., HostToDevice)` |
| **D2H 计数读取（thrust 路径）** | `spconv_cpp/.../SpconvOps_apply_thrust_unique_to_indice_pairs_uniq.cu` | @25–38（`thrust::unique`，返回 `int` 到主机） |
| **D2H 计数读取（hash 路径）** | `spconv_cpp/.../SparseConvIndicesKernel_unique_hash.cu` | @14–36（`uniq_cnt.cpu(tvctx)`） |
| Rulebook 构建入口 | `spconv_cpp/.../SpconvOps.h` | `get_indice_pairs_implicit_gemm()` @544 |
| 子流形（无 DDS） | `spconv_cpp/.../SparseConvIndicesKernel_generate_subm_conv_inds.cu` | 输出计数 == 输入计数 |
| 下采样阶段（有 DDS） | `generate_conv_inds_stage1/1_5/stage2` | unique/sort 步骤就是数据相关点 |

所以 `[trainStationN]` 只由一种机制产生：`getOutputShapes()` 针对 4 个下采样层调用 `declareSizeTensor()`，而其值由 `enqueue()` 内部的 unique/sort D2H 读取产生。

### 6.2 已经存在的关键使能条件

`SpconvOps::get_indice_pairs_implicit_gemm()` **已经接受一个 `preallocated` map**，会复用调用方提供的 rulebook 张量而不重新计算：
`"PairFwd"`、`"IndiceNumPerLoc"`、`"HashKOrKV"`、`"PairMask"`
（`SpconvOps_get_indice_pairs_implicit_gemm.cc` @63–76, 127–136）。plugin 的 `enqueue()` 目前从不填充这个 map——所以预计算路径在库里 **已经盖好了一半**；缺的那一块是把它接通到 plugin 的 I/O 和导出图。

### 6.3 候选实现路线

- **路线 A —— 完整预计算（对应 §4，PTv3 风格）。** 在预处理中对 4 个下采样阶段做一次仅坐标的前向传播 → 预计算 rulebook + 每阶段计数；作为引擎输入喂入；plugin 的 `getOutputShapes()` 从输入维度推导输出维度（不再 `declareSizeTensor`）；`enqueue()` 消费 `preallocated`。一次预处理同步替代 4 次图内同步。结果最干净；改动最大（plugin I/O + 导出图 + 运行时预处理）。由于是顺序的几何级联（阶段 N+1 需要阶段 N 的坐标），预处理过程比 PTv3 更复杂。
- **路线 B —— 静态上界形状（更轻量）。** `getOutputShapes()` 返回一个每阶段的常量上界，而不是 `declareSizeTensor`；内核对该上界做填充/掩码；去掉 D2H/H2D。不改导出输入签名。能移除 trainStation，但在填充上浪费算力（有效位点随下采样减少，所以一个固定上界代价高）。
- **路线 C —— 把稀疏编码器移出 TensorRT。** 像 NVIDIA CUDA-BEVFusion 那样，在引擎之外用原生 libspconv 跑 spconv 主干，再把稠密输出喂回 TRT 引擎。因为稀疏部分没有 TRT 图，所以没有 trainStation。架构改动最大；让 spconv 与 TRT 彻底解耦。

### 6.4 建议

**路线 A** 是 PTv3 两个 PR 的忠实对应，给出最干净、完全静态的引擎，而且库已经支持预分配 rulebook——但它需要同时改动 plugin I/O、ONNX 导出和运行时预处理（一个不可拆分的配套，像 AWML#206 + autoware_universe#12727）。路线 B 是一个不错的渐进式第一步，可以在投入完整的导出/运行时契约变更之前，先单独验证 trainStation 的移除。

## 7. 路线 A —— 详细实现计划（逐文件）

### 7.0 架构决策（重要的简化）

DDS **并非源自** `ImplicitGemm` 卷积 plugin——那个 plugin 已经从一个 *输入* 维度推导其输出尺度：
`implicit_gemm_plugin.cpp:269–286` → `outputs[0].d[0] = inputs[3].d[0]`（pair_mask 的 dim0），`outputs[0].d[1] = inputs[1].d[0]`（C_out）。DDS **仅由** `GetIndicePairsImplicitGemm::getOutputShapes()` 调用 `declareSizeTensor(4, …)` 产生（`get_indices_pairs_implicit_gemm_plugin.cpp:217–238`），随后通过 pair 张量传播到下游每一层。

**推论：** 如果 rulebook（`pair_fwd`、`pair_mask_fwd`、`mask_argsort_fwd`、`out_indices`、`num_act_out`）成为真正的 **图输入**（形状在 `enqueueV3` 之前由 `setInputShape` 解析），则 size tensor 消失，`ImplicitGemm` **无需改动**——它从输入推导的输出形状已经正确。所以路线 A = *把 GetIndicePairs 节点从图中移除，把它们的输出暴露为图输入，并在预处理中预计算它们。* 这与 PTv3 完全一致（预计算 → 图输入 → 绑定），而 `GetIndicePairs` plugin 只是不再在图中实例化（保留在注册表中以向后兼容）。

### 7.1 需要预计算的层结构（来自 AWML BEVFusion 配置）

`pts_middle_encoder`（`BEVFusionSparseEncoder`），`sparse_shape=[1440,1440,41]`，除注明外 kernel=3：

| 阶段 | 层 | 类型 | Stride | 改变坐标? |
|-------|--------|------|--------|-----------------|
| conv_input | 1 | SubMConv3d | 1 | 否 |
| encoder_layer1 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | 仅下采样 |
| encoder_layer2 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | 仅下采样 |
| encoder_layer3 | subm,subm + downsample | SubM×2 + SparseConv3d | (2,2,1) | 仅下采样 |
| encoder_layer4 | subm,subm | SubM×2 | 1 | 否 |
| conv_out | 1 | SparseConv3d k=(1,1,3) | (1,1,2) | 仅下采样 |

今天只有 4 个 stride>1 的层携带 DDS（与 §2.1 的 4 个 `DeviceToShapeHostCopy` 吻合）。子流形层复用前一阶段的坐标（它们的 rulebook 是基于同一坐标的几何）。

### 7.2 运行时端（autoware_bevfusion）—— 对应 autoware_universe#12727

- **新模块** `lib/preprocess/sparse_rulebook_precompute.{hpp,cu}`：
  - 输入：由体素化已产出的体素坐标（`coors`，[A,4]）（`bevfusion_trt.cpp:initPtr/preProcess`）。
  - 对每个稀疏层按声明顺序，调用 `SpconvOps::get_indice_pairs_implicit_gemm(...)`（与 plugin 使用的同一调用，`get_indices_pairs_implicit_gemm_plugin.cpp:392/432`）写入稳定的预分配设备缓冲；把每个下采样层的 `out_indices` 串接为下一层的输入坐标。
  - 收集每一层的 `num_act_out`；对所有计数做 **一次** `cudaMemcpyAsync`+`cudaStreamSynchronize`（替代 4 次图内同步）。
- **`lib/bevfusion_trt.cpp`**：
  - `initTrt()`（约 L141–187）：在优化 profile 中以 `[min,opt,max]`（max = `out_indices_num_limit_`）注册新的 rulebook 图输入，像 PTv3 的每阶段输入那样。
  - 新增 `bindSerializedRulebookAddresses()`：为每个 rulebook 缓冲调用 `setTensorAddress()`。
  - `preProcess()`：调用预计算模块，然后在 `enqueueV3()` 之前用同步回来的计数对每个 rulebook 输入做 `setInputShape()`。
- **配置/schema**（`config/bevfusion_lidar.param.yaml`、`schema/*.json`）：添加稀疏编码器层描述列表（每层 ksize/stride/padding/subm），让预处理知道级联结构——对应 PTv3 的 `pooling_strides`。

### 7.3 导出端（AWML）—— 对应 AWML#206

- **`projects/SparseConvolution/sparse_functional.py`**（`GetIndicePairsImplicitGemm.symbolic` @243–292）：添加一个 `export_precomputed` 路径，不再发射 `autoware::GetIndicePairsImplicitGemm` 算子，而是从 **命名图输入** 返回 5 个张量（`rulebook_{i}_out_indices`、`_pair_fwd`、`_pair_mask`、`_mask_argsort`、`_num_act_out`）。
- **`projects/BEVFusion/bevfusion/sparse_encoder.py`**（`forward` @147+）：导出模式下，从注入的输入拉取每层 rulebook，而不是计算它；`ImplicitGemm` 调用保持不变（它们已经把 pair 张量作为参数）。
- **`projects/BEVFusion/deploy/exporter.py`**（`_export_main_body` @187+，`torch.onnx.export` @173，`_fix_onnx_graph`）：声明新输入 + 动态轴；为类似 `indptr` 的 `[N+1]`/`num_act_out` 张量使用独立的符号维度；确保图中不再残留 `GetIndicePairs` 节点。
- 在 BEVFusion 导出 README 中记录新的输入契约（对应 AWML#206 的 README 章节）。

### 7.4 Plugin 端

- `ImplicitGemmPlugin`：**无需改动**（输入推导的形状已正确）。
- `GetIndicesPairsImplicitGemmPlugin`：路线 A 不需改动（节点已从图中移除）。`out_indices_num_limit_ = 256000` 上界成为 rulebook 输入的 profile 最大值。

### 7.5 验证

- 移植 spconv 等价性测试的思路（PTv3 的 `serialized_pooling_metadata_test.cpp`）：一个 gtest，运行预处理 rulebook 级联，并检查它与图内 `SpconvOps::get_indice_pairs_implicit_gemm` 的输出在某个固定点云上逐字节匹配。
- 端到端：重建 ONNX（新契约）→ 重建引擎 → 在新的 nsys 抓取中确认 `[trainStation]` 标记和 4 个 `DeviceToShapeHostCopy` 已消失，且检测输出不变。

### 7.6 构建 / 测试约束

这是一个不可拆分的配套（导出 + 运行时），需要 CUDA 构建、TensorRT 引擎重建、ONNX 重新导出，以及一个数据集来验证——这些都无法在本分析沙箱中运行。实现必须逐切片（slice-by-slice）落地，并在目标机器上跑构建/测试循环。先做导出契约（像 AWML#206），再做运行时，最后重新 profile。

## 8. 实现状态

### 切片 1 —— 导出图手术 ✅ 已实现并通过 ONNX 验证

`AWML/deployment/projects/bevfusion/export/sparse_trainstation_transform.py`（`remove_trainstation_dds`）。删除 4 个下采样 `GetIndicePairsImplicitGemm` 节点，并把它们被消费的输出（`out[0..3]`；`out[4]` num_act_out 无消费者，丢弃）提升为图输入，每阶段共享一个符号维度。

在 `awml-bevfusion` 容器中针对基线 `bevfusion_sparse.onnx` 验证：
- 21 → **17** 个 `GetIndicePairsImplicitGemm`（剩余的全是 `subm=1`，即无 `declareSizeTensor`、无 DDS）。
- `ImplicitGemm` 不变（21 个）；其 12 条输入边现在来自新图输入（4 阶段 × pair_fwd/pair_mask/mask_argsort）。
- 图输入 3 → **19**（+16：4 阶段 × out_indices/pair_fwd/pair_mask/mask_argsort）。
- `onnx.checker` OK；`shape_inference(strict_mode=True)` OK → 图端到端一致。

每阶段新输入名（`l1/l2/l3/out`），INT32：
`…GetIndicePairsImplicitGemm_output_{0,1,2,3}`，形状为
`[N,4] / [KV,N] / [N,1] / [N]`（KV=27 用于 l1–l3，3 用于 conv_out）。

### 构建 / 测试工作流（本机）

- 容器 `awml-bevfusion`（`awml-bevfusion:full`）；宿主机 `AWML` 挂载在 `/workspace`，所以 AWML 的编辑实时生效。Plugin `.so` 在 `/opt/plugins/libautoware_tensorrt_plugins.so`（从 fork `vividf/autoware.universe@feat/implicit_gemm_int8` 预构建；通过 `projects/BEVFusion/plugins/build_plugin_inside_container.sh` 重建）。
- 导出/构建 CLI：
  `python -m deployment.cli.main bevfusion <deploy_cfg> <model_cfg>`
  deploy cfg：`deployment/projects/bevfusion/config/deploy_config_split_fp16_opt_trainstation.py`。
- 稀疏 ONNX = 仅 `pts_middle_encoder`；从稠密部分拆出。基线图：21 GetIndicePairs + 21 ImplicitGemm；输入 voxels/coors/num_points_per_voxel；输出 lidar_bev。

### 切片 1c —— 引擎构建 + trainStation 移除 ✅ 已证明

在容器中从基线 ONNX 与手术修改后的 ONNX 分别构建 FP16 稀疏引擎（plugin `/opt/plugins/libautoware_tensorrt_plugins.so`），然后导出 TensorRT **engine-inspector** 的层信息（无需 nsys 的结构性证明——`trainStation` 是 TRT 内部 Myelin 区域名，会原样出现在引擎层列表中）：

| 引擎 | 总层数 | `trainStation` 层数 |
|--------|-------------:|----------------------:|
| 基线（`bevfusion_sparse.onnx`） | 135 | **6**（`[trainStation1]`…`[trainStation6]`） |
| 修改后（`bevfusion_sparse_nots.onnx`） | 125 | **0** |

基线的 6 个 trainStation 与车载 nsys profile（§2）中看到的 6 个吻合。移除 4 个下采样 `GetIndicePairsImplicitGemm` 节点消除了 **全部** trainStation。两个引擎都干净构建（全部 21+21 个 plugin 实例化；19 个输入，优化 profile 一致——注意 voxels/coors/num_points_per_voxel 共享 dim_param `voxels_num`，所以它们的 profile 必须完全相同）。

> 临时脚手架：`AWML/_ts_tmp/{build_sparse_engine.py,inspect_engine.py}`。

### 切片 1d —— 数值等价性 ✅ 已证明

`AWML/_ts_tmp/validate_equiv.py`：给两个引擎喂同样的合成稀疏输入（4 万个随机体素）。修改后的引擎额外接收 4 个下采样 rulebook，这些通过 `sparse_functional.GetIndicePairsImplicitGemm.apply` 在 4 个下采样阶段级联预计算（即切片 2 CUDA 运行时的精确参考逻辑）。结果：

```
基线 (1,256,180,180)  vs  修改后 (1,256,180,180)
max abs diff = 0.0088   mean = 0.00014   relative max = 0.0034   -> 匹配（fp16 级别）
```

证实：(1) Python 预计算与基线图内计算的结果一致；(2) 修改后的引擎正确消费外部 rulebook；(3) 图手术保持了语义。预计算级联（喂入 conv 坐标 → 每个下采样阶段调用 `get_indice_pairs_implicit_gemm`，向前串接 `out_indices`；坐标归一化 `[z,y,x] → [batch,x,y,z]`；每阶段 spatial_shape `1440→720→360→180`）就是切片 2 C++/CUDA 运行时的参考。

**路线 A 在导出+引擎层面已端到端验证：trainStation 已移除，且输出数值等价。**

### 切片 1b —— 导出流水线集成 ✅ 已完成并通过官方 CLI 验证

- `onnx_export_pipeline.py::_postprocess_sparse_onnx_fp`：当设置 `deploy_cfg.spconv_remove_trainstation` 时，对稀疏 ONNX 运行 `remove_trainstation_dds`（独立于 ReLU-fuse 标志；可与之干净组合）。
- `deploy_config_split_fp16_opt_trainstation.py`：`spconv_remove_trainstation = True` + 以编程方式把 16 个 rulebook 输入注入 `components.bevfusion_sparse.tensorrt_profile`（N∈[1,256000]；KV=27 用于 l1–l3，3 用于 conv_out）。
- 端到端运行官方 CLI（`python -m deployment.cli.main bevfusion <trainstation cfg> <model cfg>`）：日志显示 "trainStation/DDS removal done (removed 4 … added 16 rulebook graph inputs)"，之后两个引擎都构建。**对 CLI 产出的 `bevfusion_sparse.engine` 做 engine-inspector：127 层，0 个 trainStation 层。**（与 ImplicitGemm ReLU 融合共存，13 个 relu。）

**导出端完成（切片 1/1b/1c/1d）：官方流水线现在通过一个 deploy-cfg 标志即可产出无 trainStation、数值等价的稀疏引擎。** 剩余工作是提供这 16 个 rulebook 输入的运行时。

### 切片 2（Python 运行时）—— rulebook 预计算接入 deploy 评估流水线

部署自己的 Python TensorRT 流水线（`pipelines/tensorrt.py`）也需要提供这 16 个 rulebook 输入（在接通之前，评估会以 "Address is not set for input … GetIndicePairsImplicitGemm_output_0" 失败）。新增：
- `pipelines/sparse_rulebook_precompute.py`：`compute_rulebook_inputs(coors_zyx, input_names)`——在 4 个下采样阶段级联 `GetIndicePairsImplicitGemm`（已验证的切片 1d 逻辑），返回 `{input_name: int32 np.ndarray}`。`has_rulebook_inputs()` 作为门控（基线下为 no-op）。
- `pipelines/tensorrt.py::_trt_infer_voxel_inputs`：当稀疏引擎暴露 rulebook 输入时，从同一个 `coors` 预计算，并在 `enqueueV3` 之前加入绑定 map。

这让 mAP 可以在容器内验证（无需 autoware 构建），也是 C++/CUDA autoware_bevfusion 移植的精确参考。

**干净的 A/B（两者都 `export.mode="none"`，引擎预构建，GPU 计时，5 个样本）：**

| 阶段 | 基线（trainStation 开） | 移除 trainStation | Δ |
|-------|---------------------------:|---------------------:|---|
| mAP Center-BEV / Plane | 0.9066 / 0.9502 | 0.9068 / 0.9503 | 一致（fp16 噪声） |
| 稀疏编码器 | 9.37 ± 0.33 ms | 8.00 ± 0.40 ms | −1.4 ms（约 15%） |
| 稠密引擎（未变——对照） | 7.25 ms | 7.04 ms | 约相等 ✓ |
| 模型总计 | 16.63 ms | 15.03 ms | −1.6 ms |

**mAP 保持不变；稀疏编码器在这块 GPU 上快约 15%。** 稠密引擎（两者逐字节相同）在噪声范围内一致，确认对比是干净的（之前的一次 A/B 因基线跑了 `mode="both"` 而被污染——评估前的一次重型引擎构建抬高了 *所有* 阶段，包括未变的稠密 44→7 ms；那次运行的延迟差值无效，但其 mAP 不受影响）。

**关于延迟数字的诚实说明：**
- 这是一块强力的 dGPU。车载目标（原始 nsys profile，§2）显示 6 个 trainStation 造成约 30% 的 GPU 空闲，所以那里的相对收益预计会比 15% 更大。
- Python 原型的 rulebook 预计算时间 **未** 计入「稀疏编码器」（该阶段只是 TRT enqueue）。它替代了基线在图内做的工作（那部分 *本来* 就在基线的 9.37 ms 里），并把 4 次图中间同步合并为 1 次预处理同步——但一个完全公平的端到端数字必须把预计算成本计入预处理。决定性的、与硬件无关的结果是结构性的：trainStation 6→0 且 mAP 不变。

### 切片 2b —— autoware_bevfusion C++/CUDA 运行时 ✅ 已实现（构建/验证见切片 2c）

把已验证的 Python 预计算移植到车载节点。在 `autoware.universe/perception/autoware_bevfusion/` 中新增 + 编辑的文件：

- **`preprocess/sparse_rulebook_precompute.{hpp,cu}`**（新）：`SparseRulebookPrecompute`——拥有稳定的每阶段设备缓冲（out_indices/pair_fwd/pair_mask/mask_argsort，按 256000 上界分配）和一个共享 spconv 工作区；`buildBatchedCoordsKernel` 转换 `coors`（`[z,y,x]` → `[batch,x,y,z]`）；`compute()` 在 4 个下采样阶段级联 `SpconvOps::get_indice_pairs_implicit_gemm`（镜像 plugin 的非 subm `enqueue` 路径），向前串接 `out_indices`；暴露每阶段计数 + 设备指针。`default_bevfusion_downsample_stages()` 编码这 4 个阶段（ksize/stride/padding/spatial 1440→720→360→180）。
- **`bevfusion_trt.{hpp,cpp}`**：`addSparseRulebookNetworkIO` / `addSparseRulebookProfileDims`（声明 16 个输入 + `[min,opt,max]` profile，max = limit）、`bindSparseRulebookAddresses`（一次性 `setTensorAddress` 到稳定缓冲）、`setSparseRulebookInputShapes`（用同步回的计数对每阶段 `setInputShape`）。`preProcess` 在体素化之后立即调用 `compute()`；全部由 `config_.sparse_remove_trainstation_` 门控（否则为 no-op → 基线引擎仍可用）。
- **`bevfusion_config.hpp`**：普通成员 `sparse_remove_trainstation_`、`sparse_out_indices_num_limit_`（256000）、`sparse_coors_is_zyx_`。
- **`bevfusion_node.cpp`** + **`config/ml_package_bevfusion_lidar.param.yaml`** + **`schema/ml_package_bevfusion.schema.json`**：`sparse_remove_trainstation` ROS 参数（默认 false）。
- **`CMakeLists.txt`**：把新的 `.cu` 加入 `${PROJECT_NAME}_cuda_lib`（已链接 `spconv::spconv`）。

未在此处构建/验证（awml-bevfusion 容器未挂载 autoware.universe，且需要 colcon/autoware + spconv 构建）。该 `.cu` 忠实镜像了已证明的 plugin `enqueue` 和已验证的 Python 级联；首次构建时需确认的集成点：`SpconvOps` 的确切 API 签名、`coors` 顺序（`sparse_coors_is_zyx_`），以及 spconv 工作区大小。

### 切片 2c —— 首次在 autoware 环境构建 + 端到端运行 ✅ 已证明（pilot-auto.x2）

在 `pilot-auto.x2` 中针对合并的单文件引擎（`bevfusion_lidar.onnx`，以 `spconv_remove_trainstation=True` 导出）构建 `autoware_bevfusion`，并在真实的 `concatenated/pointcloud` rosbag 上端到端运行。出现三个问题——恰好就是切片 2b 标记的“首次构建需确认”的三点——外加启用该路径所需的配置。修正（位于 `autoware.universe/perception/autoware_bevfusion/`）：

1. **`SpconvOps` API 签名——`std::string` vs `const char*`（编译错误）。**
   `bevfusion_trt.cpp`（`bindSparseRulebookAddresses` / `setSparseRulebookInputShapes`）中的
   `setTensorAddress(...)` / `setInputShape(...)` 传入的是 `s.onnx_base + "_output_N"`（`std::string`），
   但已安装的 `autoware_tensorrt_common` 只暴露 `(const char*, ...)` / `(int32_t, ...)` 重载，没有
   `std::string` 的隐式转换。**修正：** 每个名字外面加 `(...).c_str()`（共 8 处）。

2. **合并引擎的张量名前缀（引擎能构建，但 profile 设到了错误的张量上）。**
   `onnx.compose.merge_models` 用 `sparse/` 对稀疏子图做命名空间化，而合并步骤只把 3 个*声明的* `io.inputs`
   （`voxels`/`coors`/`num_points_per_voxel`）改回原名——所以后加（由 `sparse_trainstation_transform` 添加）的
   16 个 rulebook 输入保留了前缀，变成 `sparse//pts_middle_encoder/.../GetIndicePairsImplicitGemm_output_*`
   （双斜杠 `//`）。运行时硬编码的 `default_bevfusion_downsample_stages()` 基名没有前缀，于是 profile 被注册到了
   不存在的张量上，真正的输入没有 profile → `Error Code 4: ... is missing dimensions in profile 0`。
   **修正（放在导出端，保持运行时干净）：** 在 `onnx_export_pipeline._merge_split_onnx` 中，在声明输入/输出改名之后，
   把每个剩余图输入的 `sparse/` 命名空间剥掉，使 rulebook 输入保留原始无前缀的 `GetIndicePairsImplicitGemm`
   节点名（`gs` 按对象身份改名，消费节点也会同步更新）。合并 ONNX 的输入名于是同时匹配运行时硬编码的阶段名
   与 deploy-cfg `tensorrt_profile` 名。运行时不再需要任何前缀开关——`autoware_bevfusion` 直接按原名绑定。
   （AWML 评估不受影响：`sparse_rulebook_precompute.has_rulebook_inputs` / `compute_rulebook_inputs` 按
   `GetIndicePairsImplicitGemm_output_` 标记 + 节点 `infix` 匹配，二者都与前缀无关。）
   *需用修正后的流水线重新导出 ONNX 并重建引擎。*

3. **spconv 工作区对下采样阶段配置过小（首帧运行时中止）。**
   `SparseRulebookPrecompute` 把 `N`（= `out_indices_num_limit_`，256000）当作 `max_act_out_in_theory`
   传给 `get_indice_gen_workspace_size` / `get_indice_gen_tensors_from_workspace`，于是内部
   `indice_pairs_uniq` 缓冲按 `N*1.1 = 281600` 切分。但第一阶段实际需要
   `get_handcrafted_max_act_out(num_in, ...) ≈ 808121`，触发 spconv `StaticAllocator` 的
   `res.nbytes() >= total ... assert faild. alloc failed, tensor size too small [2, 808121] [2, 281600]`。
   plugin 的 `enqueue` 是用 `SpconvOps::get_handcrafted_max_act_out(num_act_in, ...)` 而非 `N` 来定大小。
   **修正：** 镜像 plugin——`computeStage()` 用
   `max_act_out_theory = get_handcrafted_max_act_out(num_in, ksize, stride, padding, dilation)` 同时喂给
   工作区大小与张量切分；`allocateStageBuffers()` 按最坏情况（各阶段 `get_handcrafted_max_act_out(N, ...)` 的最大值）
   分配共享工作区，由于运行时 `num_in ≤ N`，可覆盖每个阶段的切分。

**启用该路径所需的配置**（写在*实际被加载*的 ml-package 参数文件——默认 launch 下是
`~/autoware_data/bevfusion/ml_package_bevfusion_lidar.param.yaml`，由 `model_path = $(data_path)/bevfusion`
解析得到，**不是** package 的 `config/` 副本）：

```yaml
sparse_remove_trainstation: true
```

**已验证：** 引擎构建时 16 个 `sparse//...GetIndicePairsImplicitGemm_output_*` 的 profile 全部设置成功
（`Engine generation completed`），节点加载该引擎，重放点云 rosbag 驱动推理**无崩溃**且 `/objects` 输出
有检测结果（走的是 `compatibleCallback` PointCloud2 路径——也就是之前在第 0 帧崩溃的那条）。

> 构建注意（本机）：shell 会自动激活 conda base，其 `colcon` 缺少 `colcon_core`。`colcon build` 前需把
> miniconda 从 `PATH` 移除（并 unset `PYTHONPATH`/`CONDA_PREFIX`），否则构建会在编译前静默空转。

### 后续切片（待办）

- **1b** 把 `remove_trainstation_dds` 接入 `onnx_export_pipeline.py`，作为由 deploy-cfg 标志（如 `spconv_remove_trainstation=True`）门控的稀疏 ONNX 后处理；把 16 个输入加入 TensorRT 优化 profile（deploy cfg `tensorrt_profile`）。
- **1c** 决定性证明：从手术修改后的 ONNX 构建引擎并抓取 nsys → 确认 `[trainStation]` / `DeviceToShapeHostCopy` 消失（喂入一次性计算的 rulebook）。
- **2** 运行时（autoware_bevfusion）：CUDA 预计算 4 个下采样 rulebook（`SpconvOps::get_indice_pairs_implicit_gemm` 级联）+ `setInputShape` + 绑定。计数仅一次同步。
- **3** 等价性 gtest + 端到端 mAP 不变。
