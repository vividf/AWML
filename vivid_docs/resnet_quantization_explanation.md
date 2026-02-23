# ResNet Residual Connection Quantization: 核心概念与原理

## 三个仓库的核心概念一致性

虽然三个仓库的实现方式不同，但**核心概念完全一致**：

### 核心原则
**只在 skip connection (identity branch) 上添加 QDQ，不在 main branch (conv path) 上添加 QDQ**

### 三个仓库的实现对比

| 仓库 | 实现层面 | 方法 | 核心概念 |
|------|---------|------|---------|
| **Lidar_AI_Solution** | PyTorch | Forward hook 中只量化 identity | ✅ 一致 |
| **ModelOpt** | ONNX | 图分析后移除 main branch QDQ | ✅ 一致 |
| **AWML CenterPoint** | PyTorch | Forward hook 中只量化 identity | ✅ 一致 |

**结论**: 三个仓库的核心概念完全一致，只是实现层面不同。

---

## 为什么 ResNet 需要这样处理？

### 1. TensorRT 的融合规则

TensorRT 可以将以下模式融合成单个 kernel：
```
Conv -> BN -> ReLU -> Add
```

但融合的前提是：
- **Conv 的输出必须是 INT8**（有 QDQ）
- **Add 的一个输入是 INT8**（有 QDQ）
- **Add 的另一个输入（main branch）不能有 QDQ**（必须是 FP32）

### 2. 错误的做法（两个分支都量化）

```
Conv -> BN -> ReLU -> Conv -> BN -> QDQ -> Add <- QDQ Identity
                              ↑                    ↑
                         (main branch)      (skip connection)
```

**问题**:
- Main branch 有 QDQ → TensorRT 无法识别 Conv+Add 融合模式
- 会产生大量 reformat 操作（INT8 ↔ FP32 转换）
- 性能下降，内存开销增加

### 3. 正确的做法（只量化 skip connection）

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

## 详细原理说明

### TensorRT 融合的工作原理

#### 1. 融合条件

TensorRT 的融合引擎会查找以下模式：
```
[INT8 Conv] -> [INT8 BN] -> [INT8 ReLU] -> [FP32 Add] <- [INT8 QDQ]
```

当检测到这个模式时，TensorRT 会：
1. 将整个序列融合成单个 INT8 kernel
2. 在 kernel 内部处理 Add 操作（不需要额外的 reformat）
3. 输出 INT8 结果

#### 2. 为什么 main branch 不能有 QDQ？

如果 main branch 有 QDQ：
```
[INT8 Conv] -> [INT8 BN] -> [INT8 ReLU] -> [INT8 QDQ] -> [FP32 Add] <- [INT8 QDQ]
```

TensorRT 看到的是：
- Conv 输出: INT8
- QDQ 输出: FP32（因为 QDQ 会 dequantize）
- Add 输入: FP32 + INT8（类型不匹配）

**结果**: TensorRT 无法融合，因为：
- Add 需要两个相同类型的输入
- 需要额外的 reformat 操作来对齐类型
- 融合条件不满足

#### 3. 为什么 skip connection 需要有 QDQ？

Skip connection 需要 QDQ 的原因：
1. **类型对齐**: Add 操作需要两个输入类型一致
2. **量化校准**: Skip connection 的数据分布可能与 main branch 不同，需要独立的量化参数
3. **融合支持**: TensorRT 可以识别 `[INT8 QDQ] -> [FP32 Add]` 模式

---

## 实际效果对比

### 错误做法的影响

```
ONNX Graph:
  Conv -> QDQ -> Add <- QDQ
    ↓      ↓      ↓      ↓
  INT8   FP32  FP32   FP32

TensorRT Engine:
  [Conv INT8] -> [Reformat INT8→FP32] -> [Add FP32] <- [Reformat INT8→FP32]

问题:
- 2 个 reformat 操作
- 无法融合
- 性能下降
```

### 正确做法的影响

```
ONNX Graph:
  Conv -> Add <- QDQ
    ↓      ↓      ↓
  INT8   FP32   FP32

TensorRT Engine:
  [Conv+BN+ReLU+Add INT8] (融合的单个 kernel)

优势:
- 0 个 reformat 操作
- 完全融合
- 性能最优
```

---

## 三个仓库的实现验证

### 1. Lidar_AI_Solution (PyTorch 层面)

```python
# 只量化 identity branch
if hasattr(self, "residual_quantizer"):
    identity = self.residual_quantizer(identity)  # ✅ 有 QDQ

out += identity  # conv path 不量化，✅ 无 QDQ
```

**ONNX 导出结果**:
```
Conv -> BN -> ReLU -> Conv -> BN -> Add <- QDQ Identity
                              ↑            ↑
                         (无 QDQ)      (有 QDQ)
```

### 2. ModelOpt (ONNX 层面)

```python
# 1. 先量化所有节点（包括两个分支）
quantize_static(...)  # 两个分支都有 QDQ

# 2. 识别 residual Add
non_residual_inputs = build_non_residual_input_map(graph)

# 3. 移除 main branch 的 QDQ
remove_partial_input_qdq(graph, no_quantize_inputs)
```

**最终结果**:
```
Conv -> BN -> ReLU -> Conv -> BN -> Add <- QDQ Identity
                              ↑            ↑
                         (移除 QDQ)    (保留 QDQ)
```

### 3. AWML CenterPoint (PyTorch 层面，与 Lidar_AI_Solution 对齐)

```python
# 只量化 identity branch
if hasattr(self, "residual_quantizer"):
    identity = self.residual_quantizer(identity)  # ✅ 有 QDQ

out = out + identity  # conv path 不量化，✅ 无 QDQ
```

**ONNX 导出结果**:
```
Conv -> BN -> ReLU -> Conv -> BN -> Add <- QDQ Identity
                              ↑            ↑
                         (无 QDQ)      (有 QDQ)
```

---

## 总结

### 核心概念一致性 ✅

三个仓库都遵循相同的核心原则：
- **只在 skip connection 上添加 QDQ**
- **不在 main branch 上添加 QDQ**

### 为什么这样处理？

1. **TensorRT 融合需求**: 需要 `Conv -> Add` 模式，其中 Conv 输出 INT8，Add 的 main branch 输入 FP32
2. **性能优化**: 避免 reformat 操作，实现 kernel 融合
3. **内存效率**: 减少中间张量的类型转换
4. **官方推荐**: TensorRT 官方文档明确推荐这种 QDQ 放置策略

### 实现方式差异

- **Lidar_AI_Solution & AWML CenterPoint**: PyTorch 层面实现（简单直接）
- **ModelOpt**: ONNX 层面后处理（更灵活，可处理任意 ONNX 模型）

**结论**: 虽然实现方式不同，但核心概念和最终效果完全一致。
