# 🔧 CalibrationStatusClassification TensorRT 配置修正

## ❌ 問題描述

在之前的更新中，我意外移除了 CalibrationStatusClassification 配置中的重要 TensorRT 動態形狀設定：

```python
# 被移除的重要配置
model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 5, 1080, 1920],  # Minimum supported input shape
                opt_shape=[1, 5, 1860, 2880],  # Optimal shape for performance tuning
                max_shape=[1, 5, 2160, 3840],  # Maximum supported input shape
            ),
        )
    )
],
```

## ⚠️ 影響分析

移除這些配置會對 TensorRT 性能產生負面影響：

1. **性能優化缺失**：TensorRT 需要知道輸入的最小、最優和最大形狀來進行優化
2. **動態形狀支持**：無法支持不同解析度的輸入
3. **內存分配**：無法預先分配適當的內存空間

## ✅ 修正方案

### 1. 恢復 TensorRT 形狀範圍配置

```python
# CalibrationStatusClassification/configs/deploy/deploy_config.py
backend_config = dict(
    type="tensorrt",
    common_config=dict(
        max_workspace_size=1 << 30,  # 1 GiB workspace
        precision_policy="fp16",
    ),
    # Dynamic shape configuration for different input resolutions
    # TensorRT needs shape ranges for optimization even with dynamic batch size
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 5, 1080, 1920],  # Minimum supported input shape
                    opt_shape=[1, 5, 1860, 2880],  # Optimal shape for performance tuning
                    max_shape=[1, 5, 2160, 3840],  # Maximum supported input shape
                ),
            )
        )
    ],
)
```

### 2. 更新 BaseDeploymentConfig 邏輯

```python
def update_batch_size(self, batch_size: int) -> None:
    """Update batch size in backend config model_inputs."""
    if batch_size is not None:
        existing_model_inputs = self.backend_config.model_inputs
        
        # If model_inputs is None or empty, generate from model_io
        if existing_model_inputs is None or len(existing_model_inputs) == 0:
            # Generate simple shapes from model_io
            # ... (existing logic)
        else:
            # If model_inputs already exists (e.g., TensorRT shape ranges), 
            # preserve the existing configuration
            for model_input in existing_model_inputs:
                if isinstance(model_input, dict) and "input_shapes" in model_input:
                    # TensorRT shape ranges format - preserve as is
                    # Batch size is handled by dynamic_axes
                    pass
```

## 🧪 測試結果

### ✅ 修正前後對比

**修正前**：
```python
model_inputs=None  # 缺少 TensorRT 優化信息
```

**修正後**：
```python
model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 5, 1080, 1920],
                opt_shape=[1, 5, 1860, 2880], 
                max_shape=[1, 5, 2160, 3840],
            ),
        )
    )
],
```

### ✅ 測試驗證

```bash
=== CalibrationStatusClassification Config Test ===
Model inputs before update: [{'input_shapes': {'input': {'min_shape': [1, 5, 1080, 1920], 'opt_shape': [1, 5, 1860, 2880], 'max_shape': [1, 5, 2160, 3840]}}}]
Model inputs after batch_size=1: [{'input_shapes': {'input': {'min_shape': [1, 5, 1080, 1920], 'opt_shape': [1, 5, 1860, 2880], 'max_shape': [1, 5, 2160, 3840]}}}]

ONNX Settings:
  Input names: ['input']
  Output names: ['output']
  Batch size: None
  Dynamic axes: {'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size'}}
✅ CalibrationStatusClassification config works correctly!
```

## 📋 設計原則

### 1. 智能配置處理
- **簡單配置**：當 `model_inputs=None` 時，從 `model_io` 自動生成
- **TensorRT 配置**：當已有 `input_shapes` 時，保留現有配置
- **混合配置**：支持兩種格式的混合使用

### 2. 向後兼容
- ✅ 保持與現有 TensorRT 配置的兼容性
- ✅ 支持動態 batch size 和固定 batch size
- ✅ 自動處理不同類型的 model_inputs 格式

### 3. 性能優化
- ✅ TensorRT 形狀範圍優化
- ✅ 動態解析度支持
- ✅ 內存預分配優化

## 🎯 最終狀態

- ✅ **CalibrationStatusClassification**：恢復 TensorRT 形狀範圍配置
- ✅ **CenterPoint**：保持簡單配置，自動生成 model_inputs
- ✅ **YOLOX_opt_elan**：保持簡單配置，自動生成 model_inputs
- ✅ **BaseDeploymentConfig**：智能處理不同配置格式
- ✅ **所有測試通過**：配置載入和更新都正常工作

現在 CalibrationStatusClassification 既享受了統一配置設計的優勢，又保持了 TensorRT 的性能優化！🚀
