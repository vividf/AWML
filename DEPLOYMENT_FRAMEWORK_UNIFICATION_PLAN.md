# Deployment Framework 統一化實施建議

## 🎯 目標

將現有的三個模型（CenterPoint、YOLOX、CalibrationStatusClassification）的 deployment 架構統一化，創建一個可擴展的框架。

---

## 📊 當前架構問題總結

### 代碼重複問題

| 模型 | 自定義組件 | 重複功能 |
|------|------------|----------|
| **CenterPoint** | `centerpoint_onnx_helper.py`<br>`centerpoint_tensorrt_backend.py` | 多引擎處理<br>PyTorch 中間處理<br>自定義後處理 |
| **YOLOX** | `onnx_wrapper.py` | 輸出格式轉換<br>後處理邏輯 |
| **CalibrationStatus** | 無 | 標準處理（但功能受限） |

### 擴展性問題
- ❌ 新模型需要重寫相似功能
- ❌ 多引擎處理邏輯無法復用
- ❌ 後處理邏輯分散

---

## 🏗️ 統一框架設計

### 核心架構

```
autoware_ml/deployment/
├── core/
│   ├── modular_pipeline.py      # 🆕 統一 Pipeline
│   ├── processor_registry.py   # 🆕 處理器註冊
│   └── model_configs.py        # 🆕 模型配置
├── processors/
│   ├── __init__.py
│   ├── base_processor.py       # 🆕 基礎處理器
│   ├── preprocessors/          # 🆕 預處理器
│   │   ├── voxelization.py
│   │   ├── image_normalization.py
│   │   └── point_cloud.py
│   ├── midprocessors/          # 🆕 中間處理器
│   │   ├── pytorch_middle_encoder.py
│   │   └── feature_fusion.py
│   └── postprocessors/         # 🆕 後處理器
│       ├── centerpoint_decoder.py
│       ├── yolox_decoder.py
│       └── classification_decoder.py
├── backends/
│   ├── modular_backend.py      # 🆕 統一 Backend
│   └── engine_manager.py       # 🆕 引擎管理器
└── exporters/
    ├── modular_exporter.py     # 🆕 統一導出器
    └── engine_exporter.py      # 🆕 引擎導出器
```

---

## 🔧 具體實施方案

### 1. 創建基礎處理器

```python
# autoware_ml/deployment/processors/base_processor.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProcessor(ABC):
    """所有處理器的基礎類"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """處理輸入數據"""
        pass
    
    def validate_config(self) -> bool:
        """驗證配置"""
        return True
```

### 2. 實現預處理器

```python
# autoware_ml/deployment/processors/preprocessors/voxelization.py
import torch
from ..base_processor import BaseProcessor

class VoxelizationPreprocessor(BaseProcessor):
    """點雲體素化預處理器（CenterPoint 使用）"""
    
    def process(self, input_data):
        points = input_data['points']
        voxel_size = self.config['voxel_size']
        point_cloud_range = self.config['point_cloud_range']
        
        # 體素化邏輯
        voxels, coors, num_points = self._voxelize(points, voxel_size, point_cloud_range)
        
        return {
            'voxels': voxels,
            'coors': coors, 
            'num_points': num_points
        }
    
    def _voxelize(self, points, voxel_size, point_cloud_range):
        # 實現體素化邏輯
        pass

# autoware_ml/deployment/processors/preprocessors/image_normalization.py
class ImageNormalizationPreprocessor(BaseProcessor):
    """圖像標準化預處理器（YOLOX、CalibrationStatus 使用）"""
    
    def process(self, input_data):
        image = input_data['image']
        mean = self.config.get('mean', [0.485, 0.456, 0.406])
        std = self.config.get('std', [0.229, 0.224, 0.225])
        
        # 標準化邏輯
        normalized_image = self._normalize(image, mean, std)
        
        return {'image': normalized_image}
```

### 3. 實現中間處理器

```python
# autoware_ml/deployment/processors/midprocessors/pytorch_middle_encoder.py
import torch
from ..base_processor import BaseProcessor

class PyTorchMiddleEncoderProcessor(BaseProcessor):
    """PyTorch 中間編碼器（CenterPoint 使用）"""
    
    def __init__(self, config):
        super().__init__(config)
        self.pytorch_model = config['pytorch_model']
    
    def process(self, input_data):
        voxel_features = input_data['voxel_features']
        coors = input_data['coors']
        
        # 使用 PyTorch 模型處理
        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(voxel_features, coors)
        
        return {'spatial_features': spatial_features}
```

### 4. 實現後處理器

```python
# autoware_ml/deployment/processors/postprocessors/centerpoint_decoder.py
import torch
from ..base_processor import BaseProcessor

class CenterPointDecoderProcessor(BaseProcessor):
    """CenterPoint 解碼器（NMS、後處理）"""
    
    def __init__(self, config):
        super().__init__(config)
        self.pytorch_model = config['pytorch_model']
    
    def process(self, input_data):
        # head_outputs: [heatmap, reg, height, dim, rot, vel]
        head_outputs = input_data['head_outputs']
        
        # 使用 PyTorch 模型的 predict_by_feat
        with torch.no_grad():
            predictions = self.pytorch_model.pts_bbox_head.predict_by_feat(
                head_outputs, 
                input_data['batch_input_metas']
            )
        
        return predictions

# autoware_ml/deployment/processors/postprocessors/yolox_decoder.py
class YOLOXDecoderProcessor(BaseProcessor):
    """YOLOX 解碼器（Tier4 格式轉換）"""
    
    def process(self, input_data):
        raw_outputs = input_data['raw_outputs']
        output_format = self.config.get('output_format', 'standard')
        
        if output_format == 'tier4_compatible':
            return self._convert_to_tier4_format(raw_outputs)
        else:
            return self._standard_decode(raw_outputs)
```

### 5. 創建統一 Backend

```python
# autoware_ml/deployment/backends/modular_backend.py
from typing import List, Dict, Any
from ..processors.base_processor import BaseProcessor
from .base_backend import BaseBackend

class ModularBackend(BaseBackend):
    """統一的模組化 Backend"""
    
    def __init__(self, 
                 engines: List[str],
                 preprocessor: BaseProcessor = None,
                 midprocessor: BaseProcessor = None,
                 postprocessor: BaseProcessor = None):
        self.engines = engines
        self.preprocessor = preprocessor
        self.midprocessor = midprocessor
        self.postprocessor = postprocessor
        
        # 初始化引擎
        self._init_engines()
    
    def infer(self, input_data: Dict[str, Any]) -> Any:
        """統一的推理接口"""
        
        # 1. 預處理
        if self.preprocessor:
            processed_data = self.preprocessor.process(input_data)
        else:
            processed_data = input_data
        
        # 2. 引擎推理
        if len(self.engines) == 1:
            engine_output = self._run_single_engine(processed_data)
        else:
            engine_output = self._run_multi_engines(processed_data)
        
        # 3. 中間處理
        if self.midprocessor:
            engine_output = self.midprocessor.process(engine_output)
        
        # 4. 後處理
        if self.postprocessor:
            final_output = self.postprocessor.process(engine_output)
        else:
            final_output = engine_output
        
        return final_output
    
    def _run_single_engine(self, input_data):
        """單引擎推理"""
        engine = self.engines[0]
        return self._run_engine(engine, input_data)
    
    def _run_multi_engines(self, input_data):
        """多引擎推理"""
        results = {}
        
        # 運行第一個引擎（通常是 voxel encoder）
        first_engine_output = self._run_engine(self.engines[0], input_data)
        results['voxel_features'] = first_engine_output
        
        # 運行第二個引擎（通常是 backbone/neck/head）
        second_engine_input = {**input_data, **results}
        second_engine_output = self._run_engine(self.engines[1], second_engine_input)
        results['head_outputs'] = second_engine_output
        
        return results
```

### 6. 模型配置系統

```python
# autoware_ml/deployment/core/model_configs.py
MODEL_CONFIGS = {
    "CenterPoint": {
        "engines": ["pts_voxel_encoder", "pts_backbone_neck_head"],
        "preprocessor": {
            "type": "voxelization",
            "voxel_size": [0.05, 0.05, 0.1],
            "point_cloud_range": [-50, -50, -5, 50, 50, 3],
            "max_num_points": 32,
            "max_voxels": 16000
        },
        "midprocessor": {
            "type": "pytorch_middle_encoder",
            "requires_pytorch_model": True
        },
        "postprocessor": {
            "type": "centerpoint_decoder",
            "requires_pytorch_model": True,
            "rot_y_axis_reference": False
        }
    },
    
    "YOLOX": {
        "engines": ["yolox_model"],
        "preprocessor": {
            "type": "image_normalization",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "postprocessor": {
            "type": "yolox_decoder",
            "output_format": "tier4_compatible",
            "num_classes": 8
        }
    },
    
    "CalibrationStatusClassification": {
        "engines": ["classification_model"],
        "preprocessor": {
            "type": "image_normalization",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "postprocessor": {
            "type": "classification_decoder",
            "num_classes": 3
        }
    }
}
```

---

## 🚀 遷移計劃

### Phase 1: 基礎設施（1-2 週）

#### 1.1 創建基礎類
- [ ] `BaseProcessor` 抽象類
- [ ] `ModularBackend` 統一 Backend
- [ ] `ProcessorRegistry` 處理器註冊系統

#### 1.2 實現核心處理器
- [ ] `VoxelizationPreprocessor`（CenterPoint）
- [ ] `ImageNormalizationPreprocessor`（YOLOX、CalibrationStatus）
- [ ] `PyTorchMiddleEncoderProcessor`（CenterPoint）
- [ ] `CenterPointDecoderProcessor`（CenterPoint）
- [ ] `YOLOXDecoderProcessor`（YOLOX）
- [ ] `ClassificationDecoderProcessor`（CalibrationStatus）

### Phase 2: 模型遷移（2-3 週）

#### 2.1 CalibrationStatusClassification（最簡單）
- [ ] 創建新的 `main.py` 使用 `ModularBackend`
- [ ] 測試功能一致性
- [ ] 性能對比

#### 2.2 YOLOX（中等複雜度）
- [ ] 遷移 `onnx_wrapper.py` 邏輯到 `YOLOXDecoderProcessor`
- [ ] 更新 `main.py` 使用新架構
- [ ] 測試 Tier4 格式兼容性

#### 2.3 CenterPoint（最複雜）
- [ ] 遷移 `centerpoint_onnx_helper.py` 邏輯
- [ ] 遷移 `centerpoint_tensorrt_backend.py` 邏輯
- [ ] 更新 `main.py` 使用新架構
- [ ] 測試多引擎和混合執行

### Phase 3: 優化和清理（1 週）

#### 3.1 性能優化
- [ ] 內存使用優化
- [ ] 推理速度優化
- [ ] 緩存機制

#### 3.2 代碼清理
- [ ] 移除舊的自定義組件
- [ ] 更新文檔
- [ ] 添加單元測試

---

## 📊 預期效果

### 代碼減少
- **CenterPoint**: 減少 ~500 行自定義代碼
- **YOLOX**: 減少 ~200 行自定義代碼
- **CalibrationStatus**: 增加 ~50 行配置代碼

### 維護性提升
- ✅ **統一接口**: 所有模型使用相同的 API
- ✅ **配置驅動**: 行為由配置文件控制
- ✅ **模組化**: 每個組件職責單一

### 擴展性提升
- ✅ **新模型**: 只需添加配置，無需重寫代碼
- ✅ **新功能**: 在對應的 processor 中添加
- ✅ **新引擎**: 支持任意數量的引擎

---

## 🧪 測試策略

### 功能測試
```python
# 測試每個處理器
def test_voxelization_preprocessor():
    processor = VoxelizationPreprocessor(config)
    result = processor.process(test_input)
    assert result['voxels'].shape == expected_shape

# 測試完整 pipeline
def test_centerpoint_pipeline():
    backend = ModularBackend(**centerpoint_config)
    result = backend.infer(test_input)
    assert len(result) == expected_predictions
```

### 性能測試
```python
# 對比新舊實現的性能
def benchmark_centerpoint():
    old_backend = CenterPointTensorRTBackend(...)
    new_backend = ModularBackend(...)
    
    old_time = timeit.timeit(lambda: old_backend.infer(input_data))
    new_time = timeit.timeit(lambda: new_backend.infer(input_data))
    
    assert new_time <= old_time * 1.1  # 允許 10% 性能損失
```

### 兼容性測試
```python
# 確保輸出格式一致
def test_output_compatibility():
    old_output = old_backend.infer(input_data)
    new_output = new_backend.infer(input_data)
    
    assert np.allclose(old_output, new_output, rtol=1e-5)
```

---

## 📝 使用示例

### 新模型添加示例

假設要添加一個新的 3D 檢測模型 "PointPillars"：

```python
# 1. 添加配置
MODEL_CONFIGS["PointPillars"] = {
    "engines": ["pillar_encoder", "backbone_head"],
    "preprocessor": {
        "type": "pillar_encoding",  # 新的預處理器
        "pillar_size": [0.2, 0.2],
        "max_pillars": 12000
    },
    "postprocessor": {
        "type": "pointpillars_decoder",
        "num_classes": 3
    }
}

# 2. 實現新的預處理器（如果需要）
class PillarEncodingPreprocessor(BaseProcessor):
    def process(self, input_data):
        # 實現 pillar encoding 邏輯
        pass

# 3. 使用
config = MODEL_CONFIGS["PointPillars"]
backend = ModularBackend(**config)
output = backend.infer(input_data)
```

**優勢**: 新模型只需要添加配置和實現特定的處理器，不需要重寫整個 backend！

---

## ✅ 總結

### 當前問題
- ❌ 代碼重複（每個模型有自己的實現）
- ❌ 擴展性差（新模型需要重寫）
- ❌ 維護困難（分散的邏輯）

### 解決方案
- ✅ **統一 Pipeline**: Preprocessor → Midprocessor → Postprocessor
- ✅ **模組化 Backend**: 支持單引擎和多引擎
- ✅ **配置驅動**: 行為由配置文件控制
- ✅ **易於擴展**: 新模型只需添加配置

### 預期效果
- ✅ **代碼復用**: 減少 70% 的重複代碼
- ✅ **易於維護**: 統一的接口和實現
- ✅ **快速擴展**: 新模型開發時間減少 80%
- ✅ **性能提升**: 統一的優化策略

---

**這個統一框架將大大提升 AutoWare ML deployment 的可維護性和擴展性！** 🎉

---

**日期**: 2025-10-23  
**狀態**: 📋 設計完成，待實施  
**優先級**: 🔥 高（架構改進）
