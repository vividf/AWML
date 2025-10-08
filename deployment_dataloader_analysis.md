# Deployment DataLoader 架構分析與建議

## 當前架構概述

### 1. 現有的 Deployment Framework 設計

當前的 deployment framework 採用了以下架構：

```
autoware_ml/deployment/
├── core/
│   ├── base_config.py          # 配置管理
│   ├── base_data_loader.py     # DataLoader 抽象介面
│   ├── base_evaluator.py       # Evaluator 抽象介面
│   └── verification.py         # 跨 backend 驗證
├── backends/
│   ├── pytorch_backend.py
│   ├── onnx_backend.py
│   └── tensorrt_backend.py
└── exporters/
    ├── onnx_exporter.py
    └── tensorrt_exporter.py
```

### 2. 當前 BaseDataLoader 的設計

**優點：**
- ✅ **簡潔明確**：只有 3 個核心方法 (`load_sample`, `preprocess`, `get_num_samples`)
- ✅ **Task-agnostic**：不綁定特定任務，可適用於各種場景
- ✅ **輕量級**：無額外依賴，容易理解和維護
- ✅ **靈活度高**：每個專案可以自由實現自己的資料處理邏輯

**缺點：**
- ❌ **重複實現**：每個專案需要重新實現資料載入和預處理邏輯
- ❌ **與訓練不一致**：訓練時使用 mmdet pipeline，部署時用自定義邏輯
- ❌ **測試成本高**：需要額外驗證部署時的 preprocessing 與訓練一致

### 3. MMDet DataLoader 的特點

MMDet/MMEngine 使用更完整的 pipeline 系統：

```python
# MMDet 的方式
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        pipeline=test_pipeline,
        ...
    )
)
```

**優點：**
- ✅ **可配置性強**：透過 config 檔案定義整個 pipeline
- ✅ **可重用性高**：訓練和部署可以共用相同的 transforms
- ✅ **標準化**：MMDet 生態系統的標準做法
- ✅ **豐富的 transforms**：內建大量預處理和增強操作

**缺點：**
- ❌ **複雜度高**：需要理解 MMDet 的 Registry、Transform、Dataset 機制
- ❌ **過度設計**：對於簡單的部署場景可能過於複雜
- ❌ **耦合度高**：與 MMDet 生態系統強耦合
- ❌ **batch 處理限制**：MMDet DataLoader 主要為訓練設計，可能不適合單樣本推理

---

## 現有專案的 Deployment 狀況分析

### CalibrationStatusClassification (已實現)
```python
class CalibrationDataLoader(BaseDataLoader):
    def __init__(self, info_pkl_path, model_cfg, ...):
        self._transform = CalibrationClassificationTransform(...)

    def load_sample(self, index):
        return self._samples_list[index]

    def preprocess(self, sample):
        results = self._transform.transform(sample)
        tensor = torch.from_numpy(results["fused_img"])
        return tensor
```

**特點：**
- 使用自定義的 `CalibrationClassificationTransform`
- 從 info.pkl 載入資料
- 相對簡單的資料處理流程

### CenterPoint (自定義實現)
```python
class DeploymentRunner(BaseRunner):
    def run(self):
        model = self.build_model()
        self.load_verify_checkpoint(model=model)
        model.save_onnx(save_dir=self._work_dir)
```

**特點：**
- **未使用統一的 deployment framework**
- 直接在 model 上呼叫 `save_onnx()`
- 缺少 verification 和 evaluation 功能
- 需要遷移到新架構

### YOLOX (未實現)
- 目前沒有 deployment 支援
- 需要從零開始實現

---

## 對未來延伸的影響分析

### 場景 1：使用當前的 BaseDataLoader 架構

**延伸到 YOLOX 的實現方式：**

```python
class YOLOXDataLoader(BaseDataLoader):
    """DataLoader for YOLOX detection task."""

    def __init__(self, info_file: str, model_cfg: Config, device: str = "cpu"):
        self.model_cfg = model_cfg
        self.device = device

        # 從 model_cfg 獲取 test_pipeline
        self.pipeline = self._build_pipeline()

        # 載入 annotations
        self.data_infos = self._load_annotations(info_file)

    def _build_pipeline(self):
        """從 model config 建立 preprocessing pipeline"""
        # 問題：需要手動實現或包裝 mmdet transforms
        test_cfg = self.model_cfg.get('test_dataloader', {})
        pipeline_cfg = test_cfg.get('dataset', {}).get('pipeline', [])

        # Option 1: 手動重新實現
        # Option 2: 使用 mmdet.Compose + transforms
        from mmdet.datasets.transforms import Compose
        return Compose(pipeline_cfg)

    def load_sample(self, index: int) -> Dict[str, Any]:
        """載入單一樣本"""
        data_info = self.data_infos[index]
        return {
            'img_path': data_info['filename'],
            'img_id': data_info['img_id'],
            'gt_bboxes': data_info.get('ann', {}).get('bboxes', []),
            'gt_labels': data_info.get('ann', {}).get('labels', [])
        }

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """預處理樣本"""
        # 使用 mmdet pipeline
        results = self.pipeline(sample)

        # 提取 model input
        inputs = results['inputs']
        return inputs.to(self.device)
```

**挑戰：**
1. 如何整合 mmdet 的 pipeline？
2. 如何處理 batch 和 collate？
3. 如何確保與訓練時完全一致？

### 場景 2：使用 MMDet DataLoader

**直接使用 MMDet 的方式：**

```python
from mmengine.runner import Runner
from mmdet.apis import init_detector

class YOLOXDeployment:
    def __init__(self, config_file, checkpoint_file):
        self.model = init_detector(config_file, checkpoint_file)
        self.cfg = Config.fromfile(config_file)

        # 建立 dataloader
        self.dataloader = Runner.build_dataloader(
            self.cfg.test_dataloader
        )

    def evaluate(self):
        for data_batch in self.dataloader:
            with torch.no_grad():
                results = self.model.test_step(data_batch)
```

**挑戰：**
1. **不符合當前 deployment framework 的設計**
2. Runner 和 DataLoader 為訓練設計，不適合單樣本推理
3. 難以整合 ONNX/TensorRT backend
4. 失去了 framework 的統一性

---

## 建議方案

### 🎯 **推薦方案：混合架構 (Hybrid Approach)**

保持當前的 `BaseDataLoader` 架構，但**整合 MMDet 的 Transform Pipeline**：

```python
# autoware_ml/deployment/core/base_data_loader.py (保持不變)
class BaseDataLoader(ABC):
    @abstractmethod
    def load_sample(self, index: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        pass
```

```python
# autoware_ml/deployment/utils/pipeline_builder.py (新增)
from mmdet.datasets.transforms import Compose
from mmengine.config import Config

def build_test_pipeline(model_cfg: Config):
    """
    從 model config 建立 test pipeline

    Args:
        model_cfg: Model configuration with test_dataloader or test_pipeline

    Returns:
        Compose: MMDet transform pipeline
    """
    # Option 1: 從 test_dataloader 提取
    if 'test_dataloader' in model_cfg:
        pipeline_cfg = model_cfg.test_dataloader.dataset.pipeline
    # Option 2: 直接從 test_pipeline 提取
    elif 'test_pipeline' in model_cfg:
        pipeline_cfg = model_cfg.test_pipeline
    else:
        raise ValueError("No test pipeline found in config")

    return Compose(pipeline_cfg)
```

```python
# 專案實現：projects/YOLOX/deploy/data_loader.py
from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.utils import build_test_pipeline

class YOLOXDataLoader(BaseDataLoader):
    """DataLoader for YOLOX using MMDet pipeline."""

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        model_cfg: Config,
        device: str = "cpu"
    ):
        super().__init__(config={
            "ann_file": ann_file,
            "img_prefix": img_prefix,
            "device": device
        })

        self.img_prefix = img_prefix
        self.device = device

        # 載入 annotations (COCO format)
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()

        # 使用 model config 建立 pipeline
        self.pipeline = build_test_pipeline(model_cfg)

    def load_sample(self, index: int) -> Dict[str, Any]:
        """載入單一樣本（COCO格式）"""
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        # 轉換為 mmdet 格式
        return {
            'img_id': img_id,
            'img_path': os.path.join(self.img_prefix, img_info['file_name']),
            'height': img_info['height'],
            'width': img_info['width'],
            'instances': [
                {
                    'bbox': ann['bbox'],  # [x, y, w, h]
                    'bbox_label': ann['category_id'] - 1,
                    'ignore_flag': ann.get('ignore', 0)
                }
                for ann in ann_info
            ]
        }

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """使用 MMDet pipeline 預處理"""
        # 通過 pipeline 處理
        results = self.pipeline(sample)

        # 提取 model input
        # MMDet 3.x 使用 'inputs' key
        inputs = results['inputs']

        # 轉換為推理格式
        if isinstance(inputs, torch.Tensor):
            return inputs.unsqueeze(0).to(self.device)
        else:
            return torch.from_numpy(inputs).unsqueeze(0).to(self.device)

    def get_num_samples(self) -> int:
        return len(self.img_ids)

    def get_ground_truth(self, index: int) -> Dict[str, Any]:
        """獲取 ground truth（用於 evaluation）"""
        sample = self.load_sample(index)
        return {
            'gt_bboxes': [inst['bbox'] for inst in sample['instances']],
            'gt_labels': [inst['bbox_label'] for inst in sample['instances']]
        }
```

```python
# projects/CenterPoint/deploy/data_loader.py
from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.utils import build_test_pipeline

class CenterPointDataLoader(BaseDataLoader):
    """DataLoader for CenterPoint 3D detection."""

    def __init__(
        self,
        info_file: str,
        model_cfg: Config,
        device: str = "cpu"
    ):
        super().__init__(config={
            "info_file": info_file,
            "device": device
        })

        # 載入 info.pkl
        with open(info_file, 'rb') as f:
            data = pickle.load(f)
        self.data_infos = data['data_list']

        # 建立 pipeline (from mmdet3d)
        self.pipeline = build_test_pipeline(model_cfg)
        self.device = device

    def load_sample(self, index: int) -> Dict[str, Any]:
        """載入點雲和標註"""
        info = self.data_infos[index]

        return {
            'lidar_points': {
                'lidar_path': info['lidar_points']['lidar_path']
            },
            'gt_bboxes_3d': info.get('ann_info', {}).get('gt_bboxes_3d', []),
            'gt_labels_3d': info.get('ann_info', {}).get('gt_labels_3d', [])
        }

    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """使用 MMDet3D pipeline 預處理"""
        results = self.pipeline(sample)

        # 提取 voxels, coordinates, num_points
        voxels = results['voxels']
        coordinates = results['coors']
        num_points = results['num_points']

        # 根據 model 需求組織輸入
        # CenterPoint 可能需要特殊的輸入格式
        return {
            'voxels': torch.from_numpy(voxels).to(self.device),
            'coordinates': torch.from_numpy(coordinates).to(self.device),
            'num_points': torch.from_numpy(num_points).to(self.device)
        }

    def get_num_samples(self) -> int:
        return len(self.data_infos)
```

### 這個混合架構的優點

1. ✅ **保持統一介面**：所有專案使用相同的 `BaseDataLoader` API
2. ✅ **重用 MMDet transforms**：直接使用訓練時的 pipeline 配置
3. ✅ **確保一致性**：preprocessing 邏輯與訓練完全相同
4. ✅ **靈活性**：特殊專案（如 CalibrationStatusClassification）可以使用自定義 transform
5. ✅ **可測試性**：容易驗證 deployment 和訓練的一致性
6. ✅ **降低維護成本**：不需要重複實現 transforms

---

## 對現有專案的影響與遷移計劃

### CalibrationStatusClassification
**狀態：已實現，無需修改**
- 當前實現已經很好，可以保持現狀
- 未來可選擇性地改用 MMPretrain pipeline（如果需要）

### CenterPoint
**狀態：需要重構**

**當前問題：**
- 使用獨立的 `DeploymentRunner`，未整合到統一 framework
- 缺少 verification 和 evaluation 功能
- 直接呼叫 `model.save_onnx()`

**建議遷移步驟：**

1. **建立 CenterPointDataLoader** (使用混合架構)
   ```python
   # projects/CenterPoint/deploy/data_loader.py
   class CenterPointDataLoader(BaseDataLoader):
       # 如上面的範例實現
   ```

2. **建立 CenterPointEvaluator**
   ```python
   # projects/CenterPoint/deploy/evaluator.py
   class Detection3DEvaluator(BaseEvaluator):
       def evaluate(self, model_path, data_loader, ...):
           # 計算 mAP, NDS 等 3D detection metrics
   ```

3. **建立 deployment config**
   ```python
   # projects/CenterPoint/deploy/deploy_config.py
   export = dict(
       mode='both',
       verify=True,
       device='cuda:0',
       work_dir='work_dirs/centerpoint_deployment'
   )

   runtime_io = dict(
       info_pkl='data/t4dataset/centerpoint_infos_val.pkl'
   )

   # ... onnx_config, backend_config, evaluation
   ```

4. **建立 main.py 使用統一 pipeline**
   ```python
   # projects/CenterPoint/deploy/main.py
   from autoware_ml.deployment.core import BaseDeploymentConfig
   from autoware_ml.deployment.exporters import ONNXExporter, TensorRTExporter

   def main():
       # Parse args
       # Load configs
       # Create data loader
       # Export & verify
       # Evaluate
   ```

### YOLOX
**狀態：從零開始**

按照上述混合架構實現：
1. `projects/YOLOX/deploy/data_loader.py` - YOLOXDataLoader
2. `projects/YOLOX/deploy/evaluator.py` - Detection2DEvaluator
3. `projects/YOLOX/deploy/deploy_config.py` - deployment config
4. `projects/YOLOX/deploy/main.py` - 統一的 deployment script

---

## 需要新增的共用工具

### 1. Pipeline Builder Utility
```python
# autoware_ml/deployment/utils/pipeline_builder.py

def build_test_pipeline(model_cfg: Config, backend: str = 'pytorch'):
    """
    Build test pipeline from model config.

    Args:
        model_cfg: Model configuration
        backend: Target backend ('pytorch', 'onnx', 'tensorrt')

    Returns:
        Pipeline for preprocessing
    """
    # 實現邏輯
```

### 2. Detection Evaluator 基礎類別
```python
# autoware_ml/deployment/core/detection_evaluator.py

class BaseDetectionEvaluator(BaseEvaluator):
    """Base class for detection evaluators with common metrics."""

    def compute_ap(self, predictions, ground_truths, iou_threshold=0.5):
        """Compute Average Precision"""
        pass

    def compute_map(self, predictions, ground_truths, iou_thresholds):
        """Compute mean Average Precision"""
        pass
```

### 3. Common Metrics
```python
# autoware_ml/deployment/metrics/detection_metrics.py

def compute_coco_metrics(predictions, ground_truths):
    """Compute COCO-style detection metrics"""
    pass

def compute_3d_detection_metrics(predictions, ground_truths):
    """Compute 3D detection metrics (mAP, NDS, etc.)"""
    pass
```

---

## 總結與建議

### ✅ 推薦做法

**採用混合架構**：
1. **保持** `BaseDataLoader` 的簡潔介面
2. **整合** MMDet/MMDet3D 的 Transform Pipeline
3. **提供** `build_test_pipeline()` 工具函數
4. **允許** 特殊專案使用自定義 transforms

### 📋 實施步驟

**Phase 1: 基礎建設**
- [ ] 實現 `pipeline_builder.py` 工具
- [ ] 實現 `BaseDetectionEvaluator`
- [ ] 新增 detection metrics 模組
- [ ] 撰寫使用文檔和範例

**Phase 2: 遷移 CenterPoint**
- [ ] 實現 `CenterPointDataLoader`
- [ ] 實現 `Detection3DEvaluator`
- [ ] 建立 deployment config
- [ ] 測試與驗證

**Phase 3: 實現 YOLOX**
- [ ] 實現 `YOLOXDataLoader`
- [ ] 實現 `Detection2DEvaluator`
- [ ] 建立 deployment config
- [ ] 測試與驗證

**Phase 4: 文檔與推廣**
- [ ] 更新 deployment README
- [ ] 新增 tutorial 文檔
- [ ] 為其他專案提供遷移指南

### ⚠️ 注意事項

1. **版本兼容性**：確保 MMDet/MMDet3D 版本一致
2. **性能考量**：MMDet pipeline 可能有額外開銷，需要測試
3. **特殊需求**：某些專案（如多模態）可能需要特殊處理
4. **向後兼容**：不要破壞現有的 CalibrationStatusClassification 實現

---

## 問題討論

**Q1: 是否所有專案都必須使用 MMDet pipeline？**
A: 不是。`BaseDataLoader` 是抽象介面，專案可以選擇：
   - 使用 MMDet pipeline（推薦，確保一致性）
   - 使用自定義 transforms（如 CalibrationStatusClassification）

**Q2: 如何處理不同的輸入格式（image, point cloud, multi-modal）？**
A: `preprocess()` 方法返回的格式由專案決定：
   - 2D Detection: `torch.Tensor` (B, C, H, W)
   - 3D Detection: `Dict[str, torch.Tensor]` (voxels, coordinates, etc.)
   - Multi-modal: `Dict` with multiple modalities

**Q3: 如何確保 deployment 和訓練完全一致？**
A:
   - 使用相同的 config 檔案
   - 使用 `build_test_pipeline()` 從 model config 提取 pipeline
   - 新增自動化測試比對訓練和部署的輸出

**Q4: 性能是否會受影響？**
A:
   - MMDet pipeline 有一些開銷（主要是 dict 操作）
   - 對於 inference，影響通常可忽略（< 1ms）
   - 可以透過 profiling 驗證
   - 如果有性能問題，可以考慮 cache 或優化

---

**作者建議：採用混合架構，既保持了設計的簡潔性，又能重用 MMDet 生態系統的 transforms，是最適合未來擴展的方案。**
