# Deployment DataLoader 使用教學

本教學說明如何使用混合架構 (BaseDataLoader + MMDet Pipeline) 為新專案建立 deployment 支援。

## 目錄

1. [快速開始](#快速開始)
2. [核心概念](#核心概念)
3. [實作步驟](#實作步驟)
4. [完整範例](#完整範例)
5. [常見問題](#常見問題)

---

## 快速開始

### 為什麼使用混合架構？

**混合架構** 結合了兩個優勢：
- 🎯 **統一介面** (BaseDataLoader)：所有專案使用相同的 API
- 🔄 **重用 Pipeline** (MMDet)：直接使用訓練時的預處理邏輯

**關鍵好處**：
✅ 確保 deployment 與訓練的預處理**完全一致**  
✅ 不需要重新實作 transforms  
✅ 降低維護成本和 bug 風險  

### 適用場景

| 專案類型 | 推薦方案 | 範例 |
|---------|---------|------|
| MMDet 系列 (2D detection) | 混合架構 | YOLOX, FRNet |
| MMDet3D 系列 (3D detection) | 混合架構 | CenterPoint, BEVFusion |
| 特殊多模態 | 自定義 DataLoader | CalibrationStatusClassification |

---

## 核心概念

### 1. BaseDataLoader 介面

所有 DataLoader 必須實作 3 個核心方法：

```python
class BaseDataLoader(ABC):
    @abstractmethod
    def load_sample(self, index: int) -> Dict[str, Any]:
        """載入原始資料"""
        pass

    @abstractmethod
    def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
        """預處理資料"""
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        """取得樣本數量"""
        pass
```

### 2. 混合架構的關鍵

使用 `build_preprocessing_pipeline()` 從 model config 自動建立 MMDet pipeline：

```python
from autoware_ml.deployment.core import build_preprocessing_pipeline

# 從 model config 建立 pipeline
pipeline = build_preprocessing_pipeline(model_cfg)

# 使用 pipeline 預處理
results = pipeline(sample_data)
```

### 3. 資料流程

```
load_sample(index)
     │
     ├─> 從檔案/資料庫載入原始資料
     │   (image path, annotations, etc.)
     │
     ▼
  raw_sample (Dict)
     │
     ├─> preprocess(raw_sample)
     │        │
     │        ├─> 通過 MMDet pipeline
     │        │   (resize, normalize, pad, etc.)
     │        │
     │        ▼
     │   processed_results (Dict)
     │        │
     │        ├─> 提取 model input tensor
     │        │
     │        ▼
     ▼
  tensor (ready for inference)
```

---

## 實作步驟

### Step 1: 建立 DataLoader 檔案

建立 `projects/{PROJECT}/deploy/data_loader.py`：

```python
from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.core import build_preprocessing_pipeline
from mmengine.config import Config
import torch

class YourProjectDataLoader(BaseDataLoader):
    """DataLoader for your project."""

    def __init__(
        self,
        data_file: str,        # 資料檔案路徑
        model_cfg: Config,     # Model config
        device: str = "cpu"    # 裝置
    ):
        super().__init__(config={
            "data_file": data_file,
            "device": device
        })

        # 1. 載入資料索引
        self.data_infos = self._load_data_index(data_file)

        # 2. 建立 MMDet pipeline ⭐
        self.pipeline = build_preprocessing_pipeline(model_cfg)

        self.device = device
```

### Step 2: 實作 load_sample()

從資料檔案載入單一樣本：

```python
def load_sample(self, index: int) -> Dict[str, Any]:
    """載入原始資料（未預處理）"""

    # 取得資料資訊
    info = self.data_infos[index]

    # 轉換為 MMDet 格式
    # 這個格式必須符合 pipeline 的輸入要求
    sample = {
        'img_path': info['image_path'],       # 必要
        'img_id': info['id'],                  # 選擇性
        'instances': [                         # 選擇性（evaluation 用）
            {
                'bbox': [x, y, w, h],
                'bbox_label': class_id,
            }
            for ann in info['annotations']
        ]
    }

    return sample
```

**重要**: `load_sample()` 返回的格式必須符合 MMDet pipeline 的輸入格式。

### Step 3: 實作 preprocess()

使用 pipeline 預處理資料：

```python
def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
    """使用 MMDet pipeline 預處理"""

    # 1. 通過 pipeline
    results = self.pipeline(sample)

    # 2. 提取 model input
    # MMDet 3.x 使用 'inputs' key
    inputs = results['inputs']

    # 3. 轉換為 tensor（如果需要）
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs)

    # 4. 加上 batch dimension（如果需要）
    if inputs.ndim == 3:  # (C, H, W)
        inputs = inputs.unsqueeze(0)  # (1, C, H, W)

    # 5. 移到指定裝置
    return inputs.to(self.device)
```

### Step 4: 實作 get_num_samples()

```python
def get_num_samples(self) -> int:
    """返回總樣本數"""
    return len(self.data_infos)
```

### Step 5: （選擇性）實作 get_ground_truth()

如果需要 evaluation，實作這個方法：

```python
def get_ground_truth(self, index: int) -> Dict[str, Any]:
    """取得 ground truth（用於 evaluation）"""
    sample = self.load_sample(index)

    # 提取 annotations
    gt_bboxes = [inst['bbox'] for inst in sample['instances']]
    gt_labels = [inst['bbox_label'] for inst in sample['instances']]

    return {
        'gt_bboxes': np.array(gt_bboxes),
        'gt_labels': np.array(gt_labels)
    }
```

---

## 完整範例

### 範例 1: 2D Detection (YOLOX)

```python
# projects/YOLOX/deploy/data_loader.py

from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.core import build_preprocessing_pipeline
from pycocotools.coco import COCO
import os
import torch

class YOLOXDataLoader(BaseDataLoader):
    """YOLOX 2D object detection DataLoader"""

    def __init__(self, ann_file: str, img_prefix: str, model_cfg, device="cpu"):
        super().__init__(config={
            "ann_file": ann_file,
            "img_prefix": img_prefix,
            "device": device
        })

        # 載入 COCO annotations
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.img_prefix = img_prefix

        # 建立 pipeline ⭐
        self.pipeline = build_preprocessing_pipeline(model_cfg)
        self.device = device

    def load_sample(self, index: int):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        return {
            'img_path': os.path.join(self.img_prefix, img_info['file_name']),
            'img_id': img_id,
            'height': img_info['height'],
            'width': img_info['width'],
            'instances': [
                {
                    'bbox': ann['bbox'],
                    'bbox_label': ann['category_id'] - 1
                }
                for ann in ann_info
            ]
        }

    def preprocess(self, sample):
        results = self.pipeline(sample)
        inputs = results['inputs']

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        return inputs.to(self.device)

    def get_num_samples(self):
        return len(self.img_ids)
```

### 範例 2: 3D Detection (CenterPoint)

```python
# projects/CenterPoint/deploy/data_loader.py

from autoware_ml.deployment.core import BaseDataLoader
from autoware_ml.deployment.core import build_preprocessing_pipeline
import pickle
import torch

class CenterPointDataLoader(BaseDataLoader):
    """CenterPoint 3D detection DataLoader"""

    def __init__(self, info_file: str, model_cfg, device="cpu"):
        super().__init__(config={
            "info_file": info_file,
            "device": device
        })

        # 載入 info.pkl
        with open(info_file, 'rb') as f:
            data = pickle.load(f)
        self.data_infos = data['data_list']

        # 建立 pipeline ⭐
        self.pipeline = build_preprocessing_pipeline(model_cfg)
        self.device = device

    def load_sample(self, index: int):
        info = self.data_infos[index]

        return {
            'lidar_points': {
                'lidar_path': info['lidar_points']['lidar_path']
            },
            'sample_idx': info.get('sample_idx', index),
            # Ground truth (如果有)
            'gt_bboxes_3d': info.get('ann_info', {}).get('gt_bboxes_3d', []),
            'gt_labels_3d': info.get('ann_info', {}).get('gt_labels_3d', [])
        }

    def preprocess(self, sample):
        # 通過 pipeline (包含 voxelization)
        results = self.pipeline(sample)

        # 提取 voxel inputs
        inputs = {
            'voxels': torch.from_numpy(results['voxels']).to(self.device),
            'coors': torch.from_numpy(results['coors']).to(self.device),
            'num_points': torch.from_numpy(results['num_points']).to(self.device)
        }

        # 加上 batch index 到 coordinates
        if inputs['coors'].shape[1] == 3:
            batch_idx = torch.zeros(
                (inputs['coors'].shape[0], 1),
                dtype=inputs['coors'].dtype,
                device=self.device
            )
            inputs['coors'] = torch.cat([batch_idx, inputs['coors']], dim=1)

        return inputs

    def get_num_samples(self):
        return len(self.data_infos)
```

---

## 使用 DataLoader

### 在 deployment script 中使用

```python
# projects/YOLOX/deploy/main.py

from mmengine.config import Config
from autoware_ml.deployment.core import BaseDeploymentConfig
from .data_loader import YOLOXDataLoader

def main(args):
    # 載入 configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)

    config = BaseDeploymentConfig(deploy_cfg)

    # 建立 DataLoader
    data_loader = YOLOXDataLoader(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        model_cfg=model_cfg,
        device=config.export_config.device
    )

    # 測試載入一個樣本
    sample = data_loader.load_sample(0)
    print(f"Sample keys: {sample.keys()}")

    # 預處理
    tensor = data_loader.preprocess(sample)
    print(f"Tensor shape: {tensor.shape}")

    # 或使用便利方法
    tensor = data_loader.load_and_preprocess(0)

    # ... 用於 export, verification, evaluation
```

---

## 常見問題

### Q1: 如何確認 pipeline 格式正確？

檢查 model config 中的 `test_pipeline` 或 `test_dataloader.dataset.pipeline`：

```python
# model_config.py
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=(...))
]
```

`load_sample()` 返回的資料必須包含第一個 transform 需要的 keys。

### Q2: 如何除錯 pipeline？

```python
# 在 preprocess() 中加上 debug
def preprocess(self, sample):
    print(f"Input sample keys: {sample.keys()}")

    results = self.pipeline(sample)

    print(f"Pipeline output keys: {results.keys()}")
    print(f"Inputs shape: {results['inputs'].shape}")

    # ... 繼續處理
```

### Q3: Pipeline 輸出的格式是什麼？

MMDet 3.x 的 pipeline 通常輸出：
```python
{
    'inputs': torch.Tensor or np.ndarray,  # Model input
    'data_samples': DataSample,            # Metadata and annotations
    'img_metas': dict,                     # Image metadata
    ...
}
```

你只需要提取 `'inputs'` 即可。

### Q4: 如何處理不同的輸入格式（多模態）？

對於多模態模型（例如 BEVFusion），可以返回 dict：

```python
def preprocess(self, sample):
    results = self.pipeline(sample)

    return {
        'img': results['img_inputs'].to(self.device),
        'points': results['points'].to(self.device)
    }
```

### Q5: 能否不使用 MMDet pipeline？

可以！在 `__init__` 中設定 `use_pipeline=False`：

```python
class CustomDataLoader(BaseDataLoader):
    def __init__(self, ..., use_pipeline=True):
        if use_pipeline:
            self.pipeline = build_preprocessing_pipeline(model_cfg)
        else:
            self.pipeline = None

    def preprocess(self, sample):
        if self.pipeline:
            return self._mmdet_preprocess(sample)
        else:
            return self._custom_preprocess(sample)

    def _custom_preprocess(self, sample):
        # 自定義預處理邏輯
        ...
```

### Q6: 如何測試 DataLoader？

建立簡單的測試 script：

```python
# projects/YOLOX/deploy/test_dataloader.py

from mmengine.config import Config
from .data_loader import YOLOXDataLoader

def test_dataloader():
    # 載入 config
    model_cfg = Config.fromfile('projects/YOLOX/configs/yolox_s_8xb8-300e_coco.py')

    # 建立 DataLoader
    loader = YOLOXDataLoader(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        model_cfg=model_cfg
    )

    # 測試載入
    print(f"Total samples: {loader.get_num_samples()}")

    # 測試單一樣本
    sample = loader.load_sample(0)
    print(f"Sample keys: {sample.keys()}")

    # 測試預處理
    tensor = loader.preprocess(sample)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")

    print("✅ DataLoader test passed!")

if __name__ == '__main__':
    test_dataloader()
```

---

## 下一步

完成 DataLoader 實作後：

1. ✅ 實作 Evaluator (參考 `base_evaluator.py`)
2. ✅ 建立 deployment config
3. ✅ 建立 main.py script
4. ✅ 測試 ONNX/TensorRT export
5. ✅ 驗證與 PyTorch 一致性

參考完整範例：
- YOLOX: `projects/YOLOX/deploy/`
- CenterPoint: `projects/CenterPoint/deploy/`
- CalibrationStatusClassification: `projects/CalibrationStatusClassification/deploy/`

---

## 總結

**混合架構的關鍵點**：

1. 📦 **繼承 BaseDataLoader**
2. 🔧 **使用 `build_preprocessing_pipeline(model_cfg)`** 建立 pipeline
3. 📥 **load_sample()** 返回符合 pipeline 輸入格式的資料
4. ⚙️ **preprocess()** 使用 pipeline 處理，提取 model input
5. 🎯 **確保與訓練時的預處理完全一致**

這樣就能為任何 MMDet 系列模型快速建立 deployment 支援！🚀
