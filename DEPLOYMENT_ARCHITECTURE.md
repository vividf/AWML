# Deployment 架構文檔

## 目錄
- [概述](#概述)
- [整體架構](#整體架構)
- [核心組件](#核心組件)
- [專案實作](#專案實作)
- [部署流程](#部署流程)
- [檔案結構](#檔案結構)
- [各檔案功能說明](#各檔案功能說明)

---

## 概述

AWML Deployment Framework 是一個統一的、任務無關的部署框架，用於將訓練好的 PyTorch 模型匯出到 ONNX 和 TensorRT，並進行跨後端的驗證與評估。框架採用分層架構設計，將通用邏輯與專案特定實作分離，提供可擴展且易於維護的部署解決方案。

### 主要特性
- **統一執行器**：`DeploymentRunner` 協調匯出 → 驗證 → 評估流程
- **配置驅動**：基於 `mmengine` Config 的配置系統
- **後端匯出器**：統一的 ONNX 和 TensorRT 匯出介面
- **任務管道**：支援分類、2D 檢測、3D 檢測等任務
- **跨後端驗證**：可選的數值一致性檢查
- **效能評估**：計算準確度指標與延遲統計

---

## 整體架構

### 架構層次

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                           │
│  - CenterPoint, YOLOX-ELAN, Calibration                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│           DeploymentRunner (統一執行器)                  │
│  - 協調整個部署流程                                      │
│  - 管理模型載入、匯出、驗證、評估                        │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────────┐
│  Exporters  │ │  Pipelines │ │  Core Components   │
│             │ │            │ │                     │
│ - ONNX      │ │ - PyTorch  │ │ - BaseDataLoader   │
│ - TensorRT  │ │ - ONNX     │ │ - BaseEvaluator    │
│             │ │ - TensorRT │ │ - BaseConfig       │
└─────────────┘ └────────────┘ └─────────────────────┘
```

### 設計原則

1. **關注點分離**：通用邏輯在 `autoware_ml/deployment/`，專案特定實作在 `projects/*/deploy/`
2. **依賴注入**：透過回調函數允許專案自訂行為
3. **抽象基類**：定義清晰的介面，確保一致性
4. **配置驅動**：所有行為由配置檔案控制

---

## 核心組件

### 1. `autoware_ml/deployment/core/`

#### `base_config.py`
**功能**：部署配置的基礎類別
- `BaseDeploymentConfig`：主要配置容器
  - 管理 `export_config`、`runtime_config`、`backend_config`
  - 提供 ONNX 和 TensorRT 設定提取方法
  - 驗證配置有效性
- `ExportConfig`：匯出設定（模式、裝置、工作目錄）
- `RuntimeConfig`：運行時 I/O 設定
- `BackendConfig`：後端特定設定（精度策略、工作空間大小）

#### `base_data_loader.py`
**功能**：資料載入器抽象基類
- `BaseDataLoader`：定義資料載入介面
  - `load_sample(index)`：載入單一樣本
  - `preprocess(sample)`：預處理樣本資料
  - `get_num_samples()`：取得樣本總數
- 各專案實作：`CenterPointDataLoader`、`YOLOXOptElanDataLoader`、`CalibrationDataLoader`

#### `base_evaluator.py`
**功能**：評估器抽象基類
- `BaseEvaluator`：定義評估介面
  - `evaluate()`：執行完整評估
  - `verify()`：跨後端驗證（可選）
  - `print_results()`：格式化輸出結果
  - `compute_latency_stats()`：計算延遲統計
- 各專案實作：`CenterPointEvaluator`、`YOLOXOptElanEvaluator`、`ClassificationEvaluator`

#### `base_pipeline.py`
**功能**：部署管道抽象基類
- `BaseDeploymentPipeline`：統一的推理介面
  - `preprocess()`：預處理（抽象方法）
  - `run_model()`：模型推理（抽象方法，後端特定）
  - `postprocess()`：後處理（抽象方法）
  - `infer()`：完整推理流程（preprocess → run_model → postprocess）
  - `warmup()`、`benchmark()`：效能測試工具

#### `detection_2d_pipeline.py` / `detection_3d_pipeline.py` / `classification_pipeline.py`
**功能**：任務特定的管道基類
- 繼承自 `BaseDeploymentPipeline`
- 提供任務特定的預處理/後處理邏輯
- 例如：`Detection3DPipeline` 處理點雲資料、3D 檢測後處理

#### `preprocessing_builder.py`
**功能**：預處理管道建構器
- 根據任務類型建構標準化預處理流程

### 2. `autoware_ml/deployment/exporters/`

#### `base_exporter.py`
**功能**：匯出器抽象基類
- `BaseExporter`：定義匯出介面
  - `export()`：執行匯出（抽象方法）
  - `validate_export()`：驗證匯出結果

#### `onnx_exporter.py`
**功能**：ONNX 匯出器
- `ONNXExporter`：實作 PyTorch → ONNX 匯出
  - 使用 `torch.onnx.export()`
  - 支援動態批次大小
  - 處理多輸入/多輸出
  - 可配置 opset 版本、常數摺疊等

#### `tensorrt_exporter.py`
**功能**：TensorRT 匯出器
- `TensorRTExporter`：實作 ONNX → TensorRT 匯出
  - 使用 TensorRT Python API
  - 支援精度策略（FP32、FP16、INT8）
  - 支援動態形狀
  - 可配置工作空間大小

### 3. `autoware_ml/deployment/pipelines/`

**功能**：模型特定的管道實作

#### `centerpoint/`
- `centerpoint_pipeline.py`：CenterPoint 管道基類
  - 共享預處理（體素化）
  - 共享後處理（預測解析）
  - 抽象方法：`run_voxel_encoder()`、`run_backbone_neck_head()`
- `centerpoint_pytorch.py`：PyTorch 後端實作
- `centerpoint_onnx.py`：ONNX 後端實作（使用兩個 ONNX 模型）
- `centerpoint_tensorrt.py`：TensorRT 後端實作（使用兩個 TensorRT 引擎）

#### `yolox/`
- `yolox_pipeline.py`：YOLOX 管道基類
- `yolox_pytorch.py`、`yolox_onnx.py`、`yolox_tensorrt.py`：各後端實作

#### `calibration/`
- `calibration_pipeline.py`：校準分類管道基類
- `calibration_pytorch.py`、`calibration_onnx.py`、`calibration_tensorrt.py`：各後端實作

### 4. `autoware_ml/deployment/runners/`

#### `deployment_runner.py`
**功能**：統一部署執行器
- `DeploymentRunner`：協調整個部署流程
  - `load_pytorch_model()`：載入 PyTorch 模型（可自訂）
  - `export_onnx()`：匯出 ONNX（可自訂）
  - `export_tensorrt()`：匯出 TensorRT（可自訂）
  - `run_verification()`：執行跨後端驗證
  - `run_evaluation()`：執行模型評估
  - `run()`：執行完整部署流程

**設計特點**：
- 支援透過回調函數自訂行為
- 可繼承並覆寫方法（如 `CenterPointDeploymentRunner`）
- 自動處理配置驗證、錯誤處理、日誌記錄

---

## 專案實作

### 1. CenterPoint (`projects/CenterPoint/deploy/`)

#### `main.py`
**功能**：CenterPoint 部署入口
- 解析命令列參數（包括 `--replace-onnx-models`、`--rot-y-axis-reference`）
- 建立 `CenterPointDeploymentRunner`
- 執行部署流程

**特殊處理**：
- 自訂模型載入：可替換為 ONNX 相容模型
- 自訂 ONNX 匯出：使用模型的 `save_onnx()` 方法（產生兩個 ONNX 檔案）
- 自訂 TensorRT 匯出：將兩個 ONNX 檔案分別轉換為 TensorRT 引擎
- 自訂評估：根據後端使用不同的模型配置

#### `data_loader.py`
**功能**：CenterPoint 資料載入器
- `CenterPointDataLoader`：載入點雲資料
  - 從 info 檔案讀取樣本
  - 預處理點雲（體素化準備）
  - 支援 T4Dataset 格式

#### `evaluator.py`
**功能**：CenterPoint 評估器
- `CenterPointEvaluator`：3D 檢測評估
  - 使用 CenterPoint 管道進行推理
  - 計算檢測數量、延遲統計
  - 支援跨後端驗證（比較原始輸出）

### 2. YOLOX-ELAN (`projects/YOLOX_opt_elan/deploy/`)

#### `main.py`
**功能**：YOLOX-ELAN 部署入口
- 使用標準 `DeploymentRunner`
- 自訂 ONNX 匯出：使用 `YOLOXONNXWrapper` 包裝模型
- 替換 ReLU6 為 ReLU（ONNX 相容性）

#### `data_loader.py`
**功能**：YOLOX 資料載入器
- `YOLOXOptElanDataLoader`：載入影像資料
  - 從 COCO 格式註解檔案讀取
  - 影像預處理（resize、normalize）

#### `evaluator.py`
**功能**：YOLOX 評估器
- `YOLOXOptElanEvaluator`：2D 檢測評估
  - 使用 YOLOX 管道進行推理
  - 計算 mAP、延遲統計

#### `onnx_wrapper.py`
**功能**：ONNX 匯出包裝器
- `YOLOXONNXWrapper`：包裝 YOLOX 模型以符合 ONNX 匯出需求

### 3. Calibration (`projects/CalibrationStatusClassification/deploy/`)

#### `main.py`
**功能**：校準分類部署入口
- 最簡單的實作，使用標準 `DeploymentRunner`
- 僅自訂模型載入函數

#### `data_loader.py`
**功能**：校準資料載入器
- `CalibrationDataLoader`：載入校準狀態資料
  - 從 pickle 檔案讀取
  - 支援誤校準機率設定

#### `evaluator.py`
**功能**：分類評估器
- `ClassificationEvaluator`：分類任務評估
  - 計算準確度、混淆矩陣
  - 延遲統計

---

## 部署流程

### 完整流程圖

```
┌─────────────────────────────────────────────────────────────┐
│  1. 初始化階段                                                │
│     - 解析命令列參數                                         │
│     - 載入配置檔案 (deploy_cfg, model_cfg)                   │
│     - 建立 BaseDeploymentConfig                             │
│     - 建立 DataLoader 和 Evaluator                          │
│     - 建立 DeploymentRunner                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  2. 模型載入階段 (如果需要匯出或評估)                        │
│     - 呼叫 load_pytorch_model()                             │
│     - 專案特定：可能替換模型組件（如 CenterPoint）          │
│     - 載入權重到指定裝置                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  3. ONNX 匯出階段 (如果 export.mode 包含 'onnx')            │
│     - 呼叫 export_onnx()                                    │
│     - 取得樣本輸入                                           │
│     - 使用 ONNXExporter 或自訂函數                           │
│     - 儲存 ONNX 模型                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  4. TensorRT 匯出階段 (如果 export.mode 包含 'trt')         │
│     - 呼叫 export_tensorrt()                                │
│     - 使用 TensorRTExporter 或自訂函數                      │
│     - 從 ONNX 轉換為 TensorRT 引擎                         │
│     - 儲存 TensorRT 引擎                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  5. 驗證階段 (如果 export.verify = True)                    │
│     - 呼叫 run_verification()                               │
│     - 載入 PyTorch、ONNX、TensorRT 管道                     │
│     - 對多個樣本執行推理                                     │
│     - 比較原始輸出（return_raw_outputs=True）               │
│     - 報告通過/失敗統計                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  6. 評估階段 (如果 evaluation.enabled = True)               │
│     - 呼叫 run_evaluation()                                │
│     - 對每個後端（PyTorch/ONNX/TensorRT）執行評估            │
│     - 使用 Evaluator.evaluate()                             │
│     - 計算任務特定指標（mAP、準確度等）                      │
│     - 計算延遲統計                                           │
│     - 輸出跨後端比較報告                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│  7. 結果彙總                                                  │
│     - 彙整匯出路徑                                          │
│     - 彙整驗證結果                                          │
│     - 彙整評估結果                                          │
│     - 輸出最終報告                                          │
└─────────────────────────────────────────────────────────────┘
```

### 流程細節

#### 階段 1：初始化
```python
# 解析參數
args = parse_args()

# 載入配置
deploy_cfg = Config.fromfile(args.deploy_cfg)
model_cfg = Config.fromfile(args.model_cfg)
config = BaseDeploymentConfig(deploy_cfg)

# 建立組件
data_loader = ProjectDataLoader(...)
evaluator = ProjectEvaluator(...)
runner = DeploymentRunner(...)
```

#### 階段 2：模型載入
```python
# 標準流程
pytorch_model = runner.load_pytorch_model(checkpoint_path)

# CenterPoint 特殊處理
if replace_onnx_models:
    # 替換模型組件為 ONNX 相容版本
    model_config.type = "CenterPointONNX"
    model_config.pts_voxel_encoder.type = "PillarFeatureNetONNX"
    ...
```

#### 階段 3：ONNX 匯出
```python
# 標準流程（使用 ONNXExporter）
exporter = ONNXExporter(onnx_settings, logger)
exporter.export(pytorch_model, input_tensor, output_path)

# CenterPoint 特殊處理（使用模型內建方法）
pytorch_model.save_onnx(save_dir=output_dir, ...)
# 產生：pts_voxel_encoder.onnx, pts_backbone_neck_head.onnx

# YOLOX 特殊處理（使用包裝器）
wrapped_model = YOLOXONNXWrapper(model, num_classes)
exporter.export(wrapped_model, input_tensor, output_path)
```

#### 階段 4：TensorRT 匯出
```python
# 標準流程（使用 TensorRTExporter）
exporter = TensorRTExporter(trt_settings, logger)
exporter.export(model=None, sample_input=input, 
                output_path=trt_path, onnx_path=onnx_path)

# CenterPoint 特殊處理（多檔案）
for onnx_file, trt_file in onnx_files:
    exporter.export(..., onnx_path=onnx_file_path, 
                    output_path=trt_path)
```

#### 階段 5：驗證
```python
# 建立各後端管道
pytorch_pipeline = create_pipeline("pytorch", ...)
onnx_pipeline = create_pipeline("onnx", ...)
tensorrt_pipeline = create_pipeline("tensorrt", ...)

# 對每個樣本執行推理並比較
for sample in samples:
    pytorch_output, _, _ = pytorch_pipeline.infer(
        sample, return_raw_outputs=True
    )
    onnx_output, _, _ = onnx_pipeline.infer(
        sample, return_raw_outputs=True
    )
    
    # 比較輸出
    diff = compute_difference(pytorch_output, onnx_output)
    passed = diff < tolerance
```

#### 階段 6：評估
```python
# 對每個後端執行評估
for backend, model_path in models_to_evaluate:
    results = evaluator.evaluate(
        model_path=model_path,
        data_loader=data_loader,
        num_samples=num_samples,
        backend=backend,
        device=device,
    )
    # 結果包含：準確度、mAP、延遲統計等
```

---

## 檔案結構

### 統一框架 (`autoware_ml/deployment/`)

```
autoware_ml/deployment/
├── __init__.py
├── README.md                          # 框架使用說明
│
├── core/                              # 核心抽象類別
│   ├── __init__.py
│   ├── base_config.py                 # 配置管理
│   ├── base_data_loader.py            # 資料載入器基類
│   ├── base_evaluator.py              # 評估器基類
│   ├── base_pipeline.py               # 管道基類
│   ├── classification_pipeline.py     # 分類管道
│   ├── detection_2d_pipeline.py       # 2D 檢測管道
│   ├── detection_3d_pipeline.py       # 3D 檢測管道
│   └── preprocessing_builder.py      # 預處理建構器
│
├── exporters/                         # 匯出器
│   ├── __init__.py
│   ├── base_exporter.py               # 匯出器基類
│   ├── onnx_exporter.py               # ONNX 匯出器
│   └── tensorrt_exporter.py           # TensorRT 匯出器
│
├── pipelines/                         # 模型特定管道
│   ├── __init__.py
│   ├── calibration/
│   │   ├── calibration_pipeline.py
│   │   ├── calibration_pytorch.py
│   │   ├── calibration_onnx.py
│   │   └── calibration_tensorrt.py
│   ├── centerpoint/
│   │   ├── centerpoint_pipeline.py
│   │   ├── centerpoint_pytorch.py
│   │   ├── centerpoint_onnx.py
│   │   └── centerpoint_tensorrt.py
│   └── yolox/
│       ├── yolox_pipeline.py
│       ├── yolox_pytorch.py
│       ├── yolox_onnx.py
│       └── yolox_tensorrt.py
│
└── runners/                           # 執行器
    ├── __init__.py
    └── deployment_runner.py           # 統一部署執行器
```

### 專案實作 (`projects/*/deploy/`)

#### CenterPoint
```
projects/CenterPoint/deploy/
├── __init__.py
├── main.py                            # 部署入口（自訂 Runner）
├── data_loader.py                     # CenterPointDataLoader
└── evaluator.py                       # CenterPointEvaluator
```

#### YOLOX-ELAN
```
projects/YOLOX_opt_elan/deploy/
├── __init__.py
├── main.py                            # 部署入口（標準 Runner）
├── data_loader.py                     # YOLOXOptElanDataLoader
├── evaluator.py                       # YOLOXOptElanEvaluator
├── onnx_wrapper.py                    # YOLOXONNXWrapper
├── analyze_onnx.py                    # ONNX 分析工具
└── verify_onnx.py                     # ONNX 驗證工具
```

#### Calibration
```
projects/CalibrationStatusClassification/deploy/
├── __init__.py
├── main.py                            # 部署入口（標準 Runner）
├── data_loader.py                     # CalibrationDataLoader
└── evaluator.py                       # ClassificationEvaluator
```

---

## 各檔案功能說明

### 核心檔案

| 檔案 | 功能 | 關鍵類別/方法 |
|------|------|--------------|
| `core/base_config.py` | 配置管理 | `BaseDeploymentConfig`, `ExportConfig`, `RuntimeConfig`, `BackendConfig` |
| `core/base_data_loader.py` | 資料載入介面 | `BaseDataLoader` (抽象) |
| `core/base_evaluator.py` | 評估介面 | `BaseEvaluator` (抽象) |
| `core/base_pipeline.py` | 管道基類 | `BaseDeploymentPipeline` (抽象) |
| `exporters/onnx_exporter.py` | ONNX 匯出 | `ONNXExporter.export()` |
| `exporters/tensorrt_exporter.py` | TensorRT 匯出 | `TensorRTExporter.export()` |
| `runners/deployment_runner.py` | 統一執行器 | `DeploymentRunner.run()` |

### 專案特定檔案

#### CenterPoint
| 檔案 | 功能 | 關鍵類別/方法 |
|------|------|--------------|
| `main.py` | 部署入口 | `CenterPointDeploymentRunner`, `load_pytorch_model()`, `export_onnx()`, `export_tensorrt()` |
| `data_loader.py` | 點雲資料載入 | `CenterPointDataLoader.load_sample()`, `preprocess()` |
| `evaluator.py` | 3D 檢測評估 | `CenterPointEvaluator.evaluate()`, `verify()` |

#### YOLOX-ELAN
| 檔案 | 功能 | 關鍵類別/方法 |
|------|------|--------------|
| `main.py` | 部署入口 | `load_pytorch_model()`, `export_onnx()` |
| `data_loader.py` | 影像資料載入 | `YOLOXOptElanDataLoader.load_sample()`, `preprocess()` |
| `evaluator.py` | 2D 檢測評估 | `YOLOXOptElanEvaluator.evaluate()` |
| `onnx_wrapper.py` | ONNX 匯出包裝 | `YOLOXONNXWrapper` |

#### Calibration
| 檔案 | 功能 | 關鍵類別/方法 |
|------|------|--------------|
| `main.py` | 部署入口 | `load_pytorch_model()` |
| `data_loader.py` | 校準資料載入 | `CalibrationDataLoader.load_sample()`, `preprocess()` |
| `evaluator.py` | 分類評估 | `ClassificationEvaluator.evaluate()` |

---

## 使用範例

### CenterPoint 部署
```bash
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/configs/deploy_config.py \
    projects/CenterPoint/configs/model_config.py \
    checkpoints/centerpoint.pth \
    --work-dir work_dirs/centerpoint_deploy \
    --replace-onnx-models
```

### YOLOX-ELAN 部署
```bash
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/configs/deploy_config.py \
    projects/YOLOX_opt_elan/configs/model_config.py \
    checkpoints/yolox.pth \
    --work-dir work_dirs/yolox_deploy
```

### Calibration 部署
```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/deploy/configs/deploy_config.py \
    projects/CalibrationStatusClassification/configs/model_config.py \
    checkpoints/calibration.pth \
    --work-dir work_dirs/calibration_deploy
```

---

## 總結

AWML Deployment Framework 提供了一個統一且可擴展的部署解決方案，透過分層架構和抽象介面，實現了：

1. **程式碼重用**：通用邏輯集中在框架中
2. **專案靈活性**：專案可自訂特定行為
3. **一致性**：所有專案遵循相同的介面
4. **可維護性**：清晰的架構便於維護和擴展

框架支援多種任務類型（分類、2D 檢測、3D 檢測）和多種後端（PyTorch、ONNX、TensorRT），為 AWML 專案提供了強大的部署能力。

