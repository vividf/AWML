# Deployment 程式碼架構建議

## 現況分析

### 當前結構
```
autoware_ml/deployment/          # 統一框架
├── core/                        # 基礎抽象類別
│   ├── base_data_loader.py      # BaseDataLoader
│   ├── base_evaluator.py       # BaseEvaluator
│   ├── base_config.py          # BaseDeploymentConfig
│   └── base_pipeline.py        # BaseDeploymentPipeline
├── exporters/                   # 匯出器（統一）
│   ├── onnx_exporter.py
│   └── tensorrt_exporter.py
└── pipelines/                   # 模型特定 pipeline
    ├── calibration/
    ├── centerpoint/
    └── yolox/

projects/{Project}/deploy/        # 專案特定實作
├── main.py                      # ❌ 大量重複的流程邏輯
├── data_loader.py               # ✅ 專案特定的資料載入
└── evaluator.py                 # ✅ 專案特定的評估邏輯
```

## 問題

1. **程式碼重複**：各專案的 `main.py` 有 80%+ 相似度
   - 參數解析
   - 配置載入
   - 模型載入
   - 匯出流程
   - 驗證流程
   - 評估流程

2. **維護困難**：修改部署流程需要在多個地方同步

3. **不一致性**：不同專案可能有不同的實作細節

## 建議方案：混合架構（Hybrid Approach）

### 原則
- ✅ **通用流程** → 移到 `autoware_ml/deployment/`
- ✅ **專案特定實作** → 保留在 `projects/{Project}/deploy/`

### 推薦結構

```
autoware_ml/deployment/
├── core/                        # 基礎抽象（保持不變）
├── exporters/                    # 匯出器（保持不變）
├── pipelines/                    # 模型特定 pipeline（保持不變）
├── runners/                      # 🆕 統一的部署執行器
│   ├── __init__.py
│   ├── base_runner.py           # 基礎執行器抽象類別
│   └── unified_runner.py        # 統一的部署流程執行器
└── utils/                        # 工具函數（保持不變）

projects/{Project}/deploy/
├── main.py                       # 🆕 簡化為薄包裝層（~50 行）
├── data_loader.py               # ✅ 保留：專案特定資料載入
├── evaluator.py                 # ✅ 保留：專案特定評估邏輯
└── config.py                    # ✅ 保留：專案特定配置（可選）
```

### 實作方式

#### 1. 建立統一的 DeploymentRunner

```python
# autoware_ml/deployment/runners/unified_runner.py

class UnifiedDeploymentRunner:
    """
    統一的部署流程執行器
    
    處理所有專案共通的部署流程：
    - 配置載入與驗證
    - 模型載入
    - 資料載入器初始化
    - 匯出（ONNX/TensorRT）
    - 跨後端驗證
    - 模型評估
    """
    
    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger
    ):
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger
    
    def run(self, checkpoint_path: str = None):
        """執行完整的部署流程"""
        # 1. 載入模型（如果需要）
        # 2. 匯出模型
        # 3. 驗證（如果啟用）
        # 4. 評估（如果啟用）
        pass
```

#### 2. 簡化專案的 main.py

```python
# projects/CenterPoint/deploy/main.py

from autoware_ml.deployment.runners import UnifiedDeploymentRunner
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.evaluator import CenterPointEvaluator

def main():
    args = parse_args()  # 專案特定的參數解析
    logger = setup_logging(args.log_level)
    
    # 載入配置
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)
    
    # 建立專案特定的組件
    data_loader = CenterPointDataLoader(...)
    evaluator = CenterPointEvaluator(...)
    
    # 使用統一執行器
    runner = UnifiedDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger
    )
    
    # 執行部署流程
    runner.run(checkpoint_path=args.checkpoint)
```

### 優點

1. **減少重複**：`main.py` 從 ~600 行減少到 ~50 行
2. **統一維護**：部署流程的修改只需在一處進行
3. **保持靈活性**：專案特定的邏輯（data_loader, evaluator）仍保留在專案目錄
4. **向後相容**：可以逐步遷移，不影響現有專案

### 遷移策略

#### 階段 1：建立統一執行器（不破壞現有功能）
- 在 `autoware_ml/deployment/runners/` 建立 `UnifiedDeploymentRunner`
- 實作通用的部署流程邏輯
- 保持現有專案的 `main.py` 不變

#### 階段 2：逐步遷移（一個專案一個專案）
- 選擇一個專案（建議從 CalibrationStatusClassification 開始，因為它已經比較統一）
- 重構 `main.py` 使用 `UnifiedDeploymentRunner`
- 測試確保功能一致
- 重複其他專案

#### 階段 3：清理與優化
- 移除重複的程式碼
- 統一錯誤處理
- 改進日誌記錄

## 替代方案比較

### 方案 A：全部移到 `autoware_ml/deployment/` ❌
**缺點：**
- 會讓 `autoware_ml/deployment/` 變成巨大的 monolith
- 專案特定邏輯與框架邏輯混在一起
- 難以找到專案特定的程式碼

### 方案 B：保持現狀 ❌
**缺點：**
- 持續的程式碼重複
- 維護成本高
- 容易出現不一致

### 方案 C：混合架構（推薦）✅
**優點：**
- 平衡了統一性與靈活性
- 清晰的職責分離
- 易於維護與擴展

## 結論

**建議採用混合架構：**
- 將通用的部署流程邏輯移到 `autoware_ml/deployment/runners/`
- 保留專案特定的實作（data_loader, evaluator）在 `projects/{Project}/deploy/`
- 簡化各專案的 `main.py` 為薄包裝層

這樣可以：
- ✅ 大幅減少程式碼重複
- ✅ 統一維護部署流程
- ✅ 保持專案特定邏輯的清晰性
- ✅ 易於擴展新專案


