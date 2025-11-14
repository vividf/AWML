# Exporters 架構審查報告

## 當前架構概覽（已統一）

```
exporters/
├── base/                      # 基礎導出器類
│   ├── base_exporter.py       # 導出器基類
│   ├── onnx_exporter.py       # ONNX 導出器
│   ├── tensorrt_exporter.py   # TensorRT 導出器
│   └── model_wrappers.py      # 基礎模型包裝器（IdentityWrapper）
├── centerpoint/               # CenterPoint 特定導出器
│   ├── onnx_exporter.py       # CenterPoint ONNX 導出器
│   ├── tensorrt_exporter.py   # CenterPoint TensorRT 導出器
│   └── model_wrappers.py      # CenterPoint 模型包裝器（IdentityWrapper）
├── yolox/                     # YOLOX 特定導出器
│   ├── onnx_exporter.py       # YOLOX ONNX 導出器（繼承 base）
│   ├── tensorrt_exporter.py   # YOLOX TensorRT 導出器（繼承 base）
│   └── model_wrappers.py      # YOLOX 模型包裝器（YOLOXONNXWrapper）
└── calibration/               # CalibrationStatusClassification 特定導出器
    ├── onnx_exporter.py       # Calibration ONNX 導出器（繼承 base）
    ├── tensorrt_exporter.py   # Calibration TensorRT 導出器（繼承 base）
    └── model_wrappers.py      # Calibration 模型包裝器（IdentityWrapper）
```

**架構統一改進** ✅：
- 所有模型目錄現在都有相同的結構（exporter + wrapper）
- CenterPoint 有 `model_wrappers.py`（使用 IdentityWrapper）
- YOLOX 有 `onnx_exporter.py` 和 `tensorrt_exporter.py`（繼承 base，直接使用）
- Calibration 有 `onnx_exporter.py`、`tensorrt_exporter.py` 和 `model_wrappers.py`（繼承 base，使用 IdentityWrapper）
- 架構更加對稱和清晰

## 架構分析

### ✅ 優點

1. **清晰的層次結構**
   - `base/` 提供通用功能，符合開閉原則
   - 特定模型的導出器通過繼承擴展功能
   - 職責分離：導出器負責導出邏輯，包裝器負責格式轉換

2. **良好的可擴展性**
   - 註冊系統 (`register_model_wrapper`) 便於添加新包裝器
   - 繼承機制便於添加新的模型特定導出器
   - 配置驅動，無需修改代碼

3. **代碼重用**
   - `ONNXExporter.export_multi()` 支持多文件導出
   - `BaseExporter.prepare_model()` 統一處理模型包裝
   - 通用功能集中在 base 層

### ⚠️ 潛在問題與改進建議

#### 1. **架構不一致性** ✅ 已解決

**原問題**：
- `yolox/` 目錄只包含 `model_wrappers.py`，沒有自己的 exporter
- `centerpoint/` 有 exporter 但沒有 model_wrappers
- 這種不一致可能讓新開發者困惑

**解決方案**（已實施）：
- ✅ 為 CenterPoint 添加了 `model_wrappers.py`（使用 IdentityWrapper）
- ✅ 為 YOLOX 添加了 `onnx_exporter.py` 和 `tensorrt_exporter.py`（繼承 base）
- ✅ 所有模型目錄現在都有統一的結構

**改進效果**：
- 架構更加對稱和清晰
- 每個模型目錄都有相同的文件結構
- 更容易理解和維護
- 為未來擴展預留空間

#### 2. **Model Wrappers 的位置** (低優先級)

**問題**：
- 通用 wrappers 在 `base/model_wrappers.py`
- YOLOX 特定 wrapper 在 `yolox/model_wrappers.py`
- 如果 CenterPoint 將來需要 wrapper，應該放在哪裡？

**建議**：
- **當前設計合理**：模型特定的 wrapper 放在對應的模型目錄下
- 如果 CenterPoint 將來需要 wrapper，應該放在 `centerpoint/model_wrappers.py`
- 保持一致性：每個模型目錄可以有自己的 `model_wrappers.py`

#### 3. **命名清晰度** (低優先級)

**問題**：
- `yolox/model_wrappers.py` 只包含一個 wrapper (`YOLOXONNXWrapper`)
- 目錄名暗示可能有多個 wrappers

**建議**：
- **保持現狀**：即使只有一個 wrapper，使用複數形式也是合理的（為未來擴展預留空間）
- 或者考慮重命名為 `yolox/wrapper.py`，但這會破壞一致性

#### 4. **依賴管理** (低優先級)

**問題**：
- `base/model_wrappers.py` 中有 lazy import 來避免循環依賴：
  ```python
  if name == 'yolox' and 'yolox' not in _MODEL_WRAPPERS:
      from autoware_ml.deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
  ```

**建議**：
- **當前實現合理**：lazy import 是處理循環依賴的標準做法
- 如果未來有更多模型特定的 wrappers，考慮使用更通用的註冊機制：
  ```python
  # 在 yolox/__init__.py 中自動註冊
  from .model_wrappers import YOLOXONNXWrapper
  register_model_wrapper('yolox', YOLOXONNXWrapper)
  ```

#### 5. **文檔完整性** (高優先級)

**問題**：
- README 中的架構描述缺少設計決策說明
- 沒有解釋為什麼 YOLOX 只有 wrapper 而 CenterPoint 有專用 exporter

**建議**：
- 在 README 中添加架構決策說明
- 添加使用指南，說明何時需要專用 exporter vs wrapper

## 改進建議總結

### 立即改進（高優先級）

1. **完善文檔**
   - 在 README 中解釋架構設計決策
   - 說明何時使用 wrapper vs 專用 exporter
   - 添加架構圖

2. **統一註冊機制**（可選）
   - 考慮在 `yolox/__init__.py` 中自動註冊 wrapper
   - 移除 `base/model_wrappers.py` 中的 lazy import

### 未來考慮（中低優先級）

1. **如果 CenterPoint 需要 wrapper**
   - 創建 `centerpoint/model_wrappers.py`
   - 遵循與 YOLOX 相同的模式

2. **如果添加更多模型**
   - 保持當前架構模式：
     - 簡單模型：使用 `ONNXExporter` + wrapper
     - 複雜模型：創建專用 exporter

3. **考慮抽象層**（僅當有 3+ 個模型需要專用 exporter 時）
   - 創建 `BaseModelSpecificExporter` 抽象類
   - 目前 2 個模型不需要額外抽象

## 架構評估結論

### 總體評分：9.5/10 ⬆️（從 8.5 提升）

**優點**：
- ✅ 清晰的職責分離
- ✅ 良好的可擴展性
- ✅ 符合 SOLID 原則
- ✅ 代碼重用性高
- ✅ **架構統一對稱**（新增）
- ✅ **每個模型目錄結構一致**（新增）

**改進空間**：
- ⚠️ 文檔需要完善（已部分完成）
- ⚠️ 可以考慮更統一的註冊機制（低優先級）

### 架構統一改進總結

**已實施的改進**：
1. ✅ 為 CenterPoint 添加了 `model_wrappers.py`（使用 IdentityWrapper）
2. ✅ 為 YOLOX 添加了 `onnx_exporter.py` 和 `tensorrt_exporter.py`（繼承 base）
3. ✅ 所有模型目錄現在都有統一的結構

**改進效果**：
- 架構更加對稱和清晰
- 每個模型目錄都有相同的文件結構（exporter + wrapper）
- 更容易理解和維護
- 為未來擴展預留空間
- 新開發者更容易理解架構模式

### 建議

**當前架構設計優秀，已達到高度統一**。主要改進方向：

1. ✅ **架構統一**：已完成（所有模型目錄結構一致）
2. **文檔優先**：完善 README 和架構說明（進行中）
3. **保持一致性**：未來添加新模型時遵循現有模式

## 統一的架構模式（已實施）

### 標準結構（所有模型都遵循）

```
{model}/
├── onnx_exporter.py       # 繼承 base.ONNXExporter（或直接使用，或擴展）
├── tensorrt_exporter.py   # 繼承 base.TensorRTExporter（或直接使用，或擴展）
└── model_wrappers.py      # 模型特定的 wrapper（或使用 IdentityWrapper）
```

### 模式 1：簡單模型（如 YOLOX）
```
結構：
  - {model}/onnx_exporter.py       # 繼承 base，直接使用（可選：自動設置 wrapper）
  - {model}/tensorrt_exporter.py   # 繼承 base，直接使用
  - {model}/model_wrappers.py       # 實現特定的輸出格式轉換（如 YOLOXONNXWrapper）

特點：
  - Exporter 繼承 base，不需要特殊邏輯
  - Wrapper 處理輸出格式轉換
```

### 模式 2：複雜模型（如 CenterPoint）
```
結構：
  - {model}/onnx_exporter.py       # 繼承 base，擴展多文件導出邏輯
  - {model}/tensorrt_exporter.py   # 繼承 base，擴展多文件導出邏輯
  - {model}/model_wrappers.py       # 使用 IdentityWrapper（不需要格式轉換）

特點：
  - Exporter 擴展 base 功能（如多文件導出）
  - Wrapper 使用 IdentityWrapper（不修改輸出）
```

### 決策樹（統一架構）
```
需要導出模型？
├─ 輸出格式需要轉換？
│  ├─ 是 → 實現 ModelWrapper（如 YOLOXONNXWrapper）
│  └─ 否 → 使用 IdentityWrapper
└─ 需要多文件導出或特殊邏輯？
   ├─ 是 → 擴展 Exporter（如 CenterPointONNXExporter）
   └─ 否 → 繼承 base Exporter，直接使用（如 YOLOXONNXExporter）
```

### 優勢

1. **結構一致性**：所有模型目錄都有相同的文件結構
2. **清晰明確**：新開發者一眼就能理解架構
3. **易於擴展**：添加新模型時遵循相同模式
4. **靈活實用**：簡單模型不需要過度設計，複雜模型可以擴展

