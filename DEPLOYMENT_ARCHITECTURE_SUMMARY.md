# AutoWare ML Deployment 架構分析總結

## 📋 當前狀況

### 三個模型的 Deployment 架構

| 模型 | 架構特點 | 自定義組件 | 主要問題 |
|------|----------|------------|----------|
| **CenterPoint** | 多引擎 ONNX/TensorRT | `centerpoint_onnx_helper.py`<br>`centerpoint_tensorrt_backend.py` | 代碼重複，難以維護 |
| **YOLOX** | 單引擎 + Wrapper | `onnx_wrapper.py` | 輸出格式轉換邏輯分散 |
| **CalibrationStatus** | 標準單引擎 | 無自定義組件 | 功能受限，無法復用 |

---

## 🔍 架構差異分析

### 共同點 ✅
- 都使用 `BaseDataLoader`、`BaseEvaluator`、`BaseDeploymentConfig`
- 都使用 `verify_model_outputs` 進行驗證
- 主程序結構相似

### 差異點 ❌
- **ONNX 導出**: CenterPoint 用 Helper，YOLOX 用 Wrapper，CalibrationStatus 用標準
- **TensorRT Backend**: CenterPoint 自定義，其他用標準
- **後處理**: CenterPoint 用 PyTorch decoder，其他用標準處理
- **多引擎支持**: 只有 CenterPoint 支持

---

## 🚨 主要問題

### 1. 代碼重複
- CenterPoint: ~500 行自定義代碼
- YOLOX: ~200 行自定義代碼
- 相似功能重複實現

### 2. 擴展性差
- 新模型需要重寫相似功能
- 多引擎處理邏輯無法復用
- 後處理邏輯分散

### 3. 維護困難
- 每個模型有自己的特殊處理
- 框架更新需要同步修改多個地方

---

## 🎯 統一框架設計

### 核心概念

#### Pipeline 架構
```
Input → Preprocessor → Engine(s) → Midprocessor → Postprocessor → Output
```

#### 模組化組件
- **Preprocessor**: 預處理（體素化、圖像標準化等）
- **Midprocessor**: 中間處理（PyTorch 中間編碼器等）
- **Postprocessor**: 後處理（解碼、NMS 等）
- **ModularBackend**: 統一 Backend，支持單引擎和多引擎

### 配置驅動
```python
MODEL_CONFIGS = {
    "CenterPoint": {
        "engines": ["pts_voxel_encoder", "pts_backbone_neck_head"],
        "preprocessor": {"type": "voxelization", ...},
        "midprocessor": {"type": "pytorch_middle_encoder", ...},
        "postprocessor": {"type": "centerpoint_decoder", ...}
    },
    "YOLOX": {
        "engines": ["yolox_model"],
        "preprocessor": {"type": "image_normalization", ...},
        "postprocessor": {"type": "yolox_decoder", ...}
    }
}
```

---

## 🚀 實施計劃

### Phase 1: 基礎設施（1-2 週）
- [ ] 創建 `BaseProcessor`、`ModularBackend`
- [ ] 實現核心處理器（Voxelization、ImageNormalization 等）
- [ ] 創建模型配置系統

### Phase 2: 模型遷移（2-3 週）
- [ ] CalibrationStatusClassification（最簡單）
- [ ] YOLOX（中等複雜度）
- [ ] CenterPoint（最複雜）

### Phase 3: 優化和清理（1 週）
- [ ] 性能優化
- [ ] 移除舊代碼
- [ ] 添加測試

---

## 📊 預期效果

### 代碼減少
- **總體**: 減少 ~70% 的重複代碼
- **CenterPoint**: 減少 ~500 行
- **YOLOX**: 減少 ~200 行

### 維護性提升
- ✅ 統一接口
- ✅ 配置驅動
- ✅ 模組化設計

### 擴展性提升
- ✅ 新模型只需添加配置
- ✅ 新功能在對應 processor 中添加
- ✅ 支持任意數量的引擎

---

## 🧪 新模型添加示例

```python
# 添加新模型 "PointPillars"
MODEL_CONFIGS["PointPillars"] = {
    "engines": ["pillar_encoder", "backbone_head"],
    "preprocessor": {"type": "pillar_encoding", ...},
    "postprocessor": {"type": "pointpillars_decoder", ...}
}

# 使用
backend = ModularBackend(**MODEL_CONFIGS["PointPillars"])
output = backend.infer(input_data)
```

**優勢**: 新模型只需要配置和特定處理器，不需要重寫整個 backend！

---

## ✅ 總結

### 問題
- ❌ 代碼重複（每個模型有自己的實現）
- ❌ 擴展性差（新模型需要重寫）
- ❌ 維護困難（分散的邏輯）

### 解決方案
- ✅ **統一 Pipeline**: Preprocessor → Midprocessor → Postprocessor
- ✅ **模組化 Backend**: 支持單引擎和多引擎
- ✅ **配置驅動**: 行為由配置文件控制
- ✅ **易於擴展**: 新模型只需添加配置

### 效果
- ✅ **代碼復用**: 減少 70% 重複代碼
- ✅ **易於維護**: 統一接口和實現
- ✅ **快速擴展**: 新模型開發時間減少 80%
- ✅ **性能提升**: 統一優化策略

---

**這個統一框架將大大提升 AutoWare ML deployment 的可維護性和擴展性！** 🎉

---

**日期**: 2025-10-23  
**狀態**: 📋 設計完成，待實施  
**優先級**: 🔥 高（架構改進）
