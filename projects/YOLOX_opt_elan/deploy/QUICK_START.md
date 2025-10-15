# YOLOX_opt_elan 快速開始指南

## 最簡單的使用方式

### 1. 準備數據

確保你有 T4Dataset 格式的標註文件：
```bash
data/t4dataset/samrat/yolox_infos_val.json
```

### 2. 修改配置

編輯 `deploy_config.py`，確認路徑正確：
```python
runtime_io = dict(
    ann_file="data/t4dataset/samrat/yolox_infos_val.json",
    img_prefix="",  # 如果圖片路徑在標註中是完整路徑
)
```

### 3. 運行部署

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 完整部署（導出 + 評估）
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py \
    /path/to/your/checkpoint.pth \
    --work-dir work_dirs/yolox_opt_elan_deployment
```

### 4. 查看結果

```bash
work_dirs/yolox_opt_elan_deployment/
├── yolox_opt_elan.onnx       # ONNX 模型
└── yolox_opt_elan.engine     # TensorRT 引擎
```

終端會顯示完整的評估結果：
- mAP 指標
- 每個類別的 AP
- 延遲統計
- 跨後端比較

## 與舊方法的對比

### 舊方法（不推薦）

```bash
# 複雜且功能有限
python projects/YOLOX_opt_elan/scripts/deploy_yolox_s_opt.py \
    checkpoint.pth \
    yolox_exp_file.py \
    --batch_size 1 \
    --dynamic \
    --work-dir work_dirs \
    --nms

# 結果：只有 ONNX，無評估
```

### 新方法（推薦）✅

```bash
# 簡單且功能完整
python projects/YOLOX_opt_elan/deploy/main.py \
    projects/YOLOX_opt_elan/deploy/deploy_config.py \
    model_config.py \
    checkpoint.pth \
    --work-dir work_dirs

# 結果：ONNX + TensorRT + 完整評估
```

## 常見使用場景

### 場景 1：只導出 ONNX

```python
# 修改 deploy_config.py
export = dict(
    mode='onnx',  # 只導出 ONNX
    verify=False,
    device='cuda:0',
)

evaluation = dict(
    enabled=False,  # 不評估
)
```

### 場景 2：只評估現有模型

```python
# 修改 deploy_config.py
export = dict(
    mode='none',  # 不導出
)

evaluation = dict(
    enabled=True,
    models_to_evaluate=['onnx'],  # 評估現有 ONNX
)

runtime_io = dict(
    onnx_file='work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx',
    ...
)
```

### 場景 3：完整部署（推薦）

```python
# deploy_config.py（默認配置）
export = dict(
    mode='both',  # ONNX + TensorRT
    verify=True,  # 驗證
)

evaluation = dict(
    enabled=True,
    models_to_evaluate=['pytorch', 'onnx', 'tensorrt'],
)
```

## 故障排除

### 找不到標註文件

```
FileNotFoundError: Annotation file not found: ...
```

**解決方案**：檢查 `deploy_config.py` 中的 `ann_file` 路徑

### 找不到圖片

```
ValueError: Failed to load image: ...
```

**解決方案**：
1. 如果圖片路徑在標註中是完整路徑，設置 `img_prefix=""`
2. 如果圖片路徑是相對路徑，設置 `img_prefix="path/to/images/"`

### 自定義模組導入錯誤

```
ModuleNotFoundError: No module named 'projects.YOLOX_opt_elan.yolox'
```

**解決方案**：確保模型配置中有正確的 `custom_imports`：
```python
custom_imports = dict(
    imports=[
        'projects.YOLOX_opt_elan.yolox',
        'projects.YOLOX_opt_elan.yolox.models',
    ],
    allow_failed_imports=False,
)
```

## 獲取幫助

查看完整文檔：
- [README.md](README.md) - 完整功能說明
- [YOLOX_OPT_ELAN_DEPLOYMENT_COMPARISON.md](../../../../YOLOX_OPT_ELAN_DEPLOYMENT_COMPARISON.md) - 新舊方法對比
- [Deployment Framework](../../../../autoware_ml/deployment/README.md) - 框架文檔
