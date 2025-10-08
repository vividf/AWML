# Quick Start Guide - Deployment

快速開始使用 YOLOX 和 CenterPoint 的 deployment。

## 🚀 YOLOX - 10 分鐘快速開始

### Step 1: 準備資料

確保有 COCO 格式的資料集：
```bash
data/coco/
├── annotations/instances_val2017.json
└── val2017/*.jpg
```

### Step 2: 運行 Deployment

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 完整 pipeline: Export + Evaluate
python projects/YOLOX/deploy/main.py \
    projects/YOLOX/deploy/deploy_config.py \
    projects/YOLOX/configs/yolox_s_8xb8-300e_coco.py \
    path/to/your/checkpoint.pth \
    --work-dir work_dirs/yolox_deployment \
    --device cuda:0
```

### Step 3: 查看結果

```bash
work_dirs/yolox_deployment/
├── yolox.onnx          # ONNX model
└── yolox.engine        # TensorRT engine
```

輸出會顯示：
- ✅ mAP metrics
- ✅ Per-class AP
- ✅ Latency statistics
- ✅ Cross-backend comparison

---

## 🎯 CenterPoint - 10 分鐘快速開始

### Step 1: 準備資料

確保有 T4Dataset 格式的資料：
```bash
data/t4dataset/
├── centerpoint_infos_val.pkl
└── lidar/*.bin
```

### Step 2: 運行 Deployment

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 完整 pipeline: Export + Evaluate
python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    path/to/your/checkpoint.pth \
    --work-dir work_dirs/centerpoint_deployment \
    --device cuda:0 \
    --replace-onnx-models  # ⚠️ 重要！ONNX export 必須加這個
```

### Step 3: 查看結果

```bash
work_dirs/centerpoint_deployment/
├── pillar_encoder.onnx
├── backbone.onnx
├── neck.onnx
└── head.onnx
```

輸出會顯示：
- ✅ Detection statistics
- ✅ Per-class counts
- ✅ Latency statistics

---

## ⚡ 只評估（不 Export）

### YOLOX

```bash
# 修改 deploy_config.py:
# export = dict(mode='none', ...)
# evaluation = dict(enabled=True, models_to_evaluate=['pytorch'])

python projects/YOLOX/deploy/main.py \
    projects/YOLOX/deploy/deploy_config.py \
    projects/YOLOX/configs/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth
```

### CenterPoint

```bash
# 修改 deploy_config.py:
# export = dict(mode='none', ...)
# evaluation = dict(enabled=True, models_to_evaluate=['pytorch'])

python projects/CenterPoint/deploy/main.py \
    projects/CenterPoint/deploy/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
    checkpoint.pth
```

---

## 🔧 自定義配置

### 修改樣本數量

編輯 `deploy_config.py`:
```python
evaluation = dict(
    enabled=True,
    num_samples=50,  # 改成你想要的數量，-1 = 全部
    ...
)
```

### 修改資料路徑

#### YOLOX
```python
runtime_io = dict(
    ann_file='your/path/to/annotations.json',
    img_prefix='your/path/to/images/',
    ...
)
```

#### CenterPoint
```python
runtime_io = dict(
    info_file='your/path/to/infos.pkl',
    ...
)
```

### 修改 Export 模式

```python
export = dict(
    mode='onnx',      # 'onnx', 'trt', 'both', 'none'
    verify=True,      # True/False
    device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ...
    work_dir='your/output/dir'
)
```

### 修改 TensorRT Precision

```python
backend_config = dict(
    common_config=dict(
        precision_policy='fp16',  # 'auto', 'fp16', 'fp32_tf32'
        max_workspace_size=2 << 30  # 2 GB
    )
)
```

---

## 📊 預期輸出範例

### YOLOX 成功輸出

```
================================================================================
Exporting to ONNX
================================================================================
Input shape: torch.Size([1, 3, 640, 640])
Output path: work_dirs/yolox_deployment/yolox.onnx
✅ ONNX export successful: work_dirs/yolox_deployment/yolox.onnx

================================================================================
Exporting to TensorRT
================================================================================
Building TensorRT engine (this may take a while)...
Enabled FP16 precision
✅ TensorRT export successful: work_dirs/yolox_deployment/yolox.engine

================================================================================
Running Evaluation
================================================================================
Evaluating PYTORCH model: checkpoint.pth
Number of samples: 100
Processing sample 1/100
...
Processing sample 100/100

PYTORCH Results:
================================================================================
YOLOX Evaluation Results
================================================================================

Detection Metrics:
  mAP (0.5:0.95): 0.3742
  mAP @ IoU=0.50: 0.5683
  mAP @ IoU=0.75: 0.3912

Latency Statistics:
  Mean: 5.23 ms
  ...
```

### CenterPoint 成功輸出

```
================================================================================
CenterPoint Deployment Pipeline
================================================================================
Deployment Configuration:
  Export mode: onnx
  Device: cuda:0
  Work dir: work_dirs/centerpoint_deployment
  Replace ONNX models: True

Creating data loader...
Loaded 500 samples

Loading PyTorch model...
Replacing model components with ONNX-compatible versions
✅ PyTorch model loaded successfully

================================================================================
Exporting to ONNX
================================================================================
Using model's built-in save_onnx method
Output directory: work_dirs/centerpoint_deployment
✅ ONNX export successful: work_dirs/centerpoint_deployment

================================================================================
Running Evaluation
================================================================================
Evaluating PYTORCH model: checkpoint.pth
Number of samples: 50
Processing sample 1/50
...
```

---

## ❗ 常見問題

### Q1: FileNotFoundError: Annotation file not found

**A**: 檢查 `deploy_config.py` 中的路徑是否正確
```python
runtime_io = dict(
    ann_file='data/coco/annotations/instances_val2017.json',  # 確認這個路徑
    img_prefix='data/coco/val2017/',  # 確認這個路徑
)
```

### Q2: CenterPoint ONNX export 失敗

**A**: 必須加上 `--replace-onnx-models` flag
```bash
python ... --replace-onnx-models
```

### Q3: CUDA out of memory

**A**: 減少評估樣本數量或使用 CPU
```python
evaluation = dict(
    num_samples=10,  # 減少樣本數
    ...
)

# 或使用 CPU
export = dict(
    device='cpu',
    ...
)
```

### Q4: TensorRT not found

**A**: TensorRT 是可選的，如果沒安裝可以只 export ONNX
```python
export = dict(
    mode='onnx',  # 只 export ONNX
    ...
)
```

### Q5: 想看更詳細的 log

**A**: 使用 `--log-level DEBUG`
```bash
python ... --log-level DEBUG
```

---

## 📚 更多資訊

- **完整文檔**: `DEPLOYMENT_IMPLEMENTATION_SUMMARY.md`
- **架構分析**: `deployment_dataloader_analysis.md`
- **YOLOX README**: `projects/YOLOX/deploy/README.md`
- **CenterPoint README**: `projects/CenterPoint/deploy/README.md`
- **使用教學**: `docs/tutorial/tutorial_deployment_dataloader.md`

---

## 🎯 下一步

1. ✅ 用真實資料測試
2. ✅ 調整配置符合你的需求
3. ✅ 比較不同 backend 的性能
4. ✅ 擴展到其他專案（BEVFusion, FRNet 等）

**Happy Deploying! 🚀**
