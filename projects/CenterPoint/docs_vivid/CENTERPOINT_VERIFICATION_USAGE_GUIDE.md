# CenterPoint Verification 使用指南

## 快速開始

修復已完成，現在可以運行驗證來確認修復是否有效。

## 方法 1: 使用測試腳本（推薦）

測試腳本專注於驗證功能，更容易診斷問題。

### 命令範例

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 基本用法
python projects/CenterPoint/deploy/test_verification_fix.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --onnx-dir work_dirs/centerpoint_deployment \
    --info-file data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl \
    --device cpu \
    --num-samples 1 \
    --tolerance 0.1

# 測試多個樣本
python projects/CenterPoint/deploy/test_verification_fix.py \
    --model-cfg <config> \
    --checkpoint <checkpoint> \
    --onnx-dir <onnx_dir> \
    --info-file <info_file> \
    --device cpu \
    --num-samples 5 \
    --tolerance 0.1

# 使用更嚴格的 tolerance
python projects/CenterPoint/deploy/test_verification_fix.py \
    --model-cfg <config> \
    --checkpoint <checkpoint> \
    --onnx-dir <onnx_dir> \
    --info-file <info_file> \
    --device cpu \
    --num-samples 1 \
    --tolerance 0.01

# 使用 GPU（如果可用）
python projects/CenterPoint/deploy/test_verification_fix.py \
    --model-cfg <config> \
    --checkpoint <checkpoint> \
    --onnx-dir <onnx_dir> \
    --info-file <info_file> \
    --device cuda:0 \
    --num-samples 1 \
    --tolerance 0.1
```

### 參數說明

- `--model-cfg`: 模型配置文件路徑
- `--checkpoint`: PyTorch checkpoint 文件路徑
- `--onnx-dir`: ONNX 模型目錄（包含兩個 .onnx 文件）
- `--info-file`: T4Dataset 的 info.pkl 文件路徑
- `--device`: 使用的設備 (cpu 或 cuda:0)
- `--num-samples`: 驗證的樣本數量（1-5 足夠測試）
- `--tolerance`: 允許的最大差異（0.1 = 10%）
- `--log-level`: 日誌級別 (DEBUG, INFO, WARNING, ERROR)

## 方法 2: 使用完整部署腳本

完整的部署腳本包含導出、驗證和評估。

### 命令範例

```bash
cd /home/yihsiangfang/ml_workspace/AWML

# 只驗證（不重新導出）
python projects/CenterPoint/deploy/main.py \
    --model-cfg configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py \
    --deploy-cfg projects/CenterPoint/deploy/configs/deploy_config.py \
    --checkpoint work_dirs/centerpoint/best_checkpoint.pth \
    --replace-onnx-models \
    --device cpu

# 重新導出並驗證
# 1. 修改 deploy_config.py 設定 export.mode = "onnx"
# 2. 運行相同命令
```

## 預期輸出

### 修復前（錯誤的行為）

```
Running verification...
============================================================
Verifying sample: sample_0
============================================================
Running PyTorch inference...
  PyTorch latency: 123.45 ms
  PyTorch output type: <class 'list'>
  PyTorch output length: 6

Verifying ONNX model...
  ONNX latency: 98.76 ms
  ONNX output type: <class 'list'>
  ONNX output length: 6
  Max difference: 0.000000          ← ❌ 錯誤！應該有差異
  Mean difference: 0.000000         ← ❌ 錯誤！應該有差異
  ONNX verification PASSED ✓        ← ❌ 虛假的成功
```

### 修復後（正確的行為）

```
Running verification...
============================================================
Verifying sample: sample_0
============================================================
Running PyTorch inference...
  PyTorch latency: 123.45 ms
  PyTorch output type: <class 'list'>
  PyTorch output length: 6

Verifying ONNX model...
  ONNX latency: 98.76 ms
  ONNX output type: <class 'list'>
  ONNX output length: 6

  Reference output details:
    Output[0] shape: (1, 3, 200, 200), dtype: float32
    Output[0] range: [-5.123456, 3.456789]
    Output[0] mean: 0.012345, std: 1.234567
    Output[1] shape: (1, 2, 200, 200), dtype: float32
    Output[1] range: [-2.345678, 1.234567]
    Output[1] mean: 0.003456, std: 0.567890
    ... (更多輸出信息)

  ONNX output details:
    Output[0] shape: (1, 3, 200, 200), dtype: float32
    Output[0] range: [-5.124567, 3.457890]
    Output[0] mean: 0.012346, std: 1.234568
    Output[1] shape: (1, 2, 200, 200), dtype: float32
    Output[1] range: [-2.346789, 1.235678]
    Output[1] mean: 0.003457, std: 0.567891
    ... (更多輸出信息)

  Computing differences for 6 outputs...
    Output[0] - max_diff: 0.001234, mean_diff: 0.000123
    Output[1] - max_diff: 0.002345, mean_diff: 0.000234
    Output[2] - max_diff: 0.001567, mean_diff: 0.000156
    Output[3] - max_diff: 0.001890, mean_diff: 0.000189
    Output[4] - max_diff: 0.002123, mean_diff: 0.000212
    Output[5] - max_diff: 0.001456, mean_diff: 0.000145

  Overall Max difference: 0.002345     ← ✅ 正確！有真實差異
  Overall Mean difference: 0.000177    ← ✅ 正確！有真實差異
  ONNX verification PASSED ✓           ← ✅ 真實的成功（差異在容許範圍內）
```

## 檢查點

修復後，應該確認以下幾點：

### ✅ 檢查 1: 有詳細的輸出信息

每個輸出應該顯示：
- Shape (形狀)
- Dtype (數據類型)
- Range (數值範圍)
- Mean (平均值)
- Std (標準差)

### ✅ 檢查 2: 差異值非零

- `Max difference` 應該 > 0（通常在 1e-5 到 1e-2 之間）
- `Mean difference` 應該 > 0（通常在 1e-6 到 1e-3 之間）

### ✅ 檢查 3: 差異值合理

- 如果差異太大（> 0.1），可能表示：
  - ONNX 導出有問題
  - 數值精度問題
  - 需要調整 tolerance

- 如果差異太小（< 1e-6），可能表示：
  - 模型非常簡單
  - 或者還是有問題（需要進一步檢查）

### ✅ 檢查 4: 兩個 ONNX 模型都被使用

查看日誌中是否有：
```
Got raw input features from PyTorch model: shape (N, 32, 11)
ONNX postprocess - output length: 6
  Output[0] shape: (1, 3, 200, 200), dtype: float32
  Output[1] shape: (1, 2, 200, 200), dtype: float32
  ...
```

## 常見問題

### Q1: 如果 difference 還是 0 怎麼辦？

檢查：
1. ONNX 模型是否正確導出（兩個 .onnx 文件都存在）
2. 是否使用了正確的 checkpoint
3. 是否使用了 `--replace-onnx-models` flag
4. 查看詳細日誌確認輸出類型和形狀

### Q2: 如果 difference 很大（> 0.1）怎麼辦？

可能原因：
1. ONNX 導出設定問題（constant_folding, opset_version）
2. 數值精度問題（檢查 dtype）
3. 設備差異（CPU vs CUDA）
4. 需要調整 tolerance

建議：
```bash
# 使用 CPU 避免設備差異
--device cpu

# 使用更高的 tolerance
--tolerance 0.5

# 檢查 ONNX 導出日誌
```

### Q3: 如何確認兩個 ONNX 模型都被使用？

查看 ONNX 目錄：
```bash
ls -la work_dirs/centerpoint_deployment/
# 應該看到：
# - pts_voxel_encoder.onnx          ← 第一個模型
# - pts_backbone_neck_head.onnx     ← 第二個模型
```

查看驗證日誌，應該看到兩次 ONNX inference：
1. Voxel encoder ONNX 調用
2. Backbone/neck/head ONNX 調用

### Q4: 如何解讀驗證結果？

- **PASSED with small diff (< 0.01)**: ✅ 非常好，ONNX 轉換質量高
- **PASSED with medium diff (0.01 - 0.1)**: ✅ 可以接受，正常範圍
- **PASSED with large diff (> 0.1)**: ⚠️ 需要檢查，可能有問題
- **FAILED**: ❌ 需要調查，ONNX 轉換有問題

## 下一步

如果驗證通過：
1. ✅ 記錄驗證結果
2. ✅ 繼續進行 TensorRT 轉換（如果需要）
3. ✅ 運行完整評估來測試精度

如果驗證失敗：
1. ❌ 檢查 ONNX 導出設定
2. ❌ 檢查模型配置
3. ❌ 查看詳細日誌找出問題
4. ❌ 聯繫維護者尋求幫助

## 聯繫信息

如果遇到問題或需要幫助，請：
1. 查看詳細的分析文檔: `CENTERPOINT_VERIFICATION_FIXES_SUMMARY.md`
2. 查看修復總結: `CENTERPOINT_VERIFICATION_ISSUE_ANALYSIS.md`
3. 提供完整的日誌輸出（使用 `--log-level DEBUG`）

