# VoV99 PTQ 準確度調校（mAP 大幅下降時）

PTQ 後 mAP 從 FP16 的 ~0.5 掉到 ~0.25 時，通常是**少數敏感元件**被 INT8 量化導致。依下面步驟可快速找出原因並用「部分 FP16」換回準確度。

**若已確認 quant_neck、quant_head 關掉都無法提升 mAP，且 ResNet/SECOND 同流程正常 → 多半是 VoVNet backbone 敏感，請直接看「§ 2B」用 `skip_vovnet_stages` 做二分搜尋。**

---

## 1. 可能原因（由常見到較少）

| 元件 | 說明 | 建議 |
|------|------|------|
| **Detection head** (`pts_bbox_head`) | heatmap / bbox regression 對 scale 很敏感，INT8 易掉 mAP | 先試 `quant_head=False` |
| **Backbone（VoVNet）** | stem / stage2/3/4 的 OSA、eSE 對 INT8 敏感；ResNet/SECOND 正常時多半是 VoVNet 獨有 | 用 **`skip_vovnet_stages`** 依序試（見 §2B） |
| **Neck** | 上採樣、融合對數值範圍敏感 | 可試著跳過 `pts_neck` 或特定 deblock |
| **Calibration** | 樣本數、分布、seed 影響 amax | 增加 `calibrate-samples`、試不同 `--calib-seed` 或 `--calib-shuffle` |

VoVNet 的 backbone 結構是 **pts_backbone.stem / .stage2 / .stage3 / .stage4**（沒有 `.blocks`）。跳過某段請用 **`skip_vovnet_stages`**（推薦）或 `sensitive_layers` 指定模組名稱。

---

## 2A. 快速實驗：先試「head / neck 不量化」

在 `deploy_config_int8_vov99.py` 的 `quantization` 裡改成：

```python
quant_head=False,   # 保持 detection head 為 FP16
```

其餘不變，重新跑 PTQ 與評估：

```bash
python deployment/quantization/centerpoint_quantization.py ptq \
  --config projects/CenterPoint/configs/t4dataset/Centerpoint/vov99_secfpn_4xb16_121m_j6gen2_base_amp_t4metric_v2.py \
  --checkpoint data/user/vivid/models/2_5/experiment_j6_gen2/vov_epoch_30.pth \
  --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_vov99.py \
  --calibrate-samples 1000 --batch-size 1 --calib-seed 0 \
  --output data/user/vivid/models/2_5/experiment_j6_gen2/vov_epoch_30_ptq_head_fp16.pth
```

- 若 mAP 明顯回升（例如回到 ~0.45+），代表 **head 是主要敏感來源**，可維持 `quant_head=False` 換取準確度（head 在 TRT 仍可跑 FP16）。
- 若 mAP 仍很低，代表問題多半在 **backbone**，請做 §2B。

---

## 2B. 高度懷疑 Backbone 時：用 skip_vovnet_stages（VoVNet 專用）

在 `deploy_config_int8_vov99.py` 的 `quantization` 裡設定：

```python
skip_vovnet_stages=[0],  # 0=stem, 1=stage2, 2=stage3, 3=stage4；保持這些 stage 為 FP16
```

- **第一次**：設 `skip_vovnet_stages=[0]`（只讓 stem 維持 FP16），重新 PTQ 與評估。
- 若 mAP **有回升** → stem 很敏感，可維持 `[0]`，或再試 `[0, 1]`（stem + stage2）看能否更穩。
- 若 mAP **仍低** → 改試 `[1]`（只跳 stage2），再試 `[2]`、`[3]`，或一次跳多段如 `[0, 1]`、`[0, 1, 2]`，用二分搜尋找到最少且能救回 mAP 的組合。

**範例（依序試）：** `[0]`=stem FP16；`[1]`=stage2 FP16；`[0, 1]`=stem+stage2 FP16；`[0, 1, 2]`=stem+stage2+stage3 FP16。

PTQ 跑完後 log 會出現 `Disabled quantization for: pts_backbone.stem` 等，代表該 stage 已改為 FP16。

---

## 3. 用手動 sensitive_layers 指定 backbone（與 skip_vovnet_stages 二選一）

在 `quantization` 裡加上（或改）`sensitive_layers`，讓前段保持 FP16：

```python
sensitive_layers=[
    "pts_backbone.stem",   # 只跳過 stem
    # 或 "pts_backbone.stage2",  # 只跳過 stage2
    # 或兩者都加，依實驗結果取捨
],
```

重新 PTQ 與評估。若 mAP 回升，代表 **stem / stage2** 很敏感。**建議以 `skip_vovnet_stages` 為主**，數字索引較好做二分搜尋。

---

## 4. 建議的「平衡」設定（backbone 前段 FP16）

若實驗結果是 stem 或 stage2 敏感，可固定成例如：

```python
quantization = dict(
    ...
    quant_head=True,   # 若前面試過 head 無效可保持 True
    quant_neck=True,
    skip_vovnet_stages=[0, 1],  # stem + stage2 維持 FP16，其餘 INT8
    sensitive_layers=[],
    ...
)
```

之後再依 mAP 微調要保留到 stage2 還是 stage3。

---

## 5. 用 sensitivity 找「哪一層」最傷 mAP（可選）

要精準知道是哪幾層造成掉點，可用 sensitivity 分析；目前 `centerpoint_quantization.py sensitivity` 內建的 eval 是 placeholder（回傳 0），要得到真實 mAP 需自己接專案的 eval：

```python
# 範例：在專案內跑 sensitivity，並用真實 mAP
from deployment.quantization.sensitivity import build_sensitivity_profile, get_sensitive_layers

# 1. 載入已 PTQ、已 calibration 的 model 和 val_dataloader
# 2. 定義 eval_fn(model) -> 回傳驗證集 mAP
results = build_sensitivity_profile(model, eval_fn, output_file="sensitivity.csv")
# 3. 取 delta 最大的幾層
sensitive = get_sensitive_layers(results, threshold=0.05, top_k=10)
# 4. 把 sensitive 加到 deploy config 的 sensitive_layers，重新 PTQ
```

把 `sensitive` 裡的名字加到 `sensitive_layers` 後再 PTQ，即可針對「最傷 mAP 的層」保留 FP16。

---

## 6. 其他可調參數

- **Calibration 樣本數**：例如 `--calibrate-samples 2000`（或更多），有時能改善 amax 估計。
- **Calibration 隨機性**：試不同 `--calib-seed`（如 0, 42, 123）或加上 `--calib-shuffle`，觀察 mAP 是否穩定。
- **Batch size**：`--batch-size 2` 或 4 可能讓 calibration 更穩定（若顯存允許）。

---

## 簡短結論

- **已試過 quant_neck / quant_head 無效、且 ResNet/SECOND 正常** → 優先當成 VoVNet backbone 問題，用 **`skip_vovnet_stages`** 從 `[0]` 開始試，再依需要改成 `[0,1]`、`[1]`、`[2]` 等做二分搜尋。
- **`skip_vovnet_stages`**：0=stem, 1=stage2, 2=stage3, 3=stage4；列在裡面的 stage 會保持 FP16，其餘 backbone 仍 INT8。
- 若要精準鎖定層級，再接專案 eval 跑 `build_sensitivity_profile`，把高 delta 的層加入 `sensitive_layers` 或對應成 `skip_vovnet_stages` 再 PTQ。
