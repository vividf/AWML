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

## 1.1 為什麼不量化 stem 與 stage2 時精度會明顯提升？

實務上常觀察到：**不量化 stem 和 stage2（即 `skip_vovnet_stages=[0, 1]`）時，mAP 會明顯回升**。主要原因如下。

### （1）早期層的動態範圍與量化誤差放大

- **Stem** 與 **stage2** 直接吃 **32ch @ 1020×1020** 的 BEV 特徵，數值分布（範圍、outlier、稀疏性）和後段 stage 差異大。
- INT8 只有 256 個等級；若 activation 動態範圍大或分布不勻，單一 scale（amax）難以同時照顧小值與大值，**量化誤差在早期層特別大**。
- PTQ 的 calibration 用有限樣本估計 amax；**前幾層最依賴輸入分布**，若 calibration 樣本不足或與實際部署分布有 gap，stem / stage2 的 scale 容易估差，後面 stage 的輸入已被網路「平滑」過，相對好估。

### （2）誤差會一路往後傳遞

- Stem 和 stage2 的輸出會進入 stage3 → stage4 → stage5 → neck → head。
- 前段一旦被量化壞，**誤差會累積、放大**，後段再怎麼量化得當也難以完全彌補。
- 保留 stem + stage2 為 FP16，等於讓「整條 backbone 的輸入端」保持較高精度，後面 stage3/4/5 的 INT8 才有機會在較乾淨的輸入上發揮。

### （3）解析度最高、像素數最多

- Stem 與 stage2 都在 **1020×1020** 全解析度上運算。
- 每個像素一點誤差，會乘上 **~10^6** 個位置；對 heatmap、bbox 回歸等需要**精準空間定位**的任務，早期在 high-res 上的量化誤差特別傷 mAP。
- Stage3/4/5 已是 510 / 255 / 255 等較低解析度，同樣的絕對誤差對最終 grid 的影響較小。

### （4）VoVNet 結構：concat + eSE

- **Stage2** 內含 **OSA**（多層 conv 後 concat）與 **eSE**（global pooling → FC → scale）。
- **Concat** 把多個 branch 的 activation 拼在一起，若其中一支被量化壓縮得太厲害，會拉低整條 concat 的表現。
- **eSE** 的 scale 是逐 channel 的乘法，數值通常較小、動態範圍敏感；INT8 量化容易讓 scale 偏掉，進而影響整張 feature map 的幅度。

### （5）與 ResNet / SECOND 的差異

- ResNet / SECOND 在相同 PTQ 設定下若較不掉點，多半是因為：結構較單純（無 eSE、無大量 concat）、或前段層數/通道較少，量化誤差相對可控。
- VoVNet 的 **stem + stage2** 負責「從 raw BEV 抽出第一層高解析度特徵」，這一段保留 FP16 是**用少量額外算力換取穩定 mAP** 的常見做法。

**實務建議**：若 PTQ 後 mAP 掉很多，可先設 `skip_vovnet_stages=[0, 1]`（stem + stage2 不量化），再視需要縮小為 `[0]` 或擴大為 `[0, 1, 2]`，在精度與速度之間取得平衡。

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
