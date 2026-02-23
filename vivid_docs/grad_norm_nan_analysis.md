# Grad Norm NaN 問題分析報告

## 問題概述
從 log 文件 `20260128_081731.log` 觀察到：
- **第一次 nan 出現**：iteration 2050，`grad_norm: nan`，`loss: 2.2087`（loss 仍正常），`loss_scaler: 225.2800`
- **第二次 nan 出現**：iteration 2900，`grad_norm: nan`，`loss: nan`（完全崩潰），`loss_scaler: 48.5200`
- **持續崩潰**：iteration 3300+ 後，所有指標持續為 nan，`loss_scaler` 降至 0.0000

---

## 1. 學習率過大分析

### 當前配置
```python
lr = 0.0003  # 基礎學習率
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)

# 學習率調度器
param_scheduler = [
    # 前 30% epochs (15 epochs): 0 → lr*10 = 0.003
    dict(type="CosineAnnealingLR", T_max=15, eta_min=0.003, ...),
    # 後 70% epochs (35 epochs): 0.003 → lr*1e-4 = 3e-8
    dict(type="CosineAnnealingLR", T_max=35, eta_min=3e-8, ...),
]
```

### 問題分析
1. **學習率峰值過高**：
   - 前 30% epochs 學習率會達到 **0.003**（基礎學習率的 10 倍）
   - 從 log 看，iteration 2050 時 `lr: 3.0829e-04`，iteration 2900 時 `lr: 3.1658e-04`
   - 這些學習率在 warmup 階段是合理的，但峰值 0.003 對於混合精度訓練可能過大

2. **與 grad_norm 的關係**：
   - iteration 2050 時 `grad_norm: nan`，此時學習率約 0.0003（正常範圍）
   - iteration 2900 時 `grad_norm: nan`，學習率約 0.0003（仍在正常範圍）
   - **學習率本身不是直接原因**，但可能加劇了其他問題

3. **評估**：
   - ⚠️ **中等風險**：學習率峰值 0.003 在混合精度訓練中可能過大
   - 建議：將峰值降低到 `lr * 5 = 0.0015` 或使用更保守的 warmup

---

## 2. 批次數據異常分析

### 當前配置
```python
train_dataloader = dict(
    sampler=dict(type="DistributedWeightedRandomSampler", shuffle=True),
    dataset=dict(
        type="T4FrameSamplerDataset",
        repeat_sampling_factor=0.30,  # 重複採樣因子
        frame_object_sampler=train_frame_object_sampler,  # 包含距離和高度採樣器
    ),
)

train_pipeline = [
    dict(type="GlobalRotScaleTrans",
         rot_range=[-1.571, 1.571],  # ±90度旋轉
         scale_ratio_range=[0.80, 1.20],  # 縮放範圍
         translation_std=[1.0, 1.0, 0.2]),  # 平移標準差
    ...
]
```

### 問題分析
1. **數據增強強度**：
   - 旋轉範圍 ±90度（`-1.571` 到 `1.571` 弧度）
   - 縮放範圍 0.8-1.2（20% 變化）
   - 平移標準差 [1.0, 1.0, 0.2] 米
   - **這些增強是合理的**，不太可能導致 nan

2. **採樣器影響**：
   - `repeat_sampling_factor=0.30`：某些 frame 會被重複採樣
   - `FrameObjectSampler`：根據 BEV 距離和高度過濾對象
   - **可能導致某些 batch 中對象數量極少或極多**

3. **潛在問題**：
   - 如果某個 batch 中所有對象都被過濾掉，可能導致：
     - 空的 ground truth
     - loss 計算異常
     - 梯度異常

4. **評估**：
   - ⚠️ **中等風險**：數據採樣可能導致某些 batch 異常
   - 建議：檢查 iteration 2050 和 2900 附近的數據，確認是否有異常 batch

---

## 3. 模型架構問題分析

### 當前配置
```python
pts_backbone=dict(
    type="BEVResNet",
    depth=34,
    num_stages=3,
    strides=(1, 2, 2),  # stage0=1, stage1=2, stage2=2
    deep_stem=True,  # 使用三個 3x3 conv 代替 7x7
    conv1_stride=1,  # 第一層 stride=1（無下採樣）
    with_pool=False,  # 無 maxpool
    base_channels=64,
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    # 注意：沒有 pretrained weights！
    # init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),  # 已註釋
    in_channels=32,
)
```

### 問題分析
1. **無預訓練權重**：
   - ResNet34 **沒有使用 ImageNet 預訓練權重**
   - 原因：輸入通道不匹配（ImageNet 是 3 通道，這裡是 32 通道）
   - **這是關鍵問題！** 從頭訓練 ResNet 需要非常謹慎的初始化

2. **架構修改**：
   - `deep_stem=True`：使用三個 3x3 conv 代替標準的 7x7 conv
   - `conv1_stride=1`：第一層無下採樣（BEV 友好）
   - `with_pool=False`：無 maxpool
   - 這些修改是合理的，但與標準 ResNet 不同

3. **殘差連接**：
   - ResNet34 使用 BasicBlock（expansion=1）
   - 殘差連接應該正常，但需要確認初始化是否正確

4. **評估**：
   - 🔴 **高風險**：無預訓練權重 + 從頭訓練 + 混合精度 = 高不穩定性風險
   - 建議：
     - 檢查 ResNet 的初始化方式（應該是 Kaiming 初始化）
     - 考慮使用更保守的初始化
     - 或者嘗試使用 ImageNet 預訓練權重的前幾層（需要適配）

---

## 4. 初始化不當分析

### 當前配置
```python
# ResNet backbone
pts_backbone=dict(
    # 沒有 init_cfg，應該使用默認初始化
    ...
)

# Head
pts_bbox_head=dict(
    separate_head=dict(
        type="CustomSeparateHead",
        init_bias=-4.595,  # sigmoid(-4.595) = 0.01
        ...
    ),
)
```

### 問題分析
1. **ResNet 初始化**：
   - 沒有明確指定 `init_cfg`
   - mmdet 的 ResNet 默認應該使用 Kaiming 初始化（He initialization）
   - **但從頭初始化 ResNet34 可能不夠穩定**

2. **Head 初始化**：
   - `init_bias=-4.595`：讓初始 heatmap 輸出很小（sigmoid(-4.595) = 0.01）
   - 這是合理的，避免初始預測過於自信
   - Head 使用 `init_cfg = dict(type="Kaiming", layer="Conv2d")`

3. **BatchNorm 初始化**：
   - `norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01)`
   - `eps=1e-3` 比標準的 `1e-5` 大，這可能導致數值不穩定
   - `momentum=0.01` 比標準的 `0.1` 小，更新更慢

4. **評估**：
   - 🔴 **高風險**：
     - BN 的 `eps=1e-3` 過大，可能導致數值問題
     - 從頭初始化 ResNet 需要更謹慎的初始化策略
   - 建議：
     - 將 BN `eps` 改為 `1e-5`
     - 考慮使用 Xavier 初始化或更保守的初始化
     - 檢查是否有層的初始化異常

---

## 綜合分析與建議

### 最可能的根本原因

根據 log 分析，**最可能的根本原因是組合效應**：

1. **主要問題**：**無預訓練權重 + 從頭訓練 + 混合精度訓練**
   - ResNet34 沒有 ImageNet 預訓練權重
   - 從頭訓練深度網絡在混合精度下非常不穩定
   - 第一次 nan（iteration 2050）時 loss 還正常，說明是梯度計算問題
   - 第二次 nan（iteration 2900）時 loss 也變成 nan，說明模型參數已被污染

2. **次要問題**：
   - **BN eps 過大**（`1e-3` vs 標準 `1e-5`）
   - **學習率峰值可能過高**（0.003）
   - **數據採樣可能導致某些 batch 異常**

### 建議的解決方案（按優先級）

#### 🔴 高優先級（必須修復）

1. **修復 BatchNorm eps**：
   ```python
   norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01)  # 改為標準值
   ```

2. **降低學習率峰值**：
   ```python
   # 將 warmup 峰值從 lr*10 改為 lr*5
   dict(type="CosineAnnealingLR", T_max=15, eta_min=lr * 5, ...)  # 0.0015
   ```

3. **添加梯度監控和保護**：
   ```python
   clip_grad = dict(max_norm=10, norm_type=2)  # 從 15 降低到 10
   ```

#### ⚠️ 中優先級（強烈建議）

4. **檢查數據異常**：
   - 在訓練循環中添加檢查，確保每個 batch 都有有效的 ground truth
   - 記錄 iteration 2050 和 2900 附近的數據統計

5. **改進初始化**：
   - 考慮使用更保守的初始化（如 Xavier）
   - 或者嘗試適配 ImageNet 預訓練權重（只使用前幾層）

6. **調整 loss scaling**：
   ```python
   loss_scale={
       "init_scale": 2.0**7,  # 從 256 降低到 128
       "growth_interval": 1000,  # 從 2000 降低到 1000
   }
   ```

#### 💡 低優先級（可選）

7. **使用 FP32 訓練一段時間**：
   - 先用 FP32 訓練幾個 epochs，確保模型穩定
   - 然後再切換到 FP16

8. **添加梯度檢查**：
   - 在 backward 後檢查梯度是否包含 nan/inf
   - 如果發現，跳過該 iteration

### 預期效果

修復後應該看到：
- `grad_norm` 穩定在合理範圍（通常 < 20）
- `loss_scaler` 穩定在初始值附近（256 左右）
- 訓練過程不再出現 nan

---

## 參考資料

- Log 文件：`projects/CenterPoint/configs/t4dataset/Centerpoint/20260128_081731.log`
- 配置文件：`projects/CenterPoint/configs/t4dataset/Centerpoint/resnet34_secfpn_4xb16_121m_base_amp.py`
- 模型代碼：`projects/CenterPoint/models/backbones/resnet.py`
