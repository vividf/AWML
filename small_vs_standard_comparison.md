# Small vs Standard 版本詳細比較報告

## 檔案資訊
- **Small 版本**: `pillar_020_convnext_small_secfpn_4xb8_121m_base.py`
- **Standard 版本**: `pillar_020_convnext_standard_secfpn_4xb8_121m_base.py`
- **檔案結構**: 兩者都使用 `_base_` 導入的模組化配置

---

## 1. 模型架構差異（關鍵差異）

### 1.1 pts_backbone 配置

| 參數 | Small | Standard | 說明 |
|------|-------|----------|------|
| `type` | `"ConvNeXt_PC"` | `"ConvNeXt_PC"` | ✓ 相同 |
| `in_channels` | `32` | `32` | ✓ 相同 |
| `out_channels` | `[32, 64, 128, 128, 128]` | `[32, 192, 192, 192, 192]` | ⚠️ **不同** |
| `depths` | `[3, 2, 1, 1, 1]` | `[3, 3, 2, 1, 1]` | ⚠️ **不同** |
| `out_indices` | `[2, 3, 4]` | `[2, 3, 4]` | ✓ 相同 |
| `drop_path_rate` | `0.0` | `0.4` | ⚠️ **不同** |
| `layer_scale_init_value` | `1.0` | `1.0` | ✓ 相同 |
| `gap_before_final_norm` | `False` | `False` | ✓ 相同 |
| `with_cp` | `True` | `True` | ✓ 相同 |
| `_delete_` | `True` | `True` | ✓ 相同 |

**詳細分析**:
- **Small 版本**:
  - 通道數較小: `[32, 64, 128, 128, 128]`
  - 深度較淺: `[3, 2, 1, 1, 1]` (總共 8 層)
  - 不使用 drop path: `drop_path_rate=0.0`
  - **模型參數量較少，訓練速度較快，記憶體占用較少**

- **Standard 版本**:
  - 通道數較大: `[32, 192, 192, 192, 192]`
  - 深度較深: `[3, 3, 2, 1, 1]` (總共 10 層)
  - 使用 drop path: `drop_path_rate=0.4`
  - **模型參數量較多，表達能力更強，但訓練時間更長，記憶體占用更多**

### 1.2 pts_neck 配置

| 參數 | Small | Standard | 說明 |
|------|-------|----------|------|
| `type` | `"SECONDFPN"` | `"SECONDFPN"` | ✓ 相同 |
| `in_channels` | `[128, 128, 128]` | `[192, 192, 192]` | ⚠️ **不同** (對應 backbone 輸出) |
| `out_channels` | `[128, 128, 128]` | `[128, 128, 128]` | ✓ 相同 |
| `upsample_strides` | `[1, 2, 4]` | `[1, 2, 4]` | ✓ 相同 |
| `norm_cfg` | `dict(type="BN", eps=1e-3, momentum=0.01)` | `dict(type="BN", eps=1e-3, momentum=0.01)` | ✓ 相同 |
| `upsample_cfg` | `dict(type="deconv", bias=False)` | `dict(type="deconv", bias=False)` | ✓ 相同 |
| `use_conv_for_no_stride` | `True` | `True` | ✓ 相同 |

**說明**: `pts_neck` 的 `in_channels` 必須與 `pts_backbone` 的輸出通道數匹配：
- Small: backbone 輸出 `[128, 128, 128]` → neck 輸入 `[128, 128, 128]`
- Standard: backbone 輸出 `[192, 192, 192]` → neck 輸入 `[192, 192, 192]`

### 1.3 pts_bbox_head 配置

| 參數 | Small | Standard | 說明 |
|------|-------|----------|------|
| `type` | `"CenterHead"` | `"CenterHead"` | ✓ 相同 |
| `in_channels` | `sum([128, 128, 128])` = `384` | `sum([128, 128, 128])` = `384` | ✓ 相同 |
| `tasks` | `[dict(num_class=5, ...)]` | `[dict(num_class=5, ...)]` | ✓ 相同 |
| `loss_cls` | `GaussianFocalLoss` | `GaussianFocalLoss` | ✓ 相同 |
| `loss_bbox` | `L1Loss` | `L1Loss` | ✓ 相同 |

**說明**: 雖然 backbone 輸出通道數不同，但經過 neck 後都輸出 `[128, 128, 128]`，所以 head 的 `in_channels` 相同。

---

## 2. 資料配置差異

| 參數 | Small | Standard |
|------|-------|----------|
| `info_directory_path` | `"info/"` | `"info/user_name/"` |
| `data_root` | `"data/t4dataset/"` | `"data/t4dataset/"` ✓ 相同 |

---

## 3. 訓練配置（完全相同）

| 參數 | Small | Standard |
|------|-------|----------|
| `train_batch_size` | `8` | `8` ✓ |
| `test_batch_size` | `2` | `2` ✓ |
| `num_workers` | `32` | `32` ✓ |
| `val_interval` | `5` | `5` ✓ |
| `max_epochs` | `30` | `30` ✓ |
| `train_gpu_size` | `4` | `4` ✓ |
| `lr` | `0.0003` | `0.0003` ✓ |
| `optim_wrapper` | `AdamW, lr=0.0003, weight_decay=0.01` | `AdamW, lr=0.0003, weight_decay=0.01` ✓ |
| `param_scheduler` | 相同的 CosineAnnealingLR 和 CosineAnnealingMomentum | 相同的 CosineAnnealingLR 和 CosineAnnealingMomentum ✓ |

---

## 4. 其他配置（完全相同）

| 配置項 | Small | Standard |
|--------|-------|----------|
| `point_cloud_range` | `[-121.60, -121.60, -3.0, 121.60, 121.60, 5.0]` | `[-121.60, -121.60, -3.0, 121.60, 121.60, 5.0]` ✓ |
| `voxel_size` | `[0.20, 0.20, 8.0]` | `[0.20, 0.20, 8.0]` ✓ |
| `grid_size` | `[1216, 1216, 1]` | `[1216, 1216, 1]` ✓ |
| `sweeps_num` | `1` | `1` ✓ |
| `out_size_factor` | `4` | `4` ✓ |
| `pts_voxel_encoder` | 完全相同 | 完全相同 ✓ |
| `pts_middle_encoder` | 完全相同 | 完全相同 ✓ |
| `train_pipeline` | 完全相同 | 完全相同 ✓ |
| `test_pipeline` | 完全相同 | 完全相同 ✓ |
| `eval_pipeline` | 完全相同 | 完全相同 ✓ |
| `custom_hooks` | `[MomentumInfoHook]` | `[MomentumInfoHook]` ✓ |

---

## 5. Work Directory

| 版本 | Work Directory |
|------|----------------|
| Small | `work_dirs/centerpoint/{dataset_type}/pillar_020_convnext_small_secfpn_4xb8_121m_base` |
| Standard | `work_dirs/centerpoint/{dataset_type}/pillar_020_convnext_standard_secfpn_4xb8_121m_base` |

---

## 6. 模型參數量估算

### Small 版本
- Backbone 通道: `[32, 64, 128, 128, 128]`
- Backbone 深度: `[3, 2, 1, 1, 1]` (總共 8 層)
- Neck 輸入: `[128, 128, 128]`
- **參數量**: 較少

### Standard 版本
- Backbone 通道: `[32, 192, 192, 192, 192]`
- Backbone 深度: `[3, 3, 2, 1, 1]` (總共 10 層)
- Neck 輸入: `[192, 192, 192]`
- **參數量**: 較多（約為 Small 的 2-3 倍）

---

## 7. 性能預期差異

| 特性 | Small | Standard |
|------|-------|----------|
| **模型大小** | 較小 | 較大 |
| **參數量** | 較少 | 較多 |
| **訓練速度** | 較快 | 較慢 |
| **記憶體占用** | 較少 | 較多 |
| **推理速度** | 較快 | 較慢 |
| **表達能力** | 較弱 | 較強 |
| **預期 mAP** | 較低 | 較高 |
| **過擬合風險** | 較低（無 drop path） | 較高（有 drop path 0.4） |

---

## 8. 總結

### 主要差異
1. **模型架構**:
   - Small: 較小的 backbone（通道數和深度都較小），不使用 drop path
   - Standard: 較大的 backbone（通道數和深度都較大），使用 drop path

2. **資料路徑**:
   - Small: `info_directory_path = "info/"`
   - Standard: `info_directory_path = "info/user_name/"`

3. **Work Directory**: 名稱不同（small vs standard）

### 相同部分
- 所有訓練配置（batch size, learning rate, scheduler 等）
- 資料處理 pipeline
- 其他模型組件（voxel encoder, middle encoder, bbox head）
- 評估配置

### 建議使用場景
- **Small 版本**:
  - 資源受限的環境
  - 需要快速迭代和實驗
  - 對精度要求不是極高的場景

- **Standard 版本**:
  - 有充足計算資源
  - 追求更高精度
  - 生產環境部署
