# 詳細比較報告：pillar_020_convnext_standard vs train_config.py

## 檔案結構差異

### standard 版本 (`pillar_020_convnext_standard_secfpn_4xb8_121m_base.py`)
- **結構**: 使用 `_base_` 導入的模組化配置
- **行數**: 396 行
- **特點**: 簡潔，使用變數引用和模板

### train_config.py
- **結構**: 扁平化配置，所有值已展開
- **行數**: 1190 行
- **特點**: 完整的配置，包含所有從 base 檔案繼承的值

---

## 1. 資料路徑設定

| 項目 | standard | train_config |
|------|----------|-------------|
| `info_directory_path` | `"info/user_name/"` | `'info/kokseang_3/'` |
| `data_root` | `"data/t4dataset/"` | `'data/t4dataset/'` ✓ 相同 |

---

## 2. 驗證設定

| 項目 | standard | train_config |
|------|----------|-------------|
| `val_interval` | `5` | `2` |
| `dynamic_intervals` | `[(max_epochs - 10, 2)]` | ❌ 無 |
| `max_epochs` | `30` | `30` ✓ 相同 |

---

## 3. 模型架構配置

### 3.1 pts_backbone

| 參數 | standard | train_config |
|------|----------|-------------|
| `type` | `"ConvNeXt_PC"` | `'ConvNeXt_PC'` ✓ 相同 |
| `in_channels` | `32` | `32` ✓ 相同 |
| `out_channels` | `[32, 192, 192, 192, 192]` | ❌ 使用 `arch='small'` |
| `depths` | `[3, 3, 2, 1, 1]` | ❌ 使用 `arch='small'` |
| `out_indices` | `[2, 3, 4]` | `[2, 3, 4]` ✓ 相同 |
| `drop_path_rate` | `0.4` | `0.4` ✓ 相同 |
| `layer_scale_init_value` | `1.0` | `1.0` ✓ 相同 |
| `gap_before_final_norm` | `False` | `False` ✓ 相同 |
| `with_cp` | `True` | `True` ✓ 相同 |
| `_delete_` | `True` | ❌ 無 |

**⚠️ 重要**: `train_config.py` 使用 `arch='small'`，但 `ConvNeXt_PC` 的 `__init__` 方法不支援 `arch` 參數，需要明確指定 `out_channels` 和 `depths`。這可能導致配置錯誤。

### 3.2 pts_neck

| 參數 | standard | train_config |
|------|----------|-------------|
| `type` | `"SECONDFPN"` | `'SECONDFPN'` ✓ 相同 |
| `in_channels` | `[192, 192, 192]` | `[192, 192, 192]` ✓ **相同** |
| `out_channels` | `[128, 128, 128]` | `[128, 128, 128]` ✓ **相同** |
| `upsample_strides` | `[1, 2, 4]` | `[1, 2, 4]` ✓ 相同 |
| `norm_cfg` | `dict(type="BN", eps=1e-3, momentum=0.01)` | `dict(eps=0.001, momentum=0.01, type='BN')` ✓ 相同 |
| `upsample_cfg` | `dict(type="deconv", bias=False)` | `dict(bias=False, type='deconv')` ✓ 相同 |
| `use_conv_for_no_stride` | `True` | `True` ✓ 相同 |

**✓ 結論**: pts_neck 配置完全相同，這表示 backbone 的輸出通道數應該相同。

### 3.3 pts_bbox_head

| 參數 | standard | train_config |
|------|----------|-------------|
| `type` | `"CenterHead"` | `'CenterHead'` ✓ 相同 |
| `in_channels` | `sum([128, 128, 128])` = `384` | `384` ✓ 相同 |
| `tasks` | `[dict(num_class=5, class_names=[...])]` | `[dict(num_class=5, class_names=[...])]` ✓ 相同 |
| `bbox_coder` | 簡化配置 | 詳細配置（包含 `code_size`, `max_num`, `score_threshold`） |
| `common_heads` | ❌ 無 | ✅ 有（`dim`, `height`, `reg`, `rot`, `vel`） |
| `separate_head` | ❌ 無 | ✅ 有（`final_kernel=1`, `init_bias=-2.19`） |
| `share_conv_channel` | ❌ 無 | ✅ `64` |
| `loss_cls` | `dict(type="mmdet.GaussianFocalLoss", ...)` | `dict(type='mmdet.GaussianFocalLoss', ...)` ✓ 相同 |
| `loss_bbox` | `dict(type="mmdet.L1Loss", ...)` | `dict(type='mmdet.L1Loss', ...)` ✓ 相同 |
| `norm_bbox` | `True` | `True` ✓ 相同 |

**⚠️ 差異**: `train_config.py` 有更詳細的 head 配置，包括 `common_heads`、`separate_head` 和 `share_conv_channel`。這些參數在 standard 版本中可能從 base 配置繼承。

### 3.4 pts_voxel_encoder

| 參數 | standard | train_config |
|------|----------|-------------|
| `type` | `"BackwardPillarFeatureNet"` | `'BackwardPillarFeatureNet'` ✓ 相同 |
| `in_channels` | `4` | `4` ✓ 相同 |
| `feat_channels` | `[32, 32]` | `[32, 32]` ✓ 相同 |
| `with_distance` | `False` | `False` ✓ 相同 |
| `with_cluster_center` | `True` | `True` ✓ 相同 |
| `with_voxel_center` | `True` | `True` ✓ 相同 |
| `norm_cfg` | `dict(type="BN1d", eps=1e-3, momentum=0.01)` | `dict(eps=0.001, momentum=0.01, type='BN1d')` ✓ 相同 |
| `legacy` | `False` | `False` ✓ 相同 |

**✓ 結論**: pts_voxel_encoder 配置完全相同。

---

## 4. 訓練配置 (train_cfg)

| 參數 | standard | train_config |
|------|----------|-------------|
| `grid_size` | `grid_size` (變數) | `[1216, 1216, 1]` ✓ 相同 |
| `voxel_size` | `voxel_size` (變數) | `[0.2, 0.2, 8.0]` ✓ 相同 |
| `point_cloud_range` | `point_cloud_range` (變數) | `[-121.6, -121.6, -3.0, 121.6, 121.6, 5.0]` ✓ 相同 |
| `out_size_factor` | `out_size_factor` (變數) | `4` ✓ 相同 |
| `code_weights` | ❌ 無 | ✅ `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]` |
| `dense_reg` | ❌ 無 | ✅ `1` |
| `gaussian_overlap` | ❌ 無 | ✅ `0.1` |
| `max_objs` | ❌ 無 | ✅ `500` |
| `min_radius` | ❌ 無 | ✅ `2` |

**⚠️ 差異**: `train_config.py` 包含更多訓練參數，這些可能在 standard 版本的 base 配置中定義。

---

## 5. 測試配置 (test_cfg)

| 參數 | standard | train_config |
|------|----------|-------------|
| `grid_size` | `grid_size` (變數) | `[1216, 1216, 1]` ✓ 相同 |
| `out_size_factor` | `out_size_factor` (變數) | `4` ✓ 相同 |
| `pc_range` | `point_cloud_range` (變數) | `[-121.6, -121.6, -3.0, 121.6, 121.6, 5.0]` ✓ 相同 |
| `voxel_size` | `voxel_size` (變數) | `[0.2, 0.2, 8.0]` ✓ 相同 |
| `post_center_limit_range` | `[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0]` | `[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0]` ✓ 相同 |
| `min_radius` | ❌ 無 | ✅ `[1.0]` |
| `nms_type` | ❌ 無 | ✅ `'circle'` |
| `post_max_size` | ❌ 無 | ✅ `100` |

**⚠️ 差異**: `train_config.py` 包含更多測試參數。

---

## 6. 其他重要差異

### 6.1 Checkpoint 和恢復
- **train_config**: 有 `load_from = 'work_dirs/centerpoint/T4Dataset/dev_pillar_020_small_convnetxt_secfpn_4xb8_121m_base/epoch_30.pth'`
- **standard**: 無 `load_from`

### 6.2 Activation Checkpointing
- **train_config**: 有 `activation_checkpointing = ['pts_backbone']`
- **standard**: 無此設定

### 6.3 Custom Hooks
- **train_config**: `custom_hooks = [dict(type='ExtraRuntimeInfoHook')]`
- **standard**: `custom_hooks = [dict(type="MomentumInfoHook")]`

### 6.4 Default Hooks
- **train_config**: 包含完整的 default_hooks（checkpoint, logger, param_scheduler, sampler_seed, timer, visualization）
- **standard**: 只有部分 hooks（logger, checkpoint），其他從 base 繼承

### 6.5 Auto Scale LR
- **train_config**: `auto_scale_lr = dict(base_batch_size=32, enable=False)`
- **standard**: `auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)` (動態計算)

### 6.6 Param Scheduler
- **train_config**: `eta_min` 使用計算後的具體數值（如 `0.0029999999999999996`, `3e-08`, `0.8947368421052632`）
- **standard**: `eta_min` 使用表達式（如 `lr * 10`, `lr * 1e-4`, `0.85 / 0.95`）

---

## 7. 相同部分總結

以下配置完全相同：
1. ✅ **pts_neck**: `in_channels=[192, 192, 192]`, `out_channels=[128, 128, 128]`
2. ✅ **pts_voxel_encoder**: 所有參數相同
3. ✅ **pts_middle_encoder**: 相同
4. ✅ **基本參數**: `point_cloud_range`, `voxel_size`, `grid_size`, `sweeps_num`, `out_size_factor`
5. ✅ **資料處理**: `train_pipeline`, `test_pipeline`, `eval_pipeline` 邏輯相同
6. ✅ **優化器**: `optim_wrapper` 配置相同
7. ✅ **學習率**: `lr = 0.0003` 相同
8. ✅ **Batch Size**: `train_batch_size=8`, `test_batch_size=2` 相同

---

## 8. 關鍵差異總結

### ⚠️ 潛在問題
1. **pts_backbone 配置方式不同**:
   - standard: 明確指定 `out_channels=[32, 192, 192, 192, 192]`, `depths=[3, 3, 2, 1, 1]`
   - train_config: 使用 `arch='small'`（可能無效，因為 ConvNeXt_PC 不支援 arch 參數）

2. **pts_bbox_head 詳細程度不同**:
   - train_config 有更詳細的配置（`common_heads`, `separate_head`, `share_conv_channel`）
   - standard 版本可能從 base 配置繼承這些參數

### 📝 配置差異（不影響模型架構）
1. **資料路徑**: `info/kokseang_3/` vs `info/user_name/`
2. **驗證間隔**: `val_interval=2` vs `val_interval=5`
3. **訓練參數**: train_config 有更多訓練參數（`code_weights`, `dense_reg` 等）
4. **測試參數**: train_config 有更多測試參數（`min_radius`, `nms_type` 等）
5. **Checkpoint**: train_config 有 `load_from` checkpoint
6. **Hooks**: 不同的 custom hooks

---

## 9. 結論

### 模型架構層面
- **pts_neck 配置完全相同**，這表示 backbone 的輸出通道數應該相同（`[192, 192, 192]`）
- 如果 `train_config.py` 實際運行時能正常工作，那麼 `arch='small'` 應該對應到 `out_channels=[32, 192, 192, 192, 192]` 和 `depths=[3, 3, 2, 1, 1]`

### 配置層面
- **standard 版本**: 模組化、簡潔，使用 base 配置繼承
- **train_config**: 完整展開的配置，包含所有參數

### 建議
1. 檢查 `train_config.py` 中的 `arch='small'` 是否在某處被轉換為對應的架構配置
2. 如果 `arch='small'` 無效，應該修改為明確指定 `out_channels` 和 `depths`
3. 確認 `train_config.py` 中的額外參數（`common_heads`, `separate_head` 等）是否在 standard 版本的 base 配置中定義
