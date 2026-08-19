# CenterPoint INT8 PTQ Tutorial

複現 `centerpoint_2_6_skip_stage_0_by_distance`(CenterPoint 2.6 SECOND backbone 的
INT8 PTQ release)的完整部署流程,並在每一步留下可教學的記錄:
**PTQ 校準(含逐筆 histogram 快照)→ ONNX export → TensorRT engine → PTQ 前後評估**。

一鍵重跑:

```bash
cd <AWML repo root>
bash work_dirs/centerpoint_tutorial/scripts/run_all.sh
```

## 讀什麼(onboarding 順序)

| 文件 | 內容 |
|---|---|
| [docs/01_qdq_basics.md](docs/01_qdq_basics.md) | Q/DQ 基礎:scale/amax、對稱量化、校準一次性 vs 推論流水式、fake quant、TensorRT explicit quantization |
| [docs/02_ptq_calibration_histogram.md](docs/02_ptq_calibration_histogram.md) | PTQ 校準:histogram 逐筆演變、MSE/entropy/percentile/max 怎麼挑 amax(全部配實測圖) |
| [docs/03_pipeline_walkthrough.md](docs/03_pipeline_walkthrough.md) | 完整 pipeline 實跑記錄 + 踩雷記錄(BN-fused checkpoint 的載入順序) |
| [docs/04_backbone_recipes.md](docs/04_backbone_recipes.md) | ResNet residual-add / VoV eSE+concat+maxpool / ConvNeXt 的量化特殊處理,以及與 NVIDIA 官方建議的分歧 |

## 目錄結構

```
centerpoint_tutorial/
├── README.md                ← 你在這裡
├── docs/                    ← 四份 onboarding 文件
├── figures/                 ← 校準過程的實測圖(histogram 演變、amax 收斂、方法比較)
├── scripts/
│   ├── run_all.sh                       # 一鍵重跑
│   ├── 00_reconstruct_fp_checkpoint.py  # 從 release PTQ ckpt 還原 FP 權重
│   ├── 01_ptq_with_histogram_trace.py   # PTQ 校準 + 逐筆 histogram 記錄
│   ├── 02_plot_calibration.py           # 產生 figures/
│   └── 03_compare_amax_table.py         # 重現 amax vs release amax 對照表
├── configs/
│   ├── deploy_config_fp16_tutorial.py   # 「PTQ 前」:無 Q/DQ,純 FP16 engine
│   └── deploy_config_int8_tutorial.py   # 「PTQ 後」:release recipe(paths 改到本目錄)
├── checkpoints/
│   ├── epoch_29_fp_reconstructed.pth    # BN-fused FP 權重(從 release PTQ ckpt 剝離 amax)
│   ├── original_release_amax.pth        # release 的 56 個 amax(比較基準)
│   ├── epoch_29_ptq_tutorial.pth        # 本 tutorial 重新校準出的 PTQ ckpt
│   └── epoch_29_ptq_tutorial.calib      # amax cache
├── calib_trace/
│   ├── hist_trace.pkl                   # 60 筆 × 28 個 quantizer 的 histogram 快照
│   ├── amax_trace.json                  # 逐筆 MSE-amax 軌跡
│   ├── method_comparison.json           # 四種校準方法的最終 amax
│   └── amax_comparison.md               # 重現 vs release 對照表
├── fp16/{onnx,tensorrt}/                # PTQ 前的 ONNX + engine
├── int8/{onnx,tensorrt}/                # PTQ 後的 ONNX + engine
└── logs_*.log                           # 各步驟完整 log
```

## 結果總覽(本機 60-frame val split)

### 精度(mAP, center distance BEV / plane distance)

| backend | before PTQ (FP16) | after PTQ (INT8) | Δ |
|---|---|---|---|
| PyTorch(FP / fake-quant) | 0.4973 / 0.5164 | 0.4857 / 0.5035 | −0.012 / −0.013 |
| **TensorRT engine** | **0.4996 / 0.5189** | **0.4938 / 0.5120** | **−0.006 / −0.007** |

### 速度(TensorRT, RTX PRO 6000 Blackwell, batch=1)

| stage | FP16 engine | INT8 engine | speedup |
|---|---|---|---|
| Backbone+Neck+Head(被量化的部分) | 5.92 ± 0.28 ms | 3.47 ± 0.22 ms | **1.71×** |
| 端到端 model(voxel enc + middle + backbone head) | 7.22 ms | 4.82 ms | 1.50× |

### amax 重現性(重新校準 vs release checkpoint)

| | 結果 |
|---|---|
| weight amax(26 個 per-channel quantizer) | **完全一致(max rel diff = 0)** — 權重相同 + MaxCalibrator 確定性 |
| activation amax(30 個 per-tensor quantizer) | **中位數差 0.4%**,最大 44.6%(`blocks.2.12`)— 校準資料不同(60 本機 frames vs 400 全量) |

INT8 engine 的 ONNX 有 56 對 QuantizeLinear/DequantizeLinear(= release ONNX 的結構);
FP16 engine 的 ONNX 是零 Q/DQ 的同一張 graph。

## 權威參考

規則層面的唯一權威是 NVIDIA TensorRT 官方文件
[Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)。
[docs/01](docs/01_qdq_basics.md) §6 把它的 7 條 *Q/DQ Layer-Placement Recommendations*
逐條對照到本框架的實作,[docs/04](docs/04_backbone_recipes.md) §6 記錄了三處刻意的分歧
(`fuse_bn`、ConvTranspose per-tensor、`pytorch-quantization` vs ModelOpt)。

## 與 release 的關係(誠實聲明)

- **模型權重完全相同**:PTQ 校準不改權重,所以從 release 的 `epoch_29_ptq.pth`
  剝掉 amax 就是 release 部署的那份權重。重新校準後 **weight amax 與 release
  完全一致**(per-channel max 是確定性的)— 這驗證了整條 pipeline 沒跑歪。
- **校準資料不同**:release 用完整 val set 的 400 筆;本機只有 60 筆
  (db_j6gen2_v3 兩個 scene)→ activation amax 與 release 有差異,mAP 絕對值
  也不可與 release 的 0.7391 直接比較(資料集不同)。tutorial 的重點是
  **機制與 FP16↔INT8 的相對比較**,不是絕對值的復刻。
- 用到的 release 檔案:`~/Desktop/centerpoint_2_6_1_quant/epoch_29_ptq.pth`;
  reference log:`work_dirs/centerpoint_2_6_skip_stage_0_by_distance/deployment.log`。
