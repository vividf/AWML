# 03 — 完整 pipeline 實跑記錄:PTQ → ONNX → TensorRT → 評估

> 這份是「照著做就能重現」的實跑記錄。所有輸出都在 `work_dirs/centerpoint_tutorial/`,
> 一鍵重跑:`bash work_dirs/centerpoint_tutorial/scripts/run_all.sh`(在 AWML repo root)。

## 0. 環境與素材

| 項目 | 值 |
|---|---|
| 容器 | `bevfusion-deployment:latest`(torch 2.8.0+cu129 / TensorRT 10.8.0 / pytorch-quantization 2.1.3) |
| GPU | RTX PRO 6000 Blackwell |
| 模型 | CenterPoint (SECOND backbone) 2.6, `second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2` |
| 資料 | 本機 `db_j6gen2_v3`(60-frame val split;release 用的是 5179-frame 完整 val set) |
| 起點 checkpoint | release 的 `epoch_29_ptq.pth`(PTQ 不改權重 → 剝掉 amax 即還原 FP 權重) |

容器啟動方式(所有步驟共用):

```bash
docker run --rm --gpus all --shm-size=32g \
    -v $PWD:/workspace -w /workspace \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    bevfusion-deployment:latest <command>
```

> 注意:`--shm-size=32g` 是必要的(預設 64MB 會弄死 DataLoader worker)。

## 1. 還原 FP checkpoint(step 00)

正式流程的輸入是訓練產出的 FP checkpoint(`epoch_29.pth`)。本機沒有這個檔案,
但 **PTQ 校準只新增 amax buffer、完全不動權重**,所以:

```bash
python3 work_dirs/centerpoint_tutorial/scripts/00_reconstruct_fp_checkpoint.py \
    --ptq-checkpoint ~/Desktop/centerpoint_2_6_1_quant/epoch_29_ptq.pth \
    --output work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth
```

132 個 key → 76 個權重 key(保留)+ 56 個 amax key(剝離,另存成
`original_release_amax.pth` 當比較基準)。

還原出來的權重是 **BN 已融合** 的版本(producer 在校準前就做了 fuse_bn)。載回未融合
的模型時 backbone 的 BN key 缺失 → BN 停在預設初始值(γ=1, β=0, μ=0, σ²=1)→
數值上是 no-op(誤差 ~5e-6,來自 eps)。所以這個 checkpoint 和正版 FP checkpoint
在部署路徑上等價。

## 2. PTQ 校準 + 逐筆 histogram 記錄(step 01)

正式流程一行指令:

```bash
python -m deployment.projects.centerpoint.quantization.quantize ptq \
    --deploy-cfg <deploy_config_int8_*.py>
```

tutorial 用的是加了記錄功能的等價腳本(`01_ptq_with_histogram_trace.py`),
內部流程與 `run_ptq` 一致(fuse BN → 插 Q/DQ → 校準 → 存檔),
只是每筆資料 forward 後多存一份 histogram / amax 快照。

**唯一的差異是載入順序,而且是被一個真實的 bug 逼出來的**:`run_ptq` 的輸入是
未融合的訓練 checkpoint,所以它先 `init_model(cfg, ckpt)` 再 fuse;我們的輸入是
BN 已融合的還原 checkpoint — 先載再 fuse 的話,融合後的 conv bias 在未融合模型裡
沒有落點(那些 conv 是 `bias=False`),**26 個 bias 會被 strict=False 靜默丟掉**,
然後 fuse_bn 把 bias 補成 0 — 模型看起來能跑、校準能完成、checkpoint 能存,
但 mAP = 0,而且校準出來的 activation amax 系統性偏大 3–10 倍。
正確做法與 deploy loader 相同:**先 fuse BN + 插 Q/DQ 建好樹,再 load state_dict**,
並檢查 missing keys 除了 `_amax` 外必須是空的(腳本裡直接 raise)。

deploy config 的 `quantization` block 是唯一事實來源(release recipe):

```python
quantization = dict(
    enabled=True, mode="ptq", fuse_bn=True, default_precision="int8",
    keep_fp16=["pts_voxel_encoder", "pts_backbone.blocks.0"],  # 這兩塊不量化
    disable_recipes=["add"],       # SECOND 沒有 residual add
    ptq=dict(calibrate_samples=60, batch_size=1, calib_seed=0),  # release 用 400
)
```

產物:

```
checkpoints/epoch_29_ptq_tutorial.pth    # BN-fused 權重 + 56 個 amax
checkpoints/epoch_29_ptq_tutorial.calib  # amax cache(QAT 可重用、可 debug)
calib_trace/hist_trace.pkl               # 60 筆 × 28 個 input quantizer 的 histogram 快照
calib_trace/amax_trace.json              # 逐筆 MSE-amax 軌跡
calib_trace/method_comparison.json       # mse/entropy/percentile/max 的最終 amax 對比
```

histogram → amax 的機制解讀見 [02 — PTQ 校準](02_ptq_calibration_histogram.md)。

## 3. amax 重現性驗證(step 03)

我們用 **不同的校準資料**(60 筆本機 vs release 的 400 筆完整 val)重跑校準,
對比每個 quantizer 的 amax(`calib_trace/amax_comparison.md`):

- **weight amax 完全一致**(權重相同、MaxCalibrator 是確定性的)→ 驗證整條
  fuse-BN → 插 Q/DQ → 校準的 pipeline 沒有跑歪。
- **activation amax 有差但同量級**(校準資料不同,這是預期行為)。

## 4. FP16(PTQ 前)deploy:export → engine → eval(step 4)

```bash
python -m deployment.cli.main centerpoint \
    work_dirs/centerpoint_tutorial/configs/deploy_config_fp16_tutorial.py
```

config 上有個值得一講的細節:它仍然帶著 `quantization` block,但
`keep_fp16=["*"]` — match 所有 module,於是**一個 quantizer 都不會插**。
為什麼不干脆拿掉 block?因為我們還原的 checkpoint 是 **BN 已融合** 的權重
(conv 的 bias 承載了整個 BN 的 shift),而一般 FP 載入路徑建的是**未融合**模型,
那些 conv 是 `bias=False` — 26 個 bias 會被當 unexpected key 整組丟掉,模型直接壞掉
(我們第一次跑就踩了這個雷,mAP=0)。`enabled=True, fuse_bn=True` 讓載入走
「先 fuse BN 再 load」的同一條路,key 就對上了。

CLI 依序做:PyTorch 載入 → ONNX export(兩個檔:`pts_voxel_encoder.onnx` +
`pts_backbone_neck_head.onnx`)→ TensorRT engine build(`precision_policy="fp16"`)→
backend 評估(pytorch / tensorrt)。

FP16 的 ONNX graph 裡 **沒有任何 QuantizeLinear/DequantizeLinear** — 這就是「before PTQ」。

> 教訓(值得寫進 onboarding):**fuse_bn 是部署契約的一部分**。producer、loader、
> export 三方必須對「哪些 BN 被融合」有一致答案,不然 state_dict 的 key 就對不上。
> 這正是框架把三方都導向同一個 `build_centerpoint_plan().prepare()` 的原因。

## 5. INT8(PTQ 後)deploy(step 5)

```bash
python -m deployment.cli.main centerpoint \
    work_dirs/centerpoint_tutorial/configs/deploy_config_int8_tutorial.py
```

差異:`quantization` block 開啟,checkpoint 指向 `epoch_29_ptq_tutorial.pth`。
loader 會在 **載入 state_dict 之前** 對新建模型做同一套 fuse BN + 插 Q/DQ 變換
(所以 module tree 和 checkpoint 對得上),export 時開 `use_fb_fake_quant`,
把每個 TensorQuantizer 寫成 ONNX 的 Q/DQ 對:

```
INT8 pts_backbone_neck_head.onnx 的 op 統計:
QuantizeLinear × 56, DequantizeLinear × 56, Conv × 31, Relu × 26, ConvTranspose × 1 ...
```

TensorRT 看到 Q/DQ 走 explicit-quantization,把 `DQ→Conv→ReLU→Q` fuse 成 INT8 kernel。

## 6. 結果(60-frame 本機 val split)

| backend | mAP (center dist BEV) | mAP (plane dist) | TRT backbone+head latency |
|---|---|---|---|
| PyTorch FP(fused) | 0.4973 | 0.5164 | — |
| **TensorRT FP16(before PTQ)** | **0.4996** | **0.5189** | 5.92 ± 0.28 ms |
| PyTorch fake-quant(PTQ 後) | 0.4857 | 0.5035 | — |
| **TensorRT INT8(after PTQ)** | **0.4938** | **0.5120** | **3.47 ± 0.22 ms(1.71×)** |

INT8 的精度損失:TRT 上 −0.006 mAP;fake-quant 預覽 −0.012。
amax 重現性:weight amax 與 release **完全一致**;activation amax 中位數差 **0.4%**
(不同校準資料下)— 詳表 `calib_trace/amax_comparison.md`。

參考:release 在 5179-frame 完整 val set 上的結果(`work_dirs/centerpoint_2_6_skip_stage_0_by_distance/deployment.log`):

| backend | mAP (center dist BEV) | mAP (plane dist) | mAPH (center dist BEV) |
|---|---|---|---|
| PyTorch fake-quant | 0.7401 | 0.7574 | 0.6856 |
| TensorRT INT8 | 0.7391 | 0.7555 | 0.6852 |

兩個重點讀法:

1. **PyTorch fake-quant ≈ TensorRT INT8**(Δ ≈ 0.001):fake quant 在 float 域模擬的
   數值行為,和真的 INT8 kernel 幾乎一致 — 所以 PTQ 之後不用 build engine
   就能預估 INT8 精度。
2. 本機 60-frame 的絕對值和 release 不可比(資料不同、類別分佈只有 car/truck 為主),
   要看的是 **FP16 vs INT8 的差距** 和 **backend 之間的一致性**。

→ 下一篇:[04 — 各 backbone 的量化特殊處理](04_backbone_recipes.md)
