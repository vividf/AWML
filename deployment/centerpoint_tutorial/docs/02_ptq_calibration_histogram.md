# 02 — PTQ 校準:histogram 怎麼一步步變成 amax

> 前置:[01 — Q/DQ 基礎](01_qdq_basics.md)。
> 本篇所有圖都來自真實記錄:tutorial 的校準腳本在**每一筆校準資料 forward 之後**
> 快照了所有 activation quantizer 的 histogram 內部狀態(`calib_trace/hist_trace.pkl`)。

## 1. 校準在框架裡的位置

PTQ producer(`deployment/projects/centerpoint/quantization/quantize.py run_ptq`)的流程:

```
[1] init_model(model_cfg, FP checkpoint)          # 載入 FP 模型
[2] build_centerpoint_plan(config).prepare(model) # fuse BN → 插入 Q/DQ (QuantConv2d 等)
[3] build_calib_dataloader(cfg)                   # 用 val split 當校準資料
[4] CalibrationManager.calibrate(...)             # ← 本篇主角
[5] disable keep_fp16 subtrees → save checkpoint + .calib
```

`CalibrationManager.calibrate()` 內部三步(`deployment/quantization/core/calibration.py`):

```python
self.set_quantizer_fast()      # HistogramCalibrator._torch_hist = True(GPU histogram)
self.collect_stats(...)        # 每個 batch: model.test_step(batch),quantizer 只「看」不「量」
self.compute_amax("mse")       # 從 histogram 挑出 amax,寫進 quantizer._amax
```

關鍵狀態機:collect 階段每個 `TensorQuantizer` 是 `disable_quant() + enable_calib()` —
**模型 forward 完全是 FP 行為**,quantizer 只是旁觀者,把流過的 tensor 統計進 histogram。
收集完後 `enable_quant() + disable_calib()` 切回 fake-quant 模式。

注意這是**一次性、平行**的觀察:56 個 quantizer 在同一次 FP 前向裡各自累積自己的
histogram,不是「量化完前一層再校準下一層」的接力。為什麼這樣就夠,見
[01 — Q/DQ 基礎](01_qdq_basics.md) §3。

## 2. HistogramCalibrator:每一筆資料進來時發生什麼

每個 activation quantizer 內部維護一個 **2048-bin 的 |x| histogram**:

1. 對流過的 tensor 取絕對值。
2. 若 `max|x|` 超出目前 histogram 的範圍 → **擴張 bin edges**(保持 bin 寬度、增加 bin 數)
   把舊 counts 併進新網格。
3. 把這批值 histc 進 counts。

所以 histogram 有兩個會隨資料演變的自由度:**range(edges 上限)** 和 **counts 形狀**。
下面兩張圖是實際記錄(backbone 第一個量化 conv 的輸入):

![hist evolution](../figures/hist_evolution_backbone_blocks_1_0.png)

- 前幾筆資料就把分佈的「主體」形狀定下來了(長尾、對數座標下近似直線衰減)。
- 之後的資料主要是:(a) counts 等比例長高;(b) 偶爾一筆資料帶著更大的 outlier,
  把 range 往右拉一截。

把 60 筆快照疊成 heatmap(x 軸 = 第幾筆校準資料,y 軸 = |activation|,顏色 = log count):

![hist heatmap](../figures/hist_heatmap_backbone_blocks_1_0.png)

## 3. 從 histogram 到 amax:MSE 準則

校準結束後,`compute_amax("mse")` 對每個 histogram 做一次窮舉搜尋:

```
for 每個候選 amax(從第 128 個 bin 掃到最後一個 bin):
    用這個 amax 把 histogram 的 bin centers fake-quantize 到 127 levels
    計算量化誤差 MSE = Σ count(bin) * (center - dequant(center))²
挑 MSE 最小的候選當 amax
```

直觀:clipping error(砍掉尾巴)和 rounding error(scale 太粗)的總和最小化。
對長尾分佈,最優解幾乎總是「砍掉一點尾巴」— 所以 **MSE amax < max|x|** 是常態。

四種內建方法在同一個(最終)histogram 上各會挑在哪裡:

![method comparison](../figures/method_comparison_backbone_blocks_1_0.png)

| 方法 | 一句話 | 特性 |
|---|---|---|
| `max` | amax = 觀察到的最大值 | 無 clipping,但 outlier 直接毀掉解析度 |
| `percentile` (99.9/99.99) | 砍掉固定比例的尾巴 | 簡單粗暴,對「尾巴多重」不敏感 |
| `entropy` | 最小化量化前後分佈的 KL divergence | TensorRT implicit 模式的經典方法 |
| `mse` | 最小化加權量化誤差 | **我們用這個**(對齊 CUDA-CenterPoint 的行為) |

## 4. amax 隨校準樣本數的收斂

每筆資料之後「如果現在就停,MSE 會挑什麼 amax」的軌跡:

![amax trajectory](../figures/amax_trajectory.png)

可以看到:

- 大部分層在 **前 10–20 筆**就基本收斂 — 這是「校準只需要幾百筆資料」的直接證據。
- 個別層會因為某一筆帶 outlier 的資料出現階梯狀跳動,之後 MSE 又把它拉回來
  (MSE 對尾巴 count 不敏感,除非尾巴累積出足夠的權重)。
- release recipe 用 400 筆、我們本機用 60 筆,曲線尾端已平坦 → 樣本數差異對 amax 的
  影響有限(下面 §6 有定量驗證:中位數差 0.4%)。

## 5. Weight quantizer:不需要校準資料的另一半

Weight 是常數,`MaxCalibrator` 直接對每個 output channel 取 `max|w|`(per-channel, axis=0),
第一個 batch 就定案、之後不變:

![weight amax](../figures/weight_amax_per_channel.png)

這也解釋了 tutorial 的一個驗證結果:我們重跑校準後,**weight amax 與 release checkpoint
完全一致**(权重相同 → per-channel max 相同),差異只出現在 activation amax
(校準資料集不同)。

## 6. 校準的再現性 sanity check

release checkpoint(400 筆完整 val set)vs 本 tutorial 重跑(60 筆本機資料):

![repro vs release](../figures/amax_repro_vs_release.png)

定量結果([calib_trace/amax_comparison.md](../calib_trace/amax_comparison.md)):

- **weight amax:26 個 per-channel quantizer 全部與 release 完全一致(rel diff = 0)** —
  權重相同 + MaxCalibrator 是確定性運算。這是「pipeline 正確性」的黃金驗證。
- **activation amax:中位數差 0.4%、最大 44.6%**(`blocks.2.12`,深層 stage 對
  場景內容較敏感)。以 blocks.1.0 為例:release 6.0146 vs 重現 6.0209。
- 用**不同的校準資料**還能對到 0.4%,搭配上面的收斂曲線,可以建立
  「校準是統計問題、不是精確再現問題」的直覺:只要資料大致同分佈、
  樣本數過了收斂點,amax 就是穩定的。

> 順帶一提:我們第一次跑的時候 activation amax 系統性偏大 3–10 倍,
> 追查後不是資料問題,而是 checkpoint 載入順序的 bug(BN-fused 權重被載進
> 未融合的模型,conv bias 全被丟掉)。debug 過程見
> [03 — pipeline 實跑記錄](03_pipeline_walkthrough.md) §2。
> **weight amax 當時就已完全一致** — 這正是它作為 sanity check 的價值:
> weight 對、activation 錯 → 問題在「流過網路的資料」,不在量化機制本身。

→ 下一篇:[03 — 完整 pipeline 實跑記錄](03_pipeline_walkthrough.md)
