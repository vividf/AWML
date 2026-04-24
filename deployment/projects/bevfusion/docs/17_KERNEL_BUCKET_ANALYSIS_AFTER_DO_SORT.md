# Sparse Encoder：Nsight Kernel Bucket 解析（關閉 sort 後）

本文件記錄在 **已成功關閉 pair-mask argsort（`do_sort=false`）** 之後，對 `nsys` 報表進行「手工 kernel bucket」拆分的方法、數字結論與優先級建議。

關聯文件：

- `docs/15_README_AWML_SPCONV_INT8_ACCEL_PLAN.md`（整體計畫與 `do_sort` 語意設計）
- `docs/16_PRIORITY_A_PROFILING_USAGE.md`（Priority A：`profile_sparse_encoder`、`nsys_profile_sparse.sh`）

本文補充 **`profile_sparse_encoder` / `nsys stats` / Python 聚合**這條鏈路上的「進階拆解」，適合你已經有一個 `.nsys-rep`，想要回答：

1. pair-gen **非 sort**段是否仍是主要成本？
2. **`quantize_features`**（每層 FP16→INT8 activation quant）上限收益有多大？
3. **`DeviceMergeSort*`**是否已從報表中消失？

---

## 文檔目的與適用對象

- **目的**：把 `cuda_gpu_kern_sum` 報表裡成千上百個 kernel，收斂成少數可決策的 bucket，並與「層級 profiler / CUDA-event」對照。
- **適用對象**：已能產生 `work_dirs/.../*.nsys-rep`，並希望做 **下一刀優化排序** 的工程師。

---

## 背景：`profile_sparse_encoder` vs `nsys stats`

| 工具 | 你在看什麼 | 優點 | 限制 |
|---|---|---|---|
| `deployment/projects/bevfusion/benchmark/profile_sparse_encoder.py` | TensorRT `IProfiler` per-layer + CUDA-event 總時間 | 與部署路徑一致；輸出 bucket（`pair_gen` / `implicit_gemm_int8` / …） | 看不到 **具體 CUDA kernel 名稱** |
| `nsys stats --report cuda_gpu_kern_sum` | GPU kernel 名稱 + `Total Time (ns)` + `Instances` | 可以把 **pair-gen / sort / quant / GEMM** 拆到 kernel 級 | 需要自己做 bucket 規則；與 wall-clock 之間可能有 overlap |

**重要**：`cuda_gpu_kern_sum` 的「kernel 時間加總」通常 **大於等於** CUDA-event 的 frame time，因為：

- 同一 stream 上 kernel **序列執行**時，加總接近 wall time；
- 若存在 overlap、或 trace 含有 **一次性 setup**（例如 voxelize），加總會偏離「穩態每幀」。

因此本文同時報告：

- **per-iter kernel-sum（粗）**：把整段 trace 的 kernel 時間除以估計的 iteration 數；
- 以及 **相對占比**（在「活躍 kernel」集合內的比例），用於排序優先級。

---

## 產物路徑（預設）

以下假設你在 AWML repo 根目錄（容器內為 `/workspace`）：

- **Nsight capture**：`work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA.nsys-rep`
- **輔助 SQLite**（`nsys` 自動生成/重用）：同目錄 `*.sqlite`

你可以改成自己的檔名；下文以 `NSYS_REP` 環境變數表示。

---

## 一步：匯出 `cuda_gpu_kern_sum` 為 CSV

在 **host** 或 **container**（有 `nsys`）皆可：

```bash
export NSYS_REP="work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA.nsys-rep"
export NSYS_BIN="/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys"

"${NSYS_BIN}" stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --force-export=true \
  "${NSYS_REP}" \
  > /tmp/kern_sum.csv 2> /tmp/kern_sum.err

echo "exit=$?"
head -n 15 /tmp/kern_sum.err
head -n 8 /tmp/kern_sum.csv
```

說明：

- `--force-export=true`：避免重用舊的 `.sqlite`（與 `.nsys-rep` 不一致時會得到錯表）。
- CSV 前面可能有 `NOTICE` 行；**真正的表頭**是：

  `Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name`

---

## 二步：Python 聚合（bucket 規則）

下面腳本只做三件事：

1. 跳過 CSV 前面的 notice，找到表頭；
2. 依 kernel **名稱**做 regex 分桶；
3. 用 `quantize_features_kernel` 的 **Instances** 推估 trace 內含多少次「推論迭代」，以便換算 **per-iter ms**。

### Iteration 數怎麼估？

在 `profile_sparse_encoder` 的典型設定下：

- `--warmup 10`
- `--iterations 30`

實務上 `nsys` trace **可能同時包含 warmup + iterations**（依包裝方式而定）。工程上可用下列經驗式：

- 若每個 iteration 會呼叫 **固定次數**的 `quantize_features_kernel`（通常對應稀疏塔內 INT8 implicit GEMM 層數），則：

  `ITERS_EST ≈ instances(quantize_features_kernel) / NUM_INT8_LAYERS`

本文後面表格採用 **`NUM_INT8_LAYERS = 20`**（與 `profile_sparse_encoder` bucket 統計中 `implicit_gemm_int8` 層數一致時）。

若你的模型層數不同，請改 `NUM_INT8_LAYERS`。

### 完整腳本（建議存成檔案以便重複執行）

將下列內容存成例如 `tools/nsys_kern_bucket_sparse.py`（此處僅文件內嵌，方便複製）：

```python
#!/usr/bin/env python3
"""Aggregate cuda_gpu_kern_sum.csv into coarse buckets for sparse encoder analysis."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="Output of: nsys stats --report cuda_gpu_kern_sum --format csv")
    ap.add_argument("--int8-layers", type=int, default=20, help="INT8 implicit GEMM layers per iteration (model-specific).")
    args = ap.parse_args()

    rows: list[list[str]] = []
    with args.csv.open(newline="") as f:
        rdr = csv.reader(f)
        header: list[str] | None = None
        for r in rdr:
            if r and r[0] == "Time (%)":
                header = r
                continue
            if header is None:
                continue
            if len(r) >= len(header):
                rows.append(r)

    name_i = header.index("Name")
    t_i = header.index("Total Time (ns)")
    inst_i = header.index("Instances")

    def get(row: list[str]) -> tuple[float, int, str]:
        return (
            float(row[t_i].replace(",", "")),
            int(row[inst_i].replace(",", "")),
            row[name_i],
        )

    pat_quant_act = re.compile(r"quantize_features|dequantize_features|launch_quantize|launch_dequant", re.I)
    pat_quant_wt = re.compile(r"quantize_weights|compute_w_scales|fuse_output_scale", re.I)
    pat_cast = re.compile(r"genericReformat|copyPackedKernel|ToFp16|ToInt8|copyVectorizedKernel", re.I)
    pat_sort = re.compile(r"DeviceMergeSort|thrust::.*sort", re.I)
    # spconv CUTLASS conv kernels often look like Ampere_s8s8... or Turing_s8s8...
    pat_gemm_i8 = re.compile(r"s8s8[a-z0-9]*|_C301LLL|cumm::conv::main::(Ampere|Turing)_", re.I)
    pat_gemm_fp = re.compile(r"f16f16f16f16|f16f16|f16s32|cumm::conv.*(fp16|f16)", re.I)
    pat_pair_gen = re.compile(
        r"pair|hash|calc_(conv|subm).*_(indices|mask)|build_.*hash|clear_map|arange|scatter|unique|fill_kernel|gather",
        re.I,
    )
    pat_relu_fused = re.compile(r"generatedNativePointwise|__myl_|pointwise", re.I)
    pat_voxelize_oneshot = re.compile(r"point_to_voxelidx|determin_voxel", re.I)

    def bucket_of(name: str) -> str:
        if pat_voxelize_oneshot.search(name):
            return "voxelize_oneshot"
        if pat_quant_act.search(name):
            return "quant_act"
        if pat_quant_wt.search(name):
            return "quant_wt_static"
        if pat_cast.search(name):
            return "cast"
        if pat_sort.search(name):
            return "sort"
        if pat_gemm_i8.search(name):
            return "gemm_int8"
        if pat_gemm_fp.search(name):
            return "gemm_fp"
        if pat_relu_fused.search(name):
            return "relu_fused"
        if pat_pair_gen.search(name):
            return "pair_gen"
        return "other"

    totals: dict[str, float] = {}
    totals_inst: dict[int, dict[str, int]] = {}  # placeholder to keep mypy calm - unused
    totals_inst = {}
    for row in rows:
        t, inst, name = get(row)
        b = bucket_of(name)
        totals[b] = totals.get(b, 0.0) + t
        totals_inst[b] = totals_inst.get(b, 0) + inst

    inst_quant_act = totals_inst.get("quant_act", 0)
    iters_est = max(1, round(inst_quant_act / args.int8_layers))

    grand_all = sum(totals.values())
    grand_active = grand_all - totals.get("voxelize_oneshot", 0.0) - totals.get("quant_wt_static", 0.0)

    print(f"CSV rows (data): {len(rows)}")
    print(f"Estimated ITERS ~= {iters_est}  (quant_act inst={inst_quant_act} / int8_layers={args.int8_layers})")
    print()

    order = [
        "gemm_int8",
        "pair_gen",
        "sort",
        "relu_fused",
        "quant_act",
        "cast",
        "gemm_fp",
        "other",
        "voxelize_oneshot",
        "quant_wt_static",
    ]

    print(f"{'bucket':<18} {'total_ms':>10} {'share_active':>13} {'per_iter_ms':>12} {'inst':>8}")
    for b in order:
        if b not in totals:
            continue
        t_ms = totals[b] / 1e6
        share = (totals[b] / grand_active * 100.0) if grand_active > 0 and b not in ("voxelize_oneshot", "quant_wt_static") else 0.0
        per_iter = t_ms / iters_est if b not in ("voxelize_oneshot", "quant_wt_static") else 0.0
        note = ""
        if b in ("voxelize_oneshot", "quant_wt_static"):
            note = " (one-shot / startup)"
        print(f"{b:<18} {t_ms:10.2f} {share:12.1f}% {per_iter:11.3f} {totals_inst.get(b, 0):8d}{note}")

    print()
    print(f"Kernel-sum (active, rough per-iter): {grand_active/iters_est/1e6:.3f} ms/iter")
    print(f"Grand total (includes one-shot buckets): {grand_all/1e6:.2f} ms (NOT comparable to per-frame latency directly)")

    rows_sorted = sorted(rows, key=lambda r: -get(r)[0])[:30]
    print()
    print("=== Top 30 kernels by Total Time ===")
    for row in rows_sorted:
        t, inst, name = get(row)
        print(f"  [{bucket_of(name):<16}] {t/1e6:8.2f} ms  inst={inst:6d}  {name[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

執行：

```bash
python3 tools/nsys_kern_bucket_sparse.py /tmp/kern_sum.csv --int8-layers 20
```

---

## Bucket 語意（務必讀）

| Bucket | 典型 kernel 名稱片段 | 含義 |
|---|---|---|
| `gemm_int8` | `Ampere_s8s8...`, `Turing_s8s8...`, `_C301LLL` | spconv / CUTLASS 的 INT8(sparse) GEMM 主體 |
| `pair_gen` | `calc_subm_conv_indices_mask`, `calc_conv_indices_stage1/2`, `build_subm_conv_hash_table`, `clear_map_kernel_split` | pair-gen **非 sort**段（hash / stage / unique / scatter） |
| `sort` | `DeviceMergeSort*`, `cub::DeviceMergeSort*` | pair-mask **argsort**（關閉 `do_sort` 後應接近 0） |
| `relu_fused` | `generatedNativePointwise`, `__myl_*` | TRT/Myelin fused elementwise（常見為 relu / 小 epilogue） |
| `quant_act` | `quantize_features_kernel` | 每層 activation 的 FP16→INT8（Path B 典型成本） |
| `cast` | `genericReformat::copyVectorizedKernel` | TRT reformat / cast（邊界精度轉換） |
| `gemm_fp` | `Ampere_f16f16...` 等 | 若仍殘留 FP16 sparse conv（通常少；需要時再細查） |
| `voxelize_oneshot` | `point_to_voxelidx_kernel`, `determin_voxel_num` | **一次性** voxelize / buffer 準備；不應看成 steady-state per-frame |
| `quant_wt_static` | `quantize_weights_per_channel_kernel`, `compute_w_scales_kernel`, `fuse_output_scale_into_gemm_scale_bias_kernel` | 權重量化與 scale fuse；通常集中在 **engine 初始化一次**，不是每幀熱路 |

---

## 本文記錄的兩份對照結果（同一套 bucket、同一 `NUM_INT8_LAYERS=20`）

下列數字來自同一份方法（`cuda_gpu_kern_sum` CSV → Python 聚合），差異來源是 **不同 `.nsys-rep` capture**：

- **Case W**：仍有 `DeviceMergeSort*`（代表 **尚未**成功關 sort，或 engine/onnx 仍帶 `do_sort=1`）
- **Case N**：已確認 **sort bucket 為 0** 且 Top kernel 列表中 **見不到** `DeviceMergeSort*`

### Case W（仍有 sort）— 重點 per-iter（ITERS_EST=40）

| Bucket | per-iter（粗） | 占 active kernel-sum（粗） |
|---|---:|---:|
| `gemm_int8` | 7.650 ms | 49.8% |
| `pair_gen` | 4.288 ms | 27.9% |
| `sort` | 1.257 ms | 8.2% |
| `relu_fused` | 1.155 ms | 7.5% |
| `quant_act` | 0.628 ms | 4.1% |
| `cast` | 0.210 ms | 1.4% |
| Active kernel-sum（粗） | ~15.37 ms/iter | 100%（不含 one-shot） |

### Case N（已關 sort）— 重點 per-iter（ITERS_EST=40）

| Bucket | per-iter（粗） | 占 active kernel-sum（粗） |
|---|---:|---:|
| `gemm_int8` | 6.801 ms | ~53.0% |
| `pair_gen` | 4.023 ms | ~31.4% |
| `sort` | **0** | **0%** |
| `relu_fused` | 1.091 ms | ~8.5% |
| `quant_act` | 0.560 ms | ~4.4% |
| `cast` | 0.184 ms | ~1.4% |
| Active kernel-sum（粗） | ~12.82 ms/iter | 100%（不含 one-shot） |

### 差分解讀（W → N）

| 項目 | 觀察 | 解讀 |
|---|---|---|
| `sort` | 1.257 ms/iter → 0 | 與「關 `do_sort`」一致；應作為 **第一階段驗收標準** |
| `pair_gen` | 4.288 → 4.023（−0.265 ms/iter） | 可能是 **scheduler 噪聲**，也可能與 pair order / memory locality 有關；**不要過度解讀 0.26 ms** |
| `gemm_int8` | 7.650 → 6.801（−0.849 ms/iter） | 可能包含噪聲；也可能與 **activation 分布 / kernel 選型**在两次 capture 間略有不同有關 |
| Active sum | 15.37 → 12.82（−2.55 ms/iter） | **主因是 sort 消失（~1.26 ms）**；其餘差異多半是 trace 邊界與噪聲 |

**驗收 checklist（你應該在每次 rebuild 後檢查）**

1. `sort` bucket 應為 0（或 Top kernel 列表見不到 `DeviceMergeSort*`）。
2. `pair_gen` 仍是第二大桶（通常仍 >25% active kernel-sum）。
3. `quant_act` + `cast` 合計通常 **< 6%** active kernel-sum（本文 Case N：`0.560 + 0.184 = 0.744 ms/iter`）。

---

## 「INT8 activation chain」上限收益（把每層 FP16 I/O 改成 INT8 chain）

這條優化對應表格中的：

- `quant_act`：`quantize_features_kernel`
- `cast`：`genericReformat::copyVectorizedKernel`（以及少量邊界 reformat）

### 理論上限（粗）

以 Case N 為例：

- `quant_act`：**0.560 ms/iter**
- `cast`：**0.184 ms/iter**
- **相加上限**：**0.744 ms/iter**

若你以 `docs/15` 中記錄的 CUDA-event steady-state（例如 **~13.652 ms**）做分母，則：

- `0.744 / 13.652 ≈ 5.4%`（**非常粗**的上限；實作通常拿不到 100%，因為首層輸入與尾層輸出仍然需要邊界精度轉換）

### 更務實的上限（經驗式）

若假設：

- 20 層中只有 **18 層**的中間量化可以「鏈式」消除首尾各 1 層邊界；
- `cast` 只有 **一半**屬於「可被鏈式消除」的邊界 cast（另一半可能是 sparse↔dense / plugin 邊界必須保留）；

則粗略上限約：

- `0.560 * (18/20) + 0.184 * 0.5 ≈ 0.596 ms/iter`（約 **4.4%** of 13.652 ms）

**結論**：這條路線「量得到、做得動」，但在你目前的 profile 形態下，**上限通常小於 pair-gen 與 GEMM 主體**。因此建議排序：

1. **pair-gen 非 sort**（最大結構性桶）
2. **GEMM / kernel 選型**（最大絕對時間桶）
3. **relu / Myelin fused epilogue**（中等桶，偏 TRT graph）
4. **INT8 activation chain**（長尾，工程量大）

---

## 下一步優化建議（按 ROI）

### A. pair-gen（非 sort）：最高優先

你的 Top kernel 幾乎總會看到：

- `calc_subm_conv_indices_mask`
- `calc_conv_indices_stage1_mask_direct_table`
- `calc_conv_indices_stage2_inference_mask`
- `build_subm_conv_hash_table`
- `clear_map_kernel_split` / `arange_hash_table`

**建議手法**：

- 先用 **Nsight Compute** 對「最重的一層」做單 kernel 分析（memory vs compute vs launch）。
- 再對照 spconv 的 `direct_table` / algo 設定與版本（與 open-source 路線對齊時，通常先看這裡）。

### B. `gemm_int8`：第二優先

特徵是大量 `Ampere_s8s8...` / `Turing_s8s8...` kernel 名稱。

**建議手法**：

- 先確認 **是否混用多種 tile/arch variant**（同一份 trace 內同時出現 Ampere/Turing 字樣未必是問題，但要確認是否「不必要的 fallback」）。
- 再評估是否能透過 builder / plugin / 模型導出穩定 shape，減少多版本 kernel 混用。

### C. `relu_fused` / `__myl_*`：第三優先（TRT graph）

這類 kernel 往往對應 **激活與小型 pointwise**，有時可以透過 **fusion / 精度策略**改善，但要避免影響精度與量化節點。

### D. `quant_act` + `cast`（INT8 chain）：第四優先（上限約千分之五到千分之七 frame time）

適合在你已經把 A/B 做完、進入「擠最後幾 %」階段再做。

---

## 常見陷阱（務必讀，會直接讓你判讀錯）

### 1) `voxelize_oneshot` 會污染「per-frame」心智模型

`point_to_voxelidx_kernel` / `determin_voxel_num` 這類 kernel **Instances 往往很少**，但 `Total Time` 可能很大。

它們代表 **一次性成本**（第一次準備 voxel buffer / 計數），不應與 steady-state iterations 混算。

**建議**：分析時永遠把 `voxelize_oneshot` 從 active sum 扣除（本文 Python 腳本已這樣做）。

### 2) `quant_wt_static` 通常不是每幀成本

若你看到 `quantize_weights_per_channel_kernel` 等 kernel：

- **Instances** 若符合「層數 × 固定常數」且只在 trace 前段出現，通常是 **初始化**。

本文 Case 中它們的總時間約 **0.17~0.19 ms（整段 trace）**，不應解讀成「每幀都要量化權重」。

### 3) kernel-sum 與 CUDA-event 的 gap

`cuda_gpu_kern_sum` 加總（per-iter）與 `profile_sparse_encoder` 印出的 CUDA-event mean **不會完全一致**，原因包括：

- trace 範圍是否含 warmup / setup；
- kernel overlap / memcpy overlap；
- TensorRT 既有非 kernel 開銷（未必顯示在 `cuda_gpu_kern_sum`）。

因此請把 kernel bucket 當作 **排序工具**，把 CUDA-event 當作 **絕對時間錨點**。

### 4) regex bucket 一定不完美

例如某些 FP16 fallback kernel 可能誤入 `gemm_fp`；某些 `generatedNativePointwise` 也可能不是 relu。

當你準備動手改 code，建議把「目標 kernel 名稱」從 Top list **人工確認**一次。

---

## 與「plugin / engine 是否真帶 `do_sort`」的關聯

若你在 `build_plugin_inside_container.sh` 看到：

- `WARNING: cloned source does NOT expose the do_sort attribute`

代表本次 build **很可能仍使用舊 upstream source**（常見原因：`/tmp/autoware_tensorrt_plugins_src` 被 cache），會導致：

- ONNX 即使寫了 `do_sort_i=0`，舊 plugin 仍可能無法完整覆蓋行為；
- `nsys` 仍看得到 `DeviceMergeSort*`。

**修正**：刪除暫存目錄後重跑 build（容器內）：

```bash
rm -rf /tmp/autoware_tensorrt_plugins_src /tmp/trt_plugin_build
bash projects/BEVFusion/plugins/build_plugin_inside_container.sh
```

並確認 log 出現 **OK: cloned source exposes do_sort plugin attribute**。

（AWML repo 內的 build script 也已加入 clone meta 比對，降低誤用 stale clone 的機率。）

---

## 精度恢復：把個別 `ImplicitGemm_int8` 層切回 FP16

INT8 部署常見的一條精度救援路徑是：**從 PTQ 敏感度排行或 KPI 下降最明顯的類別回推，把少數幾個敏感 sparse-conv 層改回 FP16**。AWML 直接在 deploy_config 一個 list 勾選「哪幾層保留 FP16」，**需要重跑 PTQ**（原因見下），但不需要改 plugin C++。

### 為什麼要從 PTQ 就跳過（而不是只在 ONNX transform 跳過）

第一版只在 step 4 (`sparse_int8_onnx_transform`) 把目標層的 `ImplicitGemm` 保留為 FP16。實測發現：只要把 `conv_input.0` 切回 FP16，**mAP 反而直線下降**。根因：

1. **PTQ 校正路徑** 仍然在 `conv_input.0` 插了 `_input_quantizer` / `_weight_quantizer`。
2. 所以下一層 `encoder_layer1.0.conv1._input_quantizer._amax` 是在「上游做了 fake-quant」的分布下統計出來的。
3. 到了 **TRT inference**，`conv_input.0` 變成真正的 FP16，上游不再做 fake-quant → `encoder_layer1.0.conv1` 看到的激活分布與 PTQ 時不同 → 其 `input_scale` 是「錯的」 → INT8 error 放大 → mAP 崩。

解法：在 PTQ 階段就 **不要** 幫這些層加 quantizer，校正時它們直接以 FP 計算，下游 `_amax` 自然對齊 runtime 行為。

### 三層同步的 keep-list（stage A/B/C）

`spconv_int8_fp16_layers` 現在被 **三個階段** 共用：

- **Stage A — PTQ（step 1）**：`apply_nvidia_spconv_int8(..., exclude_patterns=...)` 跳過對應 `SparseConvolution` 的 quantizer 安裝。checkpoint **不會** 出現該層的 `_amax`。
- **Stage B — PTQ reload（step 2 / step 5，經 runner.py）**：`_prepare_encoder_for_nvidia_int8(..., exclude_patterns=...)` 以相同規則重建 quantizer 樹，確保 `load_state_dict` key 對齊。
- **Stage C — ONNX transform（step 4）**：`--fp16-layers-from-deploy-cfg` 讓 `sparse_int8_onnx_transform` 保留對應的 `ImplicitGemm` 節點（**必要**；Stage A 把 stem 從 checkpoint 裡拿掉後，step 4 若沒拿到這個 list，會因為「找不到 calibrated stem」而 **直接 fail**，屬於 fail-fast，不是 regression）。

因此：**任何時候改動 `spconv_int8_fp16_layers` 都必須 `step 1 → step 2 → step 4 → step 5` 全部重跑**；不能只重跑 step 4。

### 使用方法

在 `deployment/projects/bevfusion/config/deploy_config_split_int8.py` 裡設定：

```python
spconv_int8_fp16_layers = [
    "conv_input.0",                         # 第一個 sparse conv 最敏感時常見
    # "encoder_layer3.encoder_layer3.2",    # stride-2 downsample of stage 3
]
```

Step 1（PTQ）會自動讀 deploy_cfg；step 4 必須顯式帶 `--fp16-layers-from-deploy-cfg` 才會生效（option 2，非 option 1）。

### 比對規則（為什麼純用 name）

entry 是 **case-insensitive substring**：

- **Stage A/B**：比對 PyTorch `named_modules()` 路徑（例如 `conv_input.0`、`encoder_layer1.0.conv1`）。
- **Stage C**：比對 ONNX `node.name`（例如 `/pts_middle_encoder/conv_input/conv_input.0/ImplicitGemm`）。

因為 PyTorch ONNX exporter 把 module 路徑原樣帶進 `node.name`，**同一個 entry** 通常兩端都命中。**所有 matcher 都只看 name**（**不** 看 ONNX 的 `inputs` / `outputs`）— 這是為了避開 PyTorch exporter 的「下游 tensor 名字繼承上游 producer scope」的陷阱；過去版本不慎用 text-blob 比對，導致 `"conv_input.0"` 把下一層也一起 FP16 化，是已知會炸 mAP 的 bug。**絕對不要把 matcher 改回看 inputs/outputs。**

### 驗證是否生效（三階段都要看）

跑完 step 1 / step 2 / step 4，依序確認：

- **Step 1 / Step 2 log**（stage A / B）：

  ```
  [nvidia-quant] SKIP (kept FP16 per exclude_patterns='conv_input.0'): conv_input.0
  [nvidia-quant] FP16 exclusion summary: 1 sparse convs kept FP16 (no TensorQuantizer)
  ```

  若 pattern 打錯會看到：

  ```
  [nvidia-quant] exclude_patterns with ZERO matches (typo?): ['conv_inputt.0']
  ```

- **Step 4 log**（stage C）：

  ```
  [int8] Keep FP16 ImplicitGemm per spconv_int8_fp16_layers (pattern='conv_input.0'): name='/pts_middle_encoder/.../ImplicitGemm'
  ```

  加上 `--verbose` 後文末 `[int8-census]` 區塊會把全部 `ImplicitGemm`（FP16）/ `ImplicitGemmInt8`（INT8）節點列出。pattern 打錯時：

  ```
  [int8-audit] WARNING: spconv_int8_fp16_layers patterns did NOT match any node ...
  ```

  這個 warning 一定要修掉。

### 與 `conv_out` 的關係

`conv_out.*` 是 PTQ 規則強制 FP32，**不需** 也 **不應** 寫進 `spconv_int8_fp16_layers`。這個 list 只針對中間 `ImplicitGemm_int8` 的 FP16/INT8 切換。

---

## 建議你把哪些數字寫進 PR / issue

當你要開一張「下一步優化」issue，建議固定格式：

1. **Case id**：W（含 sort）/ N（無 sort）
2. **`.nsys-rep` 檔名與日期**
3. **`ITERS_EST` 與 `NUM_INT8_LAYERS`**
4. 五個數字：`gemm_int8 / pair_gen / sort / quant_act+cast / relu_fused`（per-iter）
5. Top 10 kernel 名稱（貼文字即可）

這會讓你在兩週後仍然能復盤「到底優化了什麼」。

---

## 版本資訊

- 本文數字與方法：2026-04 於 AWML workspace 上，使用 Nsight Systems 報表 `cuda_gpu_kern_sum`。
- `nsys` 路徑示例：`/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys`
