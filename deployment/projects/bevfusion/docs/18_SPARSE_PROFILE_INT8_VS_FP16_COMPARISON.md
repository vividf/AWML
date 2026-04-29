# Sparse Encoder：INT8 vs FP16 Priority A 報告比對

本文件記錄同一組 **真實 voxel 輸入**（`t4dataset_j6gen2_base_infos_test.pkl#0`，`num_voxels=70747`）、同一套 **warmup=10 / iterations=30** 條件下，`profile_sparse_encoder.py` 對 **INT8** 與 **FP16（無 pair-mask sort）** 兩顆 sparse engine 的量測結果；並說明為何在 bucket 加總上 **ImplicitGemm INT8 與 FP16 時間相近**，以及如何進一步用 **Nsight Systems / Compute** 拆到 kernel 級。

前置閱讀：`docs/16_PRIORITY_A_PROFILING_USAGE.md`、`docs/17_KERNEL_BUCKET_ANALYSIS_AFTER_DO_SORT.md`。

---

## 1. 測試條件（摘要）

| 項目 | INT8 | FP16 |
|------|------|------|
| Engine | `bevfusion_split_int8_deployment_sparse_full_int8/tensorrt/bevfusion_sparse.engine` | `bevfusion_deployment_split/tensorrt/bevfusion_sparse.engine` |
| Deploy config | `deploy_config_split_int8.py` | `deploy_config_split_fp16_no_sort.py` |
| Model config | `bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py`（同左） | 同左 |
| `spconv_do_sort` | `False`（export 目標 `do_sort_i=0`） | `False` |
| Profile 指令 | 見 §5 | 見 §5 |

兩邊皆為 **TensorRT `IProfiler` 的 per-layer 時間加總**（bucket 內為各層 mean 時間之和），以及 **CUDA event 包住的 `execute_async_v3` 總時間**。

---

## 2. 數字比對（你提供的兩次 run）

### 2.1 端到端（CUDA event）

| 指標 | INT8 | FP16 | 備註 |
|------|------|------|------|
| mean | **13.731 ms** | **13.215 ms** | FP16 略快約 **~3.9%** |
| median | 13.729 ms | 13.144 ms | |
| std | ±0.347 ms | ±0.293 ms | 同數量級抖動 |

### 2.2 Bucket 加總（per-iteration layer-sum）

| bucket | INT8 sum_ms | INT8 % | FP16 sum_ms | FP16 % |
|--------|-------------|--------|-------------|--------|
| **pair_gen** | 4.775 | 39.58% | 4.733 | 40.26% |
| **implicit_gemm_int8** | 4.534 | 37.59% | — | — |
| **implicit_gemm_fp** | — | — | 4.322 | 36.76% |
| relu | 0.913 | 7.56% | 0.882 | 7.50% |
| cast | 0.202 | 1.68% | 0.225 | 1.91% |
| layout | 0.011 | 0.09% | 0.006 | 0.05% |
| other | 1.629 | 13.50% | 1.589 | 13.52% |

重點：

1. **pair_gen 幾乎相同**（4.775 vs 4.733 ms，約 **0.9%**）。在 `do_sort=false` 下，這一段主要是 **index / hash / scan / pair 建構**，與資料稀疏分佈與 plugin 實作綁在一起，**與 FP16/INT8 卷積精度路線關聯較小**，因此兩條線接近是預期內的。
2. **GEMM bucket**（INT8：`implicit_gemm_int8` 加總 vs FP16：`implicit_gemm_fp` 加總）為 **4.534 vs 4.322 ms**，INT8 反而略高約 **~4.9%**。這不代表「INT8 算更慢」的單一結論，見 §3（量測語意與瓶頸）。

### 2.3 Sanity：layer-sum vs event total

FP16 報告中有：

`per-layer sum=11.756 ms, event total=13.215 ms, delta=+1.459 ms`

這代表 **TRT / plugin 啟動、未列入某層名的開銷、或與 overlap 相關的統計差**，**無法**從 bucket 加總直接還原「純 INT8 乘加」週期數。INT8 run 若也有類似 delta，解讀方式相同。

---

## 3. 為什麼 ImplicitGemm「加總時間」會跟 FP16 差不多，甚至略高？

`IProfiler` 對每個 TensorRT **layer 名稱**回報一段時間；稀疏塔的 **ImplicitGemmInt8 / ImplicitGemm** 節點裡，實際 GPU 上可能是 **多個 kernel 串接**（含間接存取、reduction、型別轉換等）。因此：

### 3.1 稀疏卷積常見是 **記憶體／索引 bound**，不是 peak INT8 算力 bound

- 不規則稀疏：大量 ** gather / scatter、indirect load、workspace**，這些與 **FP16 / INT8 的算術吞吐比** 脱鉤。
- 即使權重與啟用改為 INT8，**metadata、index、partial sum 的存取模式**仍可能主導延遲。

### 3.2 INT8 路徑的「算術」以外的成本仍會算進該 layer

可能包含（依 plugin / TRT 版本而異）：scale、累加器寬度、輸出 cast、與鄰近 fuse 的 epilogue。這些都會落在 **同一個 TRT layer 的計時**裡，不會單獨一行叫「int8_multiply_only」。

### 3.3 FP16 與 INT8 可能走不同 kernel / tiling

TensorRT 會為不同精度選不同實作；**較快的算術**若配上 **較差的 occupancy 或較多的輔助 kernel**，總 layer 時間仍可能打平。

### 3.4 層數與圖結構不完全一致

你提供的表中 **implicit_gemm_int8 count=22**、**implicit_gemm_fp count=21**（例如 `conv_out` 是否仍 FP、或 ONNX 替換策略不同），加總時間是 **跨層 sum**，不是單層公平對照。

**結論**：  
「bucket 加總相近」**不能**直接推出「INT8 與 FP16 在 GPU 上做相同次數的 INT8/FP16 乘加」；只能說在 **目前引擎與輸入**下，**TRT 對外呈現的該段 layer 時間**相近。若要回答「INT8 乘法本身幾 ns」，必須下沉到 **kernel 級**（§4）。

---

## 4. 若要拆「INT8 乘法／GEMM kernel」實際佔比：建議作法

Priority A 的 Python 工具 **刻意不做** kernel 級拆分；請與既有流程並用：

1. **Nsight Systems**（你已使用的指令，見 §5）  
   - Timeline 上看 **哪些 kernel** 跟在 `ImplicitGemm` / plugin 前後。  
   - 匯出後可用：  
     `nsys stats --report cuda_gpu_kern_sum ...`  
     依 kernel 名稱做 bucket（見 `docs/17_KERNEL_BUCKET_ANALYSIS_AFTER_DO_SORT.md` 的命名規則）。

2. **Nsight Compute**  
   - 針對 **單一熱點 kernel**（例如名稱含 `mma`、`imma`、`gemm`、或 plugin 自訂名）開 profile，看 **Memory Workload Analysis / Pipe Utilization**，判斷是 **DRAM、L2、還是 Tensor Core 數學**為主。

3. **對照語意**  
   - `IProfiler`：**layer 時間軸**（可能含多 kernel + 同步點）。  
   - `nsys stats`：**kernel 自我時間（常見為 sum）**，與 overlap 與統計方式有關，與 §2.3 的 delta 一樣要謹慎解讀。

**沒有**一個開關能直接在 TRT 裡印「INT8 multiply 占 37%」——需從 **kernel 名稱 + NC 指標** 間接佐證。

---

## 5. 本次比對使用的 nsys 指令（存檔）

### INT8

```bash
/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o /workspace/work_dirs/bevfusion_split_int8_deployment_sparse_full_int8/nsys_sparse_priorityA \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine /workspace/work_dirs/bevfusion_split_int8_deployment_sparse_full_int8/tensorrt/bevfusion_sparse.engine \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --warmup 10 \
    --iterations 30
```

### FP16（`deploy_config_split_fp16_no_sort.py`）

```bash
/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o /workspace/work_dirs/bevfusion_deployment_split/nsys_sparse_priorityA \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine /workspace/work_dirs/bevfusion_deployment_split/tensorrt/bevfusion_sparse.engine \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_fp16_no_sort.py \
    --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --warmup 10 \
    --iterations 30
```

建議在分析 `.nsys-rep` 時，對兩個 work_dir **用同一套 `cuda_gpu_kern_sum` 篩選規則**，再比對「ImplicitGemm 相關 kernel」的總時間是否與 §2.2 的 bucket 趨勢一致。

---

## 6. 總結表

| 觀察 | 說明 |
|------|------|
| 總時間 INT8 vs FP16 | 本次資料：**FP16 略快**，差距小；主要仍受 **pair_gen ~40%** 牽制。 |
| pair_gen | 兩者 **幾乎相同**；與 **do_sort=false** 後的非-sort pair 建構一致。 |
| ImplicitGemm bucket | **加總相近**，INT8 略高；**不**等同於「INT8 數學更慢」，見 §3。 |
| 若要「乘法細項」 | 必須 **Nsight Systems kernel 級 bucket +（可選）Nsight Compute**；見 §4、§7～§8。 |

---

## 7. 建議的測試與分析順序（往下做）

請盡量 **固定同一台機器、同一 driver、同一輸入**（`--sample-idx`、`--model-cfg`），只改 **engine / deploy-cfg**。建議順序：

| 步驟 | 目的 | 工具／產物 |
|------|------|------------|
| **1** | 先拿到 **TensorRT layer 級** baseline（與 §2 同一語意） | 不包 `nsys`，直接跑 `profile_sparse_encoder` → 終端機報告 + `sparse_profile.json` |
| **2** | 同上，但讓 **nsys 包住**同一支 Python，留下 timeline | `nsys profile … -o …` → **`*.nsys-rep`**（路徑見 §5） |
| **3** | 從 `.nsys-rep` 抽出 **kernel 級**加總表 | `nsys stats --report cuda_gpu_kern_sum --format csv` → **`kern_sum.csv`** |
| **4** | 把上千個 kernel 收成 **少數 bucket**（pair / gemm / quant / sort …） | 下文 §8 的 Python 聚合（或 `docs/17` 內嵌的完整版腳本） |
| **5** | **INT8 與 FP16 各跑 2～3**，對照 bucket 占比與 top-kernel 名稱是否「故事一致」 | 兩份 CSV / 兩份終端輸出並列 |
| **6**（可選） | 對 **單一熱點 kernel** 看 memory vs math | **Nsight Compute**（`ncu`）鎖 kernel 名稱 |
| **7**（可選） | 端到端 eval 是否與 profile 一致 | `BEVFUSION_TRT_SPARSE_PROFILE=1`（見 `docs/16`） |

下面給 **可直接複製** 的命令模板（請把 `NSYS_REP`、`WORKDIR` 改成你的實際路徑）。

### 7.1 步驟 1 — 純 Python（無 nsys）快速對照

```bash
cd /workspace   # 或 AWML repo 根目錄

# INT8
python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
  --engine "${WORKDIR_INT8}/tensorrt/bevfusion_sparse.engine" \
  --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  --warmup 10 --iterations 30

# FP16
python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
  --engine "${WORKDIR_FP16}/tensorrt/bevfusion_sparse.engine" \
  --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_fp16_no_sort.py \
  --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  --warmup 10 --iterations 30
```

### 7.2 步驟 2 + 3 — `nsys profile` 後匯出 `cuda_gpu_kern_sum`

```bash
# 先產生 .nsys-rep（與 §5 相同；此處只示範 INT8，FP16 改路徑即可）
/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o /workspace/work_dirs/bevfusion_split_int8_deployment_sparse_full_int8/nsys_sparse_priorityA \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine /workspace/work_dirs/bevfusion_split_int8_deployment_sparse_full_int8/tensorrt/bevfusion_sparse.engine \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
    --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --warmup 10 --iterations 30

# 再從 .nsys-rep 匯出 kernel 報表（重點：--force-export=true 避免舊 sqlite 污染）

/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --force-export=true \
  /workspace/work_dirs/bevfusion_split_int8_deployment_sparse_full_int8/nsys_sparse_priorityA.nsys-rep \
  > /tmp/kern_sum_int8.csv 2> /tmp/kern_sum_int8.err

echo "exit=$?"
head -n 15 /tmp/kern_sum_int8.err
head -n 8 /tmp/kern_sum_int8.csv
```

對 FP16 重複一次，建議輸出到 `/tmp/kern_sum_fp16.csv`，方便 diff。
```bash

# 先產生 FP16 .nsys-rep（路徑同 §5 FP16）
/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o /workspace/work_dirs/bevfusion_deployment_split/nsys_sparse_priorityA \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine /workspace/work_dirs/bevfusion_deployment_split/tensorrt/bevfusion_sparse.engine \
    --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_fp16_no_sort.py \
    --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
    --warmup 10 --iterations 30
# 再從 .nsys-rep 匯出 kernel 報表 → /tmp/kern_sum_fp16.csv
/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --force-export=true \
  /workspace/work_dirs/bevfusion_deployment_split/nsys_sparse_priorityA.nsys-rep \
  > /tmp/kern_sum_fp16.csv 2> /tmp/kern_sum_fp16.err
echo "exit=$?"
head -n 15 /tmp/kern_sum_fp16.err
head -n 8 /tmp/kern_sum_fp16.csv
```

### 7.3 步驟 6（可選）— Nsight Compute 熱點 kernel

從 §7.2 的 CSV 或下面 §8 的「Top 25 kernels」挑 **時間最長且名稱穩定**的單一 kernel，再：

```bash
ncu --set full --kernel-name-base demangled \
  -k "RegexOrNameFragment" \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine "${WORKDIR_INT8}/tensorrt/bevfusion_sparse.engine" \
    ...
```

實務上首次建議 `--launch-skip 10 --launch-count 2` 只抓穩態 iteration，避免 warmup 噪音（依 NC 版本調整參數）。

---

## 8. `nsys stats` 與你提供的 Python 片段：用途與各 regex 在做什麼

### 8.1 `nsys stats --report cuda_gpu_kern_sum` 在做什麼？

- 從 **`*.nsys-rep`**（或配套 sqlite）讀取 CUDA kernel 事件，輸出 **依 kernel 名稱聚合**的統計：`Total Time (ns)`、`Instances` 等。
- **用途**：把 TRT layer 時間（§2）對應到 **真實 GPU kernel 名稱**，才能區分「算 GEMM」「走 quant kernel」「pair/hash」「是否還有 MergeSort」。
- **限制**：同 `docs/17` — 加總時間與 CUDA-event wall time 的關係受 **overlap、是否含 warmup、是否含 one-shot voxelize** 影響；適合看 **相對占比** 與 **命名診斷**，不適合直接當「絕對每幀 ms」唯一真理。

### 8.2 你提供的 Python 腳本在做什麼？

整體流程：

1. 解析 CSV，跳過前面的 `NOTICE`，找到表頭 **`Time (%)`** 開頭那一行。
2. 對每個 kernel **`Name`** 用 **regex 規則**分到一個粗 **bucket**。
3. 把同一 bucket 的 **`Total Time (ns)`** 加總，並用 `quantize_features` 等 **Instances** **粗估** trace 裡有多少個「等效 iteration」，換算 **per-iter ms**。

因此它的 **用處**是：快速得到「**kernel 級**的 pair vs gemm vs quant vs sort」占比，補足 `profile_sparse_encoder` **看不到 kernel 名**的限制。

### 8.3 各 regex bucket 的語意（對照你的 `pat_*`）

| 規則（示意） | Bucket 名 | 代表的含義 |
|--------------|-----------|------------|
| `quantize_features`、`launch_quantize`… | `quant_act` | INT8 路線上 **啟用量化／反量化** 類 kernel（若 FP16 engine 這類通常很少或為 0）。 |
| `quantize_weights`、`compute_w_scales`… | `quant_wt_static` | **權重量化／scale**（若只在載入或第一階段出現，腳本裡常當 **one-shot** 排除在 per-iter 占比外）。 |
| `genericReformat`、`copyPackedKernel`、`ToFp16`、`ToInt8`… | `cast` | TRT **reformat / cast / memcpy 型** kernel，不等於數學主體。 |
| `DeviceMergeSort`、`thrust::*sort` | `sort` | pair-mask **argsort**；在 **`do_sort=false`** 且 plugin 生效時應 **極低或為 0**（驗收用）。 |
| `s8s8`、`cumm::conv::main::Ampere`… | `gemm_int8` | **推測**為 **INT8 implicit GEMM /稀疏卷積主算**（命名隨 cumm / TRT 版本變動，需以實際 CSV 為準）。 |
| `f16f16`、`cumm::conv.*fp16`… | `gemm_fp` | **FP16** 主算路徑。 |
| `pair`、`hash`、`indices`、`gather`… | `pair_gen` | **pair 建構 / hash / scatter** 等（與 §2 的 TRT layer `pair_gen` 對照；regex 可能 **誤傷** 其他含 `scatter` 的 kernel，需人工看 Top 列表）。 |
| `generatedNativePointwise`、`__myl_`… | `relu_fused` | elementwise / **融合 ReLU** 類。 |
| `point_to_voxelidx`… | `voxelize_oneshot` | 若 trace 含 **voxelize**，通常視為 **一次性**，不應除以 iteration。 |
| 其他 | `other` | 含 **`[trainStation1]`** 等非稀疏塔命名時會落在這裡，需回到 timeline 確認來源。 |

### 8.4 `ITERS_est = quant_act_instances / 20` 要注意什麼？

- **20** 是假設「每個 iteration 大致對應約 20 個會觸發 `quant_act` 的機會」（與文件中稀疏塔 ImplicitGemm 層數 **近似**）。
- 若你的模型 **`implicit_gemm_int8` 層數不是 20**，請改成與 **`profile_sparse_encoder` 報告里的 layer count** 一致，或直接 **手動指定 `ITERS_est = warmup + iterations`**（例如 `40`，若你確定 trace 含 10 warmup + 30 measured）——但 trace 是否含 warmup **不一定**，故 **仍以 Instances 反推較穩**，並用 §7.1 的 layer 數校正分母。
- **結論**：腳本給的 **per-iter ~ X ms** 是 **粗估**，用于 **排序優先級**；若要絕對值，請與 **CUDA-event 總時間**交叉驗證。

### 8.5 建議你如何「好好分析」

1. 先看 **Top 25 kernels** 區段：名稱是否落在預期（`cumm`、`GetIndicePairs`、`quantize_features`…）。  
2. 對照 **`sort` bucket**：應接近 0（關 sort 成功時）。  
3. **INT8 vs FP16** 並列：`gemm_int8` vs `gemm_fp`、`quant_act` 是否只在 INT8 出現、`pair_gen` 是否如 §2 一樣接近。  
4. 若 **`gemm_int8` ≈ `gemm_fp`（kernel 級）**：與 §3 一致 — 多半 **記憶體／索引** 為主，再開 **ncu** 看 **DRAM / L2 / Tensor Core** 哪條管道飽和。  

更完整的 bucket 腳本與注意事項見 **`docs/17_KERNEL_BUCKET_ANALYSIS_AFTER_DO_SORT.md`**（含與本文相同的 CSV 匯出步驟）。

---

## 相關文件

- `docs/16_PRIORITY_A_PROFILING_USAGE.md` — `profile_sparse_encoder` 用法  
- `docs/17_KERNEL_BUCKET_ANALYSIS_AFTER_DO_SORT.md` — `cuda_gpu_kern_sum` 匯出、進階 bucket、與本文 Python 同一脈絡  
- `config/deploy_config_split_int8.py` — `spconv_do_sort` 與 INT8 稀疏段  
- `config/deploy_config_split_fp16_no_sort.py` — FP16 對照、`spconv_do_sort=False`
