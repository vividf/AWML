# Priority A — Sparse Encoder 專項 Profile 操作手冊

本文件對應 `15_README_AWML_SPCONV_INT8_ACCEL_PLAN.md` 中的：

- **A1. 做 sparse encoder 專項 profile**
  - 分離 `implicit_gemm`、pair-gen、sort、elementwise、scatter 的時間
  - 不再只看總 latency
  - 成功標準：**能明確回答「時間前 3 名在哪裡」**
- **A2. 驗證目前是否有 sort 成本**
  - AWML TensorRT + plugin 路線上 `GetIndicePairsImplicitGemm` 是否顯著

提供兩個層次的量測方式：

1. **標準 Python profile 工具** — `benchmark/profile_sparse_encoder.py`
   - TensorRT `IProfiler` per-layer timing + CUDA-event 總時間
   - 自動分桶（bucket）成 `pair_gen` / `implicit_gemm_int8` / `scatter_nd` / …
   - 輸出 top-N 層、bucket 佔比、block roll-up、以及一份 JSON
2. **Nsight Systems 包裝腳本** — `benchmark/nsys_profile_sparse.sh`
   - 在 1 上面罩 `nsys profile`，讓你直接看到 implicit_gemm / argsort /
     scan / scatter 的 CUDA kernel 時間軸
3. **in-situ overlay（正式 eval 路徑）** — 新增環境變數 `BEVFUSION_TRT_SPARSE_PROFILE=1`
   - 跑 step 5 的 eval 時，直接在 log 裡印 sparse engine 的 bucket 拆分
   - 用來驗證「profile 工具跟真正跑 eval 時看到的一樣」

---

## 前置條件：完成 step 0 ~ step 5

以下流程假設你已經照目前部署腳本跑完：

```bash
# step 0 — build INT8 plugin (.so)
cd /workspace/deployment/projects/bevfusion/cpp/int8_plugin
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/conda
cmake --build build -j"$(nproc)"

# step 1 — PTQ (sparse only)
python -m deployment.projects.bevfusion.quantization.quantize ptq \
  --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
  --checkpoint work_dirs/bevfusion/bevfusion_epoch_30.pth \
  --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  --calibrate-samples 256 --batch-size 1 --calib-seed 0 \
  --output work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth

# step 2 — export float-sparse ONNX (deploy_cfg export mode = "onnx")
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py

# step 3 — rename
mv work_dirs/bevfusion_split_int8_deployment/onnx/bevfusion_sparse.onnx \
   work_dirs/bevfusion_split_int8_deployment/bevfusion_sparse_fp16.onnx

# step 4 — transform: ImplicitGemm -> ImplicitGemmInt8
python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \
  --onnx work_dirs/bevfusion_split_int8_deployment/bevfusion_sparse_fp16.onnx \
  --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \
  --output work_dirs/bevfusion_split_int8_deployment/onnx/bevfusion_sparse.onnx --verbose

# step 5 — build TRT engines (deploy_cfg export mode = "tensorrt")
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

完成後，應該會有：

- `work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine`
- `work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_dense.engine`

---

## A1 — 標準 Python profile

### 最基本用法（推薦）

使用真實 voxel input（和 eval 路徑一致）：

```bash
python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
  --engine work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine \
  --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
  --warmup 20 --iterations 200 \
  --output work_dirs/bevfusion_split_int8_deployment/sparse_profile.json
```

終端輸出範例（截斷）：

```
==============================================================================
Priority A — Sparse Encoder Profile Report
==============================================================================
Engine  : .../bevfusion_sparse.engine
Inputs  : real:info/t4dataset_j6gen2_base_infos_test.pkl#0 (num_voxels=67321)
Total GPU latency (CUDA-event, steady-state):
  mean=7.214 ± 0.032 ms  median=7.210  min=7.151  max=7.360  n=200

------------------------------------------------------------------------------
Layer-sum breakdown by op-bucket (mean per-iteration sum):
------------------------------------------------------------------------------
  bucket                count      sum_ms   % of layers
  pair_gen                  4       0.512          7.10%
  implicit_gemm_int8       14       4.932         68.43%
  implicit_gemm_fp          2       0.418          5.80%
  scatter_nd                1       0.310          4.30%
  add                       6       0.201          2.79%
  relu                      8       0.134          1.86%
  layout                    5       0.051          0.71%
  other                     7       0.648          8.99%

[A2] pair_gen (GetIndicePairsImplicitGemm) = 0.512 ms/iter (7.10% of layer-sum)
     → pair-gen 中等；若要再壓榨，優先順序低於 implicit_gemm / 邊界優化。
...
```

### 輸出說明

- **Total GPU latency** — CUDA event `time_since` 量到的穩態 sparse engine 總
  時間（不含 H2D / D2H，剛好對應 `BEVFusionTensorRTPipeline._last_split_sparse_gpu_ms`）。
- **Layer-sum breakdown** — 把 TRT `IProfiler` 回報的每一層依名稱分桶後相加。
  - `pair_gen` = `GetIndicePairsImplicitGemm` + `GetIndicePairs`（= A2 的重點）
  - `implicit_gemm_int8` = INT8 plugin 走的 `ImplicitGemmInt8`
  - `implicit_gemm_fp` = FP 版 conv（通常是 `conv_out` / `IndiceConv`）
  - `scatter_nd` = sparse 投回 dense 的 `ScatterND`
  - `add` / `relu` / `layout` / `cast` / `quant_dquant` = residual / dtype 邊界
- **Block roll-up** — 以 `conv_input` / `encoder_layer.0..3` / `conv_out`
  分群，和 `pts_middle_encoder` 結構對上，方便指認哪個 stage 吃時間。
- **Top N layers** — 排序後的單層 top-N；對應 Priority A 成功標準的
  **「時間前 3 名在哪裡」**。
- **sanity** 行 — `per-layer sum` 與 `event total` 的差量；代表 plugin launch /
  sync overhead。差太多（> 10%）通常是 per-layer 那端漏抓或有隱藏同步。

### JSON 報告

`--output sparse_profile.json` 會寫入一份：

```json
{
  "engine": ".../bevfusion_sparse.engine",
  "inputs": {"source": "real:...#0", "num_voxels": 67321, ...},
  "total_gpu_ms": {"mean_ms": 7.214, "std_ms": 0.032, "median_ms": 7.210, ...},
  "buckets": {"pair_gen": {"sum_ms": 0.512, "count": 4, "pct_of_layer_sum": 7.10}, ...},
  "blocks":  {"encoder_layer.0": {"sum_ms": 1.234, ...}, ...},
  "top_layers": [{"name": "/pts_middle_encoder/.../ImplicitGemmInt8", "mean_ms": 0.871, "bucket": "implicit_gemm_int8", "block": "encoder_layer.2"}, ...]
}
```

這份 JSON 適合丟進後續報表或 diff 不同 commit 的變化。

### 其他常用旗標

| 參數              | 用途                                                                   |
| ----------------- | ---------------------------------------------------------------------- |
| `--sample-idx N`  | 改用不同的 dataset sample（不同 voxel count 可大幅改變 sparse latency） |
| `--no-per-layer`  | 關掉 `IProfiler`（只要穩態 GPU 總時間，避免 profiler overhead）        |
| `--warmup N`      | 熱身迭代；預設 20                                                      |
| `--iterations N`  | 量測迭代；預設 200                                                     |
| `--top-n N`       | 單層 top-N 顯示數                                                      |
| `--synthetic`     | 用隨機 voxel（僅冒煙測試；**不能用來下 Priority A 結論**）             |
| `--plugin-lib X`  | 手動指定 plugin `.so`（預設從 `--deploy-cfg` 抓）                      |

---

## A2 — pair-gen / sort 成本的快速判定

Priority A 的 A2 關心「INT8 下 `do_sort=false` 有沒有機會」。這個問題在
AWML TensorRT + plugin 架構下的觀察方式是：

1. 跑 A1 的 profile 工具。
2. 看 `[A2] pair_gen (GetIndicePairsImplicitGemm)` 這行：

   | pair_gen %        | 判定                                                                |
   | ----------------- | ------------------------------------------------------------------- |
   | `>= 10%`          | 值得接著用 Nsight Compute 拆 argsort / scan kernel；有 sort 改造空間 |
   | `3% ~ 10%`        | 有感但不是主戰場；應先處理 implicit_gemm 或邊界                     |
   | `< 3%`            | 幾乎沒收益；和 8.1 的結論同一掛，不建議繼續投入                     |

文字報告會直接把這段判定寫進輸出，JSON 內可以從 `buckets.pair_gen.pct_of_layer_sum` 取得。

---

## A1 + A2 — Nsight Systems 版本

如果 `pair_gen` 佔比大於 5~10%、或你想看 kernel 等級 overlap，就跑：

```bash
# 預設會讀 env 或使用 docs 內的預設路徑
bash deployment/projects/bevfusion/benchmark/nsys_profile_sparse.sh

# 或用環境變數客製：
ENGINE=work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine \
DEPLOY_CFG=deployment/projects/bevfusion/config/deploy_config_split_int8.py \
MODEL_CFG=projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
WARMUP=10 ITERATIONS=30 \
OUTPUT_PREFIX=work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA \
bash deployment/projects/bevfusion/benchmark/nsys_profile_sparse.sh
```

## Run with host
```bash
cd /home/yihsiangfang/ml_workspace/AWML

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --name awml-bevfusion \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -v "$PWD":/workspace \
  -v "$PWD/data":/workspace/data \
  -v /opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  awml-bevfusion:full \
  bash -lc '
    /opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys profile \
      --trace=cuda,nvtx,osrt,cudnn \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      -o /workspace/work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA \
      python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
        --engine /workspace/work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine \
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \
        --model-cfg projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
        --warmup 10 \
        --iterations 30
  '
```




跑完：

```bash
# nsys stats --report gputrace     work_dirs/.../nsys_sparse_priorityA.nsys-rep | head -50
nsys stats --report cuda_gpu_trace work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA.nsys-rep | head -50
nsys stats --report cuda_kern_exec_sum work_dirs/.../nsys_sparse_priorityA.nsys-rep
```

Priority A 對應的 Nsight 檢視重點：

- Top-3 kernel 是不是 `implicit_gemm*`？是 → 主戰場就在 conv kernel（對應 New3D 結論）
- `argsort` / `scan` / `GetIndicePairs` 佔比是否 >= 10%？是 → A2 有機會
- Plugin 之間有沒有明顯 launch gap 或 D2H sync？有 → Priority B（融合 Add+Relu）
  值得做

---

## in-situ 驗證 — 在正式 eval (step 5) 裡印拆分

profile_sparse_encoder.py 跑的是只接 sparse engine 的 micro-benchmark；要確認
「真正走 eval pipeline 時看到的是同一件事」可以打開環境變數：

```bash
BEVFUSION_TRT_SPARSE_PROFILE=1 \
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

行為：

- `tensorrt.py` 會在 sparse engine 的 context 上掛 `_TRTLayerProfiler`
- 每 `BEVFUSION_TRT_SPARSE_PROFILE_EVERY`（預設 10）幀印一行 bucket 分佈
- 跑完 evaluation、pipeline cleanup 時會再印一段 **mean-per-frame** 的總結表
  （格式與 A1 工具一致，方便對照）

如果結果不同（例如 standalone 工具 pair_gen 很低但 eval 內很高），通常代表：

- Standalone 工具用的 voxel 數 / dataset 分佈和真實 eval 不同
- 真實 eval 有 voxel mean-pool、前處理等不在 TRT sparse engine 裡的成本

此時以 **in-situ overlay** 為準，並回頭調整 profile 工具的 `--sample-idx`。

---

## 產出 / 對 README_PLAN 的對應

| Priority A 條目        | 本文件提供的產出                                                     |
| ---------------------- | -------------------------------------------------------------------- |
| A1 / 「時間前 3 名」   | profile 工具 top-N 表 + JSON `top_layers` + Nsight `cuda_kern_exec_sum` |
| A1 / steady-state 測速 | profile 工具 warmup/measured 分離 + CUDA-event `total_gpu_ms`        |
| A1 / TRT layer timing  | profile 工具 bucket + block roll-up + in-situ overlay                |
| A1 / Nsight Systems    | `nsys_profile_sparse.sh` + 建議命令                                  |
| A1 / Nsight Compute    | 在 `nsys` 報告鎖定 top kernel 之後手動跑 `ncu` 驗證 A2               |
| A2 / sort 成本         | `[A2] pair_gen` 行 + 三段閾值判定                                     |

一旦拿到 top-3 名單與 A2 佔比，就可以對第 8 章做新的判斷：

- 如果 top-3 全是 `implicit_gemm_int8` 且 `pair_gen` < 3% → 和文件 4.1 結論一致，主戰場
  不在 8.1、也不在 sort，應往 **Priority B / C** 前進（plugin 邊界、獨立 engine）。
- 如果 top-3 包含 `scatter_nd` / `add` / `relu` → 支持 Priority B1（融合 Add+Relu / scatter）。
- 如果 top-3 仍以 `pair_gen` 為主 → A2 有改造空間，繼續用 Nsight Compute 拆。
