#!/usr/bin/env bash
# =============================================================================
# Priority A — Nsight Systems profiling wrapper for BEVFusion sparse encoder.
#
# 作用：
#   在 `profile_sparse_encoder.py` 之上罩一層 `nsys profile`，讓 A1 的 "top-3
#   時間分佈" 可以再對照 CUDA kernel 時間軸（實際 `implicit_gemm` / argsort /
#   scan / scatter）。
#
# 先跑完 step 0 ~ step 5（build plugin / PTQ / export / transform / trt）後：
#
#   bash deployment/projects/bevfusion/benchmark/nsys_profile_sparse.sh
#
# 可用環境變數覆寫：
#   ENGINE        — bevfusion_sparse.engine 路徑
#   DEPLOY_CFG    — deploy_config_split_int8.py
#   MODEL_CFG     — model config .py
#   CHECKPOINT    — PTQ .pth（預設讀 deploy_cfg.checkpoint_path）
#   SAMPLE_IDX    — 資料集 sample index
#   WARMUP        — 熱身迭代數（預設 10；比純 Python profile 少一些以壓縮 nsys 檔）
#   ITERATIONS    — 量測迭代數（預設 30）
#   OUTPUT_PREFIX — nsys 輸出路徑前綴（`.nsys-rep` / `.sqlite` 會自動加上）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
# REPO_ROOT="$(cd "${PROJECT_DIR}/../../../.." && pwd)"
REPO_ROOT="$(pwd)"

ENGINE="${ENGINE:-${REPO_ROOT}/work_dirs/bevfusion_split_int8_deployment/tensorrt/bevfusion_sparse.engine}"
DEPLOY_CFG="${DEPLOY_CFG:-deployment/projects/bevfusion/config/deploy_config_split_int8.py}"
MODEL_CFG="${MODEL_CFG:-projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py}"
CHECKPOINT="${CHECKPOINT:-}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-30}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${REPO_ROOT}/work_dirs/bevfusion_split_int8_deployment/nsys_sparse_priorityA}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "[nsys] 'nsys' not found in PATH. Install Nsight Systems and retry." >&2
  exit 1
fi

if [[ ! -f "${ENGINE}" ]]; then
  echo "[nsys] engine not found: ${ENGINE}" >&2
  echo "       Run step 0~5 first (see docs/16_PRIORITY_A_PROFILING_USAGE.md)." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

cd "${REPO_ROOT}"
echo "[nsys] engine       = ${ENGINE}"
echo "[nsys] deploy_cfg   = ${DEPLOY_CFG}"
echo "[nsys] model_cfg    = ${MODEL_CFG}"
echo "[nsys] sample_idx   = ${SAMPLE_IDX}"
echo "[nsys] warmup/iter  = ${WARMUP} / ${ITERATIONS}"
echo "[nsys] output       = ${OUTPUT_PREFIX}.nsys-rep"

# `cuda` + `cudnn` + `nvtx` + `osrt`：夠看 implicit_gemm / argsort / launch overhead。
# 關掉 sampling 以降低 overhead；只要 CUDA 軌道足夠交叉比對 Python 端 bucket。
nsys profile \
  --force-overwrite=true \
  --output "${OUTPUT_PREFIX}" \
  --trace=cuda,nvtx,cudnn,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -- \
  python -m deployment.projects.bevfusion.benchmark.profile_sparse_encoder \
    --engine "${ENGINE}" \
    --deploy-cfg "${DEPLOY_CFG}" \
    --model-cfg "${MODEL_CFG}" \
    --sample-idx "${SAMPLE_IDX}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[nsys] done."
echo "[nsys] inspect:"
echo "  nsys stats --report gputrace ${OUTPUT_PREFIX}.nsys-rep | head -50"
echo "  nsys stats --report cuda_kern_exec_sum ${OUTPUT_PREFIX}.nsys-rep"
echo ""
echo "對照 Priority A 判定標準："
echo "  * top-3 kernel 時間是不是 implicit_gemm?"
echo "  * argsort / scan / pair-gen kernel 佔比是否 >= 10% (A2 的 gate)?"
echo "  * plugin 間是否有明顯 launch gap / D2H 同步?"
