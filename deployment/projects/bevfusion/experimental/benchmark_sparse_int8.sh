#!/bin/bash
# =============================================================================
# BEVFusion INT8 Sparse Encoder Benchmark
#
# End-to-end workflow:
#   1. Export sparse encoder to libspconv INT8 ONNX
#   2. Build C++ benchmark
#   3. Run latency benchmark (INT8 vs FP16)
#   4. Validate PyTorch INT8 mAP
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # deployment/projects/bevfusion/experimental
PROJECT_DIR="${SCRIPT_DIR}/.."                                       # deployment/projects/bevfusion
WORKSPACE="${PROJECT_DIR}/../../../.."
BUILD_DIR="${SCRIPT_DIR}/cpp/build"                                  # experimental/cpp (Method-2 build)

# Paths (override via environment)
CONFIG="${CONFIG:-projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py}"
CHECKPOINT="${CHECKPOINT:-work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth}"
SPARSE_ONNX="${SPARSE_ONNX:-work_dirs/bevfusion/sparse_encoder_int8.xyz.onnx}"
DEPLOY_CFG="${DEPLOY_CFG:-deployment/projects/bevfusion/config/deploy_config_split_int8.py}"

echo "============================================================"
echo "BEVFusion INT8 Sparse Encoder Benchmark"
echo "============================================================"

# ---------------------------------------------------------
# Step 1: Export sparse encoder to libspconv ONNX
# ---------------------------------------------------------
echo ""
echo "[Step 1] Exporting sparse encoder to libspconv INT8 ONNX..."
echo "  Config: ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output: ${SPARSE_ONNX}"

cd "${WORKSPACE}"
python -m deployment.projects.bevfusion.experimental.export_sparse_encoder_int8 \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${SPARSE_ONNX}"

echo "  Export complete: ${SPARSE_ONNX}"

# ---------------------------------------------------------
# Step 2: Build C++ benchmark (optional, skip if no cmake)
# ---------------------------------------------------------
if command -v cmake &> /dev/null; then
    echo ""
    echo "[Step 2] Building C++ benchmark..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBSPCONV_ROOT="${WORKSPACE}/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv" \
        2>&1 | tail -5

    if make -j$(nproc) 2>&1 | tail -5; then
        echo "  Build successful"

        echo ""
        echo "[Step 3] Running C++ latency benchmark..."

        # INT8
        echo "--- INT8 ---"
        ./benchmark_sparse_int8 \
            --sparse-onnx "${WORKSPACE}/${SPARSE_ONNX}" \
            --warmup 20 \
            --iterations 200 \
            --num-voxels 40000

        # FP16 (same ONNX, but override precision)
        echo ""
        echo "--- FP16 ---"
        ./benchmark_sparse_int8 \
            --sparse-onnx "${WORKSPACE}/${SPARSE_ONNX}" \
            --fp16 \
            --warmup 20 \
            --iterations 200 \
            --num-voxels 40000
    else
        echo "  Build failed (missing TensorRT or libspconv headers). Skipping C++ benchmark."
    fi
else
    echo ""
    echo "[Step 2-3] Skipping C++ benchmark (cmake not found)."
fi

# ---------------------------------------------------------
# Step 4: Validate PyTorch INT8 mAP
# ---------------------------------------------------------
echo ""
echo "[Step 4] Validating PyTorch INT8 mAP..."
cd "${WORKSPACE}"
python -m deployment.cli.main bevfusion \
    "${DEPLOY_CFG}" \
    "${CONFIG}"

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo ""
echo "Files produced:"
echo "  - ${SPARSE_ONNX} (libspconv INT8 ONNX)"
echo ""
echo "Next steps:"
echo "  1. Compare sparse encoder latency: FP16 (Autoware plugin) vs INT8 (libspconv)"
echo "  2. Integrate libspconv_trt_bridge into Autoware perception pipeline"
echo "============================================================"
