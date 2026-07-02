#!/usr/bin/env bash
# Build the ImplicitGemmInt8 TRT plugin and run end-to-end INT8 benchmark.
#
# Usage:
#   bash deployment/projects/bevfusion_l/benchmark/build_and_test_int8_plugin.sh \
#       --config <mmengine_config> \
#       --checkpoint <ptq_checkpoint> \
#       --onnx <fp16_onnx_model>
#
# Prerequisites:
#   - CUDA toolkit, TensorRT, spconv/cumm installed
#   - PTQ checkpoint with NVIDIA _amax calibration values
#   - Standard FP16 ONNX model exported via deployment.cli.main

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/cpp/int8_plugin/build"
PLUGIN_DIR="${BUILD_DIR}"

# ─── Parse arguments ─────────────────────────────────────────────────────────
CONFIG=""
CHECKPOINT=""
ONNX=""
OUTPUT_DIR="work_dirs/bevfusion/int8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --onnx)       ONNX="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" || -z "$ONNX" ]]; then
  echo "Usage: $0 --config <cfg> --checkpoint <ckpt> --onnx <onnx>"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Build the ImplicitGemmInt8 plugin
# ═══════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "Step 1: Building ImplicitGemmInt8 TRT plugin"
echo "════════════════════════════════════════════════════════════"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

PLUGIN_LIB="${BUILD_DIR}/libimplicit_gemm_int8_plugin.so"
if [[ ! -f "$PLUGIN_LIB" ]]; then
  echo "ERROR: Plugin build failed — $PLUGIN_LIB not found"
  exit 1
fi
echo "Plugin built: ${PLUGIN_LIB}"
cd /workspace

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Transform ONNX (ImplicitGemm → ImplicitGemmInt8)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 2: Transforming ONNX to INT8 ImplicitGemmInt8 nodes"
echo "════════════════════════════════════════════════════════════"

INT8_ONNX="${OUTPUT_DIR}/sparse_encoder_int8.onnx"

python -m deployment.projects.bevfusion_l.export.sparse_int8_onnx_transform \
    --onnx "${ONNX}" \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --output "${INT8_ONNX}"

echo "INT8 ONNX: ${INT8_ONNX}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Validate PyTorch INT8 mAP (sanity check)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 3: Validating PyTorch INT8 mAP"
echo "════════════════════════════════════════════════════════════"

python -m deployment.cli.main bevfusion \
    deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \
    "${CONFIG}" \
    --eval-only pytorch 2>&1 | tail -30

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Build and run TRT engine with INT8 plugin
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 4: Testing TensorRT with ImplicitGemmInt8 plugin"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To use the INT8 plugin with your TensorRT pipeline, add to"
echo "deploy_config_split_int8.py -> tensorrt_config -> plugin_libraries:"
echo ""
echo "  plugin_libraries=["
echo "      '/opt/plugins/libautoware_tensorrt_plugins.so',"
echo "      '${PLUGIN_LIB}',"
echo "  ]"
echo ""
echo "WARNING: Do NOT use LD_PRELOAD — it causes symbol conflicts with"
echo "Python's spconv (basic_string::_M_create). The plugin is loaded"
echo "automatically by TRT via plugin_libraries at engine build time."
echo ""
echo "Then run evaluation with the INT8-transformed ONNX:"
echo "  python -m deployment.cli.main bevfusion \\"
echo "      deployment/projects/bevfusion_l/config/deploy_config_split_int8.py \\"
echo "      ${CONFIG}"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "sparse INT8 Plugin Build & Transform Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Files produced:"
echo "  Plugin:    ${PLUGIN_LIB}"
echo "  INT8 ONNX: ${INT8_ONNX}"
echo ""
echo "Next steps:"
echo "  1. Add plugin to deploy_config plugin_libraries (NOT LD_PRELOAD)"
echo "  2. Update deploy config to use the INT8 ONNX"
echo "  3. Run TRT engine build + evaluation"
echo "  4. Compare latency: FP16 ImplicitGemm vs INT8 ImplicitGemmInt8"
