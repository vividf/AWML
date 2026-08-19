#!/usr/bin/env bash
# CenterPoint PTQ tutorial — end-to-end reproduction script.
#
# Run from the AWML repo root ON THE HOST. Every heavy step runs inside the
# deployment container (bevfusion-deployment:latest = torch 2.8 + TensorRT 10.8
# + pytorch-quantization 2.1.3).
#
# Steps:
#   0. reconstruct the FP checkpoint from the released PTQ checkpoint
#   1. PTQ calibration with per-sample histogram tracing  → checkpoints/, calib_trace/
#   2. figures from the traces                            → figures/
#   3. amax comparison table (repro vs release)           → calib_trace/amax_comparison.md
#   4. FP16 deploy: ONNX export → TRT engine → eval        → fp16/
#   5. INT8 deploy: ONNX export → TRT engine → eval        → int8/
set -euo pipefail

TUTORIAL=work_dirs/centerpoint_tutorial
RELEASE_PTQ_CKPT=${RELEASE_PTQ_CKPT:-$HOME/Desktop/centerpoint_2_6_1_quant/epoch_29_ptq.pth}

in_container() {
    docker run --rm --gpus all --shm-size=32g \
        -v "$PWD:/workspace" -w /workspace \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        bevfusion-deployment:latest "$@"
}

echo "=== [0/5] Reconstruct FP checkpoint (host python, only needs torch) ==="
python3 "$TUTORIAL/scripts/00_reconstruct_fp_checkpoint.py" \
    --ptq-checkpoint "$RELEASE_PTQ_CKPT" \
    --output "$TUTORIAL/checkpoints/epoch_29_fp_reconstructed.pth"

echo "=== [1/5] PTQ calibration with histogram tracing ==="
in_container python "$TUTORIAL/scripts/01_ptq_with_histogram_trace.py" \
    --deploy-cfg "$TUTORIAL/configs/deploy_config_int8_tutorial.py" \
    --checkpoint "$TUTORIAL/checkpoints/epoch_29_fp_reconstructed.pth" \
    --output "$TUTORIAL/checkpoints/epoch_29_ptq_tutorial.pth" \
    --trace-dir "$TUTORIAL/calib_trace"

echo "=== [2/5] Calibration figures ==="
in_container python "$TUTORIAL/scripts/02_plot_calibration.py"

echo "=== [3/5] amax comparison table ==="
in_container python "$TUTORIAL/scripts/03_compare_amax_table.py" \
    | tee "$TUTORIAL/calib_trace/amax_comparison.md"

echo "=== [4/5] FP16 (before-PTQ) deploy: ONNX -> TensorRT -> eval ==="
in_container python -m deployment.cli.main centerpoint \
    "$TUTORIAL/configs/deploy_config_fp16_tutorial.py"

echo "=== [5/5] INT8 (after-PTQ) deploy: ONNX -> TensorRT -> eval ==="
in_container python -m deployment.cli.main centerpoint \
    "$TUTORIAL/configs/deploy_config_int8_tutorial.py"

echo "Done. See $TUTORIAL/README.md"
