# projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py
# Deployment configuration for ResNet18 5-channel calibration classification model
#
# 1. export: Controls export behavior (mode, verification, device, output directory)
# 2. runtime_io: Runtime I/O configuration (data paths, sample selection)
# 3. Backend configs: ONNX and TensorRT specific settings

# ==============================================================================
# Export Configuration
# ==============================================================================
export = dict(
    mode="both",  # Export mode: "onnx", "trt", or "both"
    # - "onnx": Export to ONNX only
    # - "trt": Convert to TensorRT only (requires onnx_file in runtime_io)
    # - "both": Export to ONNX then convert to TensorRT
    verify=True,  # Run verification comparing PyTorch/ONNX/TRT outputs
    device="cuda:0",  # Device for export (use "cuda:0" or "cpu")
    # Note: TensorRT always requires CUDA, will auto-switch if needed
    work_dir="/workspace/work_dirs",  # Output directory for exported models
)

# ==============================================================================
# Runtime I/O Configuration
# ==============================================================================
runtime_io = dict(
    info_pkl="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,  # Sample index to use for export and verification
    onnx_file=None,  # Optional: Path to existing ONNX file
    # - If provided with mode="trt", will convert this ONNX to TensorRT
    # - If None with mode="both", will export ONNX first then convert
)

# ==============================================================================
# Codebase Configuration
# ==============================================================================
codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")

# ==============================================================================
# TensorRT Backend Configuration
# ==============================================================================
backend_config = dict(
    type="tensorrt",
    common_config=dict(
        max_workspace_size=1 << 30,  # 1 GiB workspace for TensorRT
        # Precision policy controls how TensorRT handles numerical precision:
        # - "auto": TensorRT automatically selects precision (default)
        # - "fp16": Enable FP16 mode for faster inference with slight accuracy trade-off
        # - "fp32_tf32": Enable TF32 mode (Tensor Cores for FP32 operations on Ampere+)
        # - "strongly_typed": Enforce strict type checking (prevents automatic precision conversion)
        precision_policy="fp32_tf32",
    ),
    # Dynamic shape configuration for different input resolutions
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 5, 1080, 1920],  # Minimum supported input shape
                    opt_shape=[1, 5, 1860, 2880],  # Optimal shape for performance tuning
                    max_shape=[1, 5, 2160, 3840],  # Maximum supported input shape
                ),
            )
        )
    ],
)

# ==============================================================================
# ONNX Export Configuration
# ==============================================================================
onnx_config = dict(
    type="onnx",
    export_params=True,  # Include trained parameters in the model
    keep_initializers_as_inputs=False,  # Don't expose initializers as inputs
    opset_version=16,  # ONNX opset version (16 recommended for modern features)
    do_constant_folding=True,  # Optimize by folding constant expressions
    save_file="end2end.onnx",  # Output filename
    input_names=["input"],  # Name of input tensor
    output_names=["output"],  # Name of output tensor
    # Dynamic axes for variable input dimensions
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},  # Batch, H, W are dynamic
        "output": {0: "batch_size"},  # Batch is dynamic
    },
    input_shape=None,
)

# ==============================================================================
# Usage Examples
# ==============================================================================
#
# Basic usage (export to both ONNX and TensorRT):
#   python projects/CalibrationStatusClassification/deploy/main.py \
#       projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
#       projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
#       checkpoint.pth
#
# Export to ONNX only:
#   Set export.mode = "onnx" in this config, then run the command above
#
# Convert existing ONNX to TensorRT:
#   1) Set export.mode = "trt"
#   2) Set runtime_io.onnx_file = "/path/to/existing/model.onnx"
#   3) Run the command above
#
# Override config settings via command line:
#   python projects/CalibrationStatusClassification/deploy/main.py \
#       <deploy_cfg> <model_cfg> <checkpoint> \
#       --work-dir ./custom_output \
#       --device cuda:1 \
#       --info-pkl /path/to/custom/info.pkl \
#       --sample-idx 5
#
# Available precision policies:
#   - auto: Let TensorRT decide (default, good balance)
#   - fp16: Faster inference, ~2x speedup, small accuracy loss
#   - fp32_tf32: Use Tensor Cores on Ampere+ GPUs for FP32
#   - explicit_int8: INT8 quantization (requires calibration dataset)
#   - strongly_typed: Strict type enforcement, no automatic conversion
