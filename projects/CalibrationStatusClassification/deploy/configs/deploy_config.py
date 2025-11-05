# Deployment configuration for ResNet18 5-channel calibration classification model
#
# 1. export: Controls export behavior (mode, verification, device, output directory)
# 2. runtime_io: Runtime I/O configuration (data paths, sample selection)
# 3. Backend configs: ONNX and TensorRT specific settings

# ==============================================================================
# Export Configuration
# ==============================================================================
export = dict(
    mode="both",  # Export mode: "onnx", "trt", "both", or "none"
    # - "onnx": Export to ONNX only
    # - "trt": Convert to TensorRT only (requires onnx_file in runtime_io)
    # - "both": Export to ONNX then convert to TensorRT
    # - "none": Skip export, only run evaluation on existing models
    #           (requires evaluation.onnx_model and/or evaluation.tensorrt_model)
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
    onnx_file="/workspace/work_dirs/end2end.onnx",  # Optional: Path to existing ONNX file
    # - If provided with mode="trt", will convert this ONNX to TensorRT
    # - If None with mode="both", will export ONNX first then convert
)

# ==============================================================================
# Evaluation Configuration
# ==============================================================================
evaluation = dict(
    enabled=True,  # Enable full model evaluation (set to True to run evaluation)
    num_samples=1,  # Number of samples to evaluate from info.pkl
    verbose=True,  # Enable verbose logging showing per-sample results
    # Specify models to evaluate
    models=dict(
        onnx="/workspace/work_dirs/end2end.onnx",  # Path to ONNX model file
        tensorrt="/workspace/work_dirs/end2end.engine",  # Path to TensorRT engine file
        # pytorch="/workspace/work_dirs/best_accuracy_top1_epoch_28.pth",  # Optional: PyTorch checkpoint
    ),
)

# ==============================================================================
# Codebase Configuration
# ==============================================================================
codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")

# ==============================================================================
# Model Input/Output Configuration
# ==============================================================================
model_io = dict(
    # Input configuration
    input_name="input",
    input_shape=(5, 1860, 2880),  # (C, H, W) - batch dimension will be added automatically
    input_dtype="float32",
    
    # Output configuration  
    output_name="output",
    
    # Batch size configuration
    # Options:
    # - int: Fixed batch size (e.g., 1, 2)
    # - None: Dynamic batch size (uses dynamic_axes)
    batch_size=None,  # Dynamic batch size for flexible inference
    
    # Dynamic axes (only used when batch_size=None)
    # When batch_size is set to a number, this is automatically set to None
    # When batch_size is None, this defines dynamic batch dimensions
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"},
    },
)

# ==============================================================================
# ONNX Export Configuration
# ==============================================================================
onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=16,
    do_constant_folding=True,
    save_file="end2end.onnx",
    simplify=True,
)

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
        precision_policy="fp16",
    ),
    # Dynamic shape configuration for different input resolutions
    # TensorRT needs shape ranges for optimization even with dynamic batch size
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
