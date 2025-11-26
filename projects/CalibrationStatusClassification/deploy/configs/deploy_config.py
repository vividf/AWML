"""
CalibrationStatusClassification Deployment Configuration (v2).

This config uses the new policy-based verification architecture.
"""

# ============================================================================
# Codebase Configuration
# ============================================================================
codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")


# ============================================================================
# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
# ============================================================================
task_type = "classification"

# ============================================================================
# Checkpoint Path - Single source of truth for PyTorch model
# ============================================================================
# This is the main checkpoint path used by:
# - Export workflow: to load the PyTorch model for ONNX conversion
# - Evaluation: for PyTorch backend evaluation
# - Verification: when PyTorch is used as reference or test backend
checkpoint_path = "work_dirs/calibration_classifier/best_accuracy_top1_epoch_28.pth"

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    # Export mode:
    # - 'onnx' : export PyTorch -> ONNX
    # - 'trt'  : build TensorRT engine from an existing ONNX
    # - 'both' : export PyTorch -> ONNX -> TensorRT
    # - 'none' : no export (only evaluation / verification on existing artifacts)
    mode="both",
    # ---- Common options ----------------------------------------------------
    work_dir="work_dirs/calibration_classifier",
    # ---- ONNX source when building TensorRT only ---------------------------
    # Rule:
    # - mode == 'trt'  -> onnx_path MUST be provided (file or directory)
    # - mode in ['onnx', 'both'] -> onnx_path can be None (pipeline uses newly exported ONNX)
    onnx_path=None,  # e.g. "/workspace/work_dirs/end2end.onnx"
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_pkl="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,  # Sample index to use for export and verification
)


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

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=1,  # Number of samples to evaluate
    verbose=True,
    # Decide which backends to evaluate and on which devices.
    # Note:
    # - tensorrt.device MUST be a CUDA device (e.g., 'cuda:0')
    # - For 'none' export mode, all models must already exist on disk.
    # - PyTorch backend uses top-level checkpoint_path (no need to specify here)
    backends=dict(
        # PyTorch evaluation (uses top-level checkpoint_path)
        pytorch=dict(
            enabled=True,
            device="cuda:0",  # or 'cpu'
        ),
        # ONNX evaluation
        onnx=dict(
            enabled=True,
            device="cuda:0",  # 'cpu' or 'cuda:0'
            # If None: pipeline will infer from export.work_dir / onnx_config.save_file
            model_dir=None,
        ),
        # TensorRT evaluation
        tensorrt=dict(
            enabled=True,
            device="cuda:0",  # must be CUDA
            # If None: pipeline will infer from export.work_dir + "/tensorrt"
            engine_dir=None,
        ),
    ),
)

# ============================================================================
# Verification Configuration
# ============================================================================
# This block defines *scenarios* per export.mode, so the pipeline does not
# need many if/else branches; it just chooses the policy based on export["mode"].
# ----------------------------------------------------------------------------
verification = dict(
    # Master switch to enable/disable verification
    enabled=True,
    tolerance=1e-1,
    num_verify_samples=1,
    # Device aliases for flexible device management
    #
    # Benefits of using aliases:
    # - Change all CPU verifications to "cuda:1"? Just update devices["cpu"] = "cuda:1"
    # - Switch ONNX verification device? Just update devices["cuda"] = "cuda:1"
    # - Scenarios reference these aliases (e.g., ref_device="cpu", test_device="cuda")
    devices=dict(
        cpu="cpu",  # Alias for CPU device
        cuda="cuda:0",  # Alias for CUDA device (can be changed to cuda:1, cuda:2, etc.)
    ),
    # Verification scenarios per export mode
    #
    # Each policy is a list of comparison pairs:
    #   - ref_backend   : reference backend ('pytorch' or 'onnx')
    #   - ref_device    : device alias (e.g., "cpu", "cuda") - resolved via devices dict above
    #   - test_backend  : backend under test ('onnx' or 'tensorrt')
    #   - test_device   : device alias (e.g., "cpu", "cuda") - resolved via devices dict above
    #
    # Pipeline resolves devices like: actual_device = verification["devices"][policy["ref_device"]]
    #
    # This structure encodes:
    # - 'both':
    #     1) PyTorch(cpu) vs ONNX(cpu)
    #     2) ONNX(cuda)   vs TensorRT(cuda)
    # - 'onnx':
    #     1) PyTorch(cpu) vs ONNX(cpu)
    # - 'trt':
    #     1) ONNX(cuda)   vs TensorRT(cuda)  (using provided ONNX)
    scenarios=dict(
        both=[
            dict(
                ref_backend="pytorch",
                ref_device="cpu",
                test_backend="onnx",
                test_device="cpu",
            ),
            dict(
                ref_backend="onnx",
                ref_device="cuda",
                test_backend="tensorrt",
                test_device="cuda",
            ),
        ],
        onnx=[
            dict(
                ref_backend="pytorch",
                ref_device="cpu",
                test_backend="onnx",
                test_device="cpu",
            ),
        ],
        trt=[
            dict(
                ref_backend="onnx",
                ref_device="cuda",
                test_backend="tensorrt",
                test_device="cuda",
            ),
        ],
        none=[],
    ),
)
