"""
YOLOX_opt_elan Deployment Configuration (v2).

This config uses the new policy-based verification architecture.
"""

# ============================================================================
# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
# ============================================================================
task_type = "detection2d"

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
    work_dir="work_dirs/yolox_opt_elan_deployment",

    # ---- Source for ONNX export --------------------------------------------
    # Rule:
    # - mode in ['onnx', 'both']  -> checkpoint_path MUST be provided
    # - mode == 'trt'             -> checkpoint_path is ignored
    checkpoint_path="work_dirs/old_yolox_elan/yolox_epoch24.pth",

    # ---- ONNX source when building TensorRT only ---------------------------
    # Rule:
    # - mode == 'trt'  -> onnx_path MUST be provided (file or directory)
    # - mode in ['onnx', 'both'] -> onnx_path can be None (pipeline uses newly exported ONNX)
    onnx_path=None,  # e.g. "work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx"
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    # Path to T4Dataset annotation file
    ann_file="data/t4dataset/2d_info/yolox_infos_val.json",
    # Path to images directory (can be empty if full paths are in annotations)
    img_prefix="",
    # Sample index for export (use first sample)
    sample_idx=0,
)

# Model input/output configuration
model_io = dict(
    # Input configuration
    input_name="images",
    input_shape=(3, 960, 960),  # (C, H, W) - batch dimension will be added automatically
    input_dtype="float32",
    
    # Output configuration  
    output_name="output",
    
    # Batch size configuration
    # Options:
    # - int: Fixed batch size (e.g., 1, 6)
    # - None: Dynamic batch size (uses dynamic_axes)
    batch_size=1,  # Set to 6 to match old ONNX exactly, or None for dynamic
    
    # Dynamic axes (only used when batch_size=None)
    # When batch_size is set to a number, this is automatically set to None
    # When batch_size is None, this defines dynamic batch dimensions
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    save_file="yolox_opt_elan.onnx",
    keep_initializers_as_inputs=False,
    simplify=True,
    # Model wrapper configuration for ONNX export
    # num_classes will be automatically extracted from model.bbox_head.num_classes
    model_wrapper=dict(
        type='yolox',
        # num_classes is optional - will be auto-extracted from model if not provided
    ),
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="auto",
        # TensorRT workspace size (bytes)
        max_workspace_size=1 << 30,  # 1 GB
    ),
    # Model inputs will be generated automatically from model_io configuration
    # This will be populated by the deployment pipeline based on model_io settings
    model_inputs=None,  # Will be set automatically
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=1,      # Number of samples to evaluate
    verbose=True,

    # Decide which backends to evaluate and on which devices.
    # Note:
    # - tensorrt.device MUST be a CUDA device (e.g., 'cuda:0')
    # - For 'none' export mode, all models must already exist on disk.
    backends=dict(
        # PyTorch evaluation
        pytorch=dict(
            enabled=True,
            device="cuda:0",  # or 'cpu'
            checkpoint="work_dirs/old_yolox_elan/yolox_epoch24.pth",
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
        cpu="cpu",      # Alias for CPU device
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
