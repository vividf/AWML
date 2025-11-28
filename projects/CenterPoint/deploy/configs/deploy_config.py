"""
CenterPoint Deployment Configuration (v2).

This config is designed to:
- Make export mode behavior explicit and easy to reason about.
- Separate "what to do" (mode, which backends) from "how to do it" (paths, devices).
- Make verification & evaluation rules depend on export.mode without hardcoding them in code.
"""

# ============================================================================
# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
# ============================================================================
task_type = "detection3d"

# ============================================================================
# Checkpoint Path - Single source of truth for PyTorch model
# ============================================================================
# This is the main checkpoint path used by:
# - Export workflow: to load the PyTorch model for ONNX conversion
# - Evaluation: for PyTorch backend evaluation
# - Verification: when PyTorch is used as reference or test backend
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"

# ============================================================================
# Device settings (shared by export, evaluation, verification)
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

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
    work_dir="work_dirs/centerpoint_deployment",
    # ---- ONNX source when building TensorRT only ---------------------------
    # Rule:
    # - mode == 'trt'  -> onnx_path MUST be provided (file or directory)
    # - mode in ['onnx', 'both'] -> onnx_path can be None (pipeline uses newly exported ONNX)
    onnx_path=None,  # e.g. "work_dirs/centerpoint_deployment/centerpoint.onnx"
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    # Path to info.pkl file
    info_file="data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl",
    # Sample index for export (use first sample)
    sample_idx=1,
)

# ============================================================================
# Model Input/Output Configuration
# ============================================================================
model_io = dict(
    # Primary input configuration for 3D detection
    input_name="voxels",
    input_shape=(32, 4),  # (max_points_per_voxel, point_dim); batch dim added automatically
    input_dtype="float32",
    # Additional inputs for 3D detection
    additional_inputs=[
        dict(name="num_points", shape=(-1,), dtype="int32"),  # (num_voxels,)
        dict(name="coors", shape=(-1, 4), dtype="int32"),  # (num_voxels, 4) = (batch, z, y, x)
    ],
    # Outputs (head tensors)
    output_name="reg",  # Primary output name
    additional_outputs=["height", "dim", "rot", "vel", "hm"],
    # Batch size configuration
    # - int  : fixed batch size
    # - None : dynamic batch size with dynamic_axes
    batch_size=None,
    # Dynamic axes when batch_size=None
    dynamic_axes={
        "voxels": {0: "num_voxels"},
        "num_points": {0: "num_voxels"},
        "coors": {0: "num_voxels"},
    },
)

# ============================================================================
# ONNX Export Configuration
# ============================================================================
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    save_file="centerpoint.onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
    # CenterPoint uses multi-file ONNX (voxel encoder + backbone/head)
    # When True, model_path should be a directory containing multiple .onnx files
    # When False (default), model_path should be a single .onnx file
    multi_file=True,
)

# ============================================================================
# Backend Configuration (mainly for TensorRT)
# ============================================================================
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="auto",
        # TensorRT workspace size (bytes)
        max_workspace_size=2 << 30,  # 2 GB
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input_features=dict(
                    min_shape=[1000, 32, 11],  # Minimum supported input shape
                    opt_shape=[20000, 32, 11],  # Optimal shape for performance tuning
                    max_shape=[64000, 32, 11],  # Maximum supported input shape
                ),
                spatial_features=dict(
                    min_shape=[1, 32, 760, 760],
                    opt_shape=[1, 32, 760, 760],
                    max_shape=[1, 32, 760, 760],
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
            device=devices["cuda"],  # or 'cpu'
        ),
        # ONNX evaluation
        onnx=dict(
            enabled=True,
            device=devices["cuda"],  # 'cpu' or 'cuda:0'
            # If None: pipeline will infer from export.work_dir / onnx_config.save_file
            # model_dir=None,
            model_dir="work_dirs/centerpoint_deployment/onnx/",
        ),
        # TensorRT evaluation
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],  # must be CUDA
            # If None: pipeline will infer from export.work_dir + "/tensorrt"
            # engine_dir=None,
            engine_dir="work_dirs/centerpoint_deployment/tensorrt/",
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
    enabled=False,
    tolerance=1e-1,
    num_verify_samples=1,
    # Device aliases for flexible device management
    #
    # Benefits of using aliases:
    # - Change all CPU verifications to "cuda:1"? Just update devices["cpu"] = "cuda:1"
    # - Switch ONNX verification device? Just update devices["cuda"] = "cuda:1"
    # - Scenarios reference these aliases (e.g., ref_device="cpu", test_device="cuda")
    devices=devices,
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
