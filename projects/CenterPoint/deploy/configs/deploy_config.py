"""
CenterPoint Deployment Configuration.

This is an example deployment config for CenterPoint.
Modify according to your needs.
"""

# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
task_type = "detection3d"

# Export settings
export = dict(
    mode="both",  # Export both ONNX and TensorRT
    verify=True,  # Enable verification after export
    # Device configuration:
    # - ONNX export always uses CPU (for numerical consistency)
    # - TensorRT export always uses cuda:0 (required)
    work_dir="work_dirs/centerpoint_deployment",
)

# Runtime I/O settings
runtime_io = dict(
    # Path to info.pkl file
    info_file="data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl",
    # Sample index for export (use first sample)
    sample_idx=0,
    # Optional: path to existing ONNX file (for eval-only mode)
    # onnx_file='work_dirs/centerpoint_deployment/centerpoint.onnx',
)

# ==============================================================================
# Model Input/Output Configuration
# ==============================================================================
model_io = dict(
    # Input configuration for 3D detection
    # Note: CenterPoint has multiple inputs with variable dimensions
    input_name="voxels",  # Primary input name
    input_shape=(32, 4),  # (max_points_per_voxel, point_dim) - batch dimension will be added automatically
    input_dtype="float32",
    
    # Additional inputs for 3D detection
    additional_inputs=[
        dict(name="num_points", shape=(-1,), dtype="int32"),  # (num_voxels,)
        dict(name="coors", shape=(-1, 4), dtype="int32"),  # (num_voxels, 4) where 4 = (batch_idx, z, y, x)
    ],
    
    # Output configuration  
    output_name="reg",  # Primary output name
    additional_outputs=["height", "dim", "rot", "vel", "hm"],
    
    # Batch size configuration
    # Options:
    # - int: Fixed batch size (e.g., 1, 2)
    # - None: Dynamic batch size (uses dynamic_axes)
    batch_size=None,  # Dynamic batch size for flexible inference
    
    # Dynamic axes (only used when batch_size=None)
    # When batch_size is set to a number, this is automatically set to None
    # When batch_size is None, this defines dynamic batch dimensions
    dynamic_axes={
        "voxels": {0: "num_voxels"}, 
        "num_points": {0: "num_voxels"}, 
        "coors": {0: "num_voxels"}
    },
)

# ==============================================================================
# ONNX Export Configuration
# ==============================================================================
onnx_config = dict(
    opset_version=16,  # CenterPoint typically uses opset 13
    do_constant_folding=True,
    save_file="centerpoint.onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=True,
)

# ==============================================================================
# Backend Configuration
# ==============================================================================
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'fp32', 'strongly_typed'
        # - 'fp32': Pure FP32 (no TF32, FP16, INT8) - use for debugging numerical differences
        # - 'fp32_tf32': TF32 for FP32 operations (default for production)
        # Note: 'strongly_typed' caused errors, reverting to 'fp32_tf32'
        # The verification will use CPU PyTorch reference to handle CUDA numerical differences
        precision_policy="fp32",  # Try 'fp32' for pure FP32 to debug head output differences
        # TensorRT workspace size (bytes)
        max_workspace_size=2 << 30,  # 2 GB (3D models need more memory)
        # Disable aggressive optimizations for better numerical accuracy
        # Set to True to reduce numerical differences (may reduce performance)
        disable_optimizations=False,  # Try True if verification fails due to TensorRT optimizations
    ),
    # Model inputs will be generated automatically from model_io configuration
    # This will be populated by the deployment pipeline based on model_io settings
    model_inputs=None,  # Will be set automatically
)

# Evaluation configuration
evaluation = dict(
    enabled=False,  # Enable evaluation
    num_samples=1,  # Number of samples to evaluate (3D is slower)
    verbose=True,  # Detailed per-sample output for debugging
    # Specify models to evaluate (comment out or remove paths for backends you don't want to evaluate)
    models=dict(
        pytorch="work_dirs/centerpoint/best_checkpoint.pth",  # PyTorch checkpoint
        onnx="work_dirs/centerpoint_deployment",  # Path to ONNX model directory
        tensorrt="work_dirs/centerpoint_deployment/tensorrt",  # Path to TensorRT engine directory
    ),
    # Device configuration for evaluation:
    # - ONNX evaluation device (can choose CPU or GPU for performance testing)
    # - TensorRT evaluation always uses cuda:0 (required)
    # - PyTorch evaluation device (can choose CPU or GPU)
    onnx_device="cuda:0",  # ONNX evaluation device: "cpu" or "cuda:0"
    tensorrt_device="cuda:0",  # TensorRT evaluation device (always cuda:0)
    pytorch_device="cuda:0",  # PyTorch evaluation device: "cpu" or "cuda:0"
)

# Verification configuration
verification = dict(
    enabled=True,  # Will use export.verify
    tolerance=1e-1,  # Slightly higher tolerance for 3D detection
    num_verify_samples=1,  # Fewer samples for 3D (slower)
    # Device configuration for verification (fixed):
    # - ONNX verification always uses CPU (for numerical consistency with export)
    # - TensorRT verification always uses cuda:0 (required)
    # - PyTorch reference uses cuda:0 (for consistency with TensorRT)
)
