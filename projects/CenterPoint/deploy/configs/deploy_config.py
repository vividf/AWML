"""
CenterPoint Deployment Configuration.

This is an example deployment config for CenterPoint.
Modify according to your needs.
"""

# Export settings
export = dict(
    mode="both",  # Export both ONNX and TensorRT
    verify=True,  # Disable verification to save time
    device="cpu",  # Use CUDA for TensorRT
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
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="auto",
        # TensorRT workspace size (bytes)
        max_workspace_size=2 << 30,  # 2 GB (3D models need more memory)
    ),
    # Model inputs will be generated automatically from model_io configuration
    # This will be populated by the deployment pipeline based on model_io settings
    model_inputs=None,  # Will be set automatically
)

# Evaluation configuration
evaluation = dict(
    enabled=True,  # Enable evaluation
    num_samples=5,  # Number of samples to evaluate (3D is slower)
    verbose=True,  # Detailed per-sample output for debugging
    # Specify models to evaluate (comment out or remove paths for backends you don't want to evaluate)
    models=dict(
        pytorch="work_dirs/centerpoint/best_checkpoint.pth",  # PyTorch checkpoint
        onnx="work_dirs/centerpoint_deployment",  # Path to ONNX model directory
        tensorrt="work_dirs/centerpoint_deployment/tensorrt",  # Path to TensorRT engine directory
    ),
)

# Verification configuration
verification = dict(
    enabled=False,  # Will use export.verify
    tolerance=1e-1,  # Slightly higher tolerance for 3D detection
    num_verify_samples=1,  # Fewer samples for 3D (slower)
)
