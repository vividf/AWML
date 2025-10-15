"""
CenterPoint Deployment Configuration.

This is an example deployment config for CenterPoint.
Modify according to your needs.
"""

# Export settings
export = dict(
    mode="both",  # 'onnx', 'trt', 'both', 'none'
    verify=True,  # Enable cross-backend verification
    device="cuda:0",  # Device for export/inference
    work_dir="work_dirs/centerpoint_deployment",
)

# Runtime I/O settings
runtime_io = dict(
    # Path to info.pkl file
    info_file="data/t4dataset/centerpoint_infos_val.pkl",
    # Sample index for export (use first sample)
    sample_idx=0,
    # Optional: path to existing ONNX file (for eval-only mode)
    # onnx_file='work_dirs/centerpoint_deployment/centerpoint.onnx',
)

# ONNX configuration
onnx_config = dict(
    opset_version=13,  # CenterPoint typically uses opset 13
    do_constant_folding=True,
    input_names=["voxels", "num_points", "coors"],
    output_names=["reg", "height", "dim", "rot", "vel", "hm"],
    save_file="centerpoint.onnx",
    export_params=True,
    dynamic_axes={"voxels": {0: "num_voxels"}, "num_points": {0: "num_voxels"}, "coors": {0: "num_voxels"}},
    keep_initializers_as_inputs=False,
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="fp16",
        # TensorRT workspace size (bytes)
        max_workspace_size=2 << 30,  # 2 GB (3D models need more memory)
    ),
    # Optional: specify input shapes for TensorRT
    # Note: 3D detection has variable number of voxels
    model_inputs=[
        dict(name="voxels", shape=(-1, 32, 4), dtype="float32"),  # (num_voxels, max_points_per_voxel, point_dim)
        dict(name="num_points", shape=(-1,), dtype="int32"),  # (num_voxels,)
        dict(name="coors", shape=(-1, 4), dtype="int32"),  # (num_voxels, 4) where 4 = (batch_idx, z, y, x)
    ],
)

# Evaluation configuration
evaluation = dict(
    enabled=True,  # Enable evaluation
    num_samples=50,  # Number of samples to evaluate (3D is slower)
    verbose=False,  # Detailed per-sample output
    # Specify models to evaluate (comment out or remove paths for backends you don't want to evaluate)
    models=dict(
        # pytorch="work_dirs/centerpoint_deployment/checkpoint.pth",  # Optional: PyTorch checkpoint
        # onnx="work_dirs/centerpoint_deployment",  # Path to ONNX model directory
        # tensorrt="work_dirs/centerpoint_deployment",  # Path to TensorRT engine directory
    ),
)

# Verification configuration
verification = dict(
    enabled=True,  # Will use export.verify
    tolerance=1e-2,  # Slightly higher tolerance for 3D detection
    num_verify_samples=5,  # Fewer samples for 3D (slower)
)
