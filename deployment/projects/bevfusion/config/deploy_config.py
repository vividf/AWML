"""
BEVFusion Deployment Configuration

Example deploy config for BEVFusion LiDAR-only (main_body module).
Adapt checkpoint_path, info_file, and shape profiles to your model.
"""

# ============================================================================
# Checkpoint Path
# ============================================================================
checkpoint_path = "work_dirs/bevfusion/epoch_30.pth"

# ============================================================================
# Device settings
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="onnx",
    work_dir="work_dirs/bevfusion_deployment",
    onnx_path=None,
)

_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Component Configuration
#
# BEVFusion exports a single ONNX model (main_body) that takes
# voxels/coors/num_points_per_voxel and outputs bbox_pred/score/label_pred.
# ============================================================================
components = dict(
    bevfusion_main_body=dict(
        onnx_file="bevfusion_lidar.onnx",
        engine_file="bevfusion_lidar.engine",
        io=dict(
            inputs=[
                dict(name="voxels", dtype="float32"),
                dict(name="coors", dtype="int32"),
                dict(name="num_points_per_voxel", dtype="int32"),
            ],
            outputs=[
                dict(name="bbox_pred", dtype="float32"),
                dict(name="score", dtype="float32"),
                dict(name="label_pred", dtype="int64"),
            ],
            dynamic_axes={
                "voxels": {0: "voxels_num"},
                "coors": {0: "voxels_num"},
                "num_points_per_voxel": {0: "voxels_num"},
            },
        ),
        tensorrt_profile=dict(
            voxels=dict(
                min_shape=[1, 10, 4],
                opt_shape=[64000, 10, 4],
                max_shape=[256000, 10, 4],
            ),
            coors=dict(
                min_shape=[1, 3],
                opt_shape=[64000, 3],
                max_shape=[256000, 3],
            ),
            num_points_per_voxel=dict(
                min_shape=[1],
                opt_shape=[64000],
                max_shape=[256000],
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
    sample_idx=0,
)

# ============================================================================
# ONNX Export Settings
# ============================================================================
onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# ============================================================================
# TensorRT Build Settings
# ============================================================================
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 32,
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=5,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=False,
            device=devices["cuda"],
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=False,
            device=devices["cuda"],
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

# ============================================================================
# Verification Configuration
# ============================================================================
verification = dict(
    enabled=False,
    tolerance=1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cuda", test_backend="onnx", test_device="cuda"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cuda", test_backend="onnx", test_device="cuda"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        none=[],
    ),
)
