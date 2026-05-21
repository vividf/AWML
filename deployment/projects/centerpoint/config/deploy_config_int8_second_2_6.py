"""
CenterPoint INT8 Quantization Deployment Configuration - SECOND Backbone (2.5)

Usage:
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_second_2_5.py \
        projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py
"""

# ============================================================================
# Checkpoint Path - Use PTQ quantized checkpoint
# ============================================================================
# checkpoint_path = "models/2_5/experiment_j6_gen2/second/epoch_30_ptq.pth"
# work_dirs/centerpoint/centerpoint_2_5_best_epoch_28.pth
checkpoint_path = "vivid/bench_comparison/centerpoint_2_6/epoch_29_ptq.pth"


deploy_log_path = "deployment.log"

# ============================================================================
# Quantization Configuration
# ============================================================================
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    # SECOND backbone selective skip (prefix: pts_backbone.blocks.{i})
    # Use explicit stage list only.
    # If still unstable, extend to [0, 1] or [0, 1, 2] and re-run PTQ with same setting.
    skip_backbone_stages=[0],
    sensitive_layers=[],
)

# ============================================================================
# Device settings
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)


# Single literal for deployment output root (used before `export` exists).
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_2_6_skip_stage_0_by_distance_largebus"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
)


# ============================================================================
# Unified Component Configuration
# ============================================================================
components = dict(
    pts_voxel_encoder=dict(
        onnx_file="pts_voxel_encoder.onnx",
        engine_file="pts_voxel_encoder.engine",
        io=dict(
            inputs=[
                dict(name="input_features", dtype="float32"),
            ],
            outputs=[
                dict(name="pillar_features", dtype="float32"),
            ],
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        ),
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 11],
                opt_shape=[20000, 32, 11],
                max_shape=[96000, 32, 11],
            ),
        ),
    ),
    pts_backbone_neck_head=dict(
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(
            inputs=[
                dict(name="spatial_features", dtype="float32"),
            ],
            outputs=[
                dict(name="heatmap", dtype="float32"),
                dict(name="reg", dtype="float32"),
                dict(name="height", dtype="float32"),
                dict(name="dim", dtype="float32"),
                dict(name="rot", dtype="float32"),
                dict(name="vel", dtype="float32"),
            ],
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                "heatmap": {0: "batch_size", 2: "height", 3: "width"},
                "reg": {0: "batch_size", 2: "height", 3: "width"},
                "height": {0: "batch_size", 2: "height", 3: "width"},
                "dim": {0: "batch_size", 2: "height", 3: "width"},
                "rot": {0: "batch_size", 2: "height", 3: "width"},
                "vel": {0: "batch_size", 2: "height", 3: "width"},
            },
        ),
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 1020, 1020],
                opt_shape=[1, 32, 1020, 1020],
                max_shape=[1, 32, 1020, 1020],
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="info/kokseang_2_6_1/t4dataset_largebus_infos_test.pkl",
    sample_idx=1,
)

# ============================================================================
# ONNX Export Settings
# ============================================================================
onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    visualize_qdq_values=True,
    simplify=False,
)

# ============================================================================
# TensorRT Build Settings
# ============================================================================
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=4 << 30,
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=-1,
    num_warmup_samples=2,
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
            enabled=True,
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
    tolerance=1e-1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        none=[],
    ),
)
