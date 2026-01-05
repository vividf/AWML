"""
CenterPoint INT8 Quantization Deployment Configuration

This configuration extends the base deploy_config.py with quantization settings
for deploying PTQ (Post-Training Quantization) or QAT (Quantization-Aware Training)
models to TensorRT INT8.

Usage:
    python projects/CenterPoint/deploy/main.py \
        projects/CenterPoint/deploy/configs/deploy_config_int8.py \
        projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py
"""

# ============================================================================
# Task type for pipeline building
# ============================================================================
task_type = "detection3d"

# ============================================================================
# Checkpoint Path - Use PTQ or QAT quantized checkpoint
# ============================================================================
checkpoint_path = "work_dirs/centerpoint_ptq.pth"

# ============================================================================
# Quantization Configuration
# ============================================================================
# This tells the deployment pipeline to apply quantization transformations
# (BN fusion, Q/DQ node insertion) before loading the checkpoint.
quantization = dict(
    enabled=True,
    mode="ptq",  # 'ptq' or 'qat'
    fuse_bn=True,  # BatchNorm was fused during PTQ
    # Match the PTQ graph you exported. If these don't match, checkpoint keys
    # may not align with the model structure built during deployment.
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    # Optional: skip quantizing early backbone stages (maps to pts_backbone.blocks.<idx>)
    skip_backbone_first_stages=0,
    skip_backbone_stages=[],
    # Layers that were skipped during quantization
    # Note: ConvTranspose2d (deblocks) are excluded because TensorRT has
    # limited INT8 support for transposed convolutions
    sensitive_layers=[
        # "pts_neck.deblocks.0.0",  # ConvTranspose2d - no TRT INT8 support
        # "pts_neck.deblocks.1.0",  # ConvTranspose2d - no TRT INT8 support
        # "pts_neck.deblocks.2.0",  # ConvTranspose2d - no TRT INT8 support
    ],
)

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
    mode="both",  # Export ONNX -> TensorRT
    work_dir="work_dirs/centerpoint_int8_deployment",
    onnx_path=None,
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/info/t4dataset_j6gen2_base_infos_test.pkl",
    sample_idx=1,
)

# ============================================================================
# Model Input/Output Configuration
# ============================================================================
model_io = dict(
    input_name="voxels",
    input_shape=(32, 4),
    input_dtype="float32",
    additional_inputs=[
        dict(name="num_points", shape=(-1,), dtype="int32"),
        dict(name="coors", shape=(-1, 4), dtype="int32"),
    ],
    head_output_names=("heatmap", "reg", "height", "dim", "rot", "vel"),
    batch_size=None,
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
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
    multi_file=True,
    components=dict(
        voxel_encoder=dict(
            name="pts_voxel_encoder",
            onnx_file="pts_voxel_encoder.onnx",
            engine_file="pts_voxel_encoder.engine",
        ),
        backbone_head=dict(
            name="pts_backbone_neck_head",
            onnx_file="pts_backbone_neck_head.onnx",
            engine_file="pts_backbone_neck_head.engine",
        ),
    ),
)

# ============================================================================
# Backend Configuration - INT8 TensorRT
# ============================================================================
backend_config = dict(
    common_config=dict(
        # Use INT8 precision for quantized model
        # TensorRT will use Q/DQ nodes in ONNX to determine INT8 layers
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        # For Q/DQ INT8 export/build, prefer 'STRONGLY_TYPED'
        precision_policy="fp16",
        max_workspace_size=4 << 30,  # 4 GB for INT8 calibration
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input_features=dict(
                    min_shape=[1000, 32, 11],
                    opt_shape=[20000, 32, 11],
                    max_shape=[64000, 32, 11],
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
    enabled=False,
    num_samples=100,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=False,
            device=devices["cuda"],
            model_dir="work_dirs/centerpoint_int8_deployment/onnx/",
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir="work_dirs/centerpoint_int8_deployment/tensorrt/",
        ),
    ),
)

# ============================================================================
# Verification Configuration
# ============================================================================
verification = dict(
    enabled=False,
    tolerance=1e-1,  # INT8 may have larger tolerance than FP16
    num_verify_samples=1,
    devices=devices,
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
