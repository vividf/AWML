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
# checkpoint_path = "work_dirs/centerpoint_ptq.pth"
# checkpoint_path = "work_dirs/centerpoint-convnext/small/convext_batch_relu_epoch_2_ptq.pth"
checkpoint_path = "work_dirs/centerpoint-convnext/small/convext_batch_relu_epoch_2_ptq_fuse_downsample.pth"

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
    quant_add=True,
    quant_linear_backbone=True,
    # Optional: load calibration cache to populate amax for newly added quantizers
    # calib_cache_path="work_dirs/centerpoint-convnext/small/epoch_30_small_ptq_exp3.calib",
    # Optional: skip quantizing early backbone stages (maps to pts_backbone.blocks.<idx>)
    skip_backbone_first_stages=0,
    skip_backbone_stages=[],
    # Optional: skip ConvNeXt downsample layers (often PTQ-sensitive for stride-2 convs)
    skip_backbone_downsample_all=True,
    skip_backbone_downsample_layers=[],
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
    work_dir="work_dirs/centerpoint-convnext/small/int8_exp6",
    onnx_path="work_dirs/centerpoint-convnext/small/int8_exp6/onnx",
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/info/t4dataset_base_infos_test.pkl",
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
    opset_version=20,
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
                    # BackwardPillarFeatureNet channel calculation:
                    # base (5) + cluster_center (3) + voxel_center (2) = 10 channels
                    # Shape: (num_voxels, max_points_per_voxel, channels)
                    min_shape=[1000, 32, 10],  # Minimum supported input shape
                    opt_shape=[20000, 32, 10],  # Optimal shape for performance tuning
                    max_shape=[64000, 32, 10],  # Maximum supported input shape
                ),
                spatial_features=dict(
                    # spatial_features shape should match grid_size from training config
                    # For pillar_020_convnext_small: grid_size = [1216, 1216, 1]
                    # Note: 1216 is divisible by 8 (1216 / 8 = 152), which helps avoid dimension mismatch
                    # in neck concatenation when using upsample_strides=[1, 2, 4]
                    min_shape=[1, 32, 1216, 1216],
                    opt_shape=[1, 32, 1216, 1216],
                    max_shape=[1, 32, 1216, 1216],
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
    num_samples=100,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=True,
            device=devices["cuda"],
            model_dir="work_dirs/centerpoint-convnext/small/int8_exp6/onnx/",
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir="work_dirs/centerpoint-convnext/small/int8_exp6/tensorrt/",
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
