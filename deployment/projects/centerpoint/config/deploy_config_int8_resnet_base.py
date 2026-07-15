"""
CenterPoint INT8 Quantization Deployment Configuration - ResNet34 Backbone

Usage:
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_resnet_base.py \
        projects/CenterPoint/configs/t4dataset/Centerpoint/resnet34_secfpn_4xb16_121m_base_amp_t4metric_v2.py

Shared skeleton (components IO, verification, TensorRT build, evaluation defaults) comes from
``_deploy_config_int8_base.py``; this file holds only what differs for this model.
"""

_base_ = ["./_deploy_config_int8_base.py"]

checkpoint_path = "models/2_5/base/centerpoint_resnet34_base_2_5_epoch49_ptq.pth"

# ============================================================================
# Quantization Configuration
# ============================================================================
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    default_precision="int8",
    keep_fp16=[
        "pts_voxel_encoder",  # was quant_voxel_encoder=False
        # "pts_neck.deblocks.*.0",  # ConvTranspose2d - no TRT INT8 support
    ],
    # disable_recipes: empty — ResNet uses residual-add (BasicBlock).
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir="work_dirs/centerpoint_int8_resnet_deployment_base",
    onnx_path=None,
    sample_idx=1,
)

# Derived artifact directories
_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Per-model TensorRT profile shapes (ResNet34: 10-channel pillars, 1020x1020 BEV grid)
# ============================================================================
components = dict(
    pts_voxel_encoder=dict(
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 10],
                opt_shape=[20000, 32, 10],
                max_shape=[96000, 32, 10],
            ),
        ),
    ),
    pts_backbone_neck_head=dict(
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 1020, 1020],
                opt_shape=[1, 32, 1020, 1020],
                max_shape=[1, 32, 1020, 1020],
            ),
        ),
    ),
)

onnx_config = dict(opset_version=16)

evaluation = dict(
    num_samples=-1,
    backends=dict(
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
