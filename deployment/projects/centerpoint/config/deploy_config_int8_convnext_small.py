"""
CenterPoint INT8 Quantization Deployment Configuration - ConvNeXt Small Backbone

Usage:
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_convnext_small.py \
        projects/CenterPoint/configs/t4dataset/CenterPoint-ConvNeXtPC/pillar_020_convnext_small_secfpn_4xb8_121m_base_t4metric_v2.py

Shared skeleton (components IO, verification, TensorRT build, evaluation defaults) comes from
``_deploy_config_int8_base.py``; this file holds only what differs for this model.
"""

_base_ = ["./_deploy_config_int8_base.py"]

checkpoint_path = "work_dirs/centerpoint-convnext/epoch_5_downsample_conv_first_ptq.pth"

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
    # disable_recipes: empty — ConvNeXt uses residual-add + backbone Linear (both on).
    # calib_cache_path="work_dirs/centerpoint-convnext/small/epoch_30_small_ptq_exp3.calib",
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir="work_dirs/centerpoint-convnext/small/int8_exp7",
    onnx_path=None,
    sample_idx=1,
)

# Derived artifact directories
_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Per-model TensorRT profile shapes
#
# ConvNeXt Small uses BackwardPillarFeatureNet with 10 input channels:
#   base (5) + cluster_center (3) + voxel_center (2) = 10
# Grid size: [1216, 1216, 1]
# ============================================================================
components = dict(
    pts_voxel_encoder=dict(
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 10],
                opt_shape=[20000, 32, 10],
                max_shape=[64000, 32, 10],
            ),
        ),
    ),
    pts_backbone_neck_head=dict(
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 1216, 1216],
                opt_shape=[1, 32, 1216, 1216],
                max_shape=[1, 32, 1216, 1216],
            ),
        ),
    ),
)

onnx_config = dict(opset_version=20)

evaluation = dict(
    num_samples=100,
    backends=dict(
        pytorch=dict(enabled=True),
        onnx=dict(enabled=True, model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
