"""
CenterPoint INT8 Quantization Deployment Configuration - VoVNet 57 Backbone

Usage:
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_vov57.py \
        projects/CenterPoint/configs/t4dataset/CenterPoint-ConvNeXtPC/pillar_020_convnext_small_secfpn_4xb8_121m_base_t4metric_v2.py

Shared skeleton (components IO, verification, TensorRT build, evaluation defaults) comes from
``_deploy_config_int8_base.py``; this file holds only what differs for this model.
"""

_base_ = ["./_deploy_config_int8_base.py"]

checkpoint_path = "models/2_5/experiment_j6_gen2/vov57-v2-downsample4/epoch_30_ptq.pth"

# ============================================================================
# Quantization Configuration
# ============================================================================
# PTQ accuracy: if mAP drops a lot (e.g. 0.5 -> 0.25), widen keep_fp16 (e.g. add
# "pts_backbone.stage3" or "pts_bbox_head") — see deployment/quantization/docs/ptq_accuracy_vov99.md.
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    # INT8 by default; keep_fp16 lists subtrees to leave in FP16 (fnmatch on dotted module name; a bare
    # name keeps that module and all its children). Architecture recipes (residual-add, eSE, maxpool)
    # are always-on and class-gated — for VoVNet the eSE recipe is the single-Q-at-input INT8 path.
    default_precision="int8",
    keep_fp16=[
        "pts_voxel_encoder",  # was quant_voxel_encoder=False
        "pts_backbone.stem",  # was skip_vovnet_stages=[0]  (0=stem)
        "pts_backbone.stage2",  # was skip_vovnet_stages=[1]  (1=stage2)
        # "pts_backbone.stage3",   # widen VoVNet FP16 stages if mAP drops
        # "pts_bbox_head",         # whole head FP16 (often recovers mAP)
        # "pts_neck.deblocks.*.0",  # ConvTranspose2d - no TRT INT8 support
    ],
    # disable_recipes: empty — VoVNet uses add + eSE + maxpool + backbone Linear (all on).
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir="models/2_5/experiment_j6_gen2/vov57-v2-downsample4/int8-deployment-bench",
    onnx_path=None,
    sample_idx=1,
)

# Derived artifact directories
_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Per-model TensorRT profile shapes (VoVNet: 11-channel pillars, 1020x1020 BEV grid)
# ============================================================================
components = dict(
    pts_voxel_encoder=dict(
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 11],
                opt_shape=[20000, 32, 11],
                max_shape=[96000, 32, 11],
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

onnx_config = dict(opset_version=22)

evaluation = dict(
    num_samples=-1,
    backends=dict(
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
