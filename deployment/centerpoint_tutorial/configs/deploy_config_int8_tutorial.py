"""CenterPoint INT8 tutorial deploy config.

A copy of ``deploy_config_int8_second_2_6_quant_release.py`` with only the
artifact paths changed, so the whole PTQ → ONNX → TensorRT → evaluate loop
lands under ``work_dirs/centerpoint_tutorial/``.

Produce the PTQ checkpoint (with per-sample histogram tracing):
    python work_dirs/centerpoint_tutorial/scripts/01_ptq_with_histogram_trace.py \
        --deploy-cfg work_dirs/centerpoint_tutorial/configs/deploy_config_int8_tutorial.py \
        --checkpoint work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth \
        --output work_dirs/centerpoint_tutorial/checkpoints/epoch_29_ptq_tutorial.pth \
        --trace-dir work_dirs/centerpoint_tutorial/calib_trace

Deploy / evaluate it (ONNX export → TensorRT engine → eval):
    python -m deployment.cli.main centerpoint \
        work_dirs/centerpoint_tutorial/configs/deploy_config_int8_tutorial.py
"""

_base_ = ["../../../deployment/projects/centerpoint/config/_deploy_config_int8_base.py"]

model_cfg = (
    "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py"
)
checkpoint_path = "work_dirs/centerpoint_tutorial/checkpoints/epoch_29_ptq_tutorial.pth"

# ============================================================================
# Quantization Configuration (identical to the 2.6 release recipe, except the
# calibration sample budget: the local machine only has a 60-frame val split).
# ============================================================================
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    default_precision="int8",
    keep_fp16=[
        "pts_voxel_encoder",  # PillarFeatureNet stays FP16 (tiny, and quantizing it hurts)
        "pts_backbone.blocks.0",  # first SECOND stage stays FP16 (release recipe: skip stage 0)
    ],
    disable_recipes=["add"],  # SECOND has no residual adds; recipe explicitly off
    ptq=dict(
        checkpoint="work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth",
        calibrate_samples=60,  # release used 400; local dataset has 60 val frames
        batch_size=1,
        calib_seed=0,
    ),
)

# ============================================================================
# Export / output layout
# ============================================================================
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_tutorial/int8"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=1,
)

# SECOND: 11-channel pillars, 1020x1020 BEV grid (same as release config).
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

onnx_config = dict(opset_version=17)

evaluation = dict(
    num_samples=-1,  # all 60 local val frames
    num_warmup=2,
    backends=dict(
        pytorch=dict(enabled=True),  # fake-quant PyTorch model (the calibration-time view)
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
