"""
CenterPoint INT8 Quantization Deployment Configuration - SECOND Backbone (2.5)

One config is the artifact's manifest and drives the whole PTQ loop: what it is (``model_cfg``),
how it was made (``quantization`` placement + ``ptq`` producer block), how it deploys (export /
components / evaluation). ``checkpoint_path`` is the *output* of the PTQ producer and the *input*
of deployment — the producer defaults its ``--output`` to it, so producing and deploying use the
same artifact.

Produce the PTQ checkpoint:
    python -m deployment.projects.centerpoint.quantization.quantize ptq \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_second_2_6_quant_release.py

Deploy / evaluate it (model config comes from ``model_cfg`` below; a second positional overrides):
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_second_2_6_quant_release.py

Shared skeleton (components IO, verification, TensorRT build, evaluation defaults) comes from
``_deploy_config_int8_base.py``; this file holds only what differs for this model.
"""

_base_ = ["./_deploy_config_int8_base.py"]

# ============================================================================
# Artifact manifest — the model this artifact is, and the checkpoint it lives in.
# model_cfg: canonical model pairing (PTQ calibration + deploy/eval; CLI can override).
# checkpoint_path: PTQ producer output default AND deploy input — same artifact by construction.
# ============================================================================
model_cfg = (
    "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py"
)
# checkpoint_path = "models/2_5/experiment_j6_gen2/second/epoch_30_ptq.pth"
# work_dirs/centerpoint/centerpoint_2_5_best_epoch_28.pth
# checkpoint_path = "vivid/bench_comparison/centerpoint_2_6/epoch_29_ptq.pth"
checkpoint_path = "work_dirs/centerpoint_2_6_quant_release/epoch_29_ptq.pth"

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
        "pts_backbone.blocks.0",  # was skip_backbone_stages=[0]
        # "pts_backbone.blocks.1",  # extend to blocks.0/1/2 if still unstable
    ],
    disable_recipes=["add"],  # was quant_add unset (off); SECOND 2D backbone has no residual-add
    ptq=dict(
        # Producer half of the run (CLI flags override any of these) — the release calibration
        # recipe. Calibration data comes from the top-level model_cfg's val dataloader.
        checkpoint="work_dirs/centerpoint_2_6_quant_release/epoch_29.pth",  # FP input
        calibrate_samples=400,
        batch_size=1,
        calib_seed=0,
    ),
)

# ============================================================================
# Export Configuration
# ============================================================================
# Single literal for deployment output root (used before `export` exists).
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_2_6_quant_release"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=1,
)

# ============================================================================
# Per-model TensorRT profile shapes (SECOND: 11-channel pillars, 1020x1020 BEV grid)
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

onnx_config = dict(opset_version=17)

evaluation = dict(
    num_samples=-1,
    num_warmup=2,
    backends=dict(
        pytorch=dict(enabled=True),
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
