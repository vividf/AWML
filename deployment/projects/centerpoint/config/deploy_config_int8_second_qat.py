"""
CenterPoint INT8 QAT Deployment Configuration - SECOND Backbone (2.6)

One config drives the whole QAT loop (spec_qat.md §D2): placement (``keep_fp16`` /
``disable_recipes``, inherited from the release INT8 config) plus the ``qat`` training block.
``checkpoint_path`` is the *output* of the QAT producer and the *input* of deployment — the
producer defaults its ``--output`` to it, so producing and deploying use the same artifact.

Produce the QAT checkpoint:
    python -m deployment.projects.centerpoint.quantization.quantize qat \
        --deploy-cfg deployment/projects/centerpoint/config/deploy_config_int8_second_qat.py

Deploy / evaluate it (same pipeline as PTQ — the loader does not branch on mode; the model config
comes from the inherited top-level ``model_cfg``, a second positional overrides):
    python -m deployment.cli.main centerpoint \
        deployment/projects/centerpoint/config/deploy_config_int8_second_qat.py
"""

_base_ = ["./deploy_config_int8_second_2_6_quant_release.py"]

# The packaged QAT artifact ({"state_dict"} + sibling .calib) — producer output, deploy input.
checkpoint_path = "work_dirs/centerpoint_2_6_qat/epoch_29_qat.pth"

# ============================================================================
# Quantization Configuration — placement inherited (keep_fp16 / disable_recipes / fuse_bn);
# mode="qat" gates the training block below.
# ============================================================================
quantization = dict(
    mode="qat",
    # Drop the base config's PTQ producer block — this config's producer is the qat block below
    # (a ptq block under mode="qat" would be rejected by the schema).
    ptq=None,
    qat=dict(
        # Training half of the run (CLI flags override any of these; spec_qat.md WP1).
        train_cfg="projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py",
        checkpoint="work_dirs/centerpoint_2_6_quant_release/epoch_29.pth",  # FP init (pre-PTQ weights)
        epochs=3,  # ~10% of the original 30-epoch training (modelopt guidance, spec_qat.md §2.2)
        lr=1e-4,  # CUDA-CenterPoint one-cycle lr_max (spec_qat.md §2.1)
        calibrate_samples=400,  # CUDA-CenterPoint reference; epoch-0 calibration before fine-tune
        # calib_cache="work_dirs/centerpoint_2_6_quant_release/epoch_29_ptq.calib",  # reuse PTQ amax
        work_dir="work_dirs/centerpoint_2_6_qat/qat_training",
    ),
)

# ============================================================================
# Export / evaluation output roots (do not collide with the PTQ release dirs)
# ============================================================================
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_2_6_qat"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

export = dict(
    mode="none",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=1,
)

evaluation = dict(
    backends=dict(
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
