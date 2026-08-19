"""CenterPoint INT8 tutorial deploy config — nvidia-modelopt backend A/B run.

Identical to ``deploy_config_int8_tutorial.py`` except the artifacts: consumes the
PTQ checkpoint calibrated under ``AWML_QUANT_BACKEND=modelopt`` and lands everything
under ``work_dirs/centerpoint_tutorial/int8_modelopt/`` so the pytorch-quantization
baseline in ``int8/`` stays untouched.

Run (the backend must also be selected at deploy time so the Q/DQ module tree is
rebuilt with the same TensorQuantizer implementation that calibrated it):

    AWML_QUANT_BACKEND=modelopt python -m deployment.cli.main centerpoint \
        work_dirs/centerpoint_tutorial/configs/deploy_config_int8_modelopt.py
"""

_base_ = ["./deploy_config_int8_tutorial.py"]

checkpoint_path = "work_dirs/centerpoint_tutorial/checkpoints/epoch_29_ptq_tutorial_modelopt.pth"

quantization = dict(
    ptq=dict(
        checkpoint="work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth",
        calibrate_samples=60,
        batch_size=1,
        calib_seed=0,
    ),
)

_DEPLOY_WORK_DIR = "work_dirs/centerpoint_tutorial/int8_modelopt"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

export = dict(
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
)

evaluation = dict(
    backends=dict(
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
