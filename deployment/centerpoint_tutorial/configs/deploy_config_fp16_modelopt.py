"""CenterPoint FP16 tutorial deploy config — nvidia-modelopt backend A/B run.

Identical to ``deploy_config_fp16_tutorial.py`` except the output layout: everything lands
under ``work_dirs/centerpoint_tutorial/fp16_modelopt/`` so the pytorch-quantization baseline
in ``fp16/`` stays untouched. Exercises the backend seam's "quantized tree, all quantizers
disabled" load path (see the base config's docstring for why FP16 routes through it).

Run:

    AWML_QUANT_BACKEND=modelopt python -m deployment.cli.main centerpoint \
        work_dirs/centerpoint_tutorial/configs/deploy_config_fp16_modelopt.py
"""

_base_ = ["./deploy_config_fp16_tutorial.py"]

_DEPLOY_WORK_DIR = "work_dirs/centerpoint_tutorial/fp16_modelopt"
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
