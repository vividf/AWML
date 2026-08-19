"""CenterPoint FP16 (before-PTQ) tutorial deploy config.

The "before" half of the tutorial comparison: the SAME weights as the INT8
config, but with no Q/DQ anywhere — the exported ONNX has zero
QuantizeLinear/DequantizeLinear nodes and TensorRT builds a plain FP16 engine.

The quantization block below looks surprising for an "FP16" config, but it is
load-bearing: our reconstructed checkpoint carries BN-FUSED weights (conv bias
holds all the BN shift), while the plain FP loader builds the UNFUSED model
where those convs have bias=False — the biases would be dropped as unexpected
keys and the model would be garbage. `enabled=True, fuse_bn=True` routes the
load through the same fuse-BN-then-load path the INT8 config uses, and
`keep_fp16=["*"]` keeps every single layer out of quantization, so no
quantizer is ever inserted.

    python -m deployment.cli.main centerpoint \
        work_dirs/centerpoint_tutorial/configs/deploy_config_fp16_tutorial.py
"""

_base_ = ["../../../deployment/projects/centerpoint/config/_deploy_config_int8_base.py"]

model_cfg = (
    "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py"
)
# The reconstructed FP checkpoint: BN-fused weights, no amax buffers.
checkpoint_path = "work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth"

quantization = dict(
    enabled=True,  # only to route through the fuse-BN-aware loader (see docstring)
    mode="ptq",
    fuse_bn=True,
    default_precision="int8",
    keep_fp16=["*"],  # match EVERYTHING -> zero quantizers inserted -> pure FP16 graph
    disable_recipes=["add", "ese", "maxpool"],
)

# ============================================================================
# Export / output layout
# ============================================================================
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_tutorial/fp16"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=1,
)

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
        pytorch=dict(enabled=True),  # FP PyTorch reference
        onnx=dict(model_dir=_ONNX_DIR),
        tensorrt=dict(engine_dir=_TENSORRT_DIR),
    ),
)
