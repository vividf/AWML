"""Shared skeleton for the CenterPoint INT8 deploy configs (the ``_base_`` of ``deploy_config_int8_*.py``).

Holds every block that is identical across the INT8 variants, so a shared change (an IO tensor
name, a verification scenario, the TensorRT build settings) is a ONE-file edit. A variant file
provides only what genuinely differs per model:

- ``checkpoint_path``
- the full ``quantization`` block (``keep_fp16`` / ``disable_recipes`` tuning)
- ``export`` (mode / work_dir) and the derived evaluation artifact dirs
- ``components.*.tensorrt_profile`` shapes (pillar channels + BEV grid are per-model facts)
- ``onnx_config`` ``opset_version``
- evaluation overrides (``num_samples``, backend ``enabled`` flags)

The leading underscore keeps this file out of the ``deploy_config*.py`` glob that the parse test
and the CLI docs enumerate — it is not a runnable deploy config (no ``export`` section).
"""

deploy_log_path = "deployment.log"

# ============================================================================
# Device settings
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

# ============================================================================
# Unified Component Configuration (IO contract; shared by every CenterPoint variant)
#
# Each variant MUST add components.<name>.tensorrt_profile with its model's shapes
# (voxel-encoder pillar channels and backbone BEV grid differ per backbone).
# ============================================================================
components = dict(
    pts_voxel_encoder=dict(
        onnx_file="pts_voxel_encoder.onnx",
        engine_file="pts_voxel_encoder.engine",
        io=dict(
            inputs=[
                dict(name="input_features", dtype="float32"),
            ],
            outputs=[
                dict(name="pillar_features", dtype="float32"),
            ],
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        ),
    ),
    pts_backbone_neck_head=dict(
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(
            inputs=[
                dict(name="spatial_features", dtype="float32"),
            ],
            outputs=[
                dict(name="heatmap", dtype="float32"),
                dict(name="reg", dtype="float32"),
                dict(name="height", dtype="float32"),
                dict(name="dim", dtype="float32"),
                dict(name="rot", dtype="float32"),
                dict(name="vel", dtype="float32"),
            ],
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                "heatmap": {0: "batch_size", 2: "height", 3: "width"},
                "reg": {0: "batch_size", 2: "height", 3: "width"},
                "height": {0: "batch_size", 2: "height", 3: "width"},
                "dim": {0: "batch_size", 2: "height", 3: "width"},
                "rot": {0: "batch_size", 2: "height", 3: "width"},
                "vel": {0: "batch_size", 2: "height", 3: "width"},
            },
        ),
    ),
)

# ============================================================================
# ONNX Export Settings (variants override opset_version)
# ============================================================================
onnx_config = dict(
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# ============================================================================
# TensorRT Build Settings
# ============================================================================
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=4 << 30,
)

# ============================================================================
# Evaluation Configuration (variants override num_samples, backend enabled flags,
# and the artifact dirs derived from their work_dir)
# ============================================================================
evaluation = dict(
    enabled=True,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=False,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=False,
            device=devices["cuda"],
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
        ),
    ),
)

# ============================================================================
# Verification Configuration
# ============================================================================
verification = dict(
    enabled=False,
    tolerance=1e-1,
    num_verify_samples=1,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        none=[],
    ),
)
