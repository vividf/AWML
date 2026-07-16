"""YOLOX deployment configuration (reference: YOLOX-opt-ELAN, 960x960, 8-class).

Single-component whole-model export. Layout mirrors the CenterPoint reference config:
  1. SHARED VALUES  - paths, devices, shapes reused across sections.
  2. EXPORT         - export mode, ONNX/TensorRT build settings, the single component.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

Only the top-level names read by ``BaseDeploymentConfig`` (``checkpoint_path``, ``model_cfg``,
``deploy_log_path``, ``devices``, ``export``, ``components``, ``onnx_config``, ``tensorrt_config``,
``evaluation``, ``verification``) plus ``runtime_io`` (read by the entrypoint) take effect; names
prefixed with ``_`` are local single-source literals. Classes, score/NMS thresholds and strides are
read from the model config at runtime, so this same shape deploys any YOLOX variant.
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
# ============================================================================

# Checkpoint - single source of truth for the PyTorch model (export + PyTorch eval).
checkpoint_path = "work_dirs/old_yolox_elan/yolox_epoch24.pth"

# Canonical model config paired with this artifact (CLI positional model_cfg overrides it).
model_cfg = "projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py"

# Log file path (relative paths resolve under export.work_dir). None disables file logging.
deploy_log_path = "deployment.log"

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
_CUDA = devices["cuda"]

# Deployment output layout (single source for export outputs + eval artifact dirs).
_DEPLOY_WORK_DIR = "work_dirs/yolox_opt_elan_deployment_new"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# Model input shape [batch, channels, height, width] (BGR, 0-255; keep-ratio resized + square pad).
_INPUT_SHAPE = [1, 3, 960, 960]

# ============================================================================
# 2. EXPORT
# ============================================================================

# mode: "onnx", "trt", "both", "none"
export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=0,
)

# ONNX Export Settings (shared across all components).
# opset 16 matches the validated YOLOX export; the Tier4 output layout is emitted by
# YOLOXONNXWrapper, so no in-graph decode/NMS.
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# TensorRT Build Settings. "auto" lets TensorRT choose precision (matches the old YOLOX deploy);
# switch to "fp16" for half-precision engines.
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 30,
)

# Single whole-model component. The graph outputs [batch, num_anchors, 4+1+num_classes].
components = dict(
    model=dict(
        onnx_file="yolox.onnx",
        engine_file="yolox.engine",
        io=dict(
            inputs=[
                dict(name="images", dtype="float32"),
            ],
            outputs=[
                dict(name="output", dtype="float32"),
            ],
            dynamic_axes={
                "images": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        ),
        tensorrt_profile=dict(
            # Static square input; min == opt == max for a fixed 960x960 deployment.
            images=dict(
                min_shape=_INPUT_SHAPE,
                opt_shape=_INPUT_SHAPE,
                max_shape=_INPUT_SHAPE,
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O (read by the entrypoint, not BaseDeploymentConfig)
# ============================================================================
# info_file overrides the model config's eval ann_file; "" keeps the model config's own ann_file.
runtime_io = dict(
    info_file="",
    sample_idx=0,
)

# ============================================================================
# 3. EVALUATION
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=100,  # -1 evaluates all samples
    num_warmup=3,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=_CUDA,
        ),
        onnx=dict(
            enabled=True,
            device=_CUDA,
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=True,
            device=_CUDA,
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

# ============================================================================
# 4. VERIFICATION
# ============================================================================
verification = dict(
    enabled=True,
    tolerance=0.1,
    num_verify_samples=1,
    devices=devices,
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
