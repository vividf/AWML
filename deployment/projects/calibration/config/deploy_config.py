"""Calibration Status Classification deployment configuration (ResNet18, 5-channel).

Single-component whole-model export of an mmpretrain classifier. The input is a 5-channel fused
image (BGR + projected-LiDAR depth + intensity) at the camera's native resolution, so height/width
are dynamic and the TensorRT profile spans the expected resolution range.

Only the top-level names read by ``BaseDeploymentConfig`` (plus ``class_names`` read by
``CalibrationDeploymentConfig`` and ``runtime_io`` read by the entrypoint) take effect; ``_``-prefixed
names are local single-source literals.
"""

# ============================================================================
# 1. SHARED VALUES
# ============================================================================

# Checkpoint - single source of truth for the PyTorch model (export + PyTorch eval).
checkpoint_path = "work_dirs/calibration_classifier/best_accuracy_top1_epoch_28.pth"

# Canonical model config paired with this artifact (CLI positional model_cfg overrides it).
model_cfg = "projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py"

# Log file path (relative paths resolve under export.work_dir). None disables file logging.
deploy_log_path = "deployment.log"

# Class names in label-index order (0 miscalibrated, 1 calibrated) — the classifier model config
# records only num_classes, so the label strings live here.
class_names = ["miscalibrated", "calibrated"]

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
_CUDA = devices["cuda"]

# Deployment output layout.
_DEPLOY_WORK_DIR = "work_dirs/calibration_classifier_deployment"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# 2. EXPORT
# ============================================================================

# mode: "onnx", "trt", "both", "none"
export = dict(
    mode="none",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=0,
)

# opset 16 matches the validated classifier export; the graph outputs raw logits (softmax in
# postprocess), so IdentityWrapper is used for export.
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# FP16 TensorRT engine (matches the old calibration deploy).
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 30,
)

# Single whole-model component. 5-channel input (BGR + depth + intensity), dynamic H/W.
components = dict(
    model=dict(
        onnx_file="classification_classifier.onnx",
        engine_file="classification_classifier.engine",
        io=dict(
            inputs=[
                dict(name="input", dtype="float32"),
            ],
            outputs=[
                dict(name="output", dtype="float32"),
            ],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            },
        ),
        tensorrt_profile=dict(
            # [batch, 5 channels, height, width]; spans the expected camera resolution range.
            input=dict(
                min_shape=[1, 5, 1080, 1920],
                opt_shape=[1, 5, 1860, 2880],
                max_shape=[1, 5, 2160, 3840],
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O (read by the entrypoint) — required: the calibration info .pkl.
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,
)

# ============================================================================
# 3. EVALUATION
# ============================================================================
# num_samples counts frames; each base sample yields 2 frames (calibrated + miscalibrated), so use
# an even number for a class-balanced run (-1 evaluates all base samples × 2 variants).
evaluation = dict(
    enabled=True,
    num_samples=20,
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
    tolerance=1e-1,
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
