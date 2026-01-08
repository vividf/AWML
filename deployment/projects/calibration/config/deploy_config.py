"""
CalibrationStatusClassification Deployment Configuration.

This config uses the new unified deployment architecture with components pattern.
"""

# ============================================================================
# Codebase Configuration
# ============================================================================
codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")

# ============================================================================
# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
# ============================================================================
task_type = "classification"

# ============================================================================
# Checkpoint Path - Single source of truth for PyTorch model
# ============================================================================
checkpoint_path = "work_dirs/calibration_classifier/best_accuracy_top1_epoch_28.pth"

# ============================================================================
# Device settings (shared by export, evaluation, verification)
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir="work_dirs/calibration_classifier",
    onnx_path=None,
)

# Derived artifact directories
_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Unified Component Configuration (Single Source of Truth)
#
# CalibrationStatusClassification is an end-to-end model, so we have only one component.
# Each component defines:
#   - name: Component identifier used in export
#   - onnx_file: Output ONNX filename
#   - engine_file: Output TensorRT engine filename
#   - io: Input/output specification for ONNX export
#   - tensorrt_profile: TensorRT optimization profile (min/opt/max shapes)
# ============================================================================
components = dict(
    model=dict(
        name="calibration_classifier",
        onnx_file="end2end.onnx",
        engine_file="end2end.engine",
        io=dict(
            inputs=[
                dict(name="input", shape=[1, 5, 1860, 2880], dtype="float32"),
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
            input=dict(
                min_shape=[1, 5, 1080, 1920],
                opt_shape=[1, 5, 1860, 2880],
                max_shape=[1, 5, 2160, 3840],
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,
)

# ============================================================================
# ONNX Export Settings (shared across all components)
# ============================================================================
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=True,
)

# ============================================================================
# TensorRT Build Settings (shared across all components)
# ============================================================================
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 30,  # 1 GB
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=1,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=True,
            device=devices["cuda"],
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

# ============================================================================
# Verification Configuration
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
