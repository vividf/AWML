"""
YOLOX Deployment Configuration.

This is an example deployment config for YOLOX.
Modify according to your needs.
"""

# Export settings
export = dict(
    mode="both",  # 'onnx', 'trt', 'both', 'none'
    verify=True,  # Enable cross-backend verification
    device="cuda:0",  # Device for export/inference
    work_dir="work_dirs/yolox_deployment",
)

# Runtime I/O settings
runtime_io = dict(
    # Path to COCO annotation file
    ann_file="data/coco/annotations/instances_val2017.json",
    # Path to images directory
    img_prefix="data/coco/val2017/",
    # Sample index for export (use first sample)
    sample_idx=0,
    # Optional: path to existing ONNX file (for eval-only mode)
    # onnx_file='work_dirs/yolox_deployment/yolox.onnx',
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["outputs"],
    save_file="yolox.onnx",
    export_params=True,
    dynamic_axes={"images": {0: "batch_size"}, "outputs": {0: "batch_size"}},
    keep_initializers_as_inputs=False,
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="fp16",
        # TensorRT workspace size (bytes)
        max_workspace_size=1 << 30,  # 1 GB
    ),
    # Optional: specify input shapes for TensorRT
    model_inputs=[dict(name="images", shape=(1, 3, 640, 640), dtype="float32")],  # Adjust according to your model
)

# Evaluation configuration
evaluation = dict(
    enabled=True,  # Enable evaluation
    num_samples=100,  # Number of samples to evaluate (set to -1 for all)
    verbose=False,  # Detailed per-sample output
    # Backends to evaluate
    models_to_evaluate=["pytorch", "onnx", "tensorrt"],
)

# Verification configuration
verification = dict(
    enabled=True,  # Will use export.verify
    tolerance=1e-3,  # Output difference tolerance
    num_verify_samples=10,  # Number of samples for verification
)
