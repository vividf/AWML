"""
YOLOX_opt_elan Deployment Configuration.

This is an example deployment config for YOLOX_opt_elan object detection.
Modify according to your needs.
"""

# Export settings
export = dict(
    mode="both",  # 'onnx', 'trt', 'both', 'none'
    verify=True,  # Enable cross-backend verification
    device="cuda:0",  # Device for export/inference (CPU for Docker testing)
    work_dir="work_dirs/yolox_opt_elan_deployment",
)

# Runtime I/O settings
runtime_io = dict(
    # Path to T4Dataset annotation file
    ann_file="data/t4dataset/2d_info/2d_info_infos_val.json",
    # Path to images directory (can be empty if full paths are in annotations)
    img_prefix="",
    # Sample index for export (use first sample)
    sample_idx=0,
    # Optional: path to existing ONNX file (for eval-only mode)
    # onnx_file='work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx',
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["output"],  # Match Tier4 output name
    save_file="yolox_opt_elan.onnx",
    export_params=True,
    dynamic_axes={
        "images": {0: "batch_size"},  # Dynamic batch size
        "output": {0: "batch_size"},  # Match output name
    },
    keep_initializers_as_inputs=False,
    # Decode in inference (Tier4-compatible format)
    # If True: output format is [batch, num_predictions, 4+1+num_classes]
    #   where: [bbox_reg(4), objectness(1), class_scores(num_classes)]
    # If False: output raw head outputs (cls_scores, bbox_preds, objectnesses)
    decode_in_inference=True,
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        # Precision policy for TensorRT
        # Options: 'auto', 'fp16', 'fp32_tf32', 'strongly_typed'
        precision_policy="auto",
        # TensorRT workspace size (bytes)
        max_workspace_size=1 << 30,  # 1 GB
    ),
    # Specify input shapes for TensorRT (960x960 for YOLOX_opt_elan)
    model_inputs=[
        dict(
            name="images",
            shape=(1, 3, 960, 960),  # YOLOX_opt_elan uses 960x960 input
            dtype="float32",
        )
    ],
)

# Evaluation configuration
evaluation = dict(
    enabled=False,  # Enable evaluation
    num_samples=10,  # Number of samples to evaluate (set to -1 for all)
    verbose=False,  # Detailed per-sample output
    # Specify models to evaluate (comment out or remove paths for backends you don't want to evaluate)
    models=dict(
        onnx="/workspace/work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.onnx",  # Path to ONNX model file
        tensorrt="/workspace/work_dirs/yolox_opt_elan_deployment/yolox_opt_elan.engine",  # Path to TensorRT engine file
        # pytorch="work_dirs/old_yolox_elan/yolox_epoch24.pth",  # Optional: PyTorch checkpoint
    ),
)

# Verification configuration
verification = dict(
    enabled=True,  # Will use export.verify
    tolerance=1e-1,  # Output difference tolerance (relaxed for CUDA/CPU differences)
    num_verify_samples=10,  # Number of samples for verification
)
