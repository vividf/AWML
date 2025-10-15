"""
YOLOX_opt_elan Deployment Configuration.

This is an example deployment config for YOLOX_opt_elan object detection.
Modify according to your needs.
"""

# Export settings
export = dict(
    mode="onnx",  # 'onnx', 'trt', 'both', 'none'
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

# Model input/output configuration
model_io = dict(
    # Input configuration
    input_name="images",
    input_shape=(3, 960, 960),  # (C, H, W) - batch dimension will be added automatically
    input_dtype="float32",
    
    # Output configuration  
    output_name="output",
    
    # Batch size configuration
    # Options:
    # - int: Fixed batch size (e.g., 1, 6)
    # - None: Dynamic batch size (uses dynamic_axes)
    batch_size=6,  # Set to 6 to match old ONNX exactly, or None for dynamic
    
    # Dynamic axes (only used when batch_size=None)
    # When batch_size is set to a number, this is automatically set to None
    # When batch_size is None, this defines dynamic batch dimensions
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    save_file="yolox_opt_elan.onnx",
    keep_initializers_as_inputs=False,
    # Decode in inference (Tier4-compatible format)
    # If True: output format is [batch, num_predictions, 4+1+num_classes]
    #   where: [bbox_reg(4), objectness(1), class_scores(num_classes)]
    # If False: output raw head outputs (cls_scores, bbox_preds, objectnesses)
    # Set to False to match Tier4 YOLOX behavior exactly (no flatten operations)
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
    # Model inputs will be generated automatically from model_io configuration
    # This will be populated by the deployment pipeline based on model_io settings
    model_inputs=None,  # Will be set automatically
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
