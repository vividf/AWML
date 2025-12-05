"""
CenterPoint INT8 Deployment Configuration

This configuration extends the base deploy_config.py with INT8 quantization settings.
Use this config when deploying with PTQ/QAT quantized models.
"""

# Inherit from base config
_base_ = ["./deploy_config.py"]

# ============================================================================
# Quantization Configuration
# ============================================================================
quantization = dict(
    # Master switch to enable/disable quantization
    enabled=True,
    # Quantization mode:
    # - 'ptq' : Post-Training Quantization (calibrate pre-trained model)
    # - 'qat' : Quantization-Aware Training (fine-tune with fake quantization)
    mode="ptq",
    # Calibration settings (used for both PTQ and initial QAT calibration)
    calibration=dict(
        # Number of batches to use for calibration
        num_batches=100,
        # Calibration method for collecting statistics
        # Options: 'histogram', 'max', 'entropy'
        method="histogram",
        # Method for computing amax from statistics
        # Options: 'mse', 'entropy', 'percentile', 'max'
        amax_method="mse",
        # Path to save/load calibration cache (amax values)
        # Set to None to skip caching
        cache_path=None,
    ),
    # Layer fusion settings
    fusion=dict(
        # Fuse BatchNorm into Conv before quantization
        fuse_bn=True,
    ),
    # Sensitive layers to skip quantization
    # These are typically early layers that cause significant accuracy drop
    # Run sensitivity analysis to identify these layers
    sensitive_layers=[
        # Example: first layer of backbone
        # "pts_backbone.blocks.0.0",
    ],
    # Precision settings per layer type
    precision=dict(
        # Default precision for most layers
        default_input="int8",
        default_weight="int8",
        # First layer: keep input in FP16 to preserve input dynamic range
        first_layer_input="fp16",
        # Last layer: keep output in FP16 for postprocessing
        last_layer_output="fp16",
    ),
)

# ============================================================================
# Override Backend Configuration for INT8
# ============================================================================
backend_config = dict(
    common_config=dict(
        # Use INT8 precision policy for TensorRT
        precision_policy="int8",
        # Larger workspace for INT8 optimization
        max_workspace_size=4 << 30,  # 4 GB
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input_features=dict(
                    min_shape=[1000, 32, 11],
                    opt_shape=[20000, 32, 11],
                    max_shape=[64000, 32, 11],
                ),
                spatial_features=dict(
                    min_shape=[1, 32, 760, 760],
                    opt_shape=[1, 32, 760, 760],
                    max_shape=[1, 32, 760, 760],
                ),
            )
        )
    ],
)
