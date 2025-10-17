"""
YOLOX_opt_elan Deployment Main Script.

This script handles the complete deployment pipeline for YOLOX_opt_elan object detection:
- Export to ONNX and/or TensorRT
- Verify outputs across backends
- Evaluate model performance on T4Dataset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from mmdet.apis import init_detector
from mmengine.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.core.verification import verify_model_outputs
from autoware_ml.deployment.exporters import ONNXExporter, TensorRTExporter
from projects.YOLOX_opt_elan.deploy.data_loader import YOLOXOptElanDataLoader
from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator
from projects.YOLOX_opt_elan.deploy.onnx_wrapper import YOLOXONNXWrapper


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(model_cfg: Config, checkpoint_path: str, device: str):
    """
    Load PyTorch model from checkpoint.

    Args:
        model_cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """

    model = init_detector(model_cfg, checkpoint_path, device=device)
    model.eval()

    return model


def export_onnx(
    model, data_loader: YOLOXOptElanDataLoader, config: BaseDeploymentConfig, logger: logging.Logger, model_cfg: Config
) -> str:
    """
    Export model to ONNX format.

    Returns:
        Path to exported ONNX file
    """
    logger.info("=" * 80)
    logger.info("Exporting to ONNX")
    logger.info("=" * 80)

    # Get ONNX settings first
    onnx_settings = config.get_onnx_settings()
    
    # Get sample input and create batch based on configuration
    sample_idx = config.runtime_config.get("sample_idx", 0)
    single_input = data_loader.load_and_preprocess(sample_idx)
    
    # Get batch size from configuration
    batch_size = onnx_settings.get("batch_size", 1)
    if batch_size is None:
        # Use dynamic batch size (single sample)
        input_tensor = single_input
        logger.info(f"Using dynamic batch size (single sample)")
    else:
        # Use fixed batch size by repeating the same sample
        input_tensor = single_input.repeat(batch_size, 1, 1, 1)
        logger.info(f"Using fixed batch size: {batch_size}")
        
        # Update backend config with the batch size
        config.update_batch_size(batch_size)
    output_path = os.path.join(config.export_config.work_dir, onnx_settings["save_file"])

    os.makedirs(config.export_config.work_dir, exist_ok=True)

    # Replace ReLU6 with ReLU (Tier4 YOLOX does this at export_onnx.py line 87)
    def replace_relu6_with_relu(module):
        """Recursively replace all ReLU6 with ReLU in the module."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU6):
                setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                logger.debug(f"  Replaced {name}: ReLU6 -> ReLU")
            else:
                replace_relu6_with_relu(child)

    replace_relu6_with_relu(model)

    # Get number of classes from model config
    num_classes = model_cfg.model.bbox_head.num_classes

    # Wrap model to output Tier4 format
    wrapped_model = YOLOXONNXWrapper(
        model=model,
        num_classes=num_classes,
    )
    wrapped_model.eval()
    export_model = wrapped_model
    output_names = ["output"]  # Single output to match Tier4
    output_channels = 4 + 1 + num_classes  # bbox(4) + objectness(1) + classes
    logger.info(f"Output format: [batch_size, num_predictions, {output_channels}]")

    # Export
    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output path: {output_path}")

    torch.onnx.export(
        export_model,
        input_tensor,
        output_path,
        opset_version=onnx_settings["opset_version"],
        do_constant_folding=onnx_settings["do_constant_folding"],
        input_names=onnx_settings["input_names"],
        output_names=output_names,
        dynamic_axes=onnx_settings.get("dynamic_axes"),
        export_params=onnx_settings.get("export_params", True),
        keep_initializers_as_inputs=onnx_settings.get("keep_initializers_as_inputs", False),
    )

    logger.info(f"✅ ONNX export successful: {output_path}")

    # Apply ONNX simplifier to remove redundant operations (matching Tier4 YOLOX)
    if onnx_settings.get("simplify", True):
        import onnx
        from onnxsim import simplify
        
        logger.info("Applying ONNX simplifier to remove redundant operations...")
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_path)
        logger.info("✅ ONNX simplification successful")

    return output_path


def export_tensorrt(onnx_path: str, config: BaseDeploymentConfig, logger: logging.Logger) -> str:
    """
    Export ONNX model to TensorRT.

    Returns:
        Path to exported TensorRT engine
    """
    logger.info("=" * 80)
    logger.info("Exporting to TensorRT")
    logger.info("=" * 80)

    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not installed. Please install TensorRT.")
        return None

    trt_settings = config.get_tensorrt_settings()
    output_path = onnx_path.replace(".onnx", ".engine")

    # Build TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # Build config
    config_trt = builder.create_builder_config()

    # Handle dynamic shapes by creating an optimization profile
    # Check if network has dynamic inputs
    has_dynamic_shapes = False
    for i in range(network.num_inputs):
        input_shape = network.get_input(i).shape
        if -1 in input_shape:
            has_dynamic_shapes = True
            break

    if has_dynamic_shapes:
        logger.info("Detected dynamic shapes in network, creating optimization profile...")
        profile = builder.create_optimization_profile()

        # Get model inputs from config
        model_inputs = config.backend_config.model_inputs

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            # Find corresponding config
            input_config = next((inp for inp in model_inputs if inp["name"] == input_name), None)

            if input_config and "shape" in input_config:
                # Use shape from config
                shape = input_config["shape"]
                # For dynamic batch, set min=1, opt=1, max=batch_size
                min_shape = tuple(1 if (s == -1 or (i == 0 and s > 0)) else s for i, s in enumerate(shape))
                opt_shape = tuple(shape)
                max_shape = tuple(s if s != -1 else 1 for s in shape)  # For now, use same as opt

                logger.info(f"Setting profile for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            else:
                # Fallback: use shape from ONNX with batch=1
                min_shape = tuple(1 if s == -1 else s for s in input_shape)
                opt_shape = min_shape
                max_shape = min_shape
                logger.info(f"Setting profile for {input_name} (fallback): {min_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)

        config_trt.add_optimization_profile(profile)

    workspace_size = trt_settings["max_workspace_size"]
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    logger.info(f"Set WORKSPACE memory pool limit to {workspace_size} bytes")

    # Set precision
    precision_flags = trt_settings["policy_flags"]
    if "FP16" in precision_flags and precision_flags["FP16"]:
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision")
    if "TF32" in precision_flags and precision_flags["TF32"]:
        config_trt.set_flag(trt.BuilderFlag.TF32)
        logger.info("Enabled TF32 precision")

    logger.info("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config_trt)
    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None
    
    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"✅ TensorRT export successful: {output_path}")

    return output_path


def get_models_to_evaluate(eval_config: dict, logger: logging.Logger) -> list:
    """
    Get list of models to evaluate from config.

    Args:
        eval_config: Evaluation configuration
        logger: Logger instance

    Returns:
        List of tuples (backend_name, model_path)
    """
    models_config = eval_config.get("models", {})
    models_to_evaluate = []

    backend_mapping = {
        "pytorch": "pytorch",
        "onnx": "onnx",
        "tensorrt": "tensorrt",
    }

    for backend_key, model_path in models_config.items():
        backend_name = backend_mapping.get(backend_key.lower())
        if backend_name and model_path:
            if os.path.exists(model_path):
                models_to_evaluate.append((backend_name, model_path))
                logger.info(f"  - {backend_name}: {model_path}")
            else:
                logger.warning(f"  - {backend_name}: {model_path} (not found, skipping)")

    return models_to_evaluate


def run_evaluation(
    data_loader: YOLOXOptElanDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """Run evaluation on specified models."""
    eval_config = config.evaluation_config

    if not eval_config.get("enabled", False):
        logger.info("Evaluation disabled, skipping...")
        return

    logger.info("=" * 80)
    logger.info("Running Evaluation")
    logger.info("=" * 80)

    # Get models to evaluate from config
    models_to_evaluate = get_models_to_evaluate(eval_config, logger)

    if not models_to_evaluate:
        logger.warning("No models found for evaluation")
        return

    evaluator = YOLOXOptElanEvaluator(model_cfg)

    num_samples = eval_config.get("num_samples", 100)
    if num_samples == -1:
        num_samples = data_loader.get_num_samples()

    all_results = {}

    for backend, model_path in models_to_evaluate:
        results = evaluator.evaluate(
            model_path=model_path,
            data_loader=data_loader,
            num_samples=num_samples,
            backend=backend,
            device=config.export_config.device,
            verbose=eval_config.get("verbose", False),
        )

        all_results[backend] = results

        logger.info(f"\n{backend.upper()} Results:")
        evaluator.print_results(results)

    # Compare results across backends
    if len(all_results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Comparison")
        logger.info("=" * 80)

        for backend, results in all_results.items():
            logger.info(f"\n{backend.upper()}:")
            logger.info(f"  mAP: {results['mAP']:.4f}")
            logger.info(f"  mAP@50: {results['mAP_50']:.4f}")
            logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")


def main():
    """Main deployment pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Object Detection - Deployment Pipeline")
    logger.info("=" * 80)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)

    config = BaseDeploymentConfig(deploy_cfg)

    # Override from command line
    if args.work_dir:
        config.export_config.work_dir = args.work_dir
    if args.device:
        config.export_config.device = args.device

    logger.info("\nDeployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = YOLOXOptElanDataLoader(
        ann_file=config.runtime_config["ann_file"],
        img_prefix=config.runtime_config.get("img_prefix", ""),
        model_cfg=model_cfg,
        device=config.export_config.device,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples from T4Dataset")
    logger.info(f"Object Classes: {data_loader.get_category_names()}")

    # Load PyTorch model if needed for export
    pytorch_model = None
    if config.export_config.mode != "none":
        if args.checkpoint:
            logger.info("\nLoading PyTorch model...")
            pytorch_model = load_pytorch_model(model_cfg, args.checkpoint, config.export_config.device)
        else:
            logger.error("Checkpoint required for PyTorch model when export mode is not 'none'")
            return

    # Export ONNX
    onnx_path = None
    if config.export_config.should_export_onnx():
        onnx_path = export_onnx(pytorch_model, data_loader, config, logger, model_cfg)
    elif config.runtime_config.get("onnx_file"):
        onnx_path = config.runtime_config["onnx_file"]

    # Export TensorRT
    trt_path = None
    if config.export_config.should_export_tensorrt() and onnx_path:
        trt_path = export_tensorrt(onnx_path, config, logger)

    # Verification
    if config.export_config.verify and pytorch_model and (onnx_path or trt_path):
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Verification")
        logger.info("=" * 80)

        # Prepare test inputs
        test_inputs = {}
        num_test_samples = config.runtime_config.get("num_verification_samples", 3)

        for i in range(min(num_test_samples, data_loader.get_num_samples())):
            sample_name = f"sample_{i}"
            test_inputs[sample_name] = data_loader.load_and_preprocess(i)

        # Run verification
        verification_results = verify_model_outputs(
            pytorch_model=pytorch_model,
            test_inputs=test_inputs,
            onnx_path=onnx_path,
            tensorrt_path=trt_path,
            device=config.export_config.device,
            tolerance=config.verification_config.get("tolerance", 1e-3),
            logger=logger,
        )

        # Check if all verifications passed
        all_passed = all(verification_results.values())
        if all_passed:
            logger.info("✅ All verifications passed!")
        else:
            logger.warning("⚠️  Some verifications failed. Check the logs above for details.")

    # Evaluation
    run_evaluation(data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
