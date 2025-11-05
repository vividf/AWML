"""
YOLOX_opt_elan Deployment Main Script (New Pipeline Architecture).

This script demonstrates the new unified pipeline architecture for YOLOX deployment:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector
from mmengine.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.pipelines.yolox import (
    YOLOXPyTorchPipeline,
    YOLOXONNXPipeline,
    YOLOXTensorRTPipeline,
)
from projects.YOLOX_opt_elan.deploy.data_loader import YOLOXOptElanDataLoader
from projects.YOLOX_opt_elan.deploy.onnx_wrapper import YOLOXONNXWrapper


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(model_cfg: Config, checkpoint_path: str, device: str):
    """Load PyTorch model from checkpoint."""
    model = init_detector(model_cfg, checkpoint_path, device=device)
    model.eval()
    return model


def export_onnx_with_pipeline(
    pytorch_model,
    data_loader: YOLOXOptElanDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger
) -> str:
    """
    Export model to ONNX format using the new pipeline architecture.
    
    Returns:
        Path to exported ONNX file
    """
    logger.info("=" * 80)
    logger.info("Exporting to ONNX (New Pipeline Architecture)")
    logger.info("=" * 80)

    # Get ONNX settings
    onnx_settings = config.get_onnx_settings()
    output_path = os.path.join(config.export_config.work_dir, onnx_settings["save_file"])
    os.makedirs(config.export_config.work_dir, exist_ok=True)

    # Get sample input
    # Use the same preprocessing as inference (via pipeline.preprocess)
    # This ensures ONNX export and inference use identical input format
    sample_idx = config.runtime_config.get("sample_idx", 0)
    sample = data_loader.load_sample(sample_idx)
    single_input = data_loader.preprocess(sample)
    
    # Ensure tensor is float32 (same as pipeline preprocessing)
    if single_input.dtype != torch.float32:
        single_input = single_input.float()

    # Get batch size from configuration
    batch_size = onnx_settings.get("batch_size", 1)
    if batch_size is None:
        input_tensor = single_input
        logger.info("Using dynamic batch size")
    else:
        input_tensor = single_input.repeat(batch_size, 1, 1, 1)
        logger.info(f"Using fixed batch size: {batch_size}")

    # Replace ReLU6 with ReLU for better ONNX compatibility
    def replace_relu6_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU6):
                setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                logger.debug(f"  Replaced {name}: ReLU6 -> ReLU")
            else:
                replace_relu6_with_relu(child)

    replace_relu6_with_relu(pytorch_model)

    # Wrap model for ONNX export
    num_classes = model_cfg.model.bbox_head.num_classes
    wrapped_model = YOLOXONNXWrapper(model=pytorch_model, num_classes=num_classes)
    wrapped_model.eval()

    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output format: [batch_size, num_predictions, {4 + 1 + num_classes}]")
    logger.info(f"Output path: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        input_tensor,
        output_path,
        opset_version=onnx_settings["opset_version"],
        do_constant_folding=onnx_settings["do_constant_folding"],
        input_names=onnx_settings["input_names"],
        output_names=["output"],
        dynamic_axes=onnx_settings.get("dynamic_axes"),
        export_params=onnx_settings.get("export_params", True),
        keep_initializers_as_inputs=onnx_settings.get("keep_initializers_as_inputs", False),
    )

    logger.info(f"✅ ONNX export successful: {output_path}")

    # Apply ONNX simplifier
    if onnx_settings.get("simplify", True):
        try:
            import onnx
            from onnxsim import simplify
            
            logger.info("Applying ONNX simplifier...")
            onnx_model = onnx.load(output_path)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, output_path)
                logger.info("✅ ONNX simplification successful")
            else:
                logger.warning("⚠️  ONNX simplification check failed, using original model")
        except Exception as e:
            logger.warning(f"⚠️  ONNX simplification failed: {e}")

    return output_path


def export_tensorrt_from_onnx(
    onnx_path: str,
    config: BaseDeploymentConfig,
    logger: logging.Logger
) -> str:
    """
    Export ONNX model to TensorRT engine.
    
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
    logger.info(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # Build config
    config_trt = builder.create_builder_config()

    # Handle dynamic shapes
    has_dynamic_shapes = False
    for i in range(network.num_inputs):
        input_shape = network.get_input(i).shape
        if -1 in input_shape:
            has_dynamic_shapes = True
            break

    if has_dynamic_shapes:
        logger.info("Creating optimization profile for dynamic shapes...")
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            # Set profile for batch dimension
            min_shape = tuple(1 if s == -1 else s for s in input_shape)
            opt_shape = min_shape
            max_shape = min_shape

            logger.info(f"Setting profile for {input_name}: {min_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

        config_trt.add_optimization_profile(profile)

    # Set workspace size
    workspace_size = trt_settings["max_workspace_size"]
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Set precision
    precision_flags = trt_settings["policy_flags"]
    if precision_flags.get("FP16", False):
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision")

    # Build engine
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
            # Check if path exists and is valid for the backend
            is_valid = False
            
            if backend_name == "pytorch":
                # PyTorch: check if checkpoint file exists
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
            elif backend_name == "onnx":
                # ONNX: check if file exists (YOLOX uses single file, not directory)
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path) and model_path.endswith('.onnx')
            elif backend_name == "tensorrt":
                # TensorRT: check if engine file exists (YOLOX uses single file, not directory)
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path) and (model_path.endswith('.engine') or model_path.endswith('.trt'))
            
            if is_valid:
                models_to_evaluate.append((backend_name, model_path))
                logger.info(f"  - {backend_name}: {model_path}")
            else:
                logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")
    
    return models_to_evaluate


def run_evaluation(
    data_loader: YOLOXOptElanDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """
    Run evaluation on specified models.
    
    Args:
        data_loader: Data loader for samples
        config: Deployment configuration
        model_cfg: Model configuration
        logger: Logger instance
    """
    from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator
    
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
            if results:  # Check if results is not empty
                logger.info(f"  mAP: {results.get('mAP', 0):.4f}")
                logger.info(f"  mAP@50: {results.get('mAP_50', 0):.4f}")
                if 'latency' in results:
                    logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")
            else:
                logger.info("  No results available")


def run_verification(
    pytorch_checkpoint: str,
    onnx_path: str,
    tensorrt_path: str,
    data_loader: YOLOXOptElanDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """
    Run verification on exported models.
    
    Args:
        pytorch_checkpoint: Path to PyTorch checkpoint (reference)
        onnx_path: Path to ONNX model file
        tensorrt_path: Path to TensorRT engine file
        data_loader: Data loader for test samples
        config: Deployment configuration
        model_cfg: Model configuration
        logger: Logger instance
    """
    # Verification
    if not config.export_config.verify:
        logger.info("Verification disabled, skipping...")
        return

    if not pytorch_checkpoint:
        logger.warning("PyTorch checkpoint path not available, skipping verification")
        return
    
    if not onnx_path and not tensorrt_path:
        logger.info("No exported models to verify, skipping verification")
        return
    
    # Get verification parameters
    num_verify_samples = config.verification_config.get("num_verify_samples", 3)
    tolerance = config.verification_config.get("tolerance", 0.1)
    
    # Create evaluator
    from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator
    evaluator = YOLOXOptElanEvaluator(model_cfg)
    
    # Run verification
    verification_results = evaluator.verify(
        pytorch_model_path=pytorch_checkpoint,
        onnx_model_path=onnx_path,
        tensorrt_model_path=tensorrt_path,
        data_loader=data_loader,
        num_samples=num_verify_samples,
        device=config.export_config.device,
        tolerance=tolerance,
        verbose=False,
    )
    
    # Check if verification succeeded
    if 'summary' in verification_results:
        summary = verification_results['summary']
        if summary['failed'] == 0:
            if summary.get('skipped', 0) > 0:
                logger.info(f"\n✅ All verifications passed! ({summary['skipped']} skipped)")
            else:
                logger.info("\n✅ All verifications passed!")
        else:
            logger.warning(f"\n⚠️  {summary['failed']}/{summary['total']} verifications failed")
    else:
        logger.error("\n❌ Verification encountered errors")


def main():
    """Main deployment pipeline with new architecture."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment - New Pipeline Architecture")
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

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
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
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Check if we need model loading and export
    eval_config = config.evaluation_config
    needs_export = config.export_config.mode != "none"
    needs_onnx_eval = False
    if eval_config.get("enabled", False):
        models_to_eval = eval_config.get("models", {})
        if models_to_eval.get("onnx") or models_to_eval.get("tensorrt"):
            needs_onnx_eval = True
    
    # Load model if needed for export or ONNX/TensorRT evaluation
    pytorch_model = None
    onnx_path = None
    tensorrt_path = None
    
    if needs_export or needs_onnx_eval:
        if not args.checkpoint:
            if needs_export:
                logger.error("Checkpoint required for export")
            else:
                logger.error("Checkpoint required for ONNX/TensorRT evaluation")
            logger.error("Please provide --checkpoint argument")
            return
        
        # Load PyTorch model
        logger.info("\nLoading PyTorch model...")
        pytorch_model = load_pytorch_model(model_cfg, args.checkpoint, config.export_config.device)
        
        # Export ONNX
        if config.export_config.should_export_onnx():
            onnx_path = export_onnx_with_pipeline(pytorch_model, data_loader, config, model_cfg, logger)
        
        # Export TensorRT
        if config.export_config.should_export_tensorrt() and onnx_path:
            tensorrt_path = export_tensorrt_from_onnx(onnx_path, config, logger)
    
    # Get model paths from evaluation config if not exported
    if not onnx_path or not tensorrt_path:
        eval_models = config.evaluation_config.get("models", {})
        if not onnx_path:
            onnx_path = eval_models.get("onnx")
            if onnx_path and not os.path.exists(onnx_path):
                logger.warning(f"ONNX file from config does not exist: {onnx_path}")
                onnx_path = None
        if not tensorrt_path:
            tensorrt_path = eval_models.get("tensorrt")
            if tensorrt_path and not os.path.exists(tensorrt_path):
                logger.warning(f"TensorRT engine from config does not exist: {tensorrt_path}")
                tensorrt_path = None

    # Verification
    run_verification(
        pytorch_checkpoint=args.checkpoint,
        onnx_path=onnx_path,
        tensorrt_path=tensorrt_path,
        data_loader=data_loader,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    # Run evaluation
    run_evaluation(data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

