"""
CenterPoint Deployment Main Script.

This script handles the complete deployment pipeline for CenterPoint:
- Export to ONNX and/or TensorRT
- Verify outputs across backends
- Evaluate model performance

This is a modernized version that integrates with the unified deployment framework,
replacing the old DeploymentRunner approach.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.evaluator import CenterPointEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()

    # Add CenterPoint-specific arguments
    parser.add_argument(
        "--replace-onnx-models", action="store_true", help="Replace model components with ONNX-compatible versions"
    )
    parser.add_argument(
        "--rot-y-axis-reference", action="store_true", help="Convert rotation to y-axis clockwise reference"
    )

    args = parser.parse_args()
    return args


def load_pytorch_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: str,
    replace_onnx_models: bool = False,
    rot_y_axis_reference: bool = False,
):
    """
    Load PyTorch model from checkpoint.

    Args:
        model_cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        replace_onnx_models: Whether to replace with ONNX-compatible models
        rot_y_axis_reference: Whether to use y-axis rotation reference

    Returns:
        Loaded model
    """
    from mmdet3d.apis import init_model
    from mmengine.registry import MODELS

    # Initialize mmdet3d scope
    init_default_scope("mmdet3d")

    # Get model config
    model_config = model_cfg.model.copy()

    # Replace with ONNX models if requested
    if replace_onnx_models:
        logger = logging.getLogger(__name__)
        logger.info("Replacing model components with ONNX-compatible versions")

        # Update model type
        model_config.type = "CenterPointONNX"
        model_config.point_channels = model_config.pts_voxel_encoder.in_channels
        model_config.device = device

        # Update voxel encoder
        if model_config.pts_voxel_encoder.type == "PillarFeatureNet":
            model_config.pts_voxel_encoder.type = "PillarFeatureNetONNX"
        elif model_config.pts_voxel_encoder.type == "BackwardPillarFeatureNet":
            model_config.pts_voxel_encoder.type = "BackwardPillarFeatureNetONNX"

        # Update bbox head
        model_config.pts_bbox_head.type = "CenterHeadONNX"
        model_config.pts_bbox_head.separate_head.type = "SeparateHeadONNX"
        model_config.pts_bbox_head.rot_y_axis_reference = rot_y_axis_reference

        # Disable gradient checkpointing for ConvNeXt if present
        if hasattr(model_config.pts_backbone, "type") and model_config.pts_backbone.type == "ConvNeXt_PC":
            model_config.pts_backbone.with_cp = False

    # Build model
    model = MODELS.build(model_config)
    model.to(device)

    # Load checkpoint
    from mmengine.runner import load_checkpoint

    load_checkpoint(model, checkpoint_path, map_location=device)

    model.eval()

    return model


def export_onnx(
    model,
    data_loader: CenterPointDataLoader,
    config: BaseDeploymentConfig,
    logger: logging.Logger,
    onnx_opset_version: int = 13,
) -> str:
    """
    Export model to ONNX format.

    For CenterPoint, this uses the model's built-in save_onnx method.

    Returns:
        Path to exported ONNX file directory
    """
    logger.info("=" * 80)
    logger.info("Exporting to ONNX")
    logger.info("=" * 80)

    # Create output directory
    output_dir = config.export_config.work_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if model has save_onnx method
    if hasattr(model, "save_onnx"):
        logger.info(f"Using model's built-in save_onnx method")
        logger.info(f"Output directory: {output_dir}")

        # Export using model's method
        model.save_onnx(save_dir=output_dir, onnx_opset_version=onnx_opset_version)

        logger.info(f"✅ ONNX export successful: {output_dir}")
        return output_dir
    else:
        logger.error("Model does not have save_onnx method")
        logger.error("Please use --replace-onnx-models flag or implement ONNX export")
        return None


def export_tensorrt(onnx_dir: str, config: BaseDeploymentConfig, logger: logging.Logger) -> str:
    """
    Export ONNX model to TensorRT.

    Note: For CenterPoint with multiple ONNX files, this would need to handle
    each component separately.

    Returns:
        Path to exported TensorRT engine directory
    """
    logger.info("=" * 80)
    logger.info("Exporting to TensorRT")
    logger.info("=" * 80)

    logger.warning("TensorRT export for CenterPoint requires handling multiple ONNX files")
    logger.warning("This is a placeholder - implement multi-file TensorRT build if needed")

    # TODO: Implement TensorRT export for multi-file ONNX models

    return None


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
    data_loader: CenterPointDataLoader,
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

    evaluator = CenterPointEvaluator(model_cfg)

    num_samples = eval_config.get("num_samples", 50)
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
            logger.info(f"  Total Predictions: {results['total_predictions']}")
            logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")


def main():
    """Main deployment pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

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
    logger.info("CenterPoint Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")
    logger.info(f"  Replace ONNX models: {args.replace_onnx_models}")
    logger.info(f"  Y-axis rotation: {args.rot_y_axis_reference}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config["info_file"], model_cfg=model_cfg, device=config.export_config.device
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Load PyTorch model if needed for export
    pytorch_model = None
    if config.export_config.mode != "none":
        if args.checkpoint:
            logger.info("\nLoading PyTorch model...")
            pytorch_model = load_pytorch_model(
                model_cfg,
                args.checkpoint,
                config.export_config.device,
                replace_onnx_models=args.replace_onnx_models,
                rot_y_axis_reference=args.rot_y_axis_reference,
            )
            logger.info("✅ PyTorch model loaded successfully")
        else:
            logger.error("Checkpoint required for PyTorch model when export mode is not 'none'")
            return

    # Export ONNX
    onnx_path = None
    if config.export_config.should_export_onnx():
        onnx_opset = config.onnx_config.get("opset_version", 13)
        onnx_path = export_onnx(pytorch_model, data_loader, config, logger, onnx_opset_version=onnx_opset)
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
        
        # Create verification inputs
        verification_inputs = {}
        num_verify_samples = config.verification_config.get("num_verify_samples", 5)
        
        for i in range(min(num_verify_samples, data_loader.get_num_samples())):
            sample = data_loader.load_sample(i)
            input_data = data_loader.preprocess(sample)
            verification_inputs[f"sample_{i}"] = input_data
        
        # Run verification
        from autoware_ml.deployment.core.verification import verify_model_outputs
        
        verification_results = verify_model_outputs(
            pytorch_model=pytorch_model,
            test_inputs=verification_inputs,
            onnx_path=onnx_path,
            tensorrt_path=trt_path,
            device=config.export_config.device,
            tolerance=config.verification_config.get("tolerance", 1e-2),
            logger=logger,
        )
        
        # Print verification results
        logger.info("\nVerification Results:")
        for backend, passed in verification_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"  {backend}: {status}")

    # Evaluation
    run_evaluation(data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
