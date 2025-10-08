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

from .data_loader import CenterPointDataLoader
from .evaluator import CenterPointEvaluator


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


def run_evaluation(
    model_paths: dict,
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

    evaluator = CenterPointEvaluator(model_cfg)

    num_samples = eval_config.get("num_samples", 50)
    if num_samples == -1:
        num_samples = data_loader.get_num_samples()

    models_to_eval = eval_config.get("models_to_evaluate", ["pytorch"])

    all_results = {}

    for backend in models_to_eval:
        if backend not in model_paths or model_paths[backend] is None:
            logger.warning(f"Model for backend '{backend}' not available, skipping...")
            continue

        results = evaluator.evaluate(
            model_path=model_paths[backend],
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

    # Track model paths
    model_paths = {}

    # Load PyTorch model if needed
    pytorch_model = None
    if config.export_config.mode != "none" or "pytorch" in config.evaluation_config.get("models_to_evaluate", []):
        if args.checkpoint:
            logger.info("\nLoading PyTorch model...")
            pytorch_model = load_pytorch_model(
                model_cfg,
                args.checkpoint,
                config.export_config.device,
                replace_onnx_models=args.replace_onnx_models,
                rot_y_axis_reference=args.rot_y_axis_reference,
            )
            model_paths["pytorch"] = args.checkpoint
            logger.info("✅ PyTorch model loaded successfully")
        else:
            logger.error("Checkpoint required for PyTorch model")
            return

    # Export ONNX
    onnx_path = None
    if config.export_config.should_export_onnx():
        onnx_opset = config.onnx_config.get("opset_version", 13)
        onnx_path = export_onnx(pytorch_model, data_loader, config, logger, onnx_opset_version=onnx_opset)
        if onnx_path:
            model_paths["onnx"] = onnx_path
    elif config.runtime_config.get("onnx_file"):
        onnx_path = config.runtime_config["onnx_file"]
        model_paths["onnx"] = onnx_path

    # Export TensorRT
    trt_path = None
    if config.export_config.should_export_tensorrt() and onnx_path:
        trt_path = export_tensorrt(onnx_path, config, logger)
        if trt_path:
            model_paths["tensorrt"] = trt_path

    # Verification
    if config.export_config.verify and len(model_paths) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Verification")
        logger.info("=" * 80)
        logger.warning("Verification for 3D detection needs custom implementation")
        logger.info("TODO: Implement 3D detection verification")

    # Evaluation
    run_evaluation(model_paths, data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)

    # Print summary
    logger.info("\nGenerated Files:")
    for backend, path in model_paths.items():
        logger.info(f"  {backend.upper()}: {path}")


if __name__ == "__main__":
    main()
