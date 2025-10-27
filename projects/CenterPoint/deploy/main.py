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
import copy
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
        Tuple of (loaded model, modified model config)
    """
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint

    # Initialize mmdet3d scope
    init_default_scope("mmdet3d")

    # Get model config
    model_config = model_cfg.model.copy()

    # Replace with ONNX models if requested
    if replace_onnx_models:
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

    # Build and load model
    model = MODELS.build(model_config)
    model.to(device)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()

    # Create a new config with the modified model config
    modified_cfg = model_cfg.copy()
    modified_cfg.model = copy.deepcopy(model_config)

    return model, modified_cfg


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
        logger.info(f"Using real data for ONNX export (sample_idx=0)")

        # Export using model's method with real data
        model.save_onnx(
            save_dir=output_dir, 
            onnx_opset_version=onnx_opset_version,
            data_loader=data_loader,
            sample_idx=0
        )

        logger.info(f"✅ ONNX export successful: {output_dir}")
        return output_dir
    else:
        logger.error("Model does not have save_onnx method")
        logger.error("Please use --replace-onnx-models flag or implement ONNX export")
        return None


def export_tensorrt(onnx_dir: str, config: BaseDeploymentConfig, logger: logging.Logger) -> str:
    """
    Export ONNX models to TensorRT engines.

    CenterPoint generates two ONNX files:
    1. pts_voxel_encoder.onnx - voxel feature extraction
    2. pts_backbone_neck_head.onnx - backbone, neck, and head processing

    Returns:
        Path to exported TensorRT engine directory
    """
    logger.info("=" * 80)
    logger.info("Exporting to TensorRT")
    logger.info("=" * 80)

    # Create TensorRT output directory
    trt_dir = os.path.join(onnx_dir, "tensorrt")
    os.makedirs(trt_dir, exist_ok=True)

    # Import TensorRT exporter
    from autoware_ml.deployment.exporters.tensorrt_exporter import TensorRTExporter

    # Get TensorRT configuration
    trt_config = config.backend_config.common_config.copy()
    
    # Define the ONNX files to convert
    onnx_files = [
        ("pts_voxel_encoder.onnx", "pts_voxel_encoder.engine"),
        ("pts_backbone_neck_head.onnx", "pts_backbone_neck_head.engine")
    ]

    success_count = 0
    total_count = len(onnx_files)

    for onnx_file, trt_file in onnx_files:
        onnx_path = os.path.join(onnx_dir, onnx_file)
        trt_path = os.path.join(trt_dir, trt_file)

        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX file not found: {onnx_path}")
            continue

        logger.info(f"Converting {onnx_file} to TensorRT...")
        
        # Create TensorRT exporter
        exporter = TensorRTExporter(trt_config, logger)
        
        # Create dummy sample input for shape configuration
        # For CenterPoint, we need different sample inputs for each component
        if "voxel_encoder" in onnx_file:
            # Voxel encoder input: (num_voxels, num_max_points, point_dim)
            # Use realistic voxel counts for T4Dataset - actual shape is (num_voxels, 32, 11)
            sample_input = torch.randn(10000, 32, 11, device=config.export_config.device)
        else:
            # Backbone/neck/head input: (batch_size, channels, height, width)
            # Use realistic spatial feature dimensions - actual shape is (batch_size, 32, H, W)
            # NOTE: Actual evaluation data can produce up to 760x760, so use 800x800 for max_shape
            sample_input = torch.randn(1, 32, 200, 200, device=config.export_config.device)

        # Export to TensorRT
        success = exporter.export(
            model=None,  # Not used for TensorRT
            sample_input=sample_input,
            output_path=trt_path,
            onnx_path=onnx_path
        )

        if success:
            logger.info(f"✅ TensorRT engine saved: {trt_path}")
            success_count += 1
        else:
            logger.error(f"❌ Failed to convert {onnx_file} to TensorRT")

    if success_count == total_count:
        logger.info(f"✅ All TensorRT engines exported successfully to: {trt_dir}")
        return trt_dir
    elif success_count > 0:
        logger.warning(f"⚠️  Partial success: {success_count}/{total_count} engines exported")
        return trt_dir
    else:
        logger.error("❌ All TensorRT exports failed")
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
            # Check if path exists and is valid for the backend
            is_valid = False
            
            if backend_name == "pytorch":
                # PyTorch: check if checkpoint file exists
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
            elif backend_name == "onnx":
                # ONNX: check if directory exists and contains ONNX files
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
                    is_valid = len(onnx_files) > 0
            elif backend_name == "tensorrt":
                # TensorRT: check if directory exists and contains engine files
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    engine_files = [f for f in os.listdir(model_path) if f.endswith('.engine')]
                    is_valid = len(engine_files) > 0
            
            if is_valid:
                models_to_evaluate.append((backend_name, model_path))
                logger.info(f"  - {backend_name}: {model_path}")
            else:
                logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")

    return models_to_evaluate


def run_evaluation(
    data_loader: CenterPointDataLoader,
    config: BaseDeploymentConfig,
    original_model_cfg: Config,
    onnx_model_cfg: Config,
    logger: logging.Logger,
):
    """
    Run evaluation on specified models.
    
    Args:
        data_loader: Data loader for samples
        config: Deployment configuration
        original_model_cfg: Original model config (for PyTorch evaluation)
        onnx_model_cfg: ONNX-compatible model config (for ONNX/TensorRT evaluation)
        logger: Logger instance
    """
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

    num_samples = eval_config.get("num_samples", 50)
    if num_samples == -1:
        num_samples = data_loader.get_num_samples()

    all_results = {}

    for backend, model_path in models_to_evaluate:
        # Choose the appropriate config based on backend
        if backend == "pytorch":
            # PyTorch: use original config (no ONNX modifications)
            model_cfg = original_model_cfg
            logger.info(f"\nEvaluating {backend.upper()} with original model config")
            logger.info(f"  Model type: {model_cfg.model.type}")
            logger.info(f"  Voxel encoder: {model_cfg.model.pts_voxel_encoder.type}")
        else:
            # ONNX/TensorRT: use ONNX-compatible config
            model_cfg = onnx_model_cfg
            logger.info(f"\nEvaluating {backend.upper()} with ONNX-compatible config")
            logger.info(f"  Model type: {model_cfg.model.type}")
            logger.info(f"  Voxel encoder: {model_cfg.model.pts_voxel_encoder.type}")
        
        # Create evaluator with the appropriate config
        evaluator = CenterPointEvaluator(model_cfg)
        
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
                logger.info(f"  Total Predictions: {results.get('total_predictions', 0)}")
                if 'latency' in results:
                    logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")
            else:
                logger.info("  No results available")


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

    # Prepare model configurations
    # - original_model_cfg: for PyTorch evaluation (no ONNX modifications)
    # - onnx_model_cfg: for ONNX/TensorRT evaluation and export (with ONNX modifications)
    original_model_cfg = copy.deepcopy(model_cfg)  # Deep copy to keep original intact
    onnx_model_cfg = model_cfg  # Will be replaced if ONNX models are loaded
    pytorch_model = None
    
    # Check if we need ONNX-compatible config
    eval_config = config.evaluation_config
    needs_export = config.export_config.mode != "none"
    needs_onnx_eval = False
    if eval_config.get("enabled", False):
        models_to_eval = eval_config.get("models", {})
        if models_to_eval.get("onnx") or models_to_eval.get("tensorrt"):
            needs_onnx_eval = True
    
    # Load model if needed for export or ONNX/TensorRT evaluation
    if needs_export or needs_onnx_eval:
        if not args.checkpoint:
            if needs_export:
                logger.error("Checkpoint required for export")
            else:
                logger.error("Checkpoint required for ONNX/TensorRT evaluation")
            logger.error("Please provide --checkpoint argument")
            return
        
        logger.info("\nLoading PyTorch model...")
        
        # For export or ONNX/TensorRT evaluation: load with ONNX modifications
        if args.replace_onnx_models:
            logger.info("Loading model with ONNX-compatible configuration")
            pytorch_model, onnx_model_cfg = load_pytorch_model(
                model_cfg,
                args.checkpoint,
                config.export_config.device,
                replace_onnx_models=True,
                rot_y_axis_reference=args.rot_y_axis_reference,
            )
        else:
            logger.info("Loading model with original configuration")
            pytorch_model, _ = load_pytorch_model(
                model_cfg,
                args.checkpoint,
                config.export_config.device,
                replace_onnx_models=False,
                rot_y_axis_reference=False,
            )
        
        logger.info("✅ PyTorch model loaded successfully")

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
        
        # Create verification inputs (use raw point cloud data)
        verification_inputs = {}
        num_verify_samples = config.verification_config.get("num_verify_samples", 5)
        
        for i in range(min(num_verify_samples, data_loader.get_num_samples())):
            sample = data_loader.load_sample(i)
            # Get points directly (not preprocessed)
            if 'points' in sample:
                points = sample['points']
            else:
                # Fallback to loading from data_loader
                input_data = data_loader.preprocess(sample)
                points = input_data.get('points', input_data)
            
            verification_inputs[f"sample_{i}"] = points
        
        # Run pipeline-based verification
        # This new approach integrates verification into the evaluation pipeline,
        # allowing both to share the same inference path while differing only
        # in whether postprocessing is applied
        from autoware_ml.deployment.core.verification import verify_centerpoint_pipeline
        
        verification_results = verify_centerpoint_pipeline(
            pytorch_model=pytorch_model,
            test_inputs=verification_inputs,
            onnx_dir=onnx_path,
            tensorrt_dir=trt_path,
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
    run_evaluation(data_loader, config, original_model_cfg, onnx_model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
