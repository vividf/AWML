"""
CenterPoint Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Export to ONNX and/or TensorRT
- Verify outputs across backends
- Evaluate model performance
"""

import copy
import os
import sys
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters.tensorrt_exporter import TensorRTExporter
from autoware_ml.deployment.runners import DeploymentRunner
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
    checkpoint_path: str,
    model_cfg: Config,
    device: str,
    replace_onnx_models: bool = False,
    rot_y_axis_reference: bool = False,
    **kwargs
):
    """
    Load PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_cfg: Model configuration
        device: Device to load model on
        replace_onnx_models: Whether to replace with ONNX-compatible models
        rot_y_axis_reference: Whether to use y-axis rotation reference
        **kwargs: Additional arguments
        
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


# TODO(vividf): rethinking the onnx export for centerpoint_onnx.
def export_onnx(
    pytorch_model,
    data_loader: CenterPointDataLoader,
    config: BaseDeploymentConfig,
    logger,
    **kwargs
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

    # Get ONNX opset version
    onnx_opset_version = config.onnx_config.get("opset_version", 13)

    # Check if model has save_onnx method
    if hasattr(pytorch_model, "save_onnx"):
        logger.info(f"Using model's built-in save_onnx method")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using real data for ONNX export (sample_idx=0)")

        # Export using model's method with real data
        pytorch_model.save_onnx(
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

# TODO(vividf): rethinking the tensorrt export for centerpoint_onnx.
def export_tensorrt(
    onnx_path: str,
    config: BaseDeploymentConfig,
    data_loader: CenterPointDataLoader,
    logger,
    **kwargs
) -> str:
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
    trt_dir = os.path.join(onnx_path, "tensorrt")
    os.makedirs(trt_dir, exist_ok=True)

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
        onnx_file_path = os.path.join(onnx_path, onnx_file)
        trt_path = os.path.join(trt_dir, trt_file)

        if not os.path.exists(onnx_file_path):
            logger.warning(f"ONNX file not found: {onnx_file_path}")
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
            onnx_path=onnx_file_path
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


class CenterPointDeploymentRunner(DeploymentRunner):
    """
    CenterPoint-specific deployment runner.
    
    Handles CenterPoint-specific requirements:
    - Special model loading with ONNX-compatible replacements
    - Custom ONNX export using model's save_onnx method
    - Multi-file TensorRT export
    - Dual model configs (original and ONNX-compatible)
    """
    
    def __init__(
        self,
        data_loader: CenterPointDataLoader,
        evaluator: CenterPointEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger,
        replace_onnx_models: bool = False,
        rot_y_axis_reference: bool = False,
    ):
        # Store original model config
        self.original_model_cfg = copy.deepcopy(model_cfg)
        self.replace_onnx_models = replace_onnx_models
        self.rot_y_axis_reference = rot_y_axis_reference
        self.onnx_model_cfg = None
        
        # Create custom load function
        def load_model_fn(checkpoint_path, **kwargs):
            model, modified_cfg = load_pytorch_model(
                checkpoint_path=checkpoint_path,
                model_cfg=model_cfg,
                device=config.export_config.device,
                replace_onnx_models=replace_onnx_models,
                rot_y_axis_reference=rot_y_axis_reference,
                **kwargs
            )
            self.onnx_model_cfg = modified_cfg
            return model
        
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            load_model_fn=load_model_fn,
            export_onnx_fn=export_onnx,
            export_tensorrt_fn=export_tensorrt,
        )
    
    def run_evaluation(self, **kwargs) -> dict:
        """
        Run evaluation with appropriate model configs for each backend.
        
        CenterPoint needs different configs for PyTorch vs ONNX/TensorRT.
        """
        eval_config = self.config.evaluation_config
        
        if not eval_config.get("enabled", False):
            self.logger.info("Evaluation disabled, skipping...")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("Running Evaluation")
        self.logger.info("=" * 80)
        
        # Get models to evaluate from config
        models_to_evaluate = self.get_models_to_evaluate()
        
        if not models_to_evaluate:
            self.logger.warning("No models found for evaluation")
            return {}
        
        num_samples = eval_config.get("num_samples", 50)
        if num_samples == -1:
            num_samples = self.data_loader.get_num_samples()
        
        all_results = {}
        
        for backend, model_path in models_to_evaluate:
            # Choose the appropriate config based on backend
            if backend == "pytorch":
                # PyTorch: use original config (no ONNX modifications)
                model_cfg = self.original_model_cfg
                self.logger.info(f"\nEvaluating {backend.upper()} with original model config")
                self.logger.info(f"  Model type: {model_cfg.model.type}")
                self.logger.info(f"  Voxel encoder: {model_cfg.model.pts_voxel_encoder.type}")
            else:
                # ONNX/TensorRT: use ONNX-compatible config
                if self.onnx_model_cfg is None:
                    self.logger.warning("ONNX-compatible config not available, using original config")
                    model_cfg = self.original_model_cfg
                else:
                    model_cfg = self.onnx_model_cfg
                self.logger.info(f"\nEvaluating {backend.upper()} with ONNX-compatible config")
                self.logger.info(f"  Model type: {model_cfg.model.type}")
                self.logger.info(f"  Voxel encoder: {model_cfg.model.pts_voxel_encoder.type}")
            
            # Create evaluator with the appropriate config
            evaluator = CenterPointEvaluator(model_cfg)
            
            results = evaluator.evaluate(
                model_path=model_path,
                data_loader=self.data_loader,
                num_samples=num_samples,
                backend=backend,
                device=self.config.export_config.device,
                verbose=eval_config.get("verbose", False),
            )
            
            all_results[backend] = results
            
            self.logger.info(f"\n{backend.upper()} Results:")
            evaluator.print_results(results)
        
        # Compare results across backends
        if len(all_results) > 1:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Cross-Backend Comparison")
            self.logger.info("=" * 80)
            
            for backend, results in all_results.items():
                self.logger.info(f"\n{backend.upper()}:")
                if results:  # Check if results is not empty
                    self.logger.info(f"  Total Predictions: {results.get('total_predictions', 0)}")
                    if 'latency' in results:
                        self.logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")
                else:
                    self.logger.info("  No results available")
        
        return all_results


def main():
    """Main deployment pipeline using unified runner."""
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
        info_file=config.runtime_config["info_file"], 
        model_cfg=model_cfg, 
        device=config.export_config.device
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator (will be recreated with appropriate config during evaluation)
    evaluator = CenterPointEvaluator(model_cfg)

    # Create CenterPoint-specific runner
    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        replace_onnx_models=args.replace_onnx_models,
        rot_y_axis_reference=args.rot_y_axis_reference,
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=args.checkpoint)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
