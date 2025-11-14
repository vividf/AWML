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
from typing import Any, Dict, Optional

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters import CenterPointONNXExporter, CenterPointTensorRTExporter
from autoware_ml.deployment.runners import DeploymentRunner
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.evaluator import CenterPointEvaluator


def create_onnx_model_cfg(
    model_cfg: Config,
    device: str,
    rot_y_axis_reference: bool = False,
) -> Config:
    """
    Create an ONNX-compatible model config based on the original config.
    """
    onnx_cfg = model_cfg.copy()
    model_config = copy.deepcopy(onnx_cfg.model)

    model_config.type = "CenterPointONNX"
    model_config.point_channels = model_config.pts_voxel_encoder.in_channels
    model_config.device = device

    if model_config.pts_voxel_encoder.type == "PillarFeatureNet":
        model_config.pts_voxel_encoder.type = "PillarFeatureNetONNX"
    elif model_config.pts_voxel_encoder.type == "BackwardPillarFeatureNet":
        model_config.pts_voxel_encoder.type = "BackwardPillarFeatureNetONNX"

    model_config.pts_bbox_head.type = "CenterHeadONNX"
    model_config.pts_bbox_head.separate_head.type = "SeparateHeadONNX"
    model_config.pts_bbox_head.rot_y_axis_reference = rot_y_axis_reference

    if hasattr(model_config.pts_backbone, "type") and model_config.pts_backbone.type == "ConvNeXt_PC":
        model_config.pts_backbone.with_cp = False

    onnx_cfg.model = model_config
    return onnx_cfg


def build_model_from_cfg(model_cfg: Config, checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Build and load a model from the provided configuration.
    """
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint

    init_default_scope("mmdet3d")
    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    model.to(device)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    model.cfg = model_cfg
    return model


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()

    # Add CenterPoint-specific arguments
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

    if replace_onnx_models:
        modified_cfg = create_onnx_model_cfg(model_cfg, device, rot_y_axis_reference)
    else:
        modified_cfg = model_cfg.copy()

    model = build_model_from_cfg(modified_cfg, checkpoint_path, device)

    return model, modified_cfg


class CenterPointDeploymentRunner(DeploymentRunner):
    """
    CenterPoint-specific deployment runner.
    
    Handles CenterPoint-specific requirements:
    - Special model loading with ONNX-compatible replacements
    - Uses CenterPoint-specific exporters (inherited from base exporters)
    - Unified ONNX-compatible config for all backends
    
    Note: CenterPoint exporters are passed directly to DeploymentRunner,
    which automatically handles the multi-file export logic.
    """
    
    def __init__(
        self,
        data_loader: CenterPointDataLoader,
        evaluator: CenterPointEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger,
    ):
        # Store unified ONNX-compatible config
        self.model_cfg = copy.deepcopy(model_cfg)

        def load_model_fn(checkpoint_path, **kwargs):
            return build_model_from_cfg(self.model_cfg, checkpoint_path, device="cpu")

        # Create CenterPoint-specific exporters
        onnx_settings = config.get_onnx_settings()
        trt_settings = config.get_tensorrt_settings()
        
        onnx_exporter = CenterPointONNXExporter(onnx_settings, logger)
        tensorrt_exporter = CenterPointTensorRTExporter(trt_settings, logger)

        # Pass exporters directly to DeploymentRunner
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            load_model_fn=load_model_fn,
            onnx_exporter=onnx_exporter,
            tensorrt_exporter=tensorrt_exporter,
        )


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
    # Optionally override evaluation devices from command line
    if args.device:
        if args.device not in ("cpu", "cuda:0"):
            logger.warning(
                f"Unsupported device override '{args.device}'. "
                "Only 'cpu' or 'cuda:0' are allowed. Ignoring override."
            )
        else:
            eval_devices_cfg = config.deploy_cfg.setdefault("evaluation", {}).setdefault("devices", {})
            eval_devices_cfg["pytorch"] = args.device
            eval_devices_cfg["onnx"] = args.device

    # Always create ONNX-compatible config for all backends (export uses CPU)
    onnx_model_cfg = create_onnx_model_cfg(
        model_cfg,
        device="cpu",
        rot_y_axis_reference=args.rot_y_axis_reference,
    )

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.verification_config.get('enabled', False)}")
    eval_devices_cfg = config.evaluation_config.get("devices", {})
    logger.info("  Evaluation devices:")
    logger.info(f"    PyTorch: {eval_devices_cfg.get('pytorch', 'cpu')}")
    logger.info(f"    ONNX: {eval_devices_cfg.get('onnx', 'cpu')}")
    logger.info(f"    TensorRT: {eval_devices_cfg.get('tensorrt', 'cuda:0')}")
    logger.info(f"  Y-axis rotation: {args.rot_y_axis_reference}")
    logger.info(f"  Using ONNX-compatible config for all backends")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config["info_file"], 
        model_cfg=model_cfg,  # Data loader can use original config
        device="cpu",
        task_type=config.task_type
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator with unified ONNX-compatible config
    # Get checkpoint_path from config or command line args
    checkpoint_path = args.checkpoint or config.export_config.checkpoint_path
    evaluator = CenterPointEvaluator(
        onnx_model_cfg,
        checkpoint_path=checkpoint_path
    )

    # Create CenterPoint-specific runner
    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=onnx_model_cfg,  # Use ONNX-compatible config
        logger=logger,
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=args.checkpoint)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
