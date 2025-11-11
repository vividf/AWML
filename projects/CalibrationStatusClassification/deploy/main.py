"""
CalibrationStatusClassification Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.runners import DeploymentRunner
from projects.CalibrationStatusClassification.deploy.data_loader import CalibrationDataLoader
from projects.CalibrationStatusClassification.deploy.evaluator import ClassificationEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(checkpoint_path: str, **kwargs):
    """Load PyTorch model from checkpoint."""
    model_cfg = kwargs.get('model_cfg')
    device = kwargs.get('device', 'cuda:0')
    torch_device = torch.device(device)
    model = get_model(model_cfg, checkpoint_path, device=torch_device)
    model.eval()
    return model


def main():
    """Main deployment pipeline using unified runner."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("CalibrationStatusClassification Deployment - Unified Runner Architecture")
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
    logger.info("CalibrationStatusClassification Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")

    # Get info_pkl path
    info_pkl = config.runtime_config.get("info_pkl")
    if not info_pkl:
        logger.error("info_pkl path must be provided in config")
        return

    # Create data loader (calibrated version for export)
    logger.info("\nCreating data loader...")
    data_loader = CalibrationDataLoader(
        info_pkl_path=info_pkl,
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        device=config.export_config.device,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator
    evaluator = ClassificationEvaluator(model_cfg)

    # Create unified runner with custom model loading function
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=lambda checkpoint_path, **kwargs: load_pytorch_model(
            checkpoint_path, model_cfg=model_cfg, device=config.export_config.device, **kwargs
        ),
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

