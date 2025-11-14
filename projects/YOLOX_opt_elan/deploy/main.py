"""
YOLOX_opt_elan Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

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
from autoware_ml.deployment.runners import DeploymentRunner
from projects.YOLOX_opt_elan.deploy.data_loader import YOLOXOptElanDataLoader
from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(checkpoint_path: str, model_cfg_path: str, device: str, **kwargs):
    """
    Load PyTorch model from checkpoint.
    
    Performs YOLOX-specific preprocessing:
    - Replaces ReLU6 with ReLU for better ONNX compatibility
    """
    model = init_detector(model_cfg_path, checkpoint_path, device=device)
    model.eval()
    
    # Replace ReLU6 with ReLU for better ONNX compatibility
    def replace_relu6_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU6):
                setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
            else:
                replace_relu6_with_relu(child)
    
    replace_relu6_with_relu(model)
    
    return model


def main():
    """Main deployment pipeline using unified runner."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment - Unified Runner Architecture")
    logger.info("=" * 80)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    model_cfg_path = args.model_cfg
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
    logger.info(f"  Verify: {config.verification_config.get('enabled', False)}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = YOLOXOptElanDataLoader(
        ann_file=config.runtime_config["ann_file"],
        img_prefix=config.runtime_config.get("img_prefix", ""),
        model_cfg=model_cfg,
        device=config.export_config.device,
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator
    evaluator = YOLOXOptElanEvaluator(model_cfg, model_cfg_path=model_cfg_path)

    # Create unified runner
    # Note: export_onnx uses default implementation from DeploymentRunner
    # Model wrapper is configured in deploy_config.py (onnx_config.model_wrapper)
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=lambda checkpoint_path, **kwargs: load_pytorch_model(
            checkpoint_path, model_cfg_path=model_cfg_path, device=config.export_config.device, **kwargs
        ),
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
