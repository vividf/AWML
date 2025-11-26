"""
YOLOX_opt_elan Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

import sys
from pathlib import Path

from mmengine.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from deployment.core import BaseDeploymentConfig, setup_logging
from deployment.core.config.base_config import parse_base_args
from deployment.exporters.yolox.model_wrappers import YOLOXONNXWrapper
from deployment.runners import YOLOXDeploymentRunner
from projects.YOLOX_opt_elan.deploy.data_loader import YOLOXOptElanDataLoader
from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator
from projects.YOLOX_opt_elan.deploy.utils import extract_detection2d_metrics_config


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


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

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode.value}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.verification_config.enabled}")
    logger.info(f"  CUDA device (TensorRT): {config.export_config.cuda_device}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = YOLOXOptElanDataLoader(
        ann_file=config.runtime_config["ann_file"],
        img_prefix=config.runtime_config.get("img_prefix", ""),
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Extract Detection2DMetricsConfig for autoware_perception_evaluation
    logger.info("\nExtracting Detection2D metrics config from model config...")
    metrics_config = extract_detection2d_metrics_config(model_cfg, logger=logger)
    logger.info("Successfully created Detection2DMetricsConfig")

    # Create evaluator with extracted metrics_config
    evaluator = YOLOXOptElanEvaluator(
        model_cfg,
        model_cfg_path=model_cfg_path,
        metrics_config=metrics_config,
    )

    # Validate checkpoint path for export
    if config.export_config.should_export_onnx():
        checkpoint_path = config.export_config.checkpoint_path
        if not checkpoint_path:
            logger.error("Checkpoint path must be provided in export.checkpoint_path for ONNX/TensorRT export.")
            return

    # Create YOLOX-specific runner
    runner = YOLOXDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        onnx_wrapper_cls=YOLOXONNXWrapper,
    )

    # Execute deployment workflow
    runner.run(
        model_cfg_path=model_cfg_path,
    )


if __name__ == "__main__":
    main()
