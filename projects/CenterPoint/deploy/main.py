"""
CenterPoint Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Export to ONNX and/or TensorRT
- Verify outputs across backends
- Evaluate model performance
"""

import sys
from pathlib import Path

from mmengine.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from deployment.core import BaseDeploymentConfig, setup_logging
from deployment.core.config.base_config import parse_base_args
from deployment.core.contexts import CenterPointExportContext
from deployment.exporters.centerpoint.model_wrappers import CenterPointONNXWrapper
from deployment.runners import CenterPointDeploymentRunner
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.evaluator import CenterPointEvaluator
from projects.CenterPoint.deploy.utils import extract_t4metric_v2_config


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()

    # Add CenterPoint-specific arguments
    parser.add_argument(
        "--rot-y-axis-reference", action="store_true", help="Convert rotation to y-axis clockwise reference"
    )

    args = parser.parse_args()
    return args


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

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode.value}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.verification_config.enabled}")
    logger.info(f"  CUDA device (TensorRT): {config.devices.cuda}")
    eval_devices_cfg = config.evaluation_config.devices
    logger.info("  Evaluation devices:")
    logger.info(f"    PyTorch: {eval_devices_cfg.get('pytorch', 'cpu')}")
    logger.info(f"    ONNX: {eval_devices_cfg.get('onnx', 'cpu')}")
    logger.info(f"    TensorRT: {eval_devices_cfg.get('tensorrt', config.devices.cuda)}")
    logger.info(f"  Y-axis rotation: {args.rot_y_axis_reference}")
    logger.info(f"  Runner will build ONNX-compatible model internally")

    # Validate checkpoint path for export
    if config.export_config.should_export_onnx():
        checkpoint_path = config.checkpoint_path
        if not checkpoint_path:
            logger.error("Checkpoint path must be provided in export.checkpoint_path for ONNX/TensorRT export.")
            return

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config["info_file"],
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Extract T4MetricV2 config from model_cfg (if available)
    # This ensures deployment evaluation uses the same settings as training evaluation
    logger.info("\nExtracting T4MetricV2 config from model config...")
    metrics_config = extract_t4metric_v2_config(model_cfg, logger=logger)
    if metrics_config is None:
        logger.warning(
            "T4MetricV2 config not found in model_cfg. "
            "Using default metrics configuration for deployment evaluation."
        )
    else:
        logger.info("Successfully extracted T4MetricV2 config from model config")

    # Create evaluator with original model_cfg and extracted metrics_config
    # Runner will convert model_cfg to ONNX-compatible config and inject both model_cfg and pytorch_model
    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,  # original cfg; will be updated to ONNX cfg by runner
        metrics_config=metrics_config,  # extracted from model_cfg or None (will use defaults)
    )

    # Create CenterPoint-specific runner
    # Runner will load model and inject it into evaluator
    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,  # original cfg; runner will convert to ONNX cfg in load_pytorch_model()
        logger=logger,
        onnx_wrapper_cls=CenterPointONNXWrapper,
    )

    # Execute deployment workflow with typed context
    context = CenterPointExportContext(rot_y_axis_reference=args.rot_y_axis_reference)
    runner.run(context=context)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
