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

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters import CenterPointONNXExporter, CenterPointTensorRTExporter
from autoware_ml.deployment.exporters.centerpoint.model_wrappers import CenterPointONNXWrapper
from autoware_ml.deployment.runners import CenterPointDeploymentRunner
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.evaluator import CenterPointEvaluator


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
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.verification_config.get('enabled', False)}")
    logger.info(f"  CUDA device (TensorRT): {config.export_config.cuda_device}")
    eval_devices_cfg = config.evaluation_config.get("devices", {})
    logger.info("  Evaluation devices:")
    logger.info(f"    PyTorch: {eval_devices_cfg.get('pytorch', 'cpu')}")
    logger.info(f"    ONNX: {eval_devices_cfg.get('onnx', 'cpu')}")
    logger.info(f"    TensorRT: {eval_devices_cfg.get('tensorrt', config.export_config.cuda_device)}")
    logger.info(f"  Y-axis rotation: {args.rot_y_axis_reference}")
    logger.info(f"  Runner will build ONNX-compatible model internally")

    checkpoint_path = config.export_config.checkpoint_path
    if not checkpoint_path and config.export_config.should_export_onnx():
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

    # Checkpoint path
    # Create evaluator with original model_cfg
    # Runner will convert it to ONNX-compatible config and inject both model_cfg and pytorch_model
    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,  # original cfg; will be updated to ONNX cfg by runner
    )

    # Create exporters
    onnx_settings = config.get_onnx_settings()
    trt_settings = config.get_tensorrt_settings()

    onnx_exporter = CenterPointONNXExporter(onnx_settings, logger, model_wrapper=CenterPointONNXWrapper)
    tensorrt_exporter = CenterPointTensorRTExporter(trt_settings, logger, model_wrapper=CenterPointONNXWrapper)

    # Create CenterPoint-specific runner
    # Runner will load model and inject it into evaluator
    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,  # original cfg; runner will convert to ONNX cfg in load_pytorch_model()
        logger=logger,
        onnx_exporter=onnx_exporter,
        tensorrt_exporter=tensorrt_exporter,
    )

    # Execute deployment workflow
    runner.run(
        checkpoint_path=checkpoint_path,
        rot_y_axis_reference=args.rot_y_axis_reference,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
