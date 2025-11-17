"""
CalibrationStatusClassification Deployment Main Script (Unified Runner Architecture).

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

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters.calibration.model_wrappers import CalibrationONNXWrapper
from autoware_ml.deployment.exporters.calibration.onnx_exporter import CalibrationONNXExporter
from autoware_ml.deployment.exporters.calibration.tensorrt_exporter import CalibrationTensorRTExporter
from autoware_ml.deployment.runners import CalibrationDeploymentRunner
from projects.CalibrationStatusClassification.deploy.data_loader import CalibrationDataLoader
from projects.CalibrationStatusClassification.deploy.evaluator import ClassificationEvaluator


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
    logger.info("CalibrationStatusClassification Deployment - Unified Runner Architecture")
    logger.info("=" * 80)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    logger.info("=" * 80)
    logger.info("CalibrationStatusClassification Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.verification_config.get('enabled', False)}")
    logger.info(f"  CUDA device (TensorRT): {config.export_config.cuda_device}")

    checkpoint_path = config.export_config.checkpoint_path
    if not checkpoint_path and config.export_config.should_export_onnx():
        logger.error("Checkpoint path must be provided in export.checkpoint_path for ONNX/TensorRT export.")
        return

    # Get info_pkl path
    info_pkl = config.runtime_config.get("info_pkl")
    if not info_pkl:
        logger.error("info_pkl path must be provided in config")
        return

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = CalibrationDataLoader(
        info_pkl_path=info_pkl,
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        device="cpu",
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator
    evaluator = ClassificationEvaluator(model_cfg)

    # Create Calibration-specific exporters
    onnx_settings = config.get_onnx_settings()
    trt_settings = config.get_tensorrt_settings()

    onnx_exporter = CalibrationONNXExporter(onnx_settings, logger, model_wrapper=CalibrationONNXWrapper)
    tensorrt_exporter = CalibrationTensorRTExporter(trt_settings, logger, model_wrapper=CalibrationONNXWrapper)

    # Create Calibration-specific runner
    runner = CalibrationDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        onnx_exporter=onnx_exporter,
        tensorrt_exporter=tensorrt_exporter,
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
