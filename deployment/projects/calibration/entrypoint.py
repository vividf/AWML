"""Calibration deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.core.config.base_config import BaseDeploymentConfig, setup_logging
from deployment.core.contexts import CalibrationExportContext
from deployment.projects.calibration.data_loader import CalibrationDataLoader
from deployment.projects.calibration.evaluator import CalibrationEvaluator
from deployment.projects.calibration.metrics_utils import extract_classification_metrics_config
from deployment.projects.calibration.runner import CalibrationDeploymentRunner


def run(args: argparse.Namespace) -> int:
    """Run the Calibration deployment workflow for the unified CLI.

    This wires together the Calibration bundle components (data loader, evaluator,
    runner) and executes export/verification/evaluation according to `deploy_cfg`.

    Args:
        args: Parsed command-line arguments containing deploy_cfg and model_cfg paths.

    Returns:
        Exit code (0 for success).
    """
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    logger.info("=" * 80)
    logger.info("CalibrationStatusClassification Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    # Get info_file path
    info_file = config.runtime_config.info_file
    if not info_file:
        logger.error("info_file path must be provided in config")
        return 1

    data_loader = CalibrationDataLoader(
        info_pkl_path=info_file,
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        device="cpu",
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    metrics_config = extract_classification_metrics_config(model_cfg, logger=logger)

    # Get components config for artifact path resolution
    components_cfg = deploy_cfg.get("components", {})

    evaluator = CalibrationEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=components_cfg,
    )

    runner = CalibrationDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    context = CalibrationExportContext()
    runner.run(context=context)
    return 0
