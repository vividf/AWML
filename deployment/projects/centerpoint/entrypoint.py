"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.core.config.base_config import BaseDeploymentConfig, setup_logging
from deployment.core.contexts import CenterPointExportContext
from deployment.projects.centerpoint.config.components_schema import CenterPointDeployConfig
from deployment.projects.centerpoint.data_loader import CenterPointDataLoader
from deployment.projects.centerpoint.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.metrics_utils import extract_t4metric_v2_config
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner


def run(args: argparse.Namespace) -> int:
    """Run the CenterPoint deployment workflow for the unified CLI.

    This wires together the CenterPoint bundle components (data loader, evaluator,
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

    # Build typed config once; validation happens in from_dict
    centerpoint_deploy = CenterPointDeployConfig.from_dict(deploy_cfg)

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.num_samples} samples")

    metrics_config = extract_t4metric_v2_config(model_cfg, logger=logger)

    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=centerpoint_deploy.components,
    )

    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        centerpoint_deploy=centerpoint_deploy,
    )

    context = CenterPointExportContext(rot_y_axis_reference=bool(getattr(args, "rot_y_axis_reference", False)))
    runner.run(context=context)
    return 0
