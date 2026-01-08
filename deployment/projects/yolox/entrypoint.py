"""YOLOX deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.core.config.base_config import BaseDeploymentConfig, setup_logging
from deployment.core.contexts import YOLOXExportContext
from deployment.projects.yolox.data_loader import YOLOXDataLoader
from deployment.projects.yolox.evaluator import YOLOXEvaluator
from deployment.projects.yolox.metrics_utils import extract_detection2d_metrics_config
from deployment.projects.yolox.runner import YOLOXDeploymentRunner


def run(args: argparse.Namespace) -> int:
    """Run the YOLOX deployment workflow for the unified CLI.

    This wires together the YOLOX bundle components (data loader, evaluator,
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
    logger.info("YOLOX_opt_elan Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    data_loader = YOLOXDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    metrics_config = extract_detection2d_metrics_config(model_cfg, logger=logger)

    # Get components config for artifact path resolution
    components_cfg = deploy_cfg.get("components", {})

    evaluator = YOLOXEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=components_cfg,
    )

    runner = YOLOXDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    context = YOLOXExportContext(model_cfg=args.model_cfg)
    runner.run(context=context)
    return 0
