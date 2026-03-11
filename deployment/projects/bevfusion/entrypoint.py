"""BEVFusion deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.cli.args import setup_logging
from deployment.configs import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.projects.bevfusion.eval.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.io.data_loader import BEVFusionDataLoader
from deployment.projects.bevfusion.runner import BEVFusionDeploymentRunner

_REQUIRED_COMPONENTS = ("bevfusion_main_body",)


def _extract_metrics_config(model_cfg: Config, logger: logging.Logger):
    """Extract Detection3DMetricsConfig from model config.

    Tries T4MetricV2 first, then T4Metric. Falls back to a basic config
    if neither is found.
    """
    from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig

    class_names = model_cfg.class_names

    evaluator_cfg = getattr(model_cfg, "val_evaluator", None) or getattr(model_cfg, "test_evaluator", None)
    if evaluator_cfg is None:
        logger.warning("No evaluator config found; using basic metrics config")
        return Detection3DMetricsConfig(class_names=class_names)

    evaluator_type = getattr(evaluator_cfg, "type", None)

    if evaluator_type == "T4MetricV2":
        from deployment.projects.centerpoint.eval.metrics_utils import extract_t4metric_v2_config

        return extract_t4metric_v2_config(model_cfg, logger=logger)

    logger.info(f"Evaluator type '{evaluator_type}'; using basic Detection3DMetricsConfig")
    return Detection3DMetricsConfig(class_names=class_names)


def run(args: argparse.Namespace) -> int:
    """Run the BEVFusion deployment workflow."""
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    for comp_name in _REQUIRED_COMPONENTS:
        config.components_cfg.get_component(comp_name)

    logger.info("=" * 80)
    logger.info("BEVFusion Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    data_loader = BEVFusionDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
    )
    logger.info(f"Loaded {data_loader.num_samples} samples")

    metrics_config = _extract_metrics_config(model_cfg, logger)

    evaluator = BEVFusionEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=config.components_cfg,
    )

    bevfusion_deploy_cfg_path = getattr(args, "bevfusion_deploy_cfg", None)
    module = getattr(args, "module", "main_body")

    runner = BEVFusionDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        bevfusion_deploy_cfg_path=bevfusion_deploy_cfg_path,
        module=module,
    )

    context = ExportContext()
    runner.run(context=context)
    return 0
