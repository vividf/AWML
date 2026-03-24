"""BEVFusion deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.cli.args import setup_logging
from deployment.configs import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.projects.bevfusion.eval.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.io.component_utils import is_split_bevfusion_components
from deployment.projects.bevfusion.io.data_loader import BEVFusionDataLoader
from deployment.projects.bevfusion.runner import BEVFusionDeploymentRunner


def _validate_bevfusion_components(config: BaseDeploymentConfig) -> None:
    if is_split_bevfusion_components(config.components_cfg):
        config.components_cfg.get_component("bevfusion_sparse")
        config.components_cfg.get_component("bevfusion_dense")
    else:
        config.components_cfg.get_component("bevfusion_main_body")


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

    _validate_bevfusion_components(config)

    logger.info("=" * 80)
    logger.info("BEVFusion Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    quantization_cfg = deploy_cfg.get("quantization", None)
    if quantization_cfg and quantization_cfg.get("enabled", False):
        logger.info("Quantization: ENABLED")
        logger.info(
            f"  Mode: dense=pytorch_quantization, sparse={'spconv_int8' if quantization_cfg.get('spconv_int8') else 'fp32'}"
        )
        logger.info(f"  Fuse BN: {quantization_cfg.get('fuse_bn', True)}")
        logger.info(f"  Quant backbone: {quantization_cfg.get('quant_backbone', True)}")
        logger.info(f"  Quant neck: {quantization_cfg.get('quant_neck', True)}")
        logger.info(f"  Quant head: {quantization_cfg.get('quant_head', True)}")
        if quantization_cfg.get("spconv_int8"):
            logger.info(f"  Calibration samples: {quantization_cfg.get('num_calibration_samples', 5)}")
    else:
        logger.info("Quantization: disabled")

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
        tensorrt_plugin_libraries=config.tensorrt_config.plugin_libraries,
    )

    module = getattr(args, "module", "main_body")

    runner = BEVFusionDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        module=module,
    )

    context = ExportContext()
    runner.run(context=context)
    return 0
