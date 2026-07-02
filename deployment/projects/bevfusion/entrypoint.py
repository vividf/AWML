"""BEVFusion deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.config.base import BaseDeploymentConfig
from deployment.export.contexts import ExportContext
from deployment.projects.bevfusion.evaluation.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.evaluation.executor import BEVFusionExecutor
from deployment.projects.bevfusion.io.component_utils import (
    has_component,
    is_split_bevfusion_components,
    maybe_add_merged_main_body_component,
    should_merge_split_bevfusion,
)
from deployment.projects.bevfusion.io.data_loader import BEVFusionDataLoader
from deployment.projects.bevfusion.runner import BEVFusionDeploymentRunner
from deployment.projects.registry import project_registry


def _validate_bevfusion_components(config: BaseDeploymentConfig) -> None:
    if is_split_bevfusion_components(config.components_cfg):
        config.components_cfg.get_component("bevfusion_sparse")
        config.components_cfg.get_component("bevfusion_dense")
        if has_component(config.components_cfg, "bevfusion_main_body"):
            config.components_cfg.get_component("bevfusion_main_body")
    else:
        config.components_cfg.get_component("bevfusion_main_body")


def _apply_bevfusion_component_merge_overlay(
    config: BaseDeploymentConfig,
    deploy_cfg: Config,
    logger: logging.Logger,
) -> None:
    """Apply optional split+merge overlay driven by deploy config."""
    if not should_merge_split_bevfusion(deploy_cfg):
        return
    before_names = list(config.components_cfg.component_names())
    config.components_cfg = maybe_add_merged_main_body_component(
        deploy_cfg=deploy_cfg,
        components_cfg=config.components_cfg,
    )
    after_names = list(config.components_cfg.component_names())
    logger.info(
        "BEVFusion merge flag enabled: keeping split export and adding merged artifacts component (%s -> %s)",
        before_names,
        after_names,
    )


def _extract_metrics_config(model_cfg: Config, logger: logging.Logger):
    """Extract Detection3DMetricsConfig from model config.

    Tries T4MetricV2 first; falls back to a basic config if a different (or no) evaluator
    is configured, so non-T4MetricV2 model configs still evaluate.
    """
    from deployment.metrics.detection_3d_metrics import Detection3DMetricsConfig, extract_t4metric_v2_config

    class_names = model_cfg.class_names

    def _cfg_get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        if key in obj:
            return obj[key]
        return getattr(obj, key, default)

    evaluator_cfg = getattr(model_cfg, "val_evaluator", None) or getattr(model_cfg, "test_evaluator", None)
    if evaluator_cfg is None:
        logger.warning("No evaluator config found; using basic metrics config")
        return Detection3DMetricsConfig(class_names=class_names, frame_id="base_link")

    evaluator_type = getattr(evaluator_cfg, "type", None)

    if evaluator_type == "T4MetricV2":
        return extract_t4metric_v2_config(model_cfg)

    perception_cfg = _cfg_get(evaluator_cfg, "perception_evaluator_configs")
    frame_id = _cfg_get(evaluator_cfg, "frame_id") or _cfg_get(perception_cfg, "frame_id") or "base_link"

    logger.info(
        "Evaluator type '%s'; using Detection3DMetricsConfig fallback (frame_id=%s)",
        evaluator_type,
        frame_id,
    )
    return Detection3DMetricsConfig(class_names=class_names, frame_id=frame_id)


def _apply_spconv_do_sort(deploy_cfg: Config, logger: logging.Logger) -> None:
    """Apply the ``spconv_do_sort`` field from ``deploy_cfg`` (default ``True``) to the
    GetIndicePairsImplicitGemm symbolic/forward path.

    Controls the pair-mask argsort baked into the exported sparse graph; set ``False`` in a
    deploy config to skip it.
    """
    value = bool(deploy_cfg.get("spconv_do_sort", True))
    from projects.SparseConvolution.sparse_functional import set_do_sort

    set_do_sort(value)
    logger.info(
        "spconv_do_sort: %s (baked into GetIndicePairsImplicitGemm.do_sort_i at ONNX export)",
        value,
    )


def run(args: argparse.Namespace) -> int:
    """Run the BEVFusion deployment workflow."""
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    logger = setup_logging(args.log_level)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)
    _apply_bevfusion_component_merge_overlay(config, deploy_cfg, logger)

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    project_registry.validate_required_components("bevfusion", config.components_cfg)
    _validate_bevfusion_components(config)
    _apply_spconv_do_sort(deploy_cfg, logger)

    logger.info("=" * 80)
    logger.info("BEVFusion Deployment Pipeline")
    logger.info("=" * 80)

    info_file = (deploy_cfg.get("runtime_io", {}) or {}).get("info_file", "")
    data_loader = BEVFusionDataLoader(
        info_file=info_file,
        model_cfg=model_cfg,
    )
    logger.info("Loaded %s samples", data_loader.num_samples)

    metrics_config = _extract_metrics_config(model_cfg, logger)

    plugin_libraries = tuple((deploy_cfg.get("tensorrt_config", {}) or {}).get("plugin_libraries", ()) or ())

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner
    # (which hands it the loaded reference model after export).
    executor = BEVFusionExecutor(
        components_cfg=config.components_cfg,
        tensorrt_plugin_libraries=plugin_libraries,
    )

    evaluator = BEVFusionEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        executor=executor,
    )

    module = getattr(args, "module", "main_body")

    runner = BEVFusionDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        module=module,
        plugin_libraries=plugin_libraries,
    )

    context = ExportContext()
    runner.run(context=context)
    return 0
