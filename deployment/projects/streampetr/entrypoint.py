"""StreamPETR deployment entrypoint invoked by the unified CLI.

Self-wired (the shared ``run_detection3d_deployment`` helper hard-wires the LiDAR
``PointCloudDataLoader``; StreamPETR is camera-based, so it follows the YOLOX pattern):
parse the two configs, build the typed deploy config + file logging, build the clip-ordered
camera data loader, derive the 3D-detection metrics config, then hand the shared
``Detection3DEvaluator`` and ``StreamPETRExecutor`` to the runner.
"""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.evaluation.detection_3d_evaluator import Detection3DEvaluator
from deployment.metrics.detection_3d_metrics import Detection3DMetricsConfig, extract_t4metric_v2_config
from deployment.projects.streampetr.config.streampetr_deployment_config import StreamPETRDeploymentConfig
from deployment.projects.streampetr.evaluation.executor import StreamPETRExecutor
from deployment.projects.streampetr.io.data_loader import StreamPETRDataLoader
from deployment.projects.streampetr.runner import StreamPETRDeploymentRunner

logger = logging.getLogger(__name__)


def _build_metrics_config(model_cfg: Config) -> Detection3DMetricsConfig:
    """Build the 3D metrics config, tolerating T4Metric (v1) model configs.

    The shared extractor requires ``val_evaluator.type == "T4MetricV2"``; the current
    StreamPETR T4 configs still use ``T4Metric``. Until evaluation is enabled (migration
    spec Phase 6) fall back to a default config built from ``class_names`` — it is never
    exercised while ``evaluation.enabled`` is False.
    """
    try:
        return extract_t4metric_v2_config(model_cfg)
    except (ValueError, KeyError, AttributeError) as exc:
        logger.warning(
            "Falling back to default Detection3DMetricsConfig (%s). "
            "Switch the model config's val_evaluator to T4MetricV2 before enabling evaluation.",
            exc,
        )
        return Detection3DMetricsConfig(class_names=list(model_cfg.class_names), frame_id="base_link")


def run(args: argparse.Namespace) -> int:
    """Run the StreamPETR deployment workflow (load → export → verify → evaluate).

    Args:
        args: Parsed CLI args carrying ``deploy_cfg``, ``model_cfg`` and ``log_level``.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    # The deploy config's top-level ``model_cfg`` records the artifact's canonical pairing; the CLI
    # positional stays as an override for eval-variant runs (mirrors the 3D entrypoint).
    model_cfg_path = args.model_cfg or deploy_cfg.get("model_cfg")
    if not model_cfg_path:
        raise SystemExit(
            "No model config: pass it as the second positional argument or set a top-level "
            f"`model_cfg` in the deploy config ({args.deploy_cfg})."
        )
    model_cfg = Config.fromfile(model_cfg_path)
    config = StreamPETRDeploymentConfig(deploy_cfg)

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    logger.info("=" * 80)
    logger.info("StreamPETR Deployment Pipeline")
    logger.info("=" * 80)

    info_file = (deploy_cfg.get("runtime_io", {}) or {}).get("info_file", "")
    data_loader = StreamPETRDataLoader(model_cfg=model_cfg, info_file=info_file)
    logger.info("Loaded %s samples (clip-ordered)", data_loader.num_samples)

    metrics_config = _build_metrics_config(model_cfg)

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner (which hands
    # it the loaded reference model after export).
    executor = StreamPETRExecutor(components_cfg=config.components_cfg)
    evaluator = Detection3DEvaluator(model_cfg=model_cfg, metrics_config=metrics_config, executor=executor)
    runner = StreamPETRDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
    )

    runner.run()
    return 0
