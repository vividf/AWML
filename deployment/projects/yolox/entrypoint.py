"""YOLOX deployment entrypoint invoked by the unified CLI.

Wires YOLOX directly (there is a single 2D-detection project, so no shared 2D entrypoint helper is
warranted): parse the two configs, build the typed deploy config + file logging, build the data
loader, derive the 2D-detection metrics config and YOLOX decode params from the model config, then
hand a shared ``Detection2DEvaluator`` and ``YOLOXExecutor`` to the runner.
"""

from __future__ import annotations

import argparse

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.evaluation.detection_2d_evaluator import Detection2DEvaluator
from deployment.metrics.detection_2d_metrics import extract_detection2d_metrics_config
from deployment.projects.yolox.config.yolox_deployment_config import YOLOXDeploymentConfig
from deployment.projects.yolox.evaluation.executor import YOLOXExecutor
from deployment.projects.yolox.inference.base_inference_pipeline import YOLOXDecodeParams
from deployment.projects.yolox.io.data_loader import YOLOXDataLoader
from deployment.projects.yolox.runner import YOLOXDeploymentRunner


def run(args: argparse.Namespace) -> int:
    """Run the YOLOX deployment workflow (load → export → verify → evaluate).

    Args:
        args: Parsed CLI args carrying ``deploy_cfg``, ``model_cfg`` and ``log_level``.

    Returns:
        Process exit code (0 on success).
    """
    logger = setup_logging(args.log_level)

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
    config = YOLOXDeploymentConfig(deploy_cfg)

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    logger.info("=" * 80)
    logger.info("YOLOX Deployment Pipeline")
    logger.info("=" * 80)

    # ``runtime_io.info_file`` overrides the model config's ann_file when set; absent it is "" and
    # the loader resolves the eval info file from the model config.
    info_file = (deploy_cfg.get("runtime_io", {}) or {}).get("info_file", "")
    data_loader = YOLOXDataLoader(model_cfg=model_cfg, info_file=info_file)
    logger.info("Loaded %s samples", data_loader.num_samples)

    metrics_config = extract_detection2d_metrics_config(model_cfg)
    decode_params = YOLOXDecodeParams.from_model_cfg(model_cfg, metrics_config.class_names)

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner (which hands
    # it the loaded reference model after export).
    executor = YOLOXExecutor(components_cfg=config.components_cfg, decode_params=decode_params)
    evaluator = Detection2DEvaluator(model_cfg=model_cfg, metrics_config=metrics_config, executor=executor)
    runner = YOLOXDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
    )

    runner.run()
    return 0
