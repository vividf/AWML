"""Calibration classifier deployment entrypoint invoked by the unified CLI.

Wires the classifier directly (single classification project, so no shared classification entrypoint
helper is warranted): parse the two configs, build the typed deploy config + file logging, build the
dual-variant data loader, derive the classification metrics config from the deploy-config class
names, then hand a shared ``ClassificationEvaluator`` and ``CalibrationExecutor`` to the runner.
"""

from __future__ import annotations

import argparse

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.evaluation.classification_evaluator import ClassificationEvaluator
from deployment.metrics.classification_metrics import extract_classification_metrics_config
from deployment.projects.calibration.config.calibration_deployment_config import CalibrationDeploymentConfig
from deployment.projects.calibration.evaluation.executor import CalibrationExecutor
from deployment.projects.calibration.io.data_loader import CalibrationDataLoader
from deployment.projects.calibration.runner import CalibrationDeploymentRunner


def run(args: argparse.Namespace) -> int:
    """Run the calibration classifier deployment workflow (load → export → verify → evaluate).

    Args:
        args: Parsed CLI args carrying ``deploy_cfg``, ``model_cfg`` and ``log_level``.

    Returns:
        Process exit code (0 on success).
    """
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg_path = args.model_cfg or deploy_cfg.get("model_cfg")
    if not model_cfg_path:
        raise SystemExit(
            "No model config: pass it as the second positional argument or set a top-level "
            f"`model_cfg` in the deploy config ({args.deploy_cfg})."
        )
    # import_custom_modules=False: the classifier is pure mmpretrain, so building it needs no custom
    # registry entries, and eagerly importing the model config's custom_imports would pull in
    # training-only modules (the result-visualization hook, the training dataset) whose deps
    # (e.g. matplotlib) need not exist in a deploy image. The data loader imports the one thing
    # deployment does need — the calibration transform — lazily, when it builds the loader.
    model_cfg = Config.fromfile(model_cfg_path, import_custom_modules=False)
    config = CalibrationDeploymentConfig(deploy_cfg)

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    logger.info("=" * 80)
    logger.info("Calibration Status Classification Deployment Pipeline")
    logger.info("=" * 80)

    # The classifier's ground truth is synthetic (built by the transform), so an eval info file is
    # required; there is no model-config fallback for it.
    info_file = (deploy_cfg.get("runtime_io", {}) or {}).get("info_file", "")
    if not info_file:
        raise SystemExit(
            f"calibration requires `runtime_io.info_file` (the calibration info .pkl) in {args.deploy_cfg}."
        )

    class_names = config.class_names
    data_loader = CalibrationDataLoader(model_cfg=model_cfg, info_file=info_file, class_names=class_names)
    logger.info(
        "Loaded %s frames (%s base samples × 2 calibrated/miscalibrated variants)",
        data_loader.num_samples,
        data_loader.num_samples // 2,
    )

    metrics_config = extract_classification_metrics_config(model_cfg, class_names=class_names)

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner (which hands
    # it the loaded reference model after export).
    executor = CalibrationExecutor(components_cfg=config.components_cfg, class_names=class_names)
    evaluator = ClassificationEvaluator(model_cfg=model_cfg, metrics_config=metrics_config, executor=executor)
    runner = CalibrationDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
    )

    runner.run()
    return 0
