"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import io
import logging

import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config

from deployment.cli.args import setup_logging
from deployment.configs import BaseDeploymentConfig
from deployment.core.contexts import CenterPointExportContext
from deployment.projects.centerpoint.eval.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.eval.metrics_utils import extract_t4metric_v2_config
from deployment.projects.centerpoint.io.data_loader import CenterPointDataLoader
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner

_REQUIRED_COMPONENTS = ("pts_voxel_encoder", "pts_backbone_neck_head")


def _validate_required_components(components_cfg) -> None:
    """Validate that all CenterPoint required components exist in the config.

    Args:
        components_cfg: Components config with get_component(name).

    Raises:
        KeyError or similar: If any of _REQUIRED_COMPONENTS is missing.
    """
    for component_name in _REQUIRED_COMPONENTS:
        components_cfg.get_component(component_name)


class _StdoutTee(io.TextIOBase):
    """Duplicate stdout writes to terminal and a log file."""

    def __init__(self, stream: io.TextIOBase, log_stream: io.TextIOBase) -> None:
        self._stream = stream
        self._log_stream = log_stream

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._log_stream.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._log_stream.flush()


def run(args: argparse.Namespace) -> int:
    """Run the CenterPoint deployment workflow for the unified CLI.

    Args:
        args: Parsed command-line arguments containing deploy_cfg and model_cfg paths.

    Returns:
        Exit code (0 for success).
    """
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    output_path = deploy_cfg.get("output_path")
    if output_path:
        log_path = Path(str(output_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_stream:
            with redirect_stdout(_StdoutTee(sys.stdout, log_stream)):
                return _run_centerpoint(args, deploy_cfg)

    return _run_centerpoint(args, deploy_cfg)


def _run_centerpoint(args: argparse.Namespace, deploy_cfg: Config) -> int:
    """Execute deployment workflow using a prepared deploy config."""
    logger = setup_logging(args.log_level, deploy_cfg.get("output_path"))
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    _validate_required_components(config.components_cfg)

    quantization_cfg = deploy_cfg.get("quantization", None)

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)
    if quantization_cfg and quantization_cfg.get("enabled", False):
        logger.info(f"  Quantization: {quantization_cfg.get('mode', 'ptq')} (enabled)")
    else:
        logger.info("  Quantization: disabled")

    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
    )
    logger.info(f"Loaded {data_loader.num_samples} samples")

    metrics_config = extract_t4metric_v2_config(model_cfg, logger=logger)

    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=config.components_cfg,
    )

    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    context = CenterPointExportContext(rot_y_axis_reference=bool(getattr(args, "rot_y_axis_reference", False)))
    runner.run(context=context)
    return 0
