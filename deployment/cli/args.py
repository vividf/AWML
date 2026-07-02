"""
CLI helpers: argument parsing and logging setup.

Config schema does not depend on argparse/logging; this module is the single place for CLI concerns.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"

# Snapshot of the deployment's own root-logger configuration, taken the moment we install
# it. ``restore_deployment_logging`` re-asserts this after a third-party library hijacks the
# root logger — notably ``absl.logging``, which is pulled in by ``pytorch_quantization`` when a
# quantized model is loaded. absl installs its own root handler (emitting only WARNING+ to
# stderr) and can raise the root level, which otherwise silences every subsequent INFO record
# (the entire evaluation phase). See ``restore_deployment_logging``.
_DEPLOYMENT_LOG_HANDLERS: Optional[List[logging.Handler]] = None
_DEPLOYMENT_LOG_LEVEL: Optional[int] = None


def _record_deployment_logging() -> None:
    """Remember the current root handlers + level as the deployment's canonical config."""
    global _DEPLOYMENT_LOG_HANDLERS, _DEPLOYMENT_LOG_LEVEL
    root = logging.getLogger()
    _DEPLOYMENT_LOG_HANDLERS = list(root.handlers)
    _DEPLOYMENT_LOG_LEVEL = root.level


def restore_deployment_logging() -> None:
    """Re-assert the deployment's root-logger config after a third-party hijack.

    ``pytorch_quantization`` imports ``absl.logging``, which installs its own root handler
    (only WARNING+ reaches stderr) and can raise the root level. That silences every
    subsequent ``logging.INFO`` record — including the whole evaluation phase, so runs with
    quantization enabled print calibrator warnings and then nothing else. Call this after the
    (quantized) model is loaded to drop any foreign handlers, re-attach our console+file
    handlers, and reset the root level. No-op if logging was not set up via ``setup_logging``.
    """
    if _DEPLOYMENT_LOG_HANDLERS is None or _DEPLOYMENT_LOG_LEVEL is None:
        return
    root = logging.getLogger()
    recorded = _DEPLOYMENT_LOG_HANDLERS
    # Drop handlers we did not install (e.g. absl's), which would otherwise shadow ours.
    for handler in list(root.handlers):
        if handler not in recorded:
            root.removeHandler(handler)
    # Re-attach any of our handlers that a hijacker detached.
    for handler in recorded:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(_DEPLOYMENT_LOG_LEVEL)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=getattr(logging, level), format=_LOG_FORMAT)
    _record_deployment_logging()
    return logging.getLogger("deployment")


def add_deployment_file_logging(log_file_path: str) -> None:
    """
    Append a UTF-8 file handler to the root logger so all log records are also written to disk.

    Idempotent for the same absolute path. Creates parent directories when needed.

    Args:
        log_file_path: Absolute or resolved path to the log file.
    """
    path = Path(log_file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            handler_path = getattr(h, "baseFilename", "")
            if handler_path and Path(handler_path).resolve(strict=False) == path:
                return

    fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    fh.setLevel(root.level)
    root.addHandler(fh)

    # Fold the new file handler into the canonical snapshot so restore_deployment_logging()
    # keeps writing to disk after any later root-logger hijack (absl via pytorch_quantization).
    _record_deployment_logging()


def parse_base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Create argument parser with common deployment arguments.

    Args:
        parser: Existing ArgumentParser to add arguments to

    Returns:
        ArgumentParser with deployment arguments
    """
    parser.add_argument("deploy_cfg", help="Deploy config path")
    parser.add_argument("model_cfg", help="Model config path")
    # Optional overrides
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser
