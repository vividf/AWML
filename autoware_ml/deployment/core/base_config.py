"""
Base configuration classes for deployment framework.

This module provides the foundation for task-agnostic deployment configuration.
Task-specific deployment configs should extend BaseDeploymentConfig.
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

from mmengine.config import Config

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB

# Precision policy mapping for TensorRT
PRECISION_POLICIES = {
    "auto": {},  # No special flags, TensorRT decides
    "fp16": {"FP16": True},
    "fp32_tf32": {"TF32": True},  # TF32 for FP32 operations
    "explicit_int8": {"INT8": True},
    "strongly_typed": {"STRONGLY_TYPED": True},  # Network creation flag
}


class ExportConfig:
    """Configuration for model export settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.mode = config_dict.get("mode", "both")
        self.verify = config_dict.get("verify", False)
        self.device = config_dict.get("device", "cuda:0")
        self.work_dir = config_dict.get("work_dir", "work_dirs")

    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        return self.mode in ["onnx", "both"]

    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        return self.mode in ["trt", "both"]


class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get a runtime configuration value."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to runtime config."""
        return self._config[key]


class BackendConfig:
    """Configuration for backend-specific settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.common_config = config_dict.get("common_config", {})
        self.model_inputs = config_dict.get("model_inputs", [])

    def get_precision_policy(self) -> str:
        """Get precision policy name."""
        return self.common_config.get("precision_policy", "auto")

    def get_precision_flags(self) -> Dict[str, bool]:
        """Get TensorRT precision flags for the configured policy."""
        policy = self.get_precision_policy()
        return PRECISION_POLICIES.get(policy, {})

    def get_max_workspace_size(self) -> int:
        """Get maximum workspace size for TensorRT."""
        return self.common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE)


class BaseDeploymentConfig:
    """
    Base configuration container for deployment settings.

    This class provides a task-agnostic interface for deployment configuration.
    Task-specific configs should extend this class and add task-specific settings.
    """

    def __init__(self, deploy_cfg: Config):
        """
        Initialize deployment configuration.

        Args:
            deploy_cfg: MMEngine Config object containing deployment settings
        """
        self.deploy_cfg = deploy_cfg
        self._validate_config()

        # Initialize config sections
        self.export_config = ExportConfig(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig(deploy_cfg.get("runtime_io", {}))
        self.backend_config = BackendConfig(deploy_cfg.get("backend_config", {}))

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        # Validate required sections
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. " "Please update your config to include 'export' section."
            )

        # Validate export mode
        valid_modes = ["onnx", "trt", "both", "none"]
        mode = self.deploy_cfg.get("export", {}).get("mode", "both")
        if mode not in valid_modes:
            raise ValueError(f"Invalid export mode '{mode}'. Must be one of {valid_modes}")

        # Validate precision policy if present
        backend_cfg = self.deploy_cfg.get("backend_config", {})
        common_cfg = backend_cfg.get("common_config", {})
        precision_policy = common_cfg.get("precision_policy", "auto")
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    @property
    def evaluation_config(self) -> Dict:
        """Get evaluation configuration."""
        return self.deploy_cfg.get("evaluation", {})

    @property
    def onnx_config(self) -> Dict:
        """Get ONNX configuration."""
        return self.deploy_cfg.get("onnx_config", {})

    def get_onnx_settings(self) -> Dict[str, Any]:
        """
        Get ONNX export settings.

        Returns:
            Dictionary containing ONNX export parameters
        """
        onnx_config = self.onnx_config
        return {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": onnx_config.get("input_names", ["input"]),
            "output_names": onnx_config.get("output_names", ["output"]),
            "dynamic_axes": onnx_config.get("dynamic_axes"),
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": onnx_config.get("save_file", "model.onnx"),
        }

    def get_tensorrt_settings(self) -> Dict[str, Any]:
        """
        Get TensorRT export settings with precision policy support.

        Returns:
            Dictionary containing TensorRT export parameters
        """
        return {
            "max_workspace_size": self.backend_config.get_max_workspace_size(),
            "precision_policy": self.backend_config.get_precision_policy(),
            "policy_flags": self.backend_config.get_precision_flags(),
            "model_inputs": self.backend_config.model_inputs,
        }


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("deployment")


def parse_base_args(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """
    Create argument parser with common deployment arguments.

    Args:
        parser: Optional existing ArgumentParser to add arguments to

    Returns:
        ArgumentParser with deployment arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Deploy model to ONNX/TensorRT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("deploy_cfg", help="Deploy config path")
    parser.add_argument("model_cfg", help="Model config path")
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Model checkpoint path (optional when mode='none')"
    )

    # Optional overrides
    parser.add_argument("--work-dir", help="Override output directory from config")
    parser.add_argument("--device", help="Override device from config")
    parser.add_argument("--log-level", default="INFO", choices=list(logging._nameToLevel.keys()), help="Logging level")

    return parser
