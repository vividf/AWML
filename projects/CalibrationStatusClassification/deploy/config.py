"""
Configuration classes and constants for deployment.
"""

import argparse
import logging
import os
import os.path as osp
from typing import Dict

from mmengine.config import Config

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB
EXPECTED_CHANNELS = 5  # RGB + Depth + Intensity
LABELS = {"0": "miscalibrated", "1": "calibrated"}

# Precision policy mapping
PRECISION_POLICIES = {
    "auto": {},  # No special flags, TensorRT decides
    "fp16": {"FP16": True},
    "fp32_tf32": {"TF32": True},  # TF32 for FP32 operations
    "explicit_int8": {"INT8": True},
    "strongly_typed": {"STRONGLY_TYPED": True},  # Network creation flag
}


class DeploymentConfig:
    """Configuration container for deployment settings."""

    def __init__(self, deploy_cfg: Config):
        self.deploy_cfg = deploy_cfg
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. "
                "Please update your config to the new format with 'export', 'runtime_io' sections."
            )

        if "runtime_io" not in self.deploy_cfg:
            raise ValueError("Missing 'runtime_io' section in deploy config.")

        # Validate export mode
        valid_modes = ["onnx", "trt", "both", "none"]
        mode = self.export_config.get("mode", "both")
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
    def export_config(self) -> Dict:
        """Get export configuration (mode, verify, device, work_dir)."""
        return self.deploy_cfg.get("export", {})

    @property
    def runtime_io_config(self) -> Dict:
        """Get runtime I/O configuration (info_pkl, sample_idx, onnx_file)."""
        return self.deploy_cfg.get("runtime_io", {})

    @property
    def evaluation_config(self) -> Dict:
        """Get evaluation configuration (enabled, num_samples, verbose, model paths)."""
        return self.deploy_cfg.get("evaluation", {})

    @property
    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        mode = self.export_config.get("mode", "both")
        return mode in ["onnx", "both"]

    @property
    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        mode = self.export_config.get("mode", "both")
        return mode in ["trt", "both"]

    @property
    def should_verify(self) -> bool:
        """Check if verification is requested."""
        return self.export_config.get("verify", False)

    @property
    def device(self) -> str:
        """Get device for export."""
        return self.export_config.get("device", "cuda:0")

    @property
    def work_dir(self) -> str:
        """Get working directory."""
        return self.export_config.get("work_dir", os.getcwd())

    @property
    def onnx_settings(self) -> Dict:
        """Get ONNX export settings."""
        onnx_config = self.deploy_cfg.get("onnx_config", {})
        return {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": onnx_config.get("input_names", ["input"]),
            "output_names": onnx_config.get("output_names", ["output"]),
            "dynamic_axes": onnx_config.get("dynamic_axes"),
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": onnx_config.get("save_file", "calibration_classifier.onnx"),
        }

    @property
    def tensorrt_settings(self) -> Dict:
        """Get TensorRT export settings with precision policy support."""
        backend_config = self.deploy_cfg.get("backend_config", {})
        common_config = backend_config.get("common_config", {})

        # Get precision policy
        precision_policy = common_config.get("precision_policy", "auto")
        policy_flags = PRECISION_POLICIES.get(precision_policy, {})

        return {
            "max_workspace_size": common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
            "precision_policy": precision_policy,
            "policy_flags": policy_flags,
            "model_inputs": backend_config.get("model_inputs", []),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export CalibrationStatusClassification model to ONNX/TensorRT.",
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
    parser.add_argument("--info-pkl", help="Override info.pkl path from config")

    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("mmdeploy")
