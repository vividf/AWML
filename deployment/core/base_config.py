"""
Base configuration classes for deployment framework.

This module provides the foundation for task-agnostic deployment configuration.
Task-specific deployment configs should extend BaseDeploymentConfig.
"""

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from mmengine.config import Config

from deployment.core.backend import Backend

# Constants
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB


def _empty_mapping() -> Mapping[Any, Any]:
    """Return an immutable empty mapping."""
    return MappingProxyType({})


class PrecisionPolicy(str, Enum):
    """Precision policy options for TensorRT."""

    AUTO = "auto"
    FP16 = "fp16"
    FP32_TF32 = "fp32_tf32"
    EXPLICIT_INT8 = "explicit_int8"
    STRONGLY_TYPED = "strongly_typed"


class ExportMode(str, Enum):
    """Export workflow modes."""

    ONNX = "onnx"
    TRT = "trt"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Optional[Union[str, "ExportMode"]]) -> "ExportMode":
        """Parse strings or enum members into ExportMode (defaults to BOTH)."""
        if value is None:
            return cls.BOTH
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(f"Invalid export mode '{value}'. Must be one of {[m.value for m in cls]}.")


# Precision policy mapping for TensorRT
PRECISION_POLICIES = {
    PrecisionPolicy.AUTO.value: {},  # No special flags, TensorRT decides
    PrecisionPolicy.FP16.value: {"FP16": True},
    PrecisionPolicy.FP32_TF32.value: {"TF32": True},  # TF32 for FP32 operations
    PrecisionPolicy.EXPLICIT_INT8.value: {"INT8": True},
    PrecisionPolicy.STRONGLY_TYPED.value: {"STRONGLY_TYPED": True},  # Network creation flag
}


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for model export settings."""

    mode: ExportMode = ExportMode.BOTH
    work_dir: str = "work_dirs"
    checkpoint_path: Optional[str] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    cuda_device: str = "cuda:0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cuda_device", self._parse_cuda_device(self.cuda_device))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExportConfig":
        """Create ExportConfig from dict."""
        return cls(
            mode=ExportMode.from_value(config_dict.get("mode", ExportMode.BOTH)),
            work_dir=config_dict.get("work_dir", cls.work_dir),
            checkpoint_path=config_dict.get("checkpoint_path"),
            onnx_path=config_dict.get("onnx_path"),
            tensorrt_path=config_dict.get("tensorrt_path"),
            cuda_device=config_dict.get("cuda_device", cls.cuda_device),
        )

    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        return self.mode in (ExportMode.ONNX, ExportMode.BOTH)

    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        return self.mode in (ExportMode.TRT, ExportMode.BOTH)

    @staticmethod
    def _parse_cuda_device(device: Optional[str]) -> str:
        """Parse and normalize CUDA device string to 'cuda:N' format."""
        if device is None:
            return "cuda:0"

        if not isinstance(device, str):
            raise ValueError("cuda_device must be a string (e.g., 'cuda:0')")

        normalized = device.strip().lower()
        if normalized == "":
            normalized = "cuda:0"

        if normalized == "cuda":
            normalized = "cuda:0"

        if not normalized.startswith("cuda"):
            raise ValueError(f"Invalid cuda_device '{device}'. Must start with 'cuda'")

        if ":" in normalized:
            suffix = normalized.split(":", 1)[1]
            suffix = suffix.strip()
            if suffix == "":
                suffix = "0"
            if not suffix.isdigit():
                raise ValueError(f"Invalid CUDA device index in '{device}'")
            device_id = int(suffix)
        else:
            device_id = 0

        if device_id < 0:
            raise ValueError("CUDA device index must be non-negative")

        return f"cuda:{device_id}"

    def get_cuda_device_index(self) -> int:
        """Return CUDA device index as integer."""
        return int(self.cuda_device.split(":", 1)[1])


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    data: Mapping[str, Any]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RuntimeConfig":
        return cls(MappingProxyType(dict(config_dict)))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a runtime configuration value."""
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to runtime config."""
        return self.data[key]


@dataclass(frozen=True)
class TensorRTProfileConfig:
    """Optimization profile description for a TensorRT input tensor."""

    min_shape: Tuple[int, ...] = field(default_factory=tuple)
    opt_shape: Tuple[int, ...] = field(default_factory=tuple)
    max_shape: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorRTProfileConfig":
        return cls(
            min_shape=cls._normalize_shape(data.get("min_shape")),
            opt_shape=cls._normalize_shape(data.get("opt_shape")),
            max_shape=cls._normalize_shape(data.get("max_shape")),
        )

    @staticmethod
    def _normalize_shape(shape: Optional[Iterable[int]]) -> Tuple[int, ...]:
        if shape is None:
            return tuple()
        return tuple(int(dim) for dim in shape)

    def has_complete_profile(self) -> bool:
        return bool(self.min_shape and self.opt_shape and self.max_shape)


@dataclass(frozen=True)
class TensorRTModelInputConfig:
    """Typed container for TensorRT model input shape settings."""

    input_shapes: Mapping[str, TensorRTProfileConfig] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorRTModelInputConfig":
        input_shapes_raw = data.get("input_shapes", {}) or {}
        profile_map = {
            name: TensorRTProfileConfig.from_dict(shape_dict or {}) for name, shape_dict in input_shapes_raw.items()
        }
        return cls(input_shapes=MappingProxyType(profile_map))


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for backend-specific settings."""

    common_config: Mapping[str, Any] = field(default_factory=_empty_mapping)
    model_inputs: Tuple[TensorRTModelInputConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BackendConfig":
        common_config = dict(config_dict.get("common_config", {}))
        model_inputs_raw: Iterable[Dict[str, Any]] = config_dict.get("model_inputs", []) or []
        model_inputs: Tuple[TensorRTModelInputConfig, ...] = tuple(
            TensorRTModelInputConfig.from_dict(item) for item in model_inputs_raw
        )
        return cls(
            common_config=MappingProxyType(common_config),
            model_inputs=model_inputs,
        )

    def get_precision_policy(self) -> str:
        """Get precision policy name."""
        return self.common_config.get("precision_policy", PrecisionPolicy.AUTO.value)

    def get_precision_flags(self) -> Dict[str, bool]:
        """Get TensorRT precision flags for the configured policy."""
        policy = self.get_precision_policy()
        return PRECISION_POLICIES.get(policy, {})

    def get_max_workspace_size(self) -> int:
        """Get maximum workspace size for TensorRT."""
        return self.common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE)


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed configuration for evaluation settings."""

    enabled: bool = False
    num_samples: int = 10
    verbose: bool = False
    backends: Mapping[Any, Mapping[str, Any]] = field(default_factory=_empty_mapping)
    models: Mapping[Any, Any] = field(default_factory=_empty_mapping)
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        backends_raw = config_dict.get("backends", {}) or {}
        backends_frozen = {key: MappingProxyType(dict(value)) for key, value in backends_raw.items()}

        return cls(
            enabled=config_dict.get("enabled", False),
            num_samples=config_dict.get("num_samples", 10),
            verbose=config_dict.get("verbose", False),
            backends=MappingProxyType(backends_frozen),
            models=MappingProxyType(dict(config_dict.get("models", {}))),
            devices=MappingProxyType(dict(config_dict.get("devices", {}))),
        )


@dataclass
class VerificationConfig:
    """Typed configuration for verification settings."""

    enabled: bool = True
    num_verify_samples: int = 3
    tolerance: float = 0.1
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)
    scenarios: Mapping[ExportMode, Tuple["VerificationScenario", ...]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VerificationConfig":
        scenarios_raw = config_dict.get("scenarios", {}) or {}
        scenario_map: Dict[ExportMode, Tuple["VerificationScenario", ...]] = {}
        for mode_key, scenario_list in scenarios_raw.items():
            mode = ExportMode.from_value(mode_key)
            scenario_entries = tuple(VerificationScenario.from_dict(entry) for entry in (scenario_list or []))
            scenario_map[mode] = scenario_entries

        return cls(
            enabled=config_dict.get("enabled", True),
            num_verify_samples=config_dict.get("num_verify_samples", 3),
            tolerance=config_dict.get("tolerance", 0.1),
            devices=MappingProxyType(dict(config_dict.get("devices", {}))),
            scenarios=MappingProxyType(scenario_map),
        )

    def get_scenarios(self, mode: ExportMode) -> Tuple["VerificationScenario", ...]:
        """Return scenarios for a specific export mode."""
        return self.scenarios.get(mode, ())


@dataclass(frozen=True)
class VerificationScenario:
    """Immutable verification scenario specification."""

    ref_backend: Backend
    ref_device: str
    test_backend: Backend
    test_device: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationScenario":
        missing_keys = {"ref_backend", "ref_device", "test_backend", "test_device"} - data.keys()
        if missing_keys:
            raise ValueError(f"Verification scenario missing keys: {missing_keys}")

        return cls(
            ref_backend=Backend.from_value(data["ref_backend"]),
            ref_device=str(data["ref_device"]),
            test_backend=Backend.from_value(data["test_backend"]),
            test_device=str(data["test_device"]),
        )


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
        self.export_config = ExportConfig.from_dict(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig.from_dict(deploy_cfg.get("runtime_io", {}))
        self.backend_config = BackendConfig.from_dict(deploy_cfg.get("backend_config", {}))
        self._evaluation_config = EvaluationConfig.from_dict(deploy_cfg.get("evaluation", {}))
        self._verification_config = VerificationConfig.from_dict(deploy_cfg.get("verification", {}))

        self._validate_cuda_device()

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        # Validate required sections
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. " "Please update your config to include 'export' section."
            )

        # Validate export mode
        try:
            ExportMode.from_value(self.deploy_cfg.get("export", {}).get("mode", ExportMode.BOTH))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        # Validate precision policy if present
        backend_cfg = self.deploy_cfg.get("backend_config", {})
        common_cfg = backend_cfg.get("common_config", {})
        precision_policy = common_cfg.get("precision_policy", PrecisionPolicy.AUTO.value)
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    def _validate_cuda_device(self) -> None:
        """Validate CUDA device availability once at config stage."""
        if not self._needs_cuda_device():
            return

        cuda_device = self.export_config.cuda_device
        device_idx = self.export_config.get_cuda_device_index()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device is required (TensorRT export/verification/evaluation enabled) "
                "but torch.cuda.is_available() returned False."
            )

        device_count = torch.cuda.device_count()
        if device_idx >= device_count:
            raise ValueError(
                f"Requested CUDA device '{cuda_device}' but only {device_count} CUDA device(s) are available."
            )

    def _needs_cuda_device(self) -> bool:
        """Determine if current deployment config requires a CUDA device."""
        if self.export_config.should_export_tensorrt():
            return True

        evaluation_cfg = self.evaluation_config
        backends_cfg = evaluation_cfg.backends
        tensorrt_backend = backends_cfg.get(Backend.TENSORRT.value) or backends_cfg.get(Backend.TENSORRT, {})
        if tensorrt_backend and tensorrt_backend.get("enabled", False):
            return True

        verification_cfg = self.verification_config

        for scenario_list in verification_cfg.scenarios.values():
            for scenario in scenario_list:
                if Backend.TENSORRT in (scenario.ref_backend, scenario.test_backend):
                    return True

        return False

    @property
    def evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self._evaluation_config

    @property
    def onnx_config(self) -> Dict:
        """Get ONNX configuration."""
        return self.deploy_cfg.get("onnx_config", {})

    @property
    def verification_config(self) -> VerificationConfig:
        """Get verification configuration."""
        return self._verification_config

    def get_evaluation_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get evaluation backends configuration.

        Returns:
            Dictionary mapping backend names to their configuration
        """
        return self.evaluation_config.backends

    def get_verification_scenarios(self, export_mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """
        Get verification scenarios for the given export mode.

        Args:
            export_mode: Export mode (`ExportMode`)

        Returns:
            Tuple of verification scenarios
        """
        return self.verification_config.get_scenarios(export_mode)

    @property
    def task_type(self) -> Optional[str]:
        """Get task type for pipeline building."""
        return self.deploy_cfg.get("task_type")

    def get_onnx_settings(self) -> Dict[str, Any]:
        """
        Get ONNX export settings.

        Returns:
            Dictionary containing ONNX export parameters
        """
        onnx_config = self.onnx_config
        model_io = self.deploy_cfg.get("model_io", {})

        # Get batch size and dynamic axes from model_io
        batch_size = model_io.get("batch_size", None)
        dynamic_axes = model_io.get("dynamic_axes", None)

        # If batch_size is set to a number, disable dynamic_axes
        if batch_size is not None and isinstance(batch_size, int):
            dynamic_axes = None

        # Handle multiple inputs and outputs
        input_names = [model_io.get("input_name", "input")]
        output_names = [model_io.get("output_name", "output")]

        # Add additional inputs if specified
        additional_inputs = model_io.get("additional_inputs", [])
        for additional_input in additional_inputs:
            if isinstance(additional_input, dict):
                input_names.append(additional_input.get("name", "input"))

        # Add additional outputs if specified
        additional_outputs = model_io.get("additional_outputs", [])
        for additional_output in additional_outputs:
            if isinstance(additional_output, str):
                output_names.append(additional_output)

        settings = {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": onnx_config.get("save_file", "model.onnx"),
            "decode_in_inference": onnx_config.get("decode_in_inference", True),
            "batch_size": batch_size,
        }

        # Include model_wrapper config if present in onnx_config
        if "model_wrapper" in onnx_config:
            settings["model_wrapper"] = onnx_config["model_wrapper"]

        return settings

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
    # Optional overrides
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser
