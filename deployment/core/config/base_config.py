"""
Base configuration classes for deployment framework.

This module provides the foundation for task-agnostic deployment configuration.
Task-specific deployment configs should extend BaseDeploymentConfig.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from mmengine.config import Config

from deployment.core.backend import Backend
from deployment.exporters.common.configs import (
    ONNXExportConfig,
    TensorRTExportConfig,
    TensorRTModelInputConfig,
    TensorRTProfileConfig,
)

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
    STRONGLY_TYPED = "strongly_typed"


class ExportMode(str, Enum):
    """Export pipeline modes."""

    ONNX = "onnx"
    TRT = "trt"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Optional[Union[str, ExportMode]]) -> ExportMode:
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
    PrecisionPolicy.STRONGLY_TYPED.value: {"STRONGLY_TYPED": True},  # Network creation flag
}


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for model export settings."""

    mode: ExportMode = ExportMode.BOTH
    work_dir: str = "work_dirs"
    onnx_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> ExportConfig:
        """Create ExportConfig from dict."""
        return cls(
            mode=ExportMode.from_value(config_dict.get("mode", ExportMode.BOTH)),
            work_dir=config_dict.get("work_dir", cls.work_dir),
            onnx_path=config_dict.get("onnx_path"),
        )

    @property
    def should_export_onnx(self) -> bool:
        """Whether ONNX export is requested."""
        return self.mode in (ExportMode.ONNX, ExportMode.BOTH)

    @property
    def should_export_tensorrt(self) -> bool:
        """Whether TensorRT export is requested."""
        return self.mode in (ExportMode.TRT, ExportMode.BOTH)


@dataclass(frozen=True)
class DeviceConfig:
    """Normalized device settings shared across deployment stages."""

    cpu: str = "cpu"
    cuda: Optional[str] = "cuda:0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", self._normalize_cpu(self.cpu))
        object.__setattr__(self, "cuda", self._normalize_cuda(self.cuda))

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> DeviceConfig:
        """Create DeviceConfig from dict."""
        return cls(cpu=config_dict.get("cpu", cls.cpu), cuda=config_dict.get("cuda", cls.cuda))

    @staticmethod
    def _normalize_cpu(device: Optional[str]) -> str:
        """Normalize CPU device string."""
        if not device:
            return "cpu"
        normalized = str(device).strip().lower()
        if normalized.startswith("cuda"):
            raise ValueError("CPU device cannot be a CUDA device")
        return normalized

    @staticmethod
    def _normalize_cuda(device: Optional[str]) -> Optional[str]:
        """Normalize CUDA device string to 'cuda:N' format."""
        if device is None:
            return None
        if not isinstance(device, str):
            raise ValueError("cuda device must be a string (e.g., 'cuda:0')")
        normalized = device.strip().lower()
        if normalized == "":
            return None
        if normalized == "cuda":
            normalized = "cuda:0"
        if not normalized.startswith("cuda"):
            raise ValueError(f"Invalid CUDA device '{device}'. Must start with 'cuda'")
        suffix = normalized.split(":", 1)[1] if ":" in normalized else "0"
        suffix = suffix.strip() or "0"
        if not suffix.isdigit():
            raise ValueError(f"Invalid CUDA device index in '{device}'")
        device_id = int(suffix)
        if device_id < 0:
            raise ValueError("CUDA device index must be non-negative")
        return f"cuda:{device_id}"

    @property
    def cuda_device_index(self) -> Optional[int]:
        """Return CUDA device index as integer (if configured)."""
        if self.cuda is None:
            return None
        return int(self.cuda.split(":", 1)[1])


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    info_file: str = ""
    sample_idx: int = 0

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> RuntimeConfig:
        """Create RuntimeConfig from dictionary."""
        return cls(
            info_file=config_dict.get("info_file", ""),
            sample_idx=config_dict.get("sample_idx", 0),
        )


@dataclass(frozen=True)
class TensorRTConfig:
    """
    Configuration for TensorRT backend-specific settings.

    Uses config structure:
        tensorrt_config = dict(precision_policy="auto", max_workspace_size=1<<30)

    TensorRT profiles are defined in components.*.tensorrt_profile.

    Note:
        The deploy config key for this section is **`tensorrt_config`**.
    """

    precision_policy: str = PrecisionPolicy.AUTO.value
    max_workspace_size: int = DEFAULT_WORKSPACE_SIZE

    def __post_init__(self) -> None:
        """Validate TensorRT precision policy at construction time."""
        if self.precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{self.precision_policy}'. "
                f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> TensorRTConfig:
        return cls(
            precision_policy=config_dict.get("precision_policy", PrecisionPolicy.AUTO.value),
            max_workspace_size=config_dict.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
        )

    @property
    def precision_flags(self) -> Mapping[str, bool]:
        """TensorRT precision flags for the configured policy."""
        return PRECISION_POLICIES[self.precision_policy]


# =============================================================================
# Component config (deploy_cfg["components"]): generic for any project
# =============================================================================


@dataclass(frozen=True)
class InputSpec:
    """Single input name/dtype for a component."""

    name: str
    dtype: str = "float32"


@dataclass(frozen=True)
class OutputSpec:
    """Single output name/dtype for a component."""

    name: str
    dtype: str = "float32"


@dataclass(frozen=True)
class ComponentIO:
    """I/O specification for a component (inputs, outputs, dynamic_axes)."""

    inputs: List[InputSpec]
    outputs: List[OutputSpec]
    dynamic_axes: Dict[str, Dict[int, str]]


@dataclass(frozen=True)
class ComponentCfg:
    """Configuration for one deployable component (e.g. model, voxel_encoder, backbone_head)."""

    name: str
    onnx_file: str
    engine_file: str
    io: ComponentIO
    tensorrt_profile: Dict[str, TensorRTProfileConfig]


@dataclass(frozen=True)
class ComponentsCfg:
    """Unified component configuration: mapping of component name -> ComponentCfg.

    Generic: single-component (e.g. "model") or multi-component (e.g. "voxel_encoder", "backbone_head").
    Use from_dict(deploy_cfg["components"]) to build; project-specific code may validate required names.
    """

    _components: Mapping[str, ComponentCfg]

    def get_component(self, name: str) -> ComponentCfg:
        """Get component config by name. Raises KeyError if not found."""
        if name not in self._components:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._components.keys())}")
        return self._components[name]

    def get_artifact_filename(self, component: str, file_key: str) -> Optional[str]:
        """Return artifact filename for path resolution (onnx_file or engine_file)."""
        comp = self._components.get(component)
        if comp is None:
            return None
        return getattr(comp, file_key, None) or None

    def component_names(self) -> Iterable[str]:
        """Iterate over component names."""
        return self._components.keys()

    def items(self) -> Iterable[Tuple[str, ComponentCfg]]:
        """Iterate (name, ComponentCfg) pairs."""
        return self._components.items()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ComponentsCfg:
        """Build ComponentsCfg from deploy_cfg['components'] dict. Generic: any keys allowed."""
        if not isinstance(raw, Mapping):
            raise TypeError(f"components must be a mapping, got {type(raw).__name__}")
        parsed = {}
        for label, comp in raw.items():
            parsed[label] = cls._parse_component(comp, label)
        return cls(_components=MappingProxyType(parsed))

    @classmethod
    def _parse_component(cls, comp: Any, label: str) -> ComponentCfg:
        if not isinstance(comp, Mapping):
            raise TypeError(f"components['{label}'] must be a mapping, got {type(comp).__name__}")
        for field in ("name", "onnx_file", "engine_file", "io"):
            if field not in comp:
                raise KeyError(f"components['{label}'] must define '{field}'.")
        io_raw = comp["io"]
        if not isinstance(io_raw, Mapping):
            raise TypeError(f"components['{label}'].io must be a mapping, got {type(io_raw).__name__}")
        if "outputs" not in io_raw or not io_raw["outputs"]:
            raise KeyError(f"components['{label}'].io.outputs must be a non-empty list.")
        if "inputs" not in io_raw or not io_raw["inputs"]:
            raise KeyError(f"components['{label}'].io.inputs must be a non-empty list.")
        outputs = []
        for i, out in enumerate(io_raw["outputs"]):
            if not isinstance(out, Mapping) or "name" not in out:
                raise KeyError(f"components['{label}'].io.outputs[{i}] must define 'name'.")
            name = out["name"]
            if not name or not isinstance(name, str):
                raise ValueError(f"components['{label}'].io.outputs[{i}].name must be a non-empty string.")
            outputs.append(OutputSpec(name=name, dtype=out.get("dtype", "float32")))
        inputs = []
        for i, inp in enumerate(io_raw["inputs"]):
            if not isinstance(inp, Mapping) or "name" not in inp:
                raise KeyError(f"components['{label}'].io.inputs[{i}] must define 'name'.")
            n = inp["name"]
            if not n or not isinstance(n, str):
                raise ValueError(f"components['{label}'].io.inputs[{i}].name must be a non-empty string.")
            inputs.append(InputSpec(name=n, dtype=inp.get("dtype", "float32")))
        io = ComponentIO(
            inputs=inputs,
            outputs=outputs,
            dynamic_axes=dict(io_raw.get("dynamic_axes", {})),
        )
        profile_raw = comp.get("tensorrt_profile") or {}
        if not isinstance(profile_raw, Mapping):
            raise TypeError(f"components['{label}'].tensorrt_profile must be a mapping.")
        tensorrt_profile = {}
        for input_name, shape_cfg in profile_raw.items():
            if not isinstance(shape_cfg, Mapping):
                raise TypeError(
                    f"components['{label}'].tensorrt_profile['{input_name}'] must be a mapping, got {type(shape_cfg).__name__}."
                )
            tensorrt_profile[input_name] = TensorRTProfileConfig.from_dict(shape_cfg)
        return ComponentCfg(
            name=str(comp["name"]),
            onnx_file=str(comp["onnx_file"]),
            engine_file=str(comp["engine_file"]),
            io=io,
            tensorrt_profile=tensorrt_profile,
        )


@dataclass(frozen=True)
class OnnxConfig:
    """ONNX export settings (shared across all components)."""

    opset_version: int = 16
    do_constant_folding: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    simplify: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> OnnxConfig:
        """Build OnnxConfig from deploy_cfg['onnx_config']."""
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError(f"onnx_config must be a mapping, got {type(raw).__name__}")
        return cls(
            opset_version=int(raw.get("opset_version", 16)),
            do_constant_folding=bool(raw.get("do_constant_folding", True)),
            export_params=bool(raw.get("export_params", True)),
            keep_initializers_as_inputs=bool(raw.get("keep_initializers_as_inputs", False)),
            simplify=bool(raw.get("simplify", False)),
        )


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
    def from_dict(cls, config_dict: Mapping[str, Any]) -> EvaluationConfig:
        backends_raw = config_dict.get("backends", None)
        if backends_raw is None:
            backends_raw = {}
        if not isinstance(backends_raw, Mapping):
            raise TypeError(f"evaluation.backends must be a mapping, got {type(backends_raw).__name__}")
        backends_frozen = {key: MappingProxyType(dict(value)) for key, value in backends_raw.items()}

        models_raw = config_dict.get("models", None)
        if models_raw is None:
            models_raw = {}
        if not isinstance(models_raw, Mapping):
            raise TypeError(f"evaluation.models must be a mapping, got {type(models_raw).__name__}")

        devices_raw = config_dict.get("devices", None)
        if devices_raw is None:
            devices_raw = {}
        if not isinstance(devices_raw, Mapping):
            raise TypeError(f"evaluation.devices must be a mapping, got {type(devices_raw).__name__}")

        return cls(
            enabled=config_dict.get("enabled", False),
            num_samples=config_dict.get("num_samples", 10),
            verbose=config_dict.get("verbose", False),
            backends=MappingProxyType(backends_frozen),
            models=MappingProxyType(dict(models_raw)),
            devices=MappingProxyType(dict(devices_raw)),
        )


@dataclass(frozen=True)
class VerificationConfig:
    """Typed configuration for verification settings."""

    enabled: bool = True
    num_verify_samples: int = 3
    tolerance: float = 0.1
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)
    scenarios: Mapping[ExportMode, Tuple[VerificationScenario, ...]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> VerificationConfig:
        scenarios_raw = config_dict.get("scenarios")
        if scenarios_raw is None:
            scenarios_raw = {}
        if not isinstance(scenarios_raw, Mapping):
            raise TypeError(f"verification.scenarios must be a mapping, got {type(scenarios_raw).__name__}")

        scenario_map: Dict[ExportMode, Tuple[VerificationScenario, ...]] = {}
        for mode_key, scenario_list in scenarios_raw.items():
            mode = ExportMode.from_value(mode_key)
            if scenario_list is None:
                scenario_list = []
            elif not isinstance(scenario_list, (list, tuple)):
                raise TypeError(
                    f"verification.scenarios.{mode_key} must be a list or tuple, got {type(scenario_list).__name__}"
                )
            scenario_entries = tuple(VerificationScenario.from_dict(entry) for entry in scenario_list)
            scenario_map[mode] = scenario_entries

        devices_raw = config_dict.get("devices")
        if devices_raw is None:
            devices_raw = {}
        if not isinstance(devices_raw, Mapping):
            raise TypeError(f"verification.devices must be a mapping, got {type(devices_raw).__name__}")

        return cls(
            enabled=config_dict.get("enabled", True),
            num_verify_samples=config_dict.get("num_verify_samples", 3),
            tolerance=config_dict.get("tolerance", 0.1),
            devices=MappingProxyType(dict(devices_raw)),
            scenarios=MappingProxyType(scenario_map),
        )

    def get_scenarios(self, mode: ExportMode) -> Tuple[VerificationScenario, ...]:
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
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationScenario:
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

    Attributes:
        checkpoint_path: Single source of truth for the PyTorch checkpoint path.
                        Used by both export (for ONNX conversion) and evaluation
                        (for PyTorch backend). Defined at top-level of deploy config.
    """

    def __init__(self, deploy_cfg: Config):
        """
        Initialize deployment configuration.

        Args:
            deploy_cfg: MMEngine Config object containing deployment settings
        """
        self.deploy_cfg = deploy_cfg
        self._validate_config()

        self._checkpoint_path = deploy_cfg.get("checkpoint_path")
        self._device_config = DeviceConfig.from_dict(deploy_cfg.get("devices", {}))
        self.components_cfg = ComponentsCfg.from_dict(deploy_cfg.get("components", {}))
        self._onnx_export_config = OnnxConfig.from_dict(deploy_cfg.get("onnx_config"))

        # Initialize config sections
        self.export_config = ExportConfig.from_dict(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig.from_dict(deploy_cfg.get("runtime_io", {}))
        self.tensorrt_config = TensorRTConfig.from_dict(deploy_cfg.get("tensorrt_config", {}))
        self._evaluation_config = EvaluationConfig.from_dict(deploy_cfg.get("evaluation", {}))
        self._verification_config = VerificationConfig.from_dict(deploy_cfg.get("verification", {}))

        self._validate_cuda_device()

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. " "Please update your config to include 'export' section."
            )
        try:
            ExportMode.from_value(self.deploy_cfg.get("export", {}).get("mode", ExportMode.BOTH))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        components = self.deploy_cfg.get("components", None)
        if components is None:
            raise ValueError("Missing 'components' section in deploy config.")
        ComponentsCfg.from_dict(components)

        tensorrt_config = self.deploy_cfg.get("tensorrt_config")
        if tensorrt_config is None:
            tensorrt_config = {}
        if not isinstance(tensorrt_config, Mapping):
            raise TypeError(f"tensorrt_config must be a mapping, got {type(tensorrt_config).__name__}")
        precision_policy = tensorrt_config.get("precision_policy", PrecisionPolicy.AUTO.value)
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    def _validate_cuda_device(self) -> None:
        """Validate CUDA device availability once at config stage."""
        if not self._needs_cuda_device():
            return

        cuda_device = self.devices.cuda
        device_idx = self.devices.cuda_device_index

        if cuda_device is None or device_idx is None:
            raise RuntimeError(
                "CUDA device is required (TensorRT export/verification/evaluation enabled) but no CUDA device was"
                " configured in deploy_cfg.devices."
            )

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
        if self.export_config.should_export_tensorrt:
            return True

        evaluation_cfg = self.evaluation_config
        backends_cfg = evaluation_cfg.backends
        tensorrt_backend = backends_cfg.get(Backend.TENSORRT.value, {})
        if tensorrt_backend and tensorrt_backend.get("enabled", False):
            return True

        verification_cfg = self.verification_config

        for scenario_list in verification_cfg.scenarios.values():
            for scenario in scenario_list:
                if Backend.TENSORRT in (scenario.ref_backend, scenario.test_backend):
                    return True

        return False

    @property
    def checkpoint_path(self) -> Optional[str]:
        """
        Get checkpoint path - single source of truth for PyTorch model.

        This path is used by:
        - Export pipeline: to load the PyTorch model for ONNX conversion
        - Evaluation: for PyTorch backend evaluation
        - Verification: when PyTorch is used as reference or test backend

        Returns:
            Path to the PyTorch checkpoint file, or None if not configured
        """
        return self._checkpoint_path

    @property
    def evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self._evaluation_config

    @property
    def onnx_config(self) -> OnnxConfig:
        """Get ONNX export configuration (typed)."""
        return self._onnx_export_config

    @property
    def verification_config(self) -> VerificationConfig:
        """Get verification configuration."""
        return self._verification_config

    @property
    def devices(self) -> DeviceConfig:
        """Get normalized device settings."""
        return self._device_config

    @property
    def evaluation_backends(self) -> Mapping[Any, Mapping[str, Any]]:
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

    def resolve_component(self, component: Optional[str] = None) -> str:
        """
        Resolve to a single component name. Central point for single-component usage.

        - If component is specified: validate and return it.
        - If component is None and exactly one component exists: return that name.
        - If component is None and multiple components: raise (caller must specify or use resolve_components).
        """
        names = tuple(self.components_cfg.component_names())
        if component is not None:
            self.components_cfg.get_component(component)
            return component
        if len(names) == 1:
            return names[0]
        if len(names) == 0:
            raise ValueError("No components defined in deploy config. Add at least one entry under 'components'.")
        raise ValueError(f"Multiple components {list(names)}. Please specify which component to use.")

    def resolve_components(self, component: Optional[str] = None) -> Tuple[str, ...]:
        """
        Resolve to a tuple of component names for iteration. Central point for export/build loops.

        - If component is specified: validate and return (component,).
        - If component is None: return all component names.
        """
        if component is not None:
            self.components_cfg.get_component(component)
            return (component,)
        return tuple(self.components_cfg.component_names())

    def get_onnx_settings(self, component: Optional[str] = None) -> ONNXExportConfig:
        """
        Get ONNX export settings for a component. I/O and save_file come from ComponentCfg only.

        Uses resolve_component(component): single component auto-resolved if only one defined.
        """
        name = self.resolve_component(component)
        comp = self.components_cfg.get_component(name)
        o = self._onnx_export_config
        input_names = tuple(inp.name for inp in comp.io.inputs)
        output_names = tuple(out.name for out in comp.io.outputs)
        if not input_names:
            input_names = ("input",)
        if not output_names:
            output_names = ("output",)
        settings_dict = {
            "opset_version": o.opset_version,
            "do_constant_folding": o.do_constant_folding,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": comp.io.dynamic_axes,
            "export_params": o.export_params,
            "keep_initializers_as_inputs": o.keep_initializers_as_inputs,
            "verbose": False,
            "save_file": comp.onnx_file,
            "batch_size": None,
            "simplify": o.simplify,
        }
        return ONNXExportConfig.from_mapping(settings_dict)

    def get_tensorrt_settings(self, component: Optional[str] = None) -> TensorRTExportConfig:
        """
        Get TensorRT export settings for a component. Profile and I/O come from ComponentCfg only.

        Uses resolve_component(component): single component auto-resolved if only one defined.
        """
        name = self.resolve_component(component)
        comp = self.components_cfg.get_component(name)
        if not comp.tensorrt_profile:
            return TensorRTExportConfig.from_mapping(
                {
                    "max_workspace_size": self.tensorrt_config.max_workspace_size,
                    "precision_policy": self.tensorrt_config.precision_policy,
                    "policy_flags": self.tensorrt_config.precision_flags,
                    "model_inputs": None,
                }
            )
        input_shapes = dict(comp.tensorrt_profile)
        model_inputs = (TensorRTModelInputConfig(input_shapes=MappingProxyType(input_shapes)),)
        return TensorRTExportConfig.from_mapping(
            {
                "max_workspace_size": self.tensorrt_config.max_workspace_size,
                "precision_policy": self.tensorrt_config.precision_policy,
                "policy_flags": self.tensorrt_config.precision_flags,
                "model_inputs": model_inputs,
            }
        )


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
