"""
Typed schema for deployment config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from deployment.config.enums import (
    DEFAULT_WORKSPACE_SIZE,
    Backend,
    ExportMode,
    PrecisionPolicy,
)
from deployment.export.exporters.configs import TensorRTProfileConfig
from deployment.primitives.device import DeviceSpec


def _empty_mapping() -> Mapping[Any, Any]:
    """Return an immutable empty mapping."""
    return MappingProxyType({})


# -----------------------------------------------------------------------------
# Export / Device / Runtime
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for model export settings."""

    mode: ExportMode  # required: caller must explicitly pick what to export
    work_dir: str = "work_dirs"
    onnx_path: Optional[str] = None
    sample_idx: int = 0

    @classmethod
    def from_dict(cls, config_dict: Optional[Mapping[str, Any]]) -> ExportConfig:
        """Build ExportConfig from deploy_cfg['export']. Required: a dict with a valid `mode`."""
        if config_dict is None:
            raise ValueError("Missing 'export' section in deploy config.")
        if not isinstance(config_dict, Mapping):
            raise TypeError(f"export must be a dict, got {type(config_dict).__name__}")
        if "mode" not in config_dict:
            valid = [m.value for m in ExportMode]
            raise ValueError(f"export.mode is required; must be one of {valid}.")
        return cls(
            mode=ExportMode.from_value(config_dict["mode"]),
            work_dir=config_dict.get("work_dir", cls.work_dir),
            onnx_path=config_dict.get("onnx_path"),
            sample_idx=config_dict.get("sample_idx", cls.sample_idx),
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
    """Parsed device settings shared across deployment stages."""

    cpu: DeviceSpec = field(default_factory=lambda: DeviceSpec.from_value("cpu"))
    cuda: Optional[DeviceSpec] = field(default_factory=lambda: DeviceSpec.from_value("cuda:0"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", self._parse_cpu_device(self.cpu))
        object.__setattr__(self, "cuda", self._parse_cuda_device(self.cuda))

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> DeviceConfig:
        """Create DeviceConfig from dict."""
        return cls(cpu=config_dict.get("cpu", "cpu"), cuda=config_dict.get("cuda", "cuda:0"))

    @staticmethod
    def _parse_cpu_device(device: Any) -> DeviceSpec:
        """Parse CPU device input into DeviceSpec."""
        device_spec = DeviceSpec.from_value(device if device is not None else "cpu")
        if device_spec.is_cuda:
            raise ValueError("CPU device cannot be a CUDA device")
        return device_spec

    @staticmethod
    def _parse_cuda_device(device: Any) -> Optional[DeviceSpec]:
        """Parse CUDA device input into DeviceSpec."""
        if device is None:
            return None
        device_spec = DeviceSpec.from_value(device)
        if not device_spec.is_cuda:
            raise ValueError(f"Invalid CUDA device '{device}'.")
        return device_spec

    @property
    def cuda_device_index(self) -> Optional[int]:
        """Return CUDA device index as integer (if configured)."""
        if self.cuda is None:
            return None
        return self.cuda.index


@dataclass(frozen=True)
class OnnxConfig:
    """ONNX export settings (shared across all components)."""

    opset_version: int = 17
    do_constant_folding: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    simplify: bool = False
    # Post-export: annotate Q/DQ scale/zero_point into node names + promote them to named
    # initializers (make_qdq_readable) so INT8 scales are visible in the exported ONNX.
    visualize_qdq_values: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> OnnxConfig:
        """Build OnnxConfig from deploy_cfg['onnx_config']."""
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError(f"onnx_config must be a dict, got {type(raw).__name__}")
        return cls(
            opset_version=int(raw.get("opset_version", cls.opset_version)),
            do_constant_folding=bool(raw.get("do_constant_folding", cls.do_constant_folding)),
            export_params=bool(raw.get("export_params", cls.export_params)),
            keep_initializers_as_inputs=bool(raw.get("keep_initializers_as_inputs", cls.keep_initializers_as_inputs)),
            simplify=bool(raw.get("simplify", cls.simplify)),
            visualize_qdq_values=bool(raw.get("visualize_qdq_values", cls.visualize_qdq_values)),
        )


@dataclass(frozen=True)
class TensorRTConfig:
    """
    Configuration for TensorRT backend-specific settings.

    Uses config structure:
        tensorrt_config = dict(precision_policy="auto", max_workspace_size=1<<30,
                               plugin_libraries=["/opt/plugins/libcustom.so"])

    TensorRT profiles are defined in components.*.tensorrt_profile.

    Note:
        The deploy config key for this section is **`tensorrt_config`**.

    ``plugin_libraries`` lists custom TensorRT plugin ``.so`` paths to ``dlopen``
    before engine build/deserialize (e.g. the BEVFusion spconv INT8 plugin). Empty
    by default, so projects that need no custom plugins (e.g. CenterPoint) are unaffected.
    """

    precision_policy: PrecisionPolicy = PrecisionPolicy.AUTO
    max_workspace_size: int = DEFAULT_WORKSPACE_SIZE
    plugin_libraries: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> TensorRTConfig:
        return cls(
            precision_policy=PrecisionPolicy.from_value(config_dict.get("precision_policy")),
            max_workspace_size=config_dict.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
            plugin_libraries=tuple(config_dict.get("plugin_libraries") or ()),
        )


# -----------------------------------------------------------------------------
# Quantization (deploy_cfg["quantization"]) — optional, default-disabled
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QATConfig:
    """Typed view of the optional ``quantization["qat"]`` sub-block (spec_qat.md §D2/WP1).

    Present only when ``quantization.mode == "qat"`` — the block records the training half of a QAT
    run so one deploy config reproduces it (placement already lives in ``keep_fp16`` /
    ``disable_recipes``, which the QAT hook consumes unchanged). ``epochs`` and ``lr`` are required:
    there is no silent recipe default — the reference values (~10% of original training epochs,
    lr=1e-4 per CUDA-CenterPoint / modelopt) belong in the config, visibly.

    ``train_cfg`` / ``checkpoint`` may be omitted and supplied on the producer CLI instead;
    ``calibrate_samples`` defaults to the CUDA-CenterPoint reference (400 @ bs=1).
    """

    epochs: int
    lr: float
    train_cfg: Optional[str] = None
    checkpoint: Optional[str] = None
    calibrate_samples: int = 400
    calib_cache: Optional[str] = None
    work_dir: Optional[str] = None

    # Typo guard — same rationale as QuantizationConfig.KNOWN_KEYS.
    KNOWN_KEYS = frozenset(
        {
            "epochs",
            "lr",
            "train_cfg",
            "checkpoint",
            "calibrate_samples",
            "calib_cache",
            "work_dir",
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QATConfig:
        """Build QATConfig from ``quantization["qat"]``.

        Raises:
            TypeError: If ``raw`` is not a mapping.
            ValueError: On unknown keys, or when ``epochs`` / ``lr`` are missing.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization.qat must be a dict, got {type(raw).__name__}")
        unknown = set(raw) - cls.KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"Unknown quantization.qat key(s): {sorted(unknown)}. Valid keys: {sorted(cls.KNOWN_KEYS)}."
            )
        missing = {k for k in ("epochs", "lr") if raw.get(k) is None}
        if missing:
            raise ValueError(
                f"quantization.qat requires {sorted(missing)} — no silent recipe default. "
                "Reference values: epochs ≈ 10% of original training, lr=1e-4 (spec_qat.md §2)."
            )
        return cls(
            epochs=int(raw["epochs"]),
            lr=float(raw["lr"]),
            train_cfg=raw.get("train_cfg"),
            checkpoint=raw.get("checkpoint"),
            calibrate_samples=int(raw.get("calibrate_samples", 400)),
            calib_cache=raw.get("calib_cache"),
            work_dir=raw.get("work_dir"),
        )


@dataclass(frozen=True)
class PTQConfig:
    """Typed view of the optional ``quantization["ptq"]`` sub-block.

    The producer half of a PTQ run, recorded in the deploy config so one file reproduces the
    checkpoint at ``checkpoint_path`` — the exact sibling of :class:`QATConfig` (placement already
    lives in ``keep_fp16`` / ``disable_recipes``, shared by producer and deploy loader).
    ``calibrate_samples`` is required: it is *the* calibration-recipe knob and gets no silent
    default — same rationale as ``QATConfig.epochs`` / ``lr``.

    The model config is NOT a block key: PTQ calibrates against the same model config the artifact
    deploys with, so it lives once at the deploy config's top level (``model_cfg``, next to
    ``checkpoint_path``). ``checkpoint`` (the FP input) may be omitted and supplied on the producer
    CLI instead; CLI flags always override block values (``resolve_ptq_settings``).
    """

    calibrate_samples: int
    checkpoint: Optional[str] = None
    batch_size: int = 1
    calib_seed: Optional[int] = None
    calib_shuffle: bool = False

    # Typo guard — same rationale as QuantizationConfig.KNOWN_KEYS.
    KNOWN_KEYS = frozenset(
        {
            "calibrate_samples",
            "checkpoint",
            "batch_size",
            "calib_seed",
            "calib_shuffle",
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PTQConfig:
        """Build PTQConfig from ``quantization["ptq"]``.

        Raises:
            TypeError: If ``raw`` is not a mapping.
            ValueError: On unknown keys, or when ``calibrate_samples`` is missing.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization.ptq must be a dict, got {type(raw).__name__}")
        unknown = set(raw) - cls.KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"Unknown quantization.ptq key(s): {sorted(unknown)}. Valid keys: {sorted(cls.KNOWN_KEYS)}."
            )
        if raw.get("calibrate_samples") is None:
            raise ValueError(
                "quantization.ptq requires calibrate_samples — no silent recipe default. "
                "It is the calibration-recipe knob (CenterPoint release reference: 400 @ bs=1)."
            )
        return cls(
            calibrate_samples=int(raw["calibrate_samples"]),
            checkpoint=raw.get("checkpoint"),
            batch_size=int(raw.get("batch_size", 1)),
            calib_seed=raw.get("calib_seed"),
            calib_shuffle=bool(raw.get("calib_shuffle", False)),
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Typed view of the deploy-config ``quantization`` section.

    The single parse of the ``quantization`` dict: ``BaseDeploymentConfig`` builds this once and the
    runners pass it straight to the model loaders — nothing downstream re-parses the raw dict.
    Defaults are chosen so an absent section yields a fully-disabled config (``enabled=False``) —
    existing non-quantized deploy configs are unaffected.

    Precision placement is declarative (modelopt-style): everything the plan reaches is
    ``default_precision`` (INT8), and ``keep_fp16`` lists glob patterns (subtree match) to leave in
    FP16 — the modern replacement for the old ~13 ``quant_*`` / ``skip_*`` / ``sensitive_layers``
    booleans. Architecture recipes are always-on and class-gated; ``disable_recipes`` opts a config
    out of one. See spec.md §3.
    """

    enabled: bool = False
    mode: str = "ptq"  # "ptq" | "qat"
    fuse_bn: bool = True
    ptq_checkpoint: bool = False
    # Precision placement: INT8 by default, opt out by glob (subtree match). ``keep_fp16`` absorbs the
    # old quant_backbone/neck/head/voxel_encoder toggles, skip_backbone_*/skip_vovnet_stages, and
    # sensitive_layers. A pattern keeps the matched module and all its descendants in FP16.
    default_precision: str = "int8"
    keep_fp16: Tuple[str, ...] = ()
    # Architecture recipes (residual-add / eSE / maxpool) are attached always, gated by module class.
    # List a recipe name here to opt this config out. Recognized: "add", "ese", "maxpool".
    disable_recipes: Tuple[str, ...] = ()
    calib_cache_path: Optional[str] = None
    # Producer blocks — each present only under its matching mode (a block under the wrong mode is
    # a config lie and from_dict raises; an explicit ``ptq=None`` / ``qat=None`` is fine, so a
    # mode="qat" child config can drop a ptq block inherited via _base_).
    # Deploy-load behavior NEVER branches on these: the loader rebuilds the identical tree for PTQ
    # and QAT checkpoints alike (spec_qat.md §D6).
    ptq: Optional[PTQConfig] = None
    qat: Optional[QATConfig] = None

    # The full key set of the ``quantization`` deploy-config section. ``from_dict`` rejects anything
    # else: a misspelled key (``keep_fp16s=...``) would otherwise silently degrade to "quantize
    # everything INT8" and be visible only as a Docker-eval mAP drop. Config-key sibling of the
    # zero-match warning in ``expand_keep_fp16`` (spec.md §3.4).
    KNOWN_KEYS = frozenset(
        {
            "enabled",
            "mode",
            "fuse_bn",
            "ptq_checkpoint",
            "default_precision",
            "keep_fp16",
            "disable_recipes",
            "calib_cache_path",
            "ptq",
            "qat",
        }
    )

    @staticmethod
    def _str_tuple(value: Any) -> Tuple[str, ...]:
        return tuple(str(v) for v in value) if value else ()

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> QuantizationConfig:
        """Build QuantizationConfig from deploy_cfg['quantization']; empty/None → disabled.

        Raises:
            ValueError: If the dict contains keys outside :attr:`KNOWN_KEYS` (typo guard).
        """
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization must be a dict, got {type(raw).__name__}")
        unknown = set(raw) - cls.KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"Unknown quantization key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(cls.KNOWN_KEYS)}. "
                "A misspelled key would silently change what gets quantized."
            )
        mode = str(raw.get("mode", "ptq"))
        qat_raw = raw.get("qat")
        if qat_raw is not None and mode != "qat":
            raise ValueError(
                f'quantization has a "qat" block but mode="{mode}" — set mode="qat" or drop the block. '
                "A qat block under a non-qat mode is a config lie."
            )
        ptq_raw = raw.get("ptq")
        if ptq_raw is not None and mode != "ptq":
            raise ValueError(
                f'quantization has a "ptq" block but mode="{mode}" — set mode="ptq" or drop the block '
                "(a mode='qat' config inheriting one via _base_ can set ptq=None). "
                "A ptq block under a non-ptq mode is a config lie."
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=mode,
            fuse_bn=bool(raw.get("fuse_bn", True)),
            ptq_checkpoint=bool(raw.get("ptq_checkpoint", False)),
            default_precision=str(raw.get("default_precision", "int8")),
            keep_fp16=cls._str_tuple(raw.get("keep_fp16")),
            disable_recipes=cls._str_tuple(raw.get("disable_recipes")),
            calib_cache_path=raw.get("calib_cache_path"),
            ptq=PTQConfig.from_dict(ptq_raw) if ptq_raw is not None else None,
            qat=QATConfig.from_dict(qat_raw) if qat_raw is not None else None,
        )

    def with_overrides(self, **overrides: Any) -> QuantizationConfig:
        """Return a copy with the given fields replaced (e.g. a CLI flag overriding a deploy-cfg value)."""
        from dataclasses import replace

        return replace(self, **overrides)


def load_quantization_config(deploy_cfg_path: str) -> Tuple[QuantizationConfig, Optional[str], Optional[str]]:
    """Load ``quantization`` (and the top-level ``checkpoint_path`` / ``model_cfg``) from a deploy config.

    Centralizes the MMEngine ``Config`` access quirks (prefer ``.get`` over ``getattr``). Used by the
    PTQ / QAT producer CLIs; the deploy loaders instead receive the ``quantization`` dict and call
    :meth:`QuantizationConfig.from_dict` directly.

    Returns:
        ``(config, checkpoint_path, model_cfg)`` — the latter two are the deploy config's top-level
        artifact-manifest keys (producer output default / canonical model pairing).
    """
    from mmengine.config import Config

    deploy_cfg = Config.fromfile(deploy_cfg_path)

    def _cfg_get(key: str, default: Any = None) -> Any:
        value = deploy_cfg.get(key, None)
        if value is None:
            value = getattr(deploy_cfg, key, None)
        return default if value is None else value

    raw = _cfg_get("quantization")
    mapping = {k: raw[k] for k in raw} if raw is not None else {}

    return QuantizationConfig.from_dict(mapping), _cfg_get("checkpoint_path"), _cfg_get("model_cfg")


# -----------------------------------------------------------------------------
# Component config (deploy_cfg["components"])
# -----------------------------------------------------------------------------


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
    """Configuration for one deployable component.

    The component identifier is the key in deploy_cfg['components']; ``name`` is always set
    from that key.
    """

    name: str
    onnx_file: str
    engine_file: str
    io: ComponentIO
    tensorrt_profile: Dict[str, TensorRTProfileConfig]


@dataclass(frozen=True)
class ComponentsConfig:
    """Component configuration: mapping of component id -> ComponentCfg.

    The dict key is the component identifier (e.g. "model", "pts_voxel_encoder", "pts_backbone_neck_head").
    """

    _components: Mapping[str, ComponentCfg]

    def get_component(self, component_name: str) -> ComponentCfg:
        """Get component config by name. Raises KeyError if not found."""
        if component_name not in self._components:
            raise KeyError(f"Unknown component: {component_name}. Available: {list(self._components.keys())}")
        return self._components[component_name]

    def get_artifact_filename(self, component_name: str, file_key: str) -> Optional[str]:
        """Return artifact filename for path resolution (onnx_file or engine_file)."""
        return getattr(self.get_component(component_name), file_key)

    def component_names(self) -> Iterable[str]:
        """Iterate over component names."""
        return self._components.keys()

    def items(self) -> Iterable[Tuple[str, ComponentCfg]]:
        """Iterate (name, ComponentCfg) pairs."""
        return self._components.items()

    def with_component(self, component: ComponentCfg) -> ComponentsConfig:
        """Return a new ``ComponentsConfig`` with ``component`` added (replacing any of the same name).

        Lets callers derive a layout (e.g. BEVFusion's merged ``bevfusion_merged``) from already
        typed components without round-tripping the whole config back through raw dicts.
        """
        return ComponentsConfig(
            _components=MappingProxyType({**self._components, component.name: component}),
        )

    @staticmethod
    def _validate_dynamic_axes(raw: Any) -> Dict[str, Dict[int, str]]:
        """Validate dynamic_axes schema without coercing types."""

        def _require_type(value: Any, expected: type, message: str) -> None:
            if not isinstance(value, expected):
                raise TypeError(f"{message}, got {type(value).__name__}")

        if raw is None:
            return {}
        _require_type(raw, Mapping, "dynamic_axes must be a dict")

        result: Dict[str, Dict[int, str]] = {}
        for name, axes in raw.items():
            _require_type(name, str, "dynamic_axes key must be str")
            _require_type(axes, Mapping, f"dynamic_axes['{name}'] must be a dict")

            typed_axes: Dict[int, str] = {}
            for axis_idx, axis_name in axes.items():
                _require_type(axis_idx, int, f"dynamic_axes['{name}'] axis index must be int")
                _require_type(axis_name, str, f"dynamic_axes['{name}'][{axis_idx}] axis name must be str")
                typed_axes[axis_idx] = axis_name
            result[name] = typed_axes
        return result

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> ComponentsConfig:
        """Build ComponentsConfig from deploy_cfg['components']. Required: a non-empty dict."""
        if raw is None:
            raise ValueError("Missing 'components' section in deploy config.")
        if not isinstance(raw, Mapping):
            raise TypeError(f"components must be a dict, got {type(raw).__name__}")
        if not raw:
            raise ValueError("deploy config 'components' must define at least one component.")
        parsed = {}
        for component_name, comp_raw in raw.items():
            parsed[component_name] = cls._parse_component(comp_raw, component_name)
        return cls(_components=MappingProxyType(parsed))

    @staticmethod
    def _parse_io_specs(
        raw_specs: Iterable[Any],
        component_name: str,
        io_kind: str,
        spec_cls: type,
    ) -> List[Any]:
        """Parse an ``io.inputs``/``io.outputs`` list into typed name/dtype specs.

        Inputs and outputs are validated identically (both are name/dtype records);
        ``io_kind`` ('inputs' or 'outputs') only shapes the error messages, and
        ``spec_cls`` selects `InputSpec` or `OutputSpec`.
        """
        specs: List[Any] = []
        for i, raw in enumerate(raw_specs):
            if not isinstance(raw, Mapping) or "name" not in raw:
                raise KeyError(f"components['{component_name}'].io.{io_kind}[{i}] must define 'name'.")
            name = raw["name"]
            if not name or not isinstance(name, str):
                raise ValueError(f"components['{component_name}'].io.{io_kind}[{i}].name must be a non-empty string.")
            specs.append(spec_cls(name=name, dtype=raw.get("dtype", "float32")))
        return specs

    @classmethod
    def _parse_component(cls, comp_raw: Any, component_name: str) -> ComponentCfg:
        if not isinstance(comp_raw, Mapping):
            raise TypeError(f"components['{component_name}'] must be a dict, got {type(comp_raw).__name__}")
        for field_name in ("onnx_file", "engine_file", "io"):
            if field_name not in comp_raw:
                raise KeyError(f"components['{component_name}'] must define '{field_name}'.")
        component_id = component_name
        io_raw = comp_raw["io"]
        if not isinstance(io_raw, Mapping):
            raise TypeError(f"components['{component_name}'].io must be a dict, got {type(io_raw).__name__}")
        if "outputs" not in io_raw or not io_raw["outputs"]:
            raise KeyError(f"components['{component_name}'].io.outputs must be a non-empty list.")
        if "inputs" not in io_raw or not io_raw["inputs"]:
            raise KeyError(f"components['{component_name}'].io.inputs must be a non-empty list.")
        outputs = cls._parse_io_specs(io_raw["outputs"], component_name, "outputs", OutputSpec)
        inputs = cls._parse_io_specs(io_raw["inputs"], component_name, "inputs", InputSpec)
        dynamic_axes = cls._validate_dynamic_axes(io_raw.get("dynamic_axes") or {})
        io = ComponentIO(
            inputs=inputs,
            outputs=outputs,
            dynamic_axes=dynamic_axes,
        )
        profile_raw = comp_raw.get("tensorrt_profile") or {}
        if not isinstance(profile_raw, Mapping):
            raise TypeError(f"components['{component_name}'].tensorrt_profile must be a dict.")
        tensorrt_profile = {}
        for input_name, shape_cfg in profile_raw.items():
            if not isinstance(shape_cfg, Mapping):
                raise TypeError(
                    f"components['{component_name}'].tensorrt_profile['{input_name}'] must be a dict, got {type(shape_cfg).__name__}."
                )
            tensorrt_profile[input_name] = TensorRTProfileConfig.from_dict(shape_cfg)
        return ComponentCfg(
            name=component_id,
            onnx_file=str(comp_raw["onnx_file"]),
            engine_file=str(comp_raw["engine_file"]),
            io=io,
            tensorrt_profile=tensorrt_profile,
        )


# -----------------------------------------------------------------------------
# Evaluation & Verification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendEvalConfig:
    """Typed per-backend evaluation settings (one entry under ``evaluation.backends``).

    Attributes:
        enabled: Whether this backend participates in evaluation.
        device: Device override (e.g. ``"cuda:0"``); ``None`` uses the backend default.
        model_dir: ONNX artifact directory (consulted for the ONNX backend).
        engine_dir: TensorRT engine directory (consulted for the TensorRT backend).
    """

    enabled: bool = False
    device: Optional[str] = None
    model_dir: Optional[str] = None
    engine_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BackendEvalConfig:
        if not isinstance(raw, Mapping):
            raise TypeError(f"evaluation.backends entry must be a dict, got {type(raw).__name__}")
        return cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            device=raw.get("device"),
            model_dir=raw.get("model_dir"),
            engine_dir=raw.get("engine_dir"),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed configuration for evaluation settings."""

    enabled: bool = False
    num_samples: int = 10
    num_warmup: int = 0
    verbose: bool = False
    backends: Mapping[str, BackendEvalConfig] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> EvaluationConfig:
        backends_raw = config_dict.get("backends") or {}
        if not isinstance(backends_raw, Mapping):
            raise TypeError(f"evaluation.backends must be a dict, got {type(backends_raw).__name__}")
        # Canonicalize keys to the backend's string value so lookups by Backend.value succeed.
        backends = {
            Backend.from_value(key).value: BackendEvalConfig.from_dict(value) for key, value in backends_raw.items()
        }

        return cls(
            enabled=config_dict.get("enabled", cls.enabled),
            num_samples=config_dict.get("num_samples", cls.num_samples),
            num_warmup=config_dict.get("num_warmup", cls.num_warmup),
            verbose=config_dict.get("verbose", cls.verbose),
            backends=MappingProxyType(backends),
        )


@dataclass(frozen=True)
class VerificationScenario:
    """Immutable verification scenario specification."""

    ref_backend: Backend
    ref_device: DeviceSpec
    test_backend: Backend
    test_device: DeviceSpec

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationScenario:
        missing_keys = {"ref_backend", "ref_device", "test_backend", "test_device"} - data.keys()
        if missing_keys:
            raise ValueError(f"Verification scenario missing keys: {missing_keys}")

        return cls(
            ref_backend=Backend.from_value(data["ref_backend"]),
            ref_device=DeviceSpec.from_value(data["ref_device"]),
            test_backend=Backend.from_value(data["test_backend"]),
            test_device=DeviceSpec.from_value(data["test_device"]),
        )


@dataclass(frozen=True)
class VerificationConfig:
    """Typed configuration for verification settings."""

    enabled: bool = True
    num_verify_samples: int = 3
    tolerance: float = 0.1
    scenarios: Mapping[ExportMode, Tuple[VerificationScenario, ...]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> VerificationConfig:
        scenarios_raw = config_dict.get("scenarios")
        if scenarios_raw is None:
            scenarios_raw = {}
        if not isinstance(scenarios_raw, Mapping):
            raise TypeError(f"verification.scenarios must be a dict, got {type(scenarios_raw).__name__}")

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

        return cls(
            enabled=config_dict.get("enabled", cls.enabled),
            num_verify_samples=config_dict.get("num_verify_samples", cls.num_verify_samples),
            tolerance=config_dict.get("tolerance", cls.tolerance),
            scenarios=MappingProxyType(scenario_map),
        )

    def get_scenarios(self, mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """Return scenarios for a specific export mode."""
        return self.scenarios.get(mode, ())
