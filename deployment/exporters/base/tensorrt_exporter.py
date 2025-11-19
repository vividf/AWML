"""TensorRT model exporter."""

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import tensorrt as trt
import torch

from deployment.core.base_config import TensorRTModelInputConfig, TensorRTProfileConfig
from deployment.exporters.base.base_exporter import BaseExporter


class TensorRTExporter(BaseExporter):
    """
    TensorRT model exporter.

    Converts ONNX models to TensorRT engine format with precision policy support.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize TensorRT exporter.

        Args:
            config: TensorRT export configuration
            model_wrapper: Optional model wrapper class (usually not needed for TensorRT)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        model: torch.nn.Module,  # Not used for TensorRT, kept for interface compatibility
        sample_input: Any,
        output_path: str,
        onnx_path: str = None,
    ) -> None:
        """
        Export ONNX model to TensorRT engine.

        Args:
            model: Not used (TensorRT converts from ONNX)
            sample_input: Sample input for shape configuration
            output_path: Path to save TensorRT engine
            onnx_path: Path to source ONNX model

        Raises:
            RuntimeError: If export fails
            ValueError: If ONNX path is missing
        """
        if onnx_path is None:
            raise ValueError("onnx_path is required for TensorRT export")

        precision_policy = self.config.get("precision_policy", "auto")
        policy_flags = self.config.get("policy_flags", {})

        self.logger.info(f"Building TensorRT engine with precision policy: {precision_policy}")
        self.logger.info(f"  ONNX source: {onnx_path}")
        self.logger.info(f"  Engine output: {output_path}")

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")

        builder = trt.Builder(trt_logger)
        builder_config = builder.create_builder_config()

        max_workspace_size = self.config.get("max_workspace_size", 1 << 30)
        builder_config.set_memory_pool_limit(pool=trt.MemoryPoolType.WORKSPACE, pool_size=max_workspace_size)

        # Create network with appropriate flags
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # Handle strongly typed flag (network creation flag)
        if policy_flags.get("STRONGLY_TYPED"):
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            self.logger.info("Using strongly typed TensorRT network creation")

        network = builder.create_network(flags)

        # Apply precision flags to builder config
        for flag_name, enabled in policy_flags.items():
            if flag_name == "STRONGLY_TYPED":
                continue
            if enabled and hasattr(trt.BuilderFlag, flag_name):
                builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))
                self.logger.info(f"BuilderFlag.{flag_name} enabled")

        # Parse ONNX model first to get network structure
        parser = trt.OnnxParser(network, trt_logger)

        try:
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    self._log_parser_errors(parser)
                    raise RuntimeError("TensorRT export failed: unable to parse ONNX file")
                self.logger.info("Successfully parsed ONNX file")

            # Setup optimization profile after parsing ONNX to get actual input names
            profile = builder.create_optimization_profile()
            self._configure_input_shapes(profile, sample_input, network)
            builder_config.add_optimization_profile(profile)

            # Build engine
            self.logger.info("Building TensorRT engine (this may take a while)...")
            serialized_engine = builder.build_serialized_network(network, builder_config)

            if serialized_engine is None:
                self.logger.error("Failed to build TensorRT engine")
                raise RuntimeError("TensorRT export failed: builder returned None")

            # Save engine
            with open(output_path, "wb") as f:
                f.write(serialized_engine)

            self.logger.info(f"TensorRT engine saved to {output_path}")
            self.logger.info(f"Engine max workspace size: {max_workspace_size / (1024**3):.2f} GB")

        except Exception as e:
            self.logger.error(f"TensorRT export failed: {e}")
            raise RuntimeError("TensorRT export failed") from e

    def _configure_input_shapes(
        self,
        profile: trt.IOptimizationProfile,
        sample_input: Any,
        network: trt.INetworkDefinition = None,
    ) -> None:
        """Configure input shapes for TensorRT optimization profile."""
        model_inputs_cfg = self.config.get("model_inputs")

        if not model_inputs_cfg:
            raise ValueError("model_inputs is not set in the config")

        if isinstance(model_inputs_cfg, TensorRTModelInputConfig):
            input_entries = (model_inputs_cfg,)
        elif isinstance(model_inputs_cfg, (list, tuple)):
            input_entries = tuple(model_inputs_cfg)
        else:
            input_entries = (model_inputs_cfg,)

        first_entry = input_entries[0]
        input_shapes = self._extract_input_shapes(first_entry)

        if not input_shapes:
            raise ValueError("TensorRT model_inputs[0] missing 'input_shapes' definitions")

        for input_name, profile_cfg in input_shapes.items():
            min_shape, opt_shape, max_shape = self._resolve_profile_shapes(profile_cfg, sample_input, input_name)
            self.logger.info(f"Setting {input_name} shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        self.logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            self.logger.error(f"Parser error: {parser.get_error(error)}")

    def _extract_input_shapes(self, entry: Any) -> Mapping[str, Any]:
        if isinstance(entry, TensorRTModelInputConfig):
            return entry.input_shapes
        if isinstance(entry, Mapping):
            return entry.get("input_shapes", {}) or {}
        raise TypeError(f"Unsupported TensorRT model input entry: {type(entry)}")

    def _resolve_profile_shapes(
        self,
        profile_cfg: Any,
        sample_input: Any,
        input_name: str,
    ) -> Sequence[Sequence[int]]:
        if isinstance(profile_cfg, TensorRTProfileConfig):
            min_shape = self._shape_to_list(profile_cfg.min_shape)
            opt_shape = self._shape_to_list(profile_cfg.opt_shape)
            max_shape = self._shape_to_list(profile_cfg.max_shape)
        elif isinstance(profile_cfg, Mapping):
            min_shape = self._shape_to_list(profile_cfg.get("min_shape"))
            opt_shape = self._shape_to_list(profile_cfg.get("opt_shape"))
            max_shape = self._shape_to_list(profile_cfg.get("max_shape"))
        else:
            raise TypeError(f"Unsupported TensorRT profile type for input '{input_name}': {type(profile_cfg)}")

        return (
            self._ensure_shape(min_shape, sample_input, input_name, "min"),
            self._ensure_shape(opt_shape, sample_input, input_name, "opt"),
            self._ensure_shape(max_shape, sample_input, input_name, "max"),
        )

    @staticmethod
    def _shape_to_list(shape: Optional[Sequence[int]]) -> Optional[Sequence[int]]:
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    def _ensure_shape(
        self,
        shape: Optional[Sequence[int]],
        sample_input: Any,
        input_name: str,
        bucket: str,
    ) -> Sequence[int]:
        if shape:
            return list(shape)
        if sample_input is None or not hasattr(sample_input, "shape"):
            raise ValueError(f"{bucket}_shape missing for {input_name} and sample_input is not provided")
        inferred = list(sample_input.shape)
        self.logger.debug("Falling back to sample_input.shape=%s for %s:%s", inferred, input_name, bucket)
        return inferred
