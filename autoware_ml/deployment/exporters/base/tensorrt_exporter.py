"""TensorRT model exporter."""

import logging
from typing import Any, Dict, Optional

import tensorrt as trt
import torch

from autoware_ml.deployment.exporters.base.base_exporter import BaseExporter


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
                continue  # Already handled
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
        """
        Configure input shapes for TensorRT optimization profile.

        Args:
            profile: TensorRT optimization profile
            sample_input: Sample input tensor
            network: TensorRT network definition (optional, used to get actual input names)
        """
        model_inputs = self.config.get("model_inputs", [])

        if model_inputs:
            # VIVID(calibration classifier)
            print("model inputs: ", model_inputs)
            input_shapes = model_inputs[0].get("input_shapes", {})
            for input_name, shapes in input_shapes.items():
                min_shape = shapes.get("min_shape")
                opt_shape = shapes.get("opt_shape")
                max_shape = shapes.get("max_shape")

                if min_shape is None:
                    if sample_input is None:
                        raise ValueError(f"min_shape missing for {input_name} and sample_input is not provided")
                    min_shape = list(sample_input.shape)

                if opt_shape is None:
                    if sample_input is None:
                        raise ValueError(f"opt_shape missing for {input_name} and sample_input is not provided")
                    opt_shape = list(sample_input.shape)

                if max_shape is None:
                    if sample_input is None:
                        raise ValueError(f"max_shape missing for {input_name} and sample_input is not provided")
                    max_shape = list(sample_input.shape)

                self.logger.info(f"Setting {input_name} shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        else:
            raise ValueError("model_inputs is not set in the config")

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        self.logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            self.logger.error(f"Parser error: {parser.get_error(error)}")
