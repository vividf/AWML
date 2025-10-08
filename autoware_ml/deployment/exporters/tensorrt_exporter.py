"""TensorRT model exporter."""

import logging
from typing import Any, Dict

import tensorrt as trt
import torch

from .base_exporter import BaseExporter


class TensorRTExporter(BaseExporter):
    """
    TensorRT model exporter.

    Converts ONNX models to TensorRT engine format with precision policy support.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize TensorRT exporter.

        Args:
            config: TensorRT export configuration
            logger: Optional logger instance
        """
        super().__init__(config)
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        model: torch.nn.Module,  # Not used for TensorRT, kept for interface compatibility
        sample_input: torch.Tensor,
        output_path: str,
        onnx_path: str = None,
    ) -> bool:
        """
        Export ONNX model to TensorRT engine.

        Args:
            model: Not used (TensorRT converts from ONNX)
            sample_input: Sample input for shape configuration
            output_path: Path to save TensorRT engine
            onnx_path: Path to source ONNX model

        Returns:
            True if export succeeded
        """
        if onnx_path is None:
            self.logger.error("onnx_path is required for TensorRT export")
            return False

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

        # Setup optimization profile
        profile = builder.create_optimization_profile()
        self._configure_input_shapes(profile, sample_input)
        builder_config.add_optimization_profile(profile)

        # Parse ONNX model
        parser = trt.OnnxParser(network, trt_logger)

        try:
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    self._log_parser_errors(parser)
                    return False
                self.logger.info("Successfully parsed ONNX file")

            # Build engine
            self.logger.info("Building TensorRT engine (this may take a while)...")
            serialized_engine = builder.build_serialized_network(network, builder_config)

            if serialized_engine is None:
                self.logger.error("Failed to build TensorRT engine")
                return False

            # Save engine
            with open(output_path, "wb") as f:
                f.write(serialized_engine)

            self.logger.info(f"TensorRT engine saved to {output_path}")
            self.logger.info(f"Engine max workspace size: {max_workspace_size / (1024**3):.2f} GB")

            return True

        except Exception as e:
            self.logger.error(f"TensorRT export failed: {e}")
            return False

    def _configure_input_shapes(
        self,
        profile: trt.IOptimizationProfile,
        sample_input: torch.Tensor,
    ) -> None:
        """
        Configure input shapes for TensorRT optimization profile.

        Args:
            profile: TensorRT optimization profile
            sample_input: Sample input tensor
        """
        model_inputs = self.config.get("model_inputs", [])

        if model_inputs:
            input_shapes = model_inputs[0].get("input_shapes", {})
            for input_name, shapes in input_shapes.items():
                min_shape = shapes.get("min_shape", list(sample_input.shape))
                opt_shape = shapes.get("opt_shape", list(sample_input.shape))
                max_shape = shapes.get("max_shape", list(sample_input.shape))

                self.logger.info(f"Setting input shapes - min: {min_shape}, " f"opt: {opt_shape}, max: {max_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        else:
            # Default shapes based on input tensor
            input_shape = list(sample_input.shape)
            self.logger.info(f"Using default input shape: {input_shape}")
            profile.set_shape("input", input_shape, input_shape, input_shape)

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        self.logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            self.logger.error(f"Parser error: {parser.get_error(error)}")
