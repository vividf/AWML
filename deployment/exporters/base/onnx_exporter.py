"""ONNX model exporter."""

import logging
import os
from typing import Any, Dict, Optional

import onnx
import onnxsim
import torch

from deployment.exporters.base.base_exporter import BaseExporter


class ONNXExporter(BaseExporter):
    """
    ONNX model exporter with enhanced features.

    Exports PyTorch models to ONNX format with:
    - Optional model wrapping for ONNX-specific output formats
    - Optional model simplification
    - Multi-file export support for complex models
    - Configuration override capability
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize ONNX exporter.

        Args:
            config: ONNX export configuration
            model_wrapper: Optional model wrapper class (e.g., YOLOXONNXWrapper)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)

    def export(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        output_path: str,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            config_override: Optional config overrides for this specific export

        Raises:
            RuntimeError: If export fails
        """
        # Apply model wrapper if configured
        model = self.prepare_model(model)
        model.eval()

        self.logger.info("Exporting model to ONNX format...")
        self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {self.config.get('opset_version', 16)}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=self.config.get("export_params", True),
                    keep_initializers_as_inputs=self.config.get("keep_initializers_as_inputs", False),
                    opset_version=self.config.get("opset_version", 16),
                    do_constant_folding=self.config.get("do_constant_folding", True),
                    input_names=self.config.get("input_names", ["input"]),
                    output_names=self.config.get("output_names", ["output"]),
                    dynamic_axes=self.config.get("dynamic_axes"),
                    verbose=self.config.get("verbose", False),
                )

            self.logger.info(f"ONNX export completed: {output_path}")

            # Optional model simplification
            if self.config.get("simplify", True):
                self._simplify_model(output_path)

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise RuntimeError("ONNX export failed") from e

    def _simplify_model(self, onnx_path: str) -> None:
        """
        Simplify ONNX model using onnxsim.

        Args:
            onnx_path: Path to ONNX model file
        """
        self.logger.info("Simplifying ONNX model...")
        try:
            model_simplified, success = onnxsim.simplify(onnx_path)
            if success:
                onnx.save(model_simplified, onnx_path)
                self.logger.info(f"ONNX model simplified successfully")
            else:
                self.logger.warning("ONNX model simplification failed")
        except Exception as e:
            self.logger.warning(f"ONNX simplification error: {e}")
