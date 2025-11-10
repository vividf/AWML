"""ONNX model exporter."""

import logging
from typing import Any, Dict

import onnx
import onnxsim
import torch

from .base_exporter import BaseExporter


class ONNXExporter(BaseExporter):
    """
    ONNX model exporter.

    Exports PyTorch models to ONNX format with optional simplification.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize ONNX exporter.

        Args:
            config: ONNX export configuration
            logger: Optional logger instance
        """
        super().__init__(config)
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
    ) -> bool:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model

        Returns:
            True if export succeeded
        """
        model.eval()

        self.logger.info("Exporting model to ONNX format...")
        self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {self.config.get('opset_version', 16)}")

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
                    verbose=False,
                )

            self.logger.info(f"ONNX export completed: {output_path}")

            # Optional model simplification
            if self.config.get("simplify", True):
                self._simplify_model(output_path)

            return True

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False

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

    def validate_export(self, output_path: str) -> bool:
        """
        Validate ONNX model.

        Args:
            output_path: Path to ONNX model file

        Returns:
            True if valid
        """
        if not super().validate_export(output_path):
            return False

        try:
            model = onnx.load(output_path)
            onnx.checker.check_model(model)
            self.logger.info("ONNX model validation passed")
            return True
        except Exception as e:
            self.logger.error(f"ONNX model validation failed: {e}")
            return False