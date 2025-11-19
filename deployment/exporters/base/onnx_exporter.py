"""ONNX model exporter."""

import logging
import os
from typing import Any, Dict, List, Optional

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
        *,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        simplify: Optional[bool] = None,
        opset_version: Optional[int] = None,
        export_params: Optional[bool] = None,
        keep_initializers_as_inputs: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            input_names: Optional per-export input names
            output_names: Optional per-export output names
            dynamic_axes: Optional per-export dynamic axes mapping
            simplify: Optional override for ONNX simplification flag
            opset_version: Optional opset override
            export_params: Optional export_params override
            keep_initializers_as_inputs: Optional override for ONNX flag
            verbose: Optional override for verbose flag

        Raises:
            RuntimeError: If export fails
        """
        # Merge per-export overrides with base configuration
        export_cfg: Dict[str, Any] = dict(self.config)
        overrides = {
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "simplify": simplify,
            "opset_version": opset_version,
            "export_params": export_params,
            "keep_initializers_as_inputs": keep_initializers_as_inputs,
            "verbose": verbose,
        }
        for key, value in overrides.items():
            if value is not None:
                export_cfg[key] = value

        # Apply model wrapper if configured
        model = self.prepare_model(model)
        model.eval()

        self.logger.info("Exporting model to ONNX format...")
        if hasattr(sample_input, "shape"):
            self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {export_cfg.get('opset_version', 16)}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=export_cfg.get("export_params", True),
                    keep_initializers_as_inputs=export_cfg.get("keep_initializers_as_inputs", False),
                    opset_version=export_cfg.get("opset_version", 16),
                    do_constant_folding=export_cfg.get("do_constant_folding", True),
                    input_names=export_cfg.get("input_names", ["input"]),
                    output_names=export_cfg.get("output_names", ["output"]),
                    dynamic_axes=export_cfg.get("dynamic_axes"),
                    verbose=export_cfg.get("verbose", False),
                )

            self.logger.info(f"ONNX export completed: {output_path}")

            # Optional model simplification
            if export_cfg.get("simplify", True):
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
