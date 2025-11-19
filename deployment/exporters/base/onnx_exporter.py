"""ONNX model exporter."""

import logging
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

import onnx
import onnxsim
import torch

from deployment.exporters.base.base_exporter import BaseExporter
from deployment.exporters.base.configs import ONNXExportConfig


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
        config: ONNXExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize ONNX exporter.

        Args:
            config: ONNX export configuration dataclass instance.
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
        # Merge per-export overrides with base configuration using dataclasses.replace
        export_cfg = self.config
        if input_names is not None:
            export_cfg = replace(export_cfg, input_names=tuple(input_names))
        if output_names is not None:
            export_cfg = replace(export_cfg, output_names=tuple(output_names))
        if dynamic_axes is not None:
            export_cfg = replace(export_cfg, dynamic_axes=dynamic_axes)
        if simplify is not None:
            export_cfg = replace(export_cfg, simplify=simplify)
        if opset_version is not None:
            export_cfg = replace(export_cfg, opset_version=opset_version)
        if export_params is not None:
            export_cfg = replace(export_cfg, export_params=export_params)
        if keep_initializers_as_inputs is not None:
            export_cfg = replace(export_cfg, keep_initializers_as_inputs=keep_initializers_as_inputs)
        if verbose is not None:
            export_cfg = replace(export_cfg, verbose=verbose)

        # Apply model wrapper if configured
        model = self.prepare_model(model)
        model.eval()

        self.logger.info("Exporting model to ONNX format...")
        if hasattr(sample_input, "shape"):
            self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {export_cfg.opset_version}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=export_cfg.export_params,
                    keep_initializers_as_inputs=export_cfg.keep_initializers_as_inputs,
                    opset_version=export_cfg.opset_version,
                    do_constant_folding=export_cfg.do_constant_folding,
                    input_names=list(export_cfg.input_names),
                    output_names=list(export_cfg.output_names),
                    dynamic_axes=export_cfg.dynamic_axes,
                    verbose=export_cfg.verbose,
                )

            self.logger.info(f"ONNX export completed: {output_path}")

            # Optional model simplification
            if export_cfg.simplify:
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
                self.logger.info("ONNX model simplified successfully")
            else:
                self.logger.warning("ONNX model simplification failed")
        except Exception as e:
            self.logger.warning(f"ONNX simplification error: {e}")
