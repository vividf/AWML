"""
CenterPoint-specific deployment runner.

This module provides a specialized runner for CenterPoint models that require
multi-file ONNX/TensorRT export (voxel encoder + backbone/neck/head).
"""

import logging
import os
from typing import Any, Optional

from deployment.exporters.centerpoint.onnx_exporter import CenterPointONNXExporter
from deployment.exporters.centerpoint.tensorrt_exporter import CenterPointTensorRTExporter
from deployment.runners.deployment_runner import BaseDeploymentRunner
from projects.CenterPoint.deploy.utils import build_centerpoint_onnx_model


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """
    CenterPoint-specific deployment runner.

    Handles CenterPoint-specific requirements:
    - Multi-file ONNX export (voxel encoder + backbone/neck/head)
    - Multi-file TensorRT export
    - Uses CenterPoint-specific exporters with special export signatures
    - ONNX-compatible model loading
    """

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        rot_y_axis_reference: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Build ONNX-compatible CenterPoint model from checkpoint.

        NOTE:
        - self.model_cfg is the "original mmdet3d config"
        - This method converts it to ONNX-friendly cfg and updates self.model_cfg

        Args:
            checkpoint_path: Path to checkpoint file
            rot_y_axis_reference: Whether to use y-axis rotation reference
            device: Device string (e.g., "cpu", "cuda:0"). Defaults to "cpu" for export.
            **kwargs: Additional arguments

        Returns:
            Loaded PyTorch model (ONNX-compatible)
        """
        # Use shared utility to build ONNX-compatible model
        model, onnx_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            rot_y_axis_reference=rot_y_axis_reference,
        )

        # Update runner's internal model_cfg to ONNX-friendly version
        self.model_cfg = onnx_cfg

        # Inject ONNX-compatible config to evaluator via setter (single source of truth)
        if hasattr(self.evaluator, "set_onnx_config"):
            self.evaluator.set_onnx_config(onnx_cfg)
            self.logger.info("Updated evaluator with ONNX-compatible config via set_onnx_config()")

        # Inject PyTorch model to evaluator via setter
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with PyTorch model via set_pytorch_model()")

        return model

    def export_onnx(self, pytorch_model: Any, **kwargs) -> Optional[str]:
        """
        Export CenterPoint model to ONNX format (multi-file).

        Overrides base implementation to handle CenterPoint's multi-file export.

        Args:
            pytorch_model: PyTorch model to export (must be CenterPointONNX-compatible)
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported ONNX directory, or None if export failed
        """
        if not self.config.export_config.should_export_onnx():
            return None

        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (Multi-file)")
        self.logger.info("=" * 80)

        # Verify exporter is CenterPoint-specific
        if not isinstance(self._onnx_exporter, CenterPointONNXExporter):
            self.logger.error("CenterPointDeploymentRunner requires CenterPointONNXExporter")
            return None

        # Verify model is ONNX-compatible
        if not hasattr(pytorch_model, "_extract_features"):
            self.logger.error("❌ ONNX export requires an ONNX-compatible model (CenterPointONNX).")
            return None

        # Save to work_dir/onnx/ directory
        output_dir = os.path.join(self.config.export_config.work_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        # Use CenterPoint-specific export signature
        try:
            self._onnx_exporter.export(
                model=pytorch_model, data_loader=self.data_loader, output_dir=output_dir, sample_idx=0
            )
        except Exception:
            self.logger.error(f"❌ ONNX export failed")
            raise

        self.logger.info(f"✅ ONNX export successful: {output_dir}")
        return output_dir

    def export_tensorrt(self, onnx_path: str, **kwargs) -> Optional[str]:
        """
        Export CenterPoint ONNX models to TensorRT engines (multi-file).

        Overrides base implementation to handle CenterPoint's multi-file export.

        Args:
            onnx_path: Path to ONNX model directory (must be a directory for CenterPoint)
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported TensorRT directory, or None if export failed
        """
        if not self.config.export_config.should_export_tensorrt():
            return None

        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None

        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to TensorRT (Multi-file)")
        self.logger.info("=" * 80)

        # Verify exporter is CenterPoint-specific
        if not isinstance(self._tensorrt_exporter, CenterPointTensorRTExporter):
            self.logger.error("CenterPointDeploymentRunner requires CenterPointTensorRTExporter")
            return None

        # CenterPoint requires ONNX directory, not a single file
        if not os.path.isdir(onnx_path):
            self.logger.error("CenterPoint requires ONNX directory, not a single file")
            return None

        # Save to work_dir/tensorrt/ directory
        output_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
        os.makedirs(output_dir, exist_ok=True)

        # Use CenterPoint-specific export signature
        # TensorRT export uses configured CUDA device
        try:
            self._tensorrt_exporter.export(
                onnx_dir=onnx_path, output_dir=output_dir, device=self.config.export_config.cuda_device
            )
        except Exception:
            self.logger.error(f"❌ TensorRT export failed")
            raise

        self.logger.info(f"✅ TensorRT export successful: {output_dir}")
        return output_dir
