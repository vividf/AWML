"""
CenterPoint-specific ONNX exporter.

This module provides a specialized exporter for CenterPoint models that need
to export multiple ONNX files (voxel encoder + backbone/neck/head).
"""

import logging
import os
from dataclasses import replace
from typing import Any, Optional, Tuple

import torch

from deployment.exporters.base.configs import ONNXExportConfig
from deployment.exporters.base.onnx_exporter import ONNXExporter


class CenterPointONNXExporter(ONNXExporter):
    """
    Specialized exporter for CenterPoint multi-file ONNX export.

    Inherits from ONNXExporter and overrides export() to handle CenterPoint's
    multi-file export requirements.

    CenterPoint models are split into two ONNX files:
    1. pts_voxel_encoder.onnx - voxel feature extraction
    2. pts_backbone_neck_head.onnx - backbone, neck, and head processing
    """

    def __init__(
        self,
        config: ONNXExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize CenterPoint ONNX exporter.

        Args:
            config: ONNX export configuration dataclass instance.
            model_wrapper: Optional model wrapper class
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)

    def export(
        self,
        model: torch.nn.Module,  # CenterPointONNX model
        sample_input: Any = None,  # Not used, kept for interface compatibility
        output_path: str = None,  # Not used, kept for interface compatibility
        data_loader=None,
        output_dir: str = None,
        sample_idx: int = 0,
    ) -> None:
        """
        Export CenterPoint to multiple ONNX files.

        Overrides parent class export() to handle CenterPoint's multi-file export.

        Args:
            model: CenterPointONNX model instance
            sample_input: Not used (kept for interface compatibility)
            output_path: Not used (kept for interface compatibility)
            data_loader: Data loader for getting real input samples
            output_dir: Directory to save ONNX files
            sample_idx: Index of sample to use for export

        Raises:
            RuntimeError: If either export fails
        """
        output_dir = self._resolve_output_dir(output_dir, output_path)
        input_features, voxel_dict = self._extract_features(model, data_loader, sample_idx, output_dir)
        voxel_encoder_path = self._export_voxel_encoder(model, input_features, output_dir)
        backbone_path = self._export_backbone_neck_head(model, input_features, voxel_dict, output_dir)
        self._log_summary(voxel_encoder_path, backbone_path)

    def _resolve_output_dir(self, output_dir: Optional[str], output_path: Optional[str]) -> str:
        """
        Resolve output directory from provided arguments.

        Args:
            output_dir: Preferred output directory
            output_path: Fallback output path (directory will be extracted)

        Returns:
            Resolved output directory path

        Raises:
            ValueError: If neither output_dir nor output_path is provided
        """
        if output_dir is None:
            if output_path is None:
                raise ValueError("Either output_dir or output_path must be provided")
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _extract_features(
        self, model: torch.nn.Module, data_loader: Any, sample_idx: int, output_dir: str
    ) -> Tuple[Any, dict]:
        """
        Extract features from real data using the model.

        Args:
            model: CenterPointONNX model instance
            data_loader: Data loader for getting real input samples
            sample_idx: Index of sample to use
            output_dir: Output directory (for logging)

        Returns:
            Tuple of (input_features, voxel_dict)

        Raises:
            RuntimeError: If feature extraction fails
        """
        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (Multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using real data (sample_idx={sample_idx})")

        try:
            self.logger.info("Extracting features from real data...")
            input_features, voxel_dict = model._extract_features(data_loader, sample_idx)
            return input_features, voxel_dict
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            raise RuntimeError("Feature extraction failed") from e

    def _export_voxel_encoder(self, model: torch.nn.Module, input_features: Any, output_dir: str) -> str:
        """
        Export voxel encoder to ONNX.

        Args:
            model: CenterPointONNX model instance
            input_features: Extracted input features
            output_dir: Output directory for ONNX file

        Returns:
            Path to exported voxel encoder ONNX file

        Raises:
            RuntimeError: If export fails
        """
        self.logger.info("\n[1/2] Exporting voxel encoder...")
        voxel_encoder_path = os.path.join(output_dir, "pts_voxel_encoder.onnx")

        voxel_encoder_cfg = self._build_voxel_encoder_config()

        try:
            super().export(
                model=model.pts_voxel_encoder,
                sample_input=input_features,
                output_path=voxel_encoder_path,
                config_override=voxel_encoder_cfg,
            )
            return voxel_encoder_path
        except Exception as e:
            self.logger.error("Failed to export voxel encoder")
            raise RuntimeError("Voxel encoder export failed") from e

    def _build_voxel_encoder_config(self) -> ONNXExportConfig:
        """
        Build configuration for voxel encoder export.

        Returns:
            ONNXExportConfig for voxel encoder
        """
        return replace(
            self.config,
            input_names=("input_features",),
            output_names=("pillar_features",),
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        )

    def _export_backbone_neck_head(
        self, model: torch.nn.Module, input_features: Any, voxel_dict: dict, output_dir: str
    ) -> str:
        """
        Export backbone, neck, and head to ONNX.

        Args:
            model: CenterPointONNX model instance
            input_features: Extracted input features
            voxel_dict: Voxel dictionary from feature extraction
            output_dir: Output directory for ONNX file

        Returns:
            Path to exported backbone+neck+head ONNX file

        Raises:
            RuntimeError: If export fails
        """
        self.logger.info("\n[2/2] Exporting backbone + neck + head...")

        # Get spatial features for backbone export
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            coors = voxel_dict["coors"]
            batch_size = coors[-1, 0] + 1
            x = model.pts_middle_encoder(voxel_features, coors, batch_size)

        # Create combined backbone+neck+head module
        # Import locally to avoid circular dependencies
        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX

        backbone_neck_head = CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

        # Get output names from bbox_head
        output_names = self._get_output_names(model)

        backbone_path = os.path.join(output_dir, "pts_backbone_neck_head.onnx")
        backbone_cfg = self._build_backbone_config(output_names)

        try:
            super().export(
                model=backbone_neck_head,
                sample_input=x,
                output_path=backbone_path,
                config_override=backbone_cfg,
            )
            return backbone_path
        except Exception as e:
            self.logger.error("Failed to export backbone+neck+head")
            raise RuntimeError("Backbone+neck+head export failed") from e

    def _get_output_names(self, model: torch.nn.Module) -> Tuple[str, ...]:
        """
        Get output names from bbox_head or use defaults.

        Args:
            model: CenterPointONNX model instance

        Returns:
            Tuple of output names
        """
        output_names = model.pts_bbox_head.output_names if hasattr(model.pts_bbox_head, "output_names") else None
        if not output_names:
            # Default output names for CenterPoint
            output_names = ["reg", "height", "dim", "rot", "vel", "heatmap"]
        return tuple(output_names) if isinstance(output_names, (list, tuple)) else (output_names,)

    def _build_backbone_config(self, output_names: Tuple[str, ...]) -> ONNXExportConfig:
        """
        Build configuration for backbone+neck+head export.

        Args:
            output_names: Output tensor names

        Returns:
            ONNXExportConfig for backbone+neck+head
        """
        return replace(
            self.config,
            input_names=("spatial_features",),
            output_names=output_names,
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
            },
        )

    def _log_summary(self, voxel_encoder_path: str, backbone_path: str) -> None:
        """
        Log export summary.

        Args:
            voxel_encoder_path: Path to voxel encoder ONNX file
            backbone_path: Path to backbone+neck+head ONNX file
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ CenterPoint ONNX export successful")
        self.logger.info("=" * 80)
        self.logger.info(f"Voxel Encoder: {voxel_encoder_path}")
        self.logger.info(f"Backbone+Neck+Head: {backbone_path}")
