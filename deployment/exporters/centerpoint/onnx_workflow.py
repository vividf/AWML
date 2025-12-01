"""
CenterPoint ONNX export workflow using composition.

This workflow orchestrates multi-file ONNX export for CenterPoint models.
It uses the ModelComponentExtractor pattern to keep model-specific logic
separate from the generic export workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.workflows.base import OnnxExportWorkflow
from deployment.exporters.workflows.interfaces import ModelComponentExtractor


class CenterPointONNXExportWorkflow(OnnxExportWorkflow):
    """
    CenterPoint ONNX export workflow.

    Orchestrates multi-file ONNX export using a generic ONNXExporter and
    CenterPointComponentExtractor for model-specific logic.

    Components exported:
    - voxel encoder → pts_voxel_encoder.onnx
    - backbone+neck+head → pts_backbone_neck_head.onnx
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        component_extractor: ModelComponentExtractor,
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize CenterPoint ONNX export workflow.

        Args:
            exporter_factory: Factory class for creating exporters
            component_extractor: Extracts model components (injected from projects/)
            config: Deployment configuration
            logger: Optional logger instance
        """
        self.exporter_factory = exporter_factory
        self.component_extractor = component_extractor
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
        context: Optional[ExportContext] = None,
    ) -> Artifact:
        """
        Export CenterPoint model to multi-file ONNX format.

        Args:
            model: CenterPoint PyTorch model
            data_loader: Data loader for extracting sample features
            output_dir: Output directory for ONNX files
            config: Deployment configuration (not used, kept for interface)
            sample_idx: Sample index to use for feature extraction
            context: Export context with project-specific parameters (currently unused,
                     but available for future extensions)

        Returns:
            Artifact pointing to output directory with multi_file=True

        Raises:
            AttributeError: If component extractor doesn't have extract_features method
            RuntimeError: If feature extraction or export fails
        """
        # Note: context available for future extensions (e.g., precision hints, debug flags)
        del context  # Explicitly unused

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (Multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using sample index: {sample_idx}")

        # Extract features using component extractor helper
        # This delegates to model's _extract_features method
        try:
            self.logger.info("Extracting features from sample data...")
            if hasattr(self.component_extractor, "extract_features"):
                sample_data = self.component_extractor.extract_features(model, data_loader, sample_idx)
            else:
                raise AttributeError("Component extractor must provide extract_features method")
        except Exception as exc:
            self.logger.error(f"Failed to extract features: {exc}")
            raise RuntimeError("Feature extraction failed") from exc

        # Extract exportable components (delegates all model-specific logic)
        components = self.component_extractor.extract_components(model, sample_data)

        # Export each component using generic ONNX exporter
        exported_paths = []
        for i, component in enumerate(components, 1):
            self.logger.info(f"\n[{i}/{len(components)}] Exporting {component.name}...")

            # Create fresh exporter for each component (no caching)
            exporter = self.exporter_factory.create_onnx_exporter(
                config=self.config, wrapper_cls=IdentityWrapper, logger=self.logger  # CenterPoint doesn't need wrapper
            )

            # Determine output path
            output_path = os.path.join(output_dir, f"{component.name}.onnx")

            # Export component
            try:
                exporter.export(
                    model=component.module,
                    sample_input=component.sample_input,
                    output_path=output_path,
                    config_override=component.config_override,
                )
                exported_paths.append(output_path)
                self.logger.info(f"Exported {component.name}: {output_path}")
            except Exception as exc:
                self.logger.error(f"Failed to export {component.name}")
                raise RuntimeError(f"{component.name} export failed") from exc

        # Log summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CenterPoint ONNX export successful")
        self.logger.info("=" * 80)
        for path in exported_paths:
            self.logger.info(f"  • {os.path.basename(path)}")

        return Artifact(path=output_dir, multi_file=True)
