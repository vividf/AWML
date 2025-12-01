"""
CenterPoint ONNX export workflow using composition.

This workflow orchestrates multi-file ONNX export for CenterPoint models.
It uses the ModelComponentExtractor pattern to keep model-specific logic
separate from the generic export workflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.workflows.base import OnnxExportWorkflow
from deployment.exporters.workflows.interfaces import ExportableComponent, ModelComponentExtractor


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
    ) -> Artifact:
        """
        Export CenterPoint model to multi-file ONNX format.

        Args:
            model: CenterPoint PyTorch model
            data_loader: Data loader for extracting sample features
            output_dir: Output directory for ONNX files
            config: Deployment configuration (not used, kept for interface)
            sample_idx: Sample index to use for feature extraction

        Returns:
            Artifact pointing to output directory with multi_file=True

        Raises:
            AttributeError: If component extractor doesn't have extract_features method
            RuntimeError: If feature extraction or export fails
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        self._log_header(output_dir_path, sample_idx)
        sample_data = self._extract_sample_data(model, data_loader, sample_idx)
        components = self.component_extractor.extract_components(model, sample_data)

        exported_paths = self._export_components(components, output_dir_path)
        self._log_summary(exported_paths)

        return Artifact(path=str(output_dir_path), multi_file=True)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _log_header(self, output_dir: Path, sample_idx: int) -> None:
        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using sample index: {sample_idx}")

    def _extract_sample_data(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> Tuple[torch.Tensor, dict]:
        if not hasattr(self.component_extractor, "extract_features"):
            raise AttributeError("Component extractor must provide extract_features method")

        self.logger.info("Extracting features from sample data...")
        try:
            return self.component_extractor.extract_features(model, data_loader, sample_idx)
        except Exception as exc:
            self.logger.error("Failed to extract features", exc_info=exc)
            raise RuntimeError("Feature extraction failed") from exc

    def _export_components(
        self,
        components: Iterable[ExportableComponent],
        output_dir: Path,
    ) -> Tuple[str, ...]:
        exported_paths: list[str] = []
        component_list = list(components)
        total = len(component_list)

        for index, component in enumerate(component_list, start=1):
            self.logger.info(f"\n[{index}/{total}] Exporting {component.name}...")
            exporter = self._build_onnx_exporter()
            output_path = output_dir / f"{component.name}.onnx"

            try:
                exporter.export(
                    model=component.module,
                    sample_input=component.sample_input,
                    output_path=str(output_path),
                    config_override=component.config_override,
                )
            except Exception as exc:
                self.logger.error(f"Failed to export {component.name}", exc_info=exc)
                raise RuntimeError(f"{component.name} export failed") from exc

            exported_paths.append(str(output_path))
            self.logger.info(f"Exported {component.name}: {output_path}")

        return tuple(exported_paths)

    def _build_onnx_exporter(self):
        # CenterPoint does not require special wrapping, so IdentityWrapper suffices.
        return self.exporter_factory.create_onnx_exporter(
            config=self.config,
            wrapper_cls=IdentityWrapper,
            logger=self.logger,
        )

    def _log_summary(self, exported_paths: Tuple[str, ...]) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CenterPoint ONNX export successful")
        self.logger.info("=" * 80)
        for path in exported_paths:
            self.logger.info(f"  • {os.path.basename(path)}")
