"""
CenterPoint-specific component extractor.

This module contains all CenterPoint-specific logic for extracting
exportable model components. It implements the ModelComponentExtractor
interface from the deployment framework.
"""

import logging
from typing import Any, List, Tuple

import torch

from deployment.exporters.base.configs import ONNXExportConfig
from deployment.exporters.workflows.interfaces import ExportableComponent, ModelComponentExtractor

logger = logging.getLogger(__name__)


class CenterPointComponentExtractor(ModelComponentExtractor):
    """
    Extracts exportable components from CenterPoint model.

    CenterPoint uses a multi-stage architecture that requires multi-file ONNX export:
    1. Voxel Encoder: Converts voxels to features
    2. Backbone+Neck+Head: Detection head

    This extractor handles all CenterPoint-specific logic:
    - Feature extraction from model
    - Creating combined backbone+neck+head module
    - Preparing sample inputs for each component
    - Configuring ONNX export settings
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize extractor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        """
        Extract CenterPoint components for ONNX export.

        Args:
            model: CenterPoint PyTorch model
            sample_data: Tuple of (input_features, voxel_dict) from preprocessing

        Returns:
            List containing two components: voxel encoder and backbone+neck+head
        """
        # Unpack sample data
        input_features, voxel_dict = sample_data

        self.logger.info("Extracting CenterPoint components for export...")

        # Component 1: Voxel Encoder
        voxel_component = self._create_voxel_encoder_component(model, input_features)

        # Component 2: Backbone+Neck+Head
        backbone_component = self._create_backbone_component(model, input_features, voxel_dict)

        self.logger.info("Extracted 2 components: voxel_encoder, backbone_neck_head")

        return [voxel_component, backbone_component]

    def _create_voxel_encoder_component(
        self, model: torch.nn.Module, input_features: torch.Tensor
    ) -> ExportableComponent:
        """Create exportable voxel encoder component."""
        return ExportableComponent(
            name="pts_voxel_encoder",
            module=model.pts_voxel_encoder,
            sample_input=input_features,
            config_override=ONNXExportConfig(
                input_names=("input_features",),
                output_names=("pillar_features",),
                dynamic_axes={
                    "input_features": {0: "num_voxels", 1: "num_max_points"},
                    "pillar_features": {0: "num_voxels"},
                },
                opset_version=16,
                do_constant_folding=True,
                simplify=True,
                save_file="pts_voxel_encoder.onnx",
            ),
        )

    def _create_backbone_component(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> ExportableComponent:
        """Create exportable backbone+neck+head component."""
        # Prepare backbone input by running through voxel and middle encoders
        backbone_input = self._prepare_backbone_input(model, input_features, voxel_dict)

        # Create combined backbone+neck+head module
        backbone_module = self._create_backbone_module(model)

        # Get output names
        output_names = self._get_output_names(model)

        return ExportableComponent(
            name="pts_backbone_neck_head",
            module=backbone_module,
            sample_input=backbone_input,
            config_override=ONNXExportConfig(
                input_names=("spatial_features",),
                output_names=output_names,
                dynamic_axes={
                    "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                },
                opset_version=16,
                do_constant_folding=True,
                simplify=True,
                save_file="pts_backbone_neck_head.onnx",
            ),
        )

    def _prepare_backbone_input(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> torch.Tensor:
        """
        Prepare input tensor for backbone export by running inference.

        This runs the voxel encoder and middle encoder to generate
        spatial features that will be the input to the backbone.
        """
        with torch.no_grad():
            # Run voxel encoder
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)

            # Get coordinates and batch size
            coors = voxel_dict["coors"]
            batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1

            # Run middle encoder (sparse convolution)
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)

        return spatial_features

    def _create_backbone_module(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Create combined backbone+neck+head module for ONNX export.

        This imports CenterPoint-specific model classes from projects/.
        """
        from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX

        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

    def _get_output_names(self, model: torch.nn.Module) -> Tuple[str, ...]:
        """Get output names from model or use defaults."""
        if hasattr(model, "pts_bbox_head") and hasattr(model.pts_bbox_head, "output_names"):
            output_names = model.pts_bbox_head.output_names
            if isinstance(output_names, (list, tuple)):
                return tuple(output_names)
            return (output_names,)

        return ("heatmap", "reg", "height", "dim", "rot", "vel")

    def extract_features(self, model: torch.nn.Module, data_loader: Any, sample_idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Extract features using model's internal method.

        This is a helper method that wraps the model's _extract_features method,
        which is used during ONNX export to get sample data.

        Args:
            model: CenterPoint model
            data_loader: Data loader
            sample_idx: Sample index

        Returns:
            Tuple of (input_features, voxel_dict)
        """
        if hasattr(model, "_extract_features"):
            return model._extract_features(data_loader, sample_idx)
        else:
            raise AttributeError(
                "CenterPoint model must have _extract_features method for ONNX export. "
                "Please ensure the model is built with ONNX compatibility."
            )
