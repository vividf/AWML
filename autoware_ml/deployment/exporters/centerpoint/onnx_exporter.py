"""
CenterPoint-specific ONNX exporter.

This module provides a specialized exporter for CenterPoint models that need
to export multiple ONNX files (voxel encoder + backbone/neck/head).
"""

import logging
import os
from typing import Any, Dict

import torch

from autoware_ml.deployment.exporters.base.onnx_exporter import ONNXExporter


class CenterPointONNXExporter(ONNXExporter):
    """
    Specialized exporter for CenterPoint multi-file ONNX export.
    
    Inherits from ONNXExporter and overrides export() to handle CenterPoint's
    multi-file export requirements.
    
    CenterPoint models are split into two ONNX files:
    1. pts_voxel_encoder.onnx - voxel feature extraction
    2. pts_backbone_neck_head.onnx - backbone, neck, and head processing
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize CenterPoint ONNX exporter.
        
        Args:
            config: ONNX export configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)
    
    def export(
        self,
        model: torch.nn.Module,  # CenterPointONNX model
        sample_input: Any = None,  # Not used, kept for interface compatibility
        output_path: str = None,  # Not used, kept for interface compatibility
        config_override: Dict[str, Any] = None,  # Not used, kept for interface compatibility
        data_loader=None,
        output_dir: str = None,
        sample_idx: int = 0,
    ) -> bool:
        """
        Export CenterPoint to multiple ONNX files.
        
        Overrides parent class export() to handle CenterPoint's multi-file export.
        
        Args:
            model: CenterPointONNX model instance
            sample_input: Not used (kept for interface compatibility)
            output_path: Not used (kept for interface compatibility)
            config_override: Not used (kept for interface compatibility)
            data_loader: Data loader for getting real input samples
            output_dir: Directory to save ONNX files
            sample_idx: Index of sample to use for export
            
        Returns:
            True if all exports succeeded
        """
        # Use output_dir if provided, otherwise fall back to output_path
        if output_dir is None:
            if output_path is None:
                raise ValueError("Either output_dir or output_path must be provided")
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        
        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (Multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using real data (sample_idx={sample_idx})")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract features from real data
            self.logger.info("Extracting features from real data...")
            input_features, voxel_dict = model._extract_features(data_loader, sample_idx)
            
            # Export voxel encoder
            self.logger.info("\n[1/2] Exporting voxel encoder...")
            voxel_encoder_config = {
                **self.config,
                'input_names': ['input_features'],
                'output_names': ['pillar_features'],
                'dynamic_axes': {
                    'input_features': {0: 'num_voxels', 1: 'num_max_points'},
                    'pillar_features': {0: 'num_voxels'},
                },
                'simplify': self.config.get('simplify', True),
            }
            
            voxel_encoder_path = os.path.join(output_dir, 'pts_voxel_encoder.onnx')
            success1 = super().export(
                model=model.pts_voxel_encoder,
                sample_input=input_features,
                output_path=voxel_encoder_path,
                config_override=voxel_encoder_config
            )
            
            if not success1:
                self.logger.error("Failed to export voxel encoder")
                return False
            
            # Get spatial features for backbone export
            self.logger.info("\n[2/2] Exporting backbone + neck + head...")
            with torch.no_grad():
                voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
                coors = voxel_dict["coors"]
                batch_size = coors[-1, 0] + 1
                x = model.pts_middle_encoder(voxel_features, coors, batch_size)
            
            # Create combined backbone+neck+head module
            # Import locally to avoid circular dependencies
            from projects.CenterPoint.models.detectors.centerpoint_onnx import CenterPointHeadONNX
            
            backbone_neck_head = CenterPointHeadONNX(
                model.pts_backbone,
                model.pts_neck,
                model.pts_bbox_head
            )
            
            # Get output names from bbox_head
            output_names = model.pts_bbox_head.output_names if hasattr(model.pts_bbox_head, 'output_names') else None
            if not output_names:
                # Default output names for CenterPoint
                output_names = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
            
            backbone_config = {
                **self.config,
                'input_names': ['spatial_features'],
                'output_names': output_names,
                'dynamic_axes': {
                    'spatial_features': {0: 'batch_size', 2: 'height', 3: 'width'},
                },
                'simplify': self.config.get('simplify', True),
            }
            
            backbone_path = os.path.join(output_dir, 'pts_backbone_neck_head.onnx')
            success2 = super().export(
                model=backbone_neck_head,
                sample_input=x,
                output_path=backbone_path,
                config_override=backbone_config
            )
            
            if not success2:
                self.logger.error("Failed to export backbone+neck+head")
                return False
            
            # All exports successful
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ CenterPoint ONNX export successful")
            self.logger.info("=" * 80)
            self.logger.info(f"Voxel Encoder: {voxel_encoder_path}")
            self.logger.info(f"Backbone+Neck+Head: {backbone_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CenterPoint ONNX export failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

