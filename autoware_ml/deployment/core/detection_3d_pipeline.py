"""
3D Object Detection Pipeline Base Class.

This module provides the base class for 3D object detection pipelines,
implementing common functionality for point cloud-based detection models like CenterPoint.
"""

from abc import abstractmethod
from typing import List, Dict, Tuple, Any
import logging

import torch
import numpy as np

from .base_pipeline import BaseDeploymentPipeline


logger = logging.getLogger(__name__)


class Detection3DPipeline(BaseDeploymentPipeline):
    """
    Base class for 3D object detection pipelines.
    
    Provides common functionality for 3D detection tasks including:
    - Point cloud preprocessing (voxelization, normalization)
    - Postprocessing (NMS, coordinate transformation)
    - Standard 3D detection output format
    
    Expected output format:
        List[Dict] where each dict contains:
        {
            'bbox_3d': [x, y, z, w, l, h, yaw],  # 3D bounding box
            'score': float,                       # Confidence score
            'label': int,                         # Class label
            'class_name': str                     # Class name (optional)
        }
    """
    
    def __init__(
        self, 
        model: Any,
        device: str = "cpu",
        num_classes: int = 10,
        class_names: List[str] = None,
        point_cloud_range: List[float] = None,
        voxel_size: List[float] = None,
        backend_type: str = "unknown"
    ):
        """
        Initialize 3D detection pipeline.
        
        Args:
            model: Model object
            device: Device for inference
            num_classes: Number of classes
            class_names: List of class names
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: Voxel size [vx, vy, vz]
            backend_type: Backend type
        """
        super().__init__(model, device, task_type="detection_3d", backend_type=backend_type)
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
    
    def preprocess(
        self, 
        points: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Standard 3D detection preprocessing.
        
        Note: For 3D detection, preprocessing is often model-specific
        (voxelization, pillar generation, etc.), so this method should
        be overridden by specific implementations.
        
        Args:
            points: Input point cloud [N, point_features]
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Dictionary containing preprocessed data
        """
        raise NotImplementedError(
            "preprocess() must be implemented by specific 3D detector pipeline.\n"
            "3D detection preprocessing varies significantly between models."
        )
    
    def run_model(self, preprocessed_input: Any) -> Any:
        """
        Run 3D detection model (backend-specific).
        
        NOTE: For 3D detection, most models use a multi-stage pipeline
        (voxel encoder → middle encoder → backbone/head) instead of a single
        run_model() call. Therefore, this method is typically not used.
        
        Instead, 3D detection pipelines override infer() and implement
        stage-specific methods like:
        - run_voxel_encoder()
        - process_middle_encoder()
        - run_backbone_head()
        
        Args:
            preprocessed_input: Preprocessed data
            
        Returns:
            Model output (backend-specific format)
        """
        raise NotImplementedError(
            "run_model() is not used for 3D detection pipelines. "
            "3D detection uses a multi-stage inference pipeline. "
            "See CenterPointDeploymentPipeline.infer() for implementation."
        )
    
    def postprocess(
        self, 
        model_output: Any,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Standard 3D detection postprocessing.
        
        Note: For 3D detection, postprocessing is often model-specific
        (CenterPoint uses predict_by_feat, PointPillars uses different logic),
        so this method should be overridden by specific implementations.
        
        Args:
            model_output: Raw model output
            metadata: Preprocessing metadata
            
        Returns:
            List of 3D detections in standard format
        """
        raise NotImplementedError(
            "postprocess() must be implemented by specific 3D detector pipeline.\n"
            "3D detection postprocessing varies significantly between models."
        )
    
    def _filter_by_score(
        self,
        detections: List[Dict],
        score_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Filter detections by confidence score.
        
        Args:
            detections: List of detection dictionaries
            score_threshold: Minimum confidence score
            
        Returns:
            Filtered detections
        """
        return [det for det in detections if det['score'] >= score_threshold]
    
    def _filter_by_range(
        self,
        detections: List[Dict],
        point_cloud_range: List[float] = None
    ) -> List[Dict]:
        """
        Filter detections by point cloud range.
        
        Args:
            detections: List of detection dictionaries
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            
        Returns:
            Filtered detections within range
        """
        if point_cloud_range is None:
            point_cloud_range = self.point_cloud_range
        
        if point_cloud_range is None:
            return detections
        
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        
        filtered = []
        for det in detections:
            x, y, z = det['bbox_3d'][:3]
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                filtered.append(det)
        
        return filtered

