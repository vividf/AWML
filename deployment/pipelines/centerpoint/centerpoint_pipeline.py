"""
CenterPoint Deployment Pipeline Base Class.

This module provides the abstract base class for CenterPoint deployment,
defining the unified pipeline that shares PyTorch processing while allowing
backend-specific optimizations for voxel encoder and backbone/head.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from deployment.pipelines.base.detection_3d_pipeline import Detection3DPipeline

logger = logging.getLogger(__name__)


class CenterPointDeploymentPipeline(Detection3DPipeline):
    """
    Abstract base class for CenterPoint deployment pipeline.

    This class defines the complete inference flow for CenterPoint, with:
    - Shared preprocessing (voxelization + input features)
    - Shared middle encoder processing
    - Shared postprocessing (predict_by_feat)
    - Abstract methods for backend-specific voxel encoder and backbone/head

    The design eliminates code duplication by centralizing PyTorch processing
    while allowing ONNX/TensorRT backends to optimize the convertible parts.
    """

    def __init__(self, pytorch_model, device: str = "cuda", backend_type: str = "unknown"):
        """
        Initialize CenterPoint pipeline.

        Args:
            pytorch_model: PyTorch model (used for preprocessing, middle encoder, postprocessing)
            device: Device for inference ('cuda' or 'cpu')
            backend_type: Backend type ('pytorch', 'onnx', 'tensorrt')
        """
        # Get class names from model config if available
        class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]  # Default T4Dataset classes
        if hasattr(pytorch_model, "CLASSES"):
            class_names = pytorch_model.CLASSES
        elif hasattr(pytorch_model, "cfg") and hasattr(pytorch_model.cfg, "class_names"):
            class_names = pytorch_model.cfg.class_names

        # Get point cloud range and voxel size from model config
        point_cloud_range = None
        voxel_size = None
        if hasattr(pytorch_model, "cfg"):
            if hasattr(pytorch_model.cfg, "point_cloud_range"):
                point_cloud_range = pytorch_model.cfg.point_cloud_range
            if hasattr(pytorch_model.cfg, "voxel_size"):
                voxel_size = pytorch_model.cfg.voxel_size

        # Initialize parent class
        super().__init__(
            model=pytorch_model,
            device=device,
            num_classes=len(class_names),
            class_names=class_names,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            backend_type=backend_type,
        )

        self.pytorch_model = pytorch_model
        self._stage_latencies = {}  # Store stage-wise latencies for detailed breakdown

    # ========== Shared Methods (All backends use same logic) ==========

    def preprocess(self, points: torch.Tensor, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Preprocess: voxelization + input features preparation.

        ONNX/TensorRT backends use this for voxelization and input feature preparation.
        PyTorch backend may override this method for end-to-end inference.

        Args:
            points: Input point cloud [N, point_features]
            **kwargs: Additional preprocessing parameters

        Returns:
            Tuple of (preprocessed_dict, metadata):
            - preprocessed_dict: Dictionary containing:
                - 'input_features': 11-dim features for voxel encoder [N_voxels, max_points, 11]
                - 'voxels': Raw voxel data
                - 'num_points': Number of points per voxel
                - 'coors': Voxel coordinates [N_voxels, 4] (batch_idx, z, y, x)
            - metadata: Empty dict (for compatibility with base class)
        """
        from mmdet3d.structures import Det3DDataSample

        # Ensure points are on correct device
        points_tensor = points.to(self.device)

        # Step 1: Voxelization using PyTorch data_preprocessor
        data_samples = [Det3DDataSample()]

        with torch.no_grad():
            batch_inputs = self.pytorch_model.data_preprocessor(
                {"inputs": {"points": [points_tensor]}, "data_samples": data_samples}
            )

        voxel_dict = batch_inputs["inputs"]["voxels"]
        voxels = voxel_dict["voxels"]
        num_points = voxel_dict["num_points"]
        coors = voxel_dict["coors"]

        # Step 2: Get input features (only for ONNX/TensorRT models)
        input_features = None
        with torch.no_grad():
            if hasattr(self.pytorch_model.pts_voxel_encoder, "get_input_features"):
                input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(voxels, num_points, coors)
        preprocessed_dict = {
            "input_features": input_features,
            "voxels": voxels,
            "num_points": num_points,
            "coors": coors,
        }

        # Return tuple format for compatibility with base class infer()
        return preprocessed_dict, {}

    def process_middle_encoder(self, voxel_features: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        """
        Process through middle encoder using PyTorch.

        All backends use the same middle encoder processing because sparse convolution
        cannot be converted to ONNX/TensorRT efficiently.

        Args:
            voxel_features: Features from voxel encoder [N_voxels, feature_dim]
            coors: Voxel coordinates [N_voxels, 4]

        Returns:
            spatial_features: Spatial features [B, C, H, W]
        """
        # Ensure tensors are on correct device
        voxel_features = voxel_features.to(self.device)
        coors = coors.to(self.device)

        # Calculate batch size
        batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1

        # Process through PyTorch middle encoder
        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)

        return spatial_features

    def postprocess(self, head_outputs: List[torch.Tensor], sample_meta: Dict) -> List[Dict]:
        """
        Postprocess: decode head outputs using PyTorch's predict_by_feat.

        All backends use the same postprocessing to ensure consistent results.
        This includes NMS, coordinate transformation, and score filtering.

        Args:
            head_outputs: List of [heatmap, reg, height, dim, rot, vel]
            sample_meta: Sample metadata (point_cloud_range, voxel_size, etc.)

        Returns:
            List of predictions with bbox_3d, score, and label
        """
        # Ensure all outputs are on correct device
        head_outputs = [out.to(self.device) for out in head_outputs]

        # Organize head outputs: [heatmap, reg, height, dim, rot, vel]
        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        heatmap, reg, height, dim, rot, vel = head_outputs

        # Check if rot_y_axis_reference conversion is needed
        # When ONNX/TensorRT outputs use rot_y_axis_reference format, we need to convert back
        # to standard format before passing to PyTorch's predict_by_feat
        if hasattr(self.pytorch_model, "pts_bbox_head"):
            rot_y_axis_reference = getattr(self.pytorch_model.pts_bbox_head, "_rot_y_axis_reference", False)

            if rot_y_axis_reference:
                # Convert dim from [w, l, h] back to [l, w, h]
                dim = dim[:, [1, 0, 2], :, :]

                # Convert rot from [-cos(x), -sin(y)] back to [sin(y), cos(x)]
                rot = rot * (-1.0)
                rot = rot[:, [1, 0], :, :]

        # Convert to mmdet3d format
        preds_dict = {"heatmap": heatmap, "reg": reg, "height": height, "dim": dim, "rot": rot, "vel": vel}
        preds_dicts = ([preds_dict],)  # Tuple[List[dict]] format

        # Prepare metadata
        from mmdet3d.structures import LiDARInstance3DBoxes

        if "box_type_3d" not in sample_meta:
            sample_meta["box_type_3d"] = LiDARInstance3DBoxes
        batch_input_metas = [sample_meta]

        # Use PyTorch's predict_by_feat for consistent decoding
        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts, batch_input_metas=batch_input_metas
            )

        # Parse predictions
        results = []
        for pred_instances in predictions_list:
            bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
            scores_3d = pred_instances.scores_3d.cpu().numpy()
            labels_3d = pred_instances.labels_3d.cpu().numpy()

            for i in range(len(bboxes_3d)):
                results.append(
                    {
                        "bbox_3d": bboxes_3d[i][:7].tolist(),  # [x, y, z, w, l, h, yaw]
                        "score": float(scores_3d[i]),
                        "label": int(labels_3d[i]),
                    }
                )

        return results

    # ========== Abstract Methods (Backend-specific implementations) ==========

    @abstractmethod
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Run voxel encoder inference.

        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized voxel encoder inference.

        Args:
            input_features: Input features [N_voxels, max_points, feature_dim]

        Returns:
            voxel_features: Voxel features [N_voxels, feature_dim]
        """
        pass

    @abstractmethod
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone + neck + head inference.

        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized backbone/neck/head inference.

        Args:
            spatial_features: Spatial features [B, C, H, W]

        Returns:
            List of head outputs: [heatmap, reg, height, dim, rot, vel]
        """
        pass

    # ========== Main Inference Pipeline ==========

    def run_model(self, preprocessed_input: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Run complete multi-stage model inference.

        This method implements all inference stages:
        1. Voxel Encoder (backend-specific)
        2. Middle Encoder (PyTorch)
        3. Backbone + Head (backend-specific)

        This method is called by the base class `infer()` method, which handles
        preprocessing, postprocessing, latency tracking, and error handling.

        Args:
            preprocessed_input: Dict from preprocess() containing:
                - 'input_features': Input features for voxel encoder [N_voxels, max_points, 11]
                - 'coors': Voxel coordinates [N_voxels, 4]
                - 'voxels': Raw voxel data
                - 'num_points': Number of points per voxel

        Returns:
            head_outputs: List of head outputs [heatmap, reg, height, dim, rot, vel]

        Note:
            Stage-wise latencies are stored in `self._stage_latencies` and will be
            merged into the overall latency breakdown by the base class `infer()`.
        """
        import time

        # Stage 1: Voxel Encoder (backend-specific)
        start = time.time()
        voxel_features = self.run_voxel_encoder(preprocessed_input["input_features"])
        self._stage_latencies["voxel_encoder_ms"] = (time.time() - start) * 1000

        # Stage 2: Middle Encoder (PyTorch - 所有后端相同)
        start = time.time()
        spatial_features = self.process_middle_encoder(voxel_features, preprocessed_input["coors"])
        self._stage_latencies["middle_encoder_ms"] = (time.time() - start) * 1000

        # Stage 3: Backbone + Head (backend-specific)
        start = time.time()
        head_outputs = self.run_backbone_head(spatial_features)
        self._stage_latencies["backbone_head_ms"] = (time.time() - start) * 1000

        return head_outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
