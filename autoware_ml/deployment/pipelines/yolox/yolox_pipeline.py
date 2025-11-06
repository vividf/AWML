"""
YOLOX Deployment Pipeline Base Class.

This module provides the abstract base class for YOLOX deployment,
defining the unified pipeline that shares preprocessing and postprocessing logic
while allowing backend-specific optimizations for model inference.
"""

from abc import abstractmethod
from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.core.detection_2d_pipeline import Detection2DPipeline


logger = logging.getLogger(__name__)


class YOLOXDeploymentPipeline(Detection2DPipeline):
    """
    Abstract base class for YOLOX deployment pipeline.
    
    This class defines the complete inference flow for YOLOX, with:
    - Shared preprocessing (image resize, normalization)
    - Shared postprocessing (bbox decoding, NMS, coordinate transform)
    - Abstract method for backend-specific model inference
    
    The design eliminates code duplication by centralizing PyTorch processing
    while allowing ONNX/TensorRT backends to optimize the model inference.
    """
    
    def __init__(
        self, 
        model: Any,
        device: str = "cuda",
        num_classes: int = 8,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300,
        backend_type: str = "unknown"
    ):
        """
        Initialize YOLOX pipeline.
        
        Args:
            model: Model object (PyTorch model, ONNX session, TensorRT engine)
            device: Device for inference ('cuda' or 'cpu')
            num_classes: Number of object classes
            class_names: List of class names
            input_size: Model input size (height, width)
            score_threshold: Confidence threshold for filtering detections
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            backend_type: Backend type ('pytorch', 'onnx', 'tensorrt')
        """
        # Default class names for T4Dataset
        if class_names is None:
            class_names = [
                "unknown", "car", "truck", "bus", "trailer",
                "motorcycle", "pedestrian", "bicycle"
            ]
        
        # Initialize parent class
        super().__init__(
            model=model,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            backend_type=backend_type
        )
        
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    # ========== Preprocessing Methods ==========
    
    def preprocess(
        self, 
        input_data: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for YOLOX inference.
        
        This method expects preprocessed tensor from MMDetection pipeline.
        All preprocessing (resize, padding, normalization) should be done
        by MMDetection's test_pipeline before calling this method.
        
        Args:
            input_data: Preprocessed tensor [1, C, H, W] from MMDetection pipeline
            **kwargs: Additional preprocessing parameters (must include 'img_info' for metadata)
            
        Returns:
            Tuple of (preprocessed_tensor, preprocessing_metadata)
            - preprocessed_tensor: [1, C, H, W]
            - preprocessing_metadata: Dict with 'scale_factor', 'original_shape', 'input_shape'
            
        Raises:
            TypeError: If input_data is not a torch.Tensor
            ValueError: If img_info is missing from kwargs
        """
        if not isinstance(input_data, torch.Tensor):
            raise TypeError(
                f"YOLOX pipeline requires preprocessed tensor from MMDetection pipeline. "
                f"Got {type(input_data)}. Please use data_loader.preprocess() first."
            )
        
        # Extract metadata from kwargs
        metadata = {}
        
        # Get img_info from kwargs (required)
        img_info = kwargs.get('img_info', {})
        if not img_info:
            raise ValueError(
                "img_info is required for YOLOX preprocessing. "
                "Please provide img_info in kwargs when calling infer()."
            )
        
        metadata['scale_factor'] = img_info.get('scale_factor', [1.0, 1.0, 1.0, 1.0])
        metadata['original_shape'] = (img_info.get('height', 1080), img_info.get('width', 1440))
        
        # Tensor from MMDet pipeline is in [0, 255] range
        # YOLOX data_preprocessor does NOT do normalization (no mean/std configured)
        # So we use the tensor as-is to match training behavior
        tensor = input_data
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        metadata['input_shape'] = tuple(tensor.shape[2:])
        
        return tensor, metadata
    
    # ========== Abstract Method (Backend-specific) ==========
    
    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> np.ndarray:
        """
        Run YOLOX model inference (backend-specific).
        
        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized model inference.
        
        Args:
            preprocessed_input: Preprocessed image tensor [1, C, H, W]
            
        Returns:
            Model output [1, num_predictions, 4+1+num_classes]
            Format: [bbox_reg(4), objectness(1), class_scores(num_classes)]
        """
        pass
    
    # ========== Postprocessing Methods ==========
    
    def postprocess(
        self, 
        model_output: np.ndarray,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Postprocess YOLOX model output to final detections.
        
        Steps:
        1. Parse model output (bbox, objectness, class scores)
        2. Compute final scores (objectness * class_score)
        3. Filter by score threshold
        4. Apply NMS
        5. Transform coordinates back to original image space
        6. Limit to max_detections
        
        Args:
            model_output: Raw model output [1, num_predictions, 4+1+num_classes]
            metadata: Preprocessing metadata
            
        Returns:
            List of detections in format:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'score': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        if metadata is None:
            metadata = {}
        
        # Model output shape: [1, num_predictions, 4+1+num_classes]
        # Remove batch dimension
        predictions = model_output[0]  # [num_predictions, 4+1+num_classes]
        
        # Parse predictions (raw YOLOX head outputs)
        bbox_reg = predictions[:, :4]  # [dx, dy, dw, dh] (raw regression relative to stride)
        objectness = predictions[:, 4]  # [num_predictions] - objectness score (sigmoid)
        class_scores = predictions[:, 5:]  # [num_predictions, num_classes] (sigmoid)

        # Decode YOLOX bbox to absolute model-space coordinates using priors
        # Priors are generated on the model input grid with strides [8, 16, 32] and offset=0
        input_h, input_w = metadata.get('input_shape', self.input_size)
        strides = [8, 16, 32]
        priors = []
        for s in strides:
            fh = input_h // s
            fw = input_w // s
            # grid centers at offset 0
            ys, xs = np.meshgrid(np.arange(fh, dtype=np.float32), np.arange(fw, dtype=np.float32), indexing='ij')
            centers_x = (xs.reshape(-1) * s)
            centers_y = (ys.reshape(-1) * s)
            stride_w = np.full_like(centers_x, s, dtype=np.float32)
            stride_h = np.full_like(centers_y, s, dtype=np.float32)
            priors.append(np.stack([centers_x, centers_y, stride_w, stride_h], axis=1))
        priors = np.concatenate(priors, axis=0).astype(np.float32)  # [N, 4]

        # Align lengths if any mismatch (safety)
        num = min(len(bbox_reg), len(priors))
        bbox_reg = bbox_reg[:num]
        objectness = objectness[:num]
        class_scores = class_scores[:num]
        priors = priors[:num]

        # Decode centers and sizes
        center_x = bbox_reg[:, 0] * priors[:, 2] + priors[:, 0]
        center_y = bbox_reg[:, 1] * priors[:, 3] + priors[:, 1]
        width = np.exp(bbox_reg[:, 2]) * priors[:, 2]
        height = np.exp(bbox_reg[:, 3]) * priors[:, 3]

        # Convert to corner format [x1, y1, x2, y2] in model input space
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        bboxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Get max class score and class id for each detection
        max_class_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Compute final scores: objectness * max_class_score
        scores = objectness * max_class_scores
        
        
        # Filter by score threshold
        valid_mask = scores >= self.score_threshold
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        
        if len(scores) == 0:
            return []
        
        # Apply NMS
        keep_indices = self._apply_nms(bboxes, scores, class_ids)
        bboxes = bboxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        
        
        # Transform coordinates back to original image space
        # MMDetection pipeline provides scale_factor for coordinate transformation
        if 'scale_factor' in metadata:
            sf = metadata['scale_factor']
            if isinstance(sf, (list, tuple, np.ndarray)) and len(sf) >= 2:
                scale_x = float(sf[0])
                scale_y = float(sf[1])
                # Rescale from model space to original image space
                bboxes[:, 0] = bboxes[:, 0] / scale_x
                bboxes[:, 1] = bboxes[:, 1] / scale_y
                bboxes[:, 2] = bboxes[:, 2] / scale_x
                bboxes[:, 3] = bboxes[:, 3] / scale_y
            else:
                raise ValueError(
                    f"Invalid scale_factor format: {sf}. "
                    "Expected list/tuple/array with at least 2 elements."
                )
        else:
            raise ValueError(
                "scale_factor is required in metadata for coordinate transformation. "
                "Please ensure img_info contains scale_factor when calling infer()."
            )
        
        # Limit to max detections
        if len(scores) > self.max_detections:
            top_indices = np.argsort(scores)[::-1][:self.max_detections]
            bboxes = bboxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]
        
        # Format results
        results = []
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            results.append({
                'bbox': bbox.tolist(),
                'score': float(score),
                'class_id': int(class_id),
                'class_name': self.class_names[int(class_id)]
            })
        
        
        return results
    
    def _apply_nms(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray
    ) -> np.ndarray:
        """
        Apply batched NMS (per-class NMS).
        
        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            
        Returns:
            Indices of boxes to keep
        """
        try:
            # Try to use MMDetection's batched NMS for consistency
            from mmcv.ops import batched_nms
            
            bboxes_tensor = torch.from_numpy(bboxes).float()
            scores_tensor = torch.from_numpy(scores).float()
            class_ids_tensor = torch.from_numpy(class_ids).long()
            
            nms_cfg = dict(type="nms", iou_threshold=self.nms_threshold)
            
            _, keep_indices = batched_nms(
                bboxes_tensor, scores_tensor, class_ids_tensor, nms_cfg
            )
            
            return keep_indices.cpu().numpy()
            
        except ImportError:
            logger.warning("mmcv.ops.batched_nms not available, using per-class NMS")
            # Fallback to per-class NMS
            return self._per_class_nms(bboxes, scores, class_ids)
    
    def _per_class_nms(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray
    ) -> np.ndarray:
        """
        Apply per-class NMS (fallback implementation).
        
        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            
        Returns:
            Indices of boxes to keep
        """
        keep_indices = []
        
        # Apply NMS per class
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_bboxes = bboxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]
            
            # Apply NMS for this class
            keep = self._nms(class_bboxes, class_scores, self.nms_threshold)
            keep_indices.extend(class_indices[keep])
        
        return np.array(keep_indices)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"device={self.device}, "
                f"input_size={self.input_size}, "
                f"num_classes={self.num_classes}, "
                f"backend={self.backend_type})")

