"""
2D Object Detection Pipeline Base Class.

This module provides the base class for 2D object detection pipelines,
implementing common preprocessing and postprocessing for models like YOLOX, YOLO, etc.
"""

from abc import abstractmethod
from typing import List, Dict, Tuple, Any
import logging

import cv2
import numpy as np
import torch

from .base_pipeline import BaseDeploymentPipeline


logger = logging.getLogger(__name__)


class Detection2DPipeline(BaseDeploymentPipeline):
    """
    Base class for 2D object detection pipelines.
    
    Provides common functionality for 2D detection tasks including:
    - Image preprocessing (resize, normalize, padding)
    - Postprocessing (NMS, coordinate transformation)
    - Standard detection output format
    
    Expected output format:
        List[Dict] where each dict contains:
        {
            'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
            'score': float,             # Confidence score
            'class_id': int,            # Class ID
            'class_name': str           # Class name (optional)
        }
    """
    
    def __init__(
        self, 
        model: Any,
        device: str = "cpu",
        num_classes: int = 80,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        backend_type: str = "unknown"
    ):
        """
        Initialize 2D detection pipeline.
        
        Args:
            model: Model object
            device: Device for inference
            num_classes: Number of classes
            class_names: List of class names
            input_size: Model input size (height, width)
            backend_type: Backend type
        """
        super().__init__(model, device, task_type="detection_2d", backend_type=backend_type)
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.input_size = input_size
        
        # Normalization parameters (can be overridden)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess(
        self, 
        image: np.ndarray,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Standard 2D detection preprocessing.
        
        Steps:
        1. Resize image to model input size (with padding to maintain aspect ratio)
        2. Normalize pixel values
        3. Convert to tensor and add batch dimension
        
        Args:
            image: Input image [H, W, C] in BGR format
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Tuple of (preprocessed_tensor, preprocessing_metadata)
            - preprocessed_tensor: [1, C, H, W]
            - preprocessing_metadata: Dict with 'scale', 'pad', 'original_shape'
        """
        original_shape = image.shape[:2]  # (H, W)
        
        # Resize with padding to maintain aspect ratio
        resized_image, scale, pad = self._resize_with_pad(image, self.input_size)
        
        # Normalize
        normalized_image = self._normalize(resized_image)
        
        # Convert to tensor
        # OpenCV uses BGR, convert to RGB
        rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()  # [C, H, W]
        tensor = tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        # Store preprocessing metadata for postprocessing
        metadata = {
            'scale': scale,
            'pad': pad,
            'original_shape': original_shape,
            'input_shape': self.input_size
        }
        
        logger.debug(f"Preprocessed image: {original_shape} → {self.input_size}, scale={scale:.3f}")
        
        return tensor, metadata
    
    def _resize_with_pad(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with padding to maintain aspect ratio.
        
        Args:
            image: Input image [H, W, C]
            target_size: Target size (height, width)
            
        Returns:
            Tuple of (resized_image, scale, pad)
            - resized_image: Padded and resized image
            - scale: Scaling factor applied
            - pad: Padding (pad_h, pad_w)
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit image into target size
        scale = min(target_h / h, target_w / w)
        
        # Calculate new size
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded, scale, (pad_h, pad_w)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixels.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Normalized image in range [0, 1] or standardized
        """
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Optionally apply mean/std normalization
        # normalized = (normalized - self.mean) / self.std
        
        return normalized
    
    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> Any:
        """
        Run detection model (backend-specific).
        
        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]
            
        Returns:
            Model output (backend-specific format)
        """
        pass
    
    def postprocess(
        self, 
        model_output: Any,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Standard 2D detection postprocessing.
        
        Steps:
        1. Parse model outputs (boxes, scores, classes)
        2. Apply NMS
        3. Transform coordinates back to original image space
        4. Filter by confidence threshold
        
        Args:
            model_output: Raw model output
            metadata: Preprocessing metadata
            
        Returns:
            List of detections in standard format
        """
        # This should be overridden by specific detectors (YOLOX, YOLO, etc.)
        # as output formats differ
        raise NotImplementedError(
            "postprocess() must be implemented by specific detector pipeline"
        )
    
    def _transform_coordinates(
        self,
        boxes: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Transform bounding box coordinates from model space to original image space.
        
        Args:
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            scale: Scale factor used in preprocessing
            pad: Padding (pad_h, pad_w)
            original_shape: Original image shape (H, W)
            
        Returns:
            Transformed boxes in original image coordinates
        """
        pad_h, pad_w = pad
        
        # Remove padding offset
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        
        # Scale back to original size
        boxes /= scale
        
        # Clip to image boundaries
        h, w = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        
        return boxes
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.45
    ) -> np.ndarray:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)

