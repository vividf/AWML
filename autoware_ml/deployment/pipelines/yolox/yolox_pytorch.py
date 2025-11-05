"""
YOLOX PyTorch Pipeline Implementation.

This module provides the PyTorch backend implementation for YOLOX deployment.
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from .yolox_pipeline import YOLOXDeploymentPipeline


logger = logging.getLogger(__name__)


class YOLOXPyTorchPipeline(YOLOXDeploymentPipeline):
    """
    YOLOX PyTorch backend implementation.
    
    This pipeline uses PyTorch for end-to-end inference with the original model.
    Useful for baseline comparison and verification.
    """
    
    def __init__(
        self, 
        pytorch_model,
        device: str = "cuda",
        num_classes: int = 8,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300
    ):
        """
        Initialize YOLOX PyTorch pipeline.
        
        Args:
            pytorch_model: PyTorch YOLOX model
            device: Device for inference
            num_classes: Number of object classes
            class_names: List of class names
            input_size: Model input size (height, width)
            score_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
        """
        super().__init__(
            model=pytorch_model,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            backend_type="pytorch"
        )
        
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        self.pytorch_model.to(self.device)
        
        logger.info(f"Initialized YOLOXPyTorchPipeline on {self.device}")
    
    def run_model(self, preprocessed_input: torch.Tensor) -> np.ndarray:
        """
        Run PyTorch model inference.
        
        Args:
            preprocessed_input: Preprocessed image tensor [1, C, H, W] in range [0, 255], BGR format
            
        Returns:
            Model output [1, num_predictions, 4+1+num_classes]
            Format: [bbox(4), objectness(1), class_scores(num_classes)]
        """
        with torch.no_grad():
            # Input is 0-255 range, BGR format (no normalization)
            # Extract features directly (matching ONNX/TRT behavior)
            feat = self.pytorch_model.extract_feat(preprocessed_input)
            
            # Get head outputs: (cls_scores, bbox_preds, objectnesses)
            cls_scores, bbox_preds, objectnesses = self.pytorch_model.bbox_head(feat)
            
            # Process each detection level (matching ONNX wrapper logic)
            outputs = []
            
            for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
                # Apply sigmoid to objectness and cls_score (NOT to bbox_pred)
                output = torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1)
                outputs.append(output)
            
            # Flatten and concatenate all levels
            batch_size = outputs[0].shape[0]
            num_channels = outputs[0].shape[1]
            outputs = torch.cat(
                [x.reshape(batch_size, num_channels, -1) for x in outputs], 
                dim=2
            ).permute(0, 2, 1)
            
            # outputs shape: [batch_size, num_predictions, 4+1+num_classes]
        
        # Convert to numpy
        output_np = outputs.cpu().numpy()
        
        logger.debug(f"PyTorch inference output shape: {output_np.shape}")
        
        return output_np

