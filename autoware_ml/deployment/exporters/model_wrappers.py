"""
Model wrappers for ONNX export.

This module provides wrapper classes that prepare models for ONNX export
with specific output formats and processing requirements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseModelWrapper(nn.Module, ABC):
    """
    Abstract base class for ONNX export model wrappers.
    
    Wrappers modify model forward pass to produce ONNX-compatible outputs
    with specific formats required by deployment backends.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize wrapper.
        
        Args:
            model: PyTorch model to wrap
            **kwargs: Wrapper-specific arguments
        """
        super().__init__()
        self.model = model
        self._wrapper_config = kwargs
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for ONNX export.
        
        Must be implemented by subclasses to define ONNX-specific output format.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get wrapper configuration."""
        return self._wrapper_config


class YOLOXONNXWrapper(BaseModelWrapper):
    """
    Wrapper for YOLOX model to match Tier4 ONNX export format.
    
    The output format matches Tier4 YOLOX exactly:
    - Shape: [batch_size, total_anchors, 4 + 1 + num_classes]
    - Content: [bbox_predictions(4), objectness(1), class_scores(num_classes)]
    - objectness and class_scores are passed through sigmoid
    - bbox_predictions are raw regression outputs (NOT decoded)
    
    Args:
        model: MMDetection YOLOX model
        num_classes: Number of object classes. If not provided, will be automatically
                     extracted from model.bbox_head.num_classes
    """
    
    def __init__(self, model: nn.Module, num_classes: int = None, **kwargs):
        # Auto-extract num_classes from model if not provided
        if num_classes is None:
            if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'num_classes'):
                num_classes = model.bbox_head.num_classes
            else:
                raise ValueError(
                    "num_classes must be provided or model must have bbox_head.num_classes attribute"
                )
        
        super().__init__(model, num_classes=num_classes, **kwargs)
        self.bbox_head = model.bbox_head
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching Tier4 YOLOX format.
        
        Args:
            x: Input tensor [batch_size, 3, H, W] in range [0, 255], BGR format
                Note: YOLOX data_preprocessor does NOT do normalization (no mean/std configured)
                So input is used as-is (0-255 range, BGR format) to match training behavior
        
        Returns:
            Concatenated predictions [batch_size, num_predictions, 4+1+num_classes]
            Format: [bbox_reg(4), objectness(1), class_scores(num_classes)]
            - bbox_reg: raw regression outputs (NOT decoded)
            - objectness: sigmoid activated
            - class_scores: sigmoid activated
        """
        # Extract features (input is 0-255 range, BGR format, no normalization)
        # This matches YOLOX training behavior where data_preprocessor does NOT normalize
        feat = self.model.extract_feat(x)
        
        # Get head outputs: (cls_scores, bbox_preds, objectnesses)
        cls_scores, bbox_preds, objectnesses = self.bbox_head(feat)
        
        # Process each detection level (matching Tier4 YOLOX logic)
        outputs = []
        
        for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
            # Apply sigmoid to objectness and cls_score (NOT to bbox_pred)
            # This matches Tier4 YOLOX yolo_head.py line 198-200
            output = torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1)
            outputs.append(output)
        
        # Flatten and concatenate all levels
        # Use static reshape to avoid Shape/Gather/Unsqueeze operations in ONNX
        # This matches Tier4 YOLOX yolo_head.py line 218-220
        batch_size = outputs[0].shape[0]
        num_channels = outputs[0].shape[1]
        outputs = torch.cat(
            [x.reshape(batch_size, num_channels, -1) for x in outputs], 
            dim=2
        ).permute(0, 2, 1)
        
        return outputs


class IdentityWrapper(BaseModelWrapper):
    """
    Identity wrapper that doesn't modify the model.
    
    Useful for models that don't need special ONNX export handling.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass without modification."""
        return self.model(*args, **kwargs)


# Model wrapper registry
_MODEL_WRAPPERS = {
    'yolox': YOLOXONNXWrapper,
    'identity': IdentityWrapper,
}


def register_model_wrapper(name: str, wrapper_class: type):
    """
    Register a custom model wrapper.
    
    Args:
        name: Wrapper name
        wrapper_class: Wrapper class (must inherit from BaseModelWrapper)
    """
    if not issubclass(wrapper_class, BaseModelWrapper):
        raise TypeError(f"Wrapper class must inherit from BaseModelWrapper, got {wrapper_class}")
    _MODEL_WRAPPERS[name] = wrapper_class


def get_model_wrapper(name: str):
    """
    Get model wrapper class by name.
    
    Args:
        name: Wrapper name
        
    Returns:
        Wrapper class
        
    Raises:
        KeyError: If wrapper name not found
    """
    if name not in _MODEL_WRAPPERS:
        raise KeyError(
            f"Model wrapper '{name}' not found. "
            f"Available wrappers: {list(_MODEL_WRAPPERS.keys())}"
        )
    return _MODEL_WRAPPERS[name]


def list_model_wrappers():
    """List all registered model wrappers."""
    return list(_MODEL_WRAPPERS.keys())

