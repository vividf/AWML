"""
Classification Pipeline Base Class.

This module provides the base class for classification pipelines,
implementing common preprocessing and postprocessing for image/point cloud classification.
"""

from abc import abstractmethod
from typing import List, Dict, Tuple, Any
import logging

import cv2
import numpy as np
import torch

from .base_pipeline import BaseDeploymentPipeline


logger = logging.getLogger(__name__)


class ClassificationPipeline(BaseDeploymentPipeline):
    """
    Base class for classification pipelines.
    
    Provides common functionality for classification tasks including:
    - Image/data preprocessing (resize, normalize)
    - Postprocessing (softmax, top-k selection)
    - Standard classification output format
    
    Expected output format:
        Dict containing:
        {
            'class_id': int,           # Predicted class ID
            'class_name': str,         # Class name
            'confidence': float,       # Confidence score
            'probabilities': np.ndarray,  # All class probabilities
            'top_k': List[Dict]        # Top-K predictions (optional)
        }
    """
    
    def __init__(
        self, 
        model: Any,
        device: str = "cpu",
        num_classes: int = 1000,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        backend_type: str = "unknown"
    ):
        """
        Initialize classification pipeline.
        
        Args:
            model: Model object
            device: Device for inference
            num_classes: Number of classes
            class_names: List of class names
            input_size: Model input size (height, width)
            backend_type: Backend type
        """
        super().__init__(model, device, task_type="classification", backend_type=backend_type)
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.input_size = input_size
        
        # ImageNet normalization (can be overridden)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess(
        self, 
        image: np.ndarray,
        **kwargs
    ) -> torch.Tensor:
        """
        Standard classification preprocessing.
        
        Steps:
        1. Resize image to model input size
        2. Normalize pixel values
        3. Convert to tensor and add batch dimension
        
        Args:
            image: Input image [H, W, C] in BGR format
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Resize
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply mean/std normalization
        normalized = (normalized - self.mean) / self.std
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        
        # Add batch dimension [1, C, H, W]
        tensor = tensor.unsqueeze(0).to(self.device)
        
        logger.debug(f"Preprocessed image to shape: {tensor.shape}")
        
        return tensor
    
    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run classification model (backend-specific).
        
        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]
            
        Returns:
            Model output (logits) [1, num_classes]
        """
        pass
    
    def postprocess(
        self, 
        model_output: torch.Tensor,
        metadata: Dict = None,
        top_k: int = 5
    ) -> Dict:
        """
        Standard classification postprocessing.
        
        Steps:
        1. Apply softmax to get probabilities
        2. Get predicted class
        3. Optionally get top-K predictions
        
        Args:
            model_output: Model output (logits) [1, num_classes]
            metadata: Additional metadata (unused for classification)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with classification results
        """
        # Convert to numpy if needed
        if isinstance(model_output, torch.Tensor):
            logits = model_output.cpu().numpy()
        else:
            logits = model_output
        
        # Remove batch dimension if present
        if logits.ndim == 2:
            logits = logits[0]
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get predicted class
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
        
        # Get top-K predictions
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        top_k_predictions = []
        for idx in top_k_indices:
            top_k_predictions.append({
                'class_id': int(idx),
                'class_name': self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                'confidence': float(probabilities[idx])
            })
        
        result = {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'top_k': top_k_predictions
        }
        
        logger.debug(f"Predicted class: {class_name} ({confidence:.3f})")
        
        return result

