"""
CalibrationStatusClassification Deployment Pipeline Base Class.

This module provides the abstract base class for CalibrationStatusClassification deployment,
defining the unified pipeline that shares preprocessing and postprocessing logic
while allowing backend-specific optimizations for model inference.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch

from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class CalibrationDeploymentPipeline(BaseDeploymentPipeline):
    """
    Abstract base class for CalibrationStatusClassification deployment pipeline.

    This class defines the complete inference flow for CalibrationStatusClassification, with:
    - Shared preprocessing (via CalibrationClassificationTransform)
    - Shared postprocessing (softmax, classification)
    - Abstract method for backend-specific model inference

    The design eliminates code duplication by centralizing processing
    while allowing ONNX/TensorRT backends to optimize the model inference.
    """

    # Label mapping for calibration status
    LABELS = {0: "miscalibrated", 1: "calibrated"}

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        num_classes: int = 2,
        class_names: List[str] = None,
        backend_type: str = "unknown",
    ):
        """
        Initialize CalibrationStatusClassification pipeline.

        Args:
            model: Model object (PyTorch model, ONNX session, TensorRT engine)
            device: Device for inference ('cuda' or 'cpu')
            num_classes: Number of classes (2 for binary classification)
            class_names: List of class names (default: ["miscalibrated", "calibrated"])
            backend_type: Backend type ('pytorch', 'onnx', 'tensorrt')
        """
        super().__init__(
            model=model,
            device=device,
            task_type="classification",
            backend_type=backend_type,
        )

        # Default class names for calibration status
        if class_names is None:
            class_names = ["miscalibrated", "calibrated"]

        self.num_classes = num_classes
        self.class_names = class_names

    def preprocess(self, input_data: Any, **kwargs) -> torch.Tensor:
        """
        Preprocess input for CalibrationStatusClassification inference.

        Supports two input types:
        1. Raw sample dict: Apply CalibrationClassificationTransform
        2. Preprocessed tensor (torch.Tensor): Skip preprocessing, use as-is

        Args:
            input_data: Input sample dict OR preprocessed tensor [1, C, H, W]
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Check if input is already a preprocessed tensor
        if isinstance(input_data, torch.Tensor):
            # Already preprocessed - use as-is
            if input_data.dtype != torch.float32:
                input_data = input_data.float()
            return input_data.to(self.device)

        # Raw sample dict - should be preprocessed by CalibrationDataLoader
        # This method expects preprocessed tensor
        if isinstance(input_data, dict):
            raise ValueError(
                "Raw sample dict should be preprocessed by CalibrationDataLoader first. "
                "Please call data_loader.preprocess(sample) before passing to pipeline."
            )

        raise ValueError(f"Unsupported input type: {type(input_data)}")

    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run CalibrationStatusClassification model inference (backend-specific).

        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized model inference.

        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]

        Returns:
            Model output (logits) [1, num_classes]
        """
        raise NotImplementedError

    def postprocess(self, model_output: torch.Tensor, metadata: Dict = None, top_k: int = 2) -> Dict:
        """
        Postprocess CalibrationStatusClassification model output.

        Steps:
        1. Apply softmax to get probabilities
        2. Get predicted class
        3. Return classification results

        Args:
            model_output: Model output (logits) [1, num_classes]
            metadata: Additional metadata (unused)
            top_k: Number of top predictions to return (default: 2 for binary classification)

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
            top_k_predictions.append(
                {
                    "class_id": int(idx),
                    "class_name": self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                    "confidence": float(probabilities[idx]),
                }
            )

        result = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_k": top_k_predictions,
        }

        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"num_classes={self.num_classes}, "
            f"backend={self.backend_type})"
        )
