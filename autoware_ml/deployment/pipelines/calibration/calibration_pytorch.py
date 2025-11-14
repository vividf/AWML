"""
CalibrationStatusClassification PyTorch Pipeline Implementation.

This module provides the PyTorch backend implementation for CalibrationStatusClassification deployment.
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.pipelines.calibration.calibration_pipeline import CalibrationDeploymentPipeline


logger = logging.getLogger(__name__)


class CalibrationPyTorchPipeline(CalibrationDeploymentPipeline):
    """
    CalibrationStatusClassification PyTorch backend implementation.
    
    This pipeline uses PyTorch for end-to-end inference with the original model.
    Useful for baseline comparison and verification.
    """
    
    def __init__(
        self, 
        pytorch_model,
        device: str = "cuda",
        num_classes: int = 2,
        class_names: List[str] = None,
    ):
        """
        Initialize CalibrationStatusClassification PyTorch pipeline.
        
        Args:
            pytorch_model: PyTorch model
            device: Device for inference
            num_classes: Number of classes (2 for binary classification)
            class_names: List of class names (default: ["miscalibrated", "calibrated"])
        """
        super().__init__(
            model=pytorch_model,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            backend_type="pytorch"
        )
        
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        self.pytorch_model.to(self.device)
        
        logger.info(f"Initialized CalibrationPyTorchPipeline on {self.device}")
    
    def run_model(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run PyTorch model inference.
        
        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]
            
        Returns:
            Model output (logits) [1, num_classes]
        """
        with torch.no_grad():
            output = self.pytorch_model(preprocessed_input)
        
        return output

