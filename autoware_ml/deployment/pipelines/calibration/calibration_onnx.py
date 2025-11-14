"""
CalibrationStatusClassification ONNX Pipeline Implementation.

This module provides the ONNX Runtime backend implementation for CalibrationStatusClassification deployment.
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.pipelines.calibration.calibration_pipeline import CalibrationDeploymentPipeline


logger = logging.getLogger(__name__)


class CalibrationONNXPipeline(CalibrationDeploymentPipeline):
    """
    CalibrationStatusClassification ONNX Runtime backend implementation.
    
    This pipeline uses ONNX Runtime for optimized inference.
    Supports both CPU and GPU execution providers.
    """
    
    def __init__(
        self, 
        onnx_path: str,
        device: str = "cuda",
        num_classes: int = 2,
        class_names: List[str] = None,
    ):
        """
        Initialize CalibrationStatusClassification ONNX pipeline.
        
        Args:
            onnx_path: Path to ONNX model file
            device: Device for inference ('cuda' or 'cpu')
            num_classes: Number of classes (2 for binary classification)
            class_names: List of class names (default: ["miscalibrated", "calibrated"])
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX pipeline. "
                "Install it with: pip install onnxruntime or onnxruntime-gpu"
            )
        
        # Setup execution providers
        if device.startswith("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Initialize parent with dummy model (we use session instead)
        super().__init__(
            model=self.session,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            backend_type="onnx"
        )
        
        logger.info(f"Initialized CalibrationONNXPipeline on {device}")
        logger.info(f"  ONNX model: {onnx_path}")
        logger.info(f"  Input: {self.input_name}, Outputs: {self.output_names}")
        logger.info(f"  Providers: {self.session.get_providers()}")
    
    def run_model(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run ONNX model inference.
        
        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]
            
        Returns:
            Model output (logits) [1, num_classes]
        """
        # Convert to numpy
        input_array = preprocessed_input.cpu().numpy().astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: input_array})
        
        # Convert back to torch tensor
        output = torch.from_numpy(outputs[0]).to(self.device)
        
        return output

