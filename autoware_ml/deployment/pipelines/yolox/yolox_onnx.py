"""
YOLOX ONNX Pipeline Implementation.

This module provides the ONNX Runtime backend implementation for YOLOX deployment.
"""

from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np

from autoware_ml.deployment.pipelines.yolox.yolox_pipeline import YOLOXDeploymentPipeline


logger = logging.getLogger(__name__)


class YOLOXONNXPipeline(YOLOXDeploymentPipeline):
    """
    YOLOX ONNX Runtime backend implementation.
    
    This pipeline uses ONNX Runtime for optimized inference.
    Supports both CPU and GPU execution providers.
    """
    
    def __init__(
        self, 
        onnx_path: str,
        device: str = "cuda",
        num_classes: int = 8,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300
    ):
        """
        Initialize YOLOX ONNX pipeline.
        
        Args:
            onnx_path: Path to ONNX model file
            device: Device for inference ('cuda' or 'cpu')
            num_classes: Number of object classes
            class_names: List of class names
            input_size: Model input size (height, width)
            score_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
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
        
        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        logger.info(f"Loaded ONNX model from: {onnx_path}")
        logger.info(f"Execution providers: {self.ort_session.get_providers()}")
        logger.info(f"Input name: {self.input_name}")
        logger.info(f"Output name: {self.output_name}")
        
        # Initialize parent class (pass ort_session as model)
        super().__init__(
            model=self.ort_session,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            backend_type="onnx"
        )
    
    def run_model(self, preprocessed_input: torch.Tensor) -> np.ndarray:
        """
        Run ONNX model inference.
        
        Args:
            preprocessed_input: Preprocessed image tensor [1, C, H, W]
            
        Returns:
            Model output [1, num_predictions, 4+1+num_classes]
            Format: [bbox(4), objectness(1), class_scores(num_classes)]
        """
        # Convert torch tensor to numpy
        input_np = preprocessed_input.cpu().numpy()
        
        # Run inference
        outputs = self.ort_session.run(
            [self.output_name],
            {self.input_name: input_np}
        )
        
        # outputs[0] shape: [batch_size, num_predictions, 4+1+num_classes]
        output_np = outputs[0]
        
        
        return output_np

