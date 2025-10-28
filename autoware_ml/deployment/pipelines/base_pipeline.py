"""
Base Deployment Pipeline for Unified Model Deployment.

This module provides the abstract base class for all deployment pipelines,
defining a unified interface across different backends (PyTorch, ONNX, TensorRT)
and task types (detection, classification, segmentation).

Architecture:
    Input → preprocess() → run_model() → postprocess() → Output

Key Design Principles:
    1. Shared Logic: preprocess/postprocess are shared across backends
    2. Backend-Specific: run_model() is implemented per backend
    3. Unified Interface: infer() provides consistent API
    4. Flexible Output: Can return raw or processed outputs
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, List
import logging
import time

import torch


logger = logging.getLogger(__name__)


class BaseDeploymentPipeline(ABC):
    """
    Abstract base class for all deployment pipelines.
    
    This class defines the unified interface for model deployment across
    different backends and task types.
    
    Attributes:
        model: Model object (PyTorch model, ONNX session, TensorRT engine, etc.)
        device: Device for inference
        task_type: Type of task ("detection_2d", "detection_3d", "classification", etc.)
        backend_type: Type of backend ("pytorch", "onnx", "tensorrt", etc.)
    """
    
    def __init__(
        self, 
        model: Any,
        device: str = "cpu",
        task_type: str = "unknown",
        backend_type: str = "unknown"
    ):
        """
        Initialize deployment pipeline.
        
        Args:
            model: Model object (backend-specific)
            device: Device for inference ('cpu', 'cuda', 'cuda:0', etc.)
            task_type: Type of task
            backend_type: Type of backend
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.task_type = task_type
        self.backend_type = backend_type
        
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
        
    # ========== Abstract Methods (Must Implement) ==========
    
    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs) -> Any:
        """
        Preprocess input data.
        
        This method should handle all preprocessing steps required before
        feeding data to the model (normalization, resizing, etc.).
        
        Args:
            input_data: Raw input (image, point cloud, etc.)
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed data ready for model
        """
        pass
    
    @abstractmethod
    def run_model(self, preprocessed_input: Any) -> Union[Any, Tuple]:
        """
        Run model inference (backend-specific).
        
        This is the only method that differs across backends.
        Each backend (PyTorch, ONNX, TensorRT) implements its own version.
        
        Args:
            preprocessed_input: Preprocessed input data
            
        Returns:
            Model output (raw tensors or backend-specific format)
        """
        pass
    
    @abstractmethod
    def postprocess(
        self, 
        model_output: Any,
        metadata: Dict = None
    ) -> Any:
        """
        Postprocess model output to final predictions.
        
        This method should handle all postprocessing steps like NMS,
        coordinate transformation, score filtering, etc.
        
        Args:
            model_output: Raw model output from run_model()
            metadata: Additional metadata (image size, point cloud range, etc.)
            
        Returns:
            Final predictions in standard format
        """
        pass
    
    # ========== Concrete Methods (Shared Logic) ==========
    
    def infer(
        self, 
        input_data: Any,
        metadata: Dict = None,
        return_raw_outputs: bool = False,
        **kwargs
    ) -> Tuple[Any, float]:
        """
        Complete inference pipeline.
        
        This method orchestrates the entire inference flow:
        1. Preprocessing
        2. Model inference
        3. Postprocessing (optional)
        
        This unified interface allows:
        - Evaluation: infer(..., return_raw_outputs=False) → get final predictions
        - Verification: infer(..., return_raw_outputs=True) → get raw outputs for comparison
        
        Args:
            input_data: Raw input data
            metadata: Additional metadata for preprocessing/postprocessing
            return_raw_outputs: If True, skip postprocessing (for verification)
            **kwargs: Additional arguments passed to preprocess()
            
        Returns:
            Tuple of (outputs, latency_ms)
            - If return_raw_outputs=True: (raw_model_output, latency)
            - If return_raw_outputs=False: (final_predictions, latency)
        """
        
        
        if metadata is None:
            metadata = {}
        
        try:
            # 1. Preprocess
            preprocessed = self.preprocess(input_data, **kwargs)
            
            start_time = time.time()
            # 2. Run model (backend-specific)
            model_output = self.run_model(preprocessed)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 3. Postprocess (optional)
            if return_raw_outputs:
                print(f"Inference completed in {latency_ms:.2f}ms (returning raw outputs)")
                return model_output, latency_ms
            else:
                predictions = self.postprocess(model_output, metadata)
                print(f"Inference completed in {latency_ms:.2f}ms (returning predictions)")
                return predictions, latency_ms
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def warmup(self, input_data: Any, num_iterations: int = 10):
        """
        Warmup the model with dummy inputs.
        
        Useful for stabilizing latency measurements, especially for GPU models.
        
        Args:
            input_data: Sample input for warmup
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up {self.__class__.__name__} with {num_iterations} iterations...")
        
        for i in range(num_iterations):
            try:
                self.infer(input_data)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        logger.info("Warmup completed")
    
    def benchmark(
        self, 
        input_data: Any, 
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_data: Sample input for benchmarking
            num_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with latency statistics (mean, std, min, max)
        """
        logger.info(f"Benchmarking {self.__class__.__name__} with {num_iterations} iterations...")
        
        # Warmup first
        self.warmup(input_data, num_iterations=10)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            _, latency = self.infer(input_data)
            latencies.append(latency)
        
        import numpy as np
        results = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'median_ms': np.median(latencies)
        }
        
        logger.info(f"Benchmark results: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        
        return results
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"device={self.device}, "
                f"task={self.task_type}, "
                f"backend={self.backend_type})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        pass

