"""PyTorch inference backend."""

import logging
import time
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base_backend import BaseBackend


class PyTorchBackend(BaseBackend):
    """
    PyTorch inference backend using standard MMDetection pipeline.

    This backend follows the same inference flow as test.py, ensuring
    consistency with MMDetection's standard evaluation pipeline.
    """

    def __init__(self, model: Union[torch.nn.Module, str], model_cfg: Optional[object] = None, device: str = "cpu", raw_output: bool = False, original_image_size: Optional[Tuple[int, int]] = None, scale_factor_config: Optional[dict] = None):
        """
        Initialize PyTorch backend.

        Args:
            model: PyTorch model instance or path to checkpoint
            model_cfg: Model configuration (required if model is a path)
            device: Device to run inference on
            raw_output: If True, return raw head outputs (for ONNX verification)
                       If False, return post-processed results (standard MMDetection)
            original_image_size: Original image size (height, width) for scale factor calculation
                                Defaults to (1080, 1440) for T4 dataset
            scale_factor_config: Configuration for scale factor calculation
                                Default: {'keep_ratio': True, 'method': 'min'}
        """
        super().__init__(model_path="", device=device)
        self._torch_device = torch.device(device)
        self.model_cfg = model_cfg
        self._logger = logging.getLogger(__name__)
        self.raw_output = raw_output
        self.original_image_size = original_image_size or (1080, 1440)  # Default to T4 dataset size
        self.scale_factor_config = scale_factor_config or {
            'keep_ratio': True,
            'method': 'min'  # 'min', 'max', 'average'
        }

        if isinstance(model, str):
            # Model path provided - load it
            self.model_path = model
            self._model = None  # Will be loaded in load_model()
            self.load_model()
        else:
            # Model instance provided
            self._model = model
            self._model.to(self._torch_device)
            self._model.eval()
        
        # Cache model type for performance
        self._model_type = self._determine_model_type()

    def load_model(self) -> None:
        """Load model from checkpoint if not already loaded."""
        if self._model is not None:
            return  # Already loaded

        if not self.model_path:
            raise RuntimeError("Model path not provided")

        if self.model_cfg is None:
            raise RuntimeError("Model config required when loading from checkpoint")

        # Load model using MMDet API
        from mmdet.apis import init_detector

        self._model = init_detector(self.model_cfg, self.model_path, device=self.device)
        self._model.eval()
        
        # Update cached model type after loading
        self._model_type = self._determine_model_type()

    def _determine_model_type(self) -> str:
        """Determine model type once during initialization."""
        if self._model is None:
            return 'unknown'
        
        # Check for detection model attributes
        if (hasattr(self._model, 'bbox_head') and 
            hasattr(self._model, 'extract_feat')):
            return 'detector'
        
        # Check for classification model attributes
        if (hasattr(self._model, 'head') and 
            hasattr(self._model, 'extract_feat')):
            return 'classifier'
        
        return 'unknown'

    def _calculate_scale_factor(self, model_input_h: int, model_input_w: int, ori_h: int, ori_w: int) -> Tuple[float, float]:
        """
        Calculate scale factor based on configuration.
        
        Args:
            model_input_h: Model input height
            model_input_w: Model input width
            ori_h: Original image height
            ori_w: Original image width
            
        Returns:
            Tuple of (scale_h, scale_w)
        """
        scale_w = model_input_w / ori_w
        scale_h = model_input_h / ori_h
        
        if self.scale_factor_config['keep_ratio']:
            method = self.scale_factor_config['method']
            if method == 'min':
                scale = min(scale_w, scale_h)
            elif method == 'max':
                scale = max(scale_w, scale_h)
            elif method == 'average':
                scale = (scale_w + scale_h) / 2
            else:
                self._logger.warning(f"Unknown scale method '{method}', using 'min'")
                scale = min(scale_w, scale_h)
            return (scale, scale)
        else:
            return (scale_h, scale_w)

    def _create_base_data_sample(self, input_tensor: torch.Tensor, batch_idx: int):
        """
        Create base data sample with common metainfo.
        
        Args:
            input_tensor: Input tensor
            batch_idx: Batch index
            
        Returns:
            Base data sample with common metainfo
        """
        from mmpretrain.structures import DataSample
        
        data_sample = DataSample()
        data_sample.set_metainfo({
            'img_shape': input_tensor.shape[2:],  # Model input size
            'batch_idx': batch_idx
        })
        return data_sample

    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor for inference

        Returns:
            Tuple of (output_array, latency_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Move input to correct device
        input_tensor = input_tensor.to(self._torch_device)

        # Run inference with timing
        with torch.no_grad():
            start_time = time.perf_counter()
            
            if self.raw_output:
                # Raw output mode - for ONNX verification
                output_array = self._get_raw_output(input_tensor)
            else:
                # Standard pipeline - for evaluation
                if self._model_type == 'detector':
                    # Detection model - use MMDetection pipeline
                    batch_data_samples = self._create_dummy_data_samples(input_tensor)
                    results = self._model.predict(input_tensor, batch_data_samples, rescale=True)
                    output_array = self._extract_predictions_from_results(results)
                elif self._model_type == 'classifier':
                    # Classification model - use MMPretrain pipeline
                    batch_data_samples = self._create_dummy_data_samples(input_tensor)
                    results = self._model.predict(input_tensor, batch_data_samples)
                    output_array = self._extract_predictions_from_results(results)
                else:
                    raise ValueError(f"Unsupported model type: {self._model_type}. "
                                   f"Model class: {type(self._model)}")
            
            end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        
        return output_array, latency_ms

    def _create_dummy_data_samples(self, input_tensor: torch.Tensor):
        """Create dummy data samples for model pipeline."""
        batch_size = input_tensor.shape[0]
        
        if self._model_type == 'detector':
            # Detection model - use MMDetection DetDataSample
            from mmdet.structures import DetDataSample
            
            data_samples = []
            
            # Get model input size (typically 960x960 for YOLOX)
            model_input_h, model_input_w = input_tensor.shape[2:]
            
            # Use configured original image size
            ori_h, ori_w = self.original_image_size
            
            # Calculate scale factors using configured method
            scale_h, scale_w = self._calculate_scale_factor(model_input_h, model_input_w, ori_h, ori_w)
            
            for i in range(batch_size):
                data_sample = DetDataSample()
                data_sample.set_metainfo({
                    'img_shape': (model_input_h, model_input_w),  # Model input size
                    'ori_shape': (ori_h, ori_w),  # Original image size
                    'scale_factor': (scale_h, scale_w),  # Use calculated scale factors
                    'batch_input_shape': (model_input_h, model_input_w)
                })
                data_samples.append(data_sample)
            
            return data_samples
        else:
            # Classification model - use MMPretrain DataSample
            data_samples = []
            for i in range(batch_size):
                data_sample = self._create_base_data_sample(input_tensor, i)
                data_samples.append(data_sample)
            
            return data_samples

    def _get_raw_output(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Get raw head outputs matching ONNX wrapper format.
        
        Args:
            input_tensor: Input tensor for inference
            
        Returns:
            numpy array with raw outputs
        """
        # Check if this is a detection model or classification model
        if self._model_type == 'detector':
            # Detection model - get raw head outputs
            feat = self._model.extract_feat(input_tensor)
            cls_scores, bbox_preds, objectnesses = self._model.bbox_head(feat)
            
            # Process each detection level (matching ONNX wrapper logic)
            outputs = []
            for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
                # Apply sigmoid to objectness and cls_score (NOT to bbox_pred)
                level_output = torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1)
                outputs.append(level_output)

            # Flatten and concatenate all levels (matching ONNX wrapper logic exactly)
            batch_size = outputs[0].shape[0]
            num_channels = outputs[0].shape[1]
            outputs = torch.cat([x.reshape(batch_size, num_channels, -1) for x in outputs], dim=2).permute(0, 2, 1)

            return outputs.cpu().numpy()
        else:
            # Classification model - get raw logits (before softmax)
            feat = self._model.extract_feat(input_tensor)
            if isinstance(feat, tuple):
                feat = feat[0]  # Take the first element if it's a tuple
            raw_logits = self._model.head(feat)
            return raw_logits.cpu().numpy()

    def _extract_predictions_from_results(self, results) -> np.ndarray:
        """
        Extract predictions from model results format.
        
        Args:
            results: List of DetDataSample/ClsDataSample objects from model.predict()
            
        Returns:
            numpy array containing predictions
        """
        if not results:
            return np.array([])
        
        # Handle single image case
        if not isinstance(results, list):
            results = [results]
        
        all_predictions = []
        
        for result in results:
            # Check if this is a detection result or classification result
            if hasattr(result, 'pred_instances') and result.pred_instances is not None:
                # Detection model result
                pred_instances = result.pred_instances
                
                # Extract bboxes, scores, labels
                bboxes = pred_instances.bboxes.cpu().numpy() if pred_instances.bboxes is not None else np.array([])
                scores = pred_instances.scores.cpu().numpy() if pred_instances.scores is not None else np.array([])
                labels = pred_instances.labels.cpu().numpy() if pred_instances.labels is not None else np.array([])
                
                # Combine into single array: [bboxes(4), scores(1), labels(1)]
                if len(bboxes) > 0:
                    predictions = np.column_stack([bboxes, scores, labels])
                else:
                    predictions = np.array([]).reshape(0, 6)
                
                all_predictions.append(predictions)
            elif hasattr(result, 'pred_score') and result.pred_score is not None:
                # Classification model result
                pred_score = result.pred_score.cpu().numpy()
                pred_label = result.pred_label.cpu().numpy()
                
                # For classification, we return the raw logits/scores
                # Shape: [num_classes] for single image
                all_predictions.append(pred_score)
            else:
                # No predictions
                all_predictions.append(np.array([]).reshape(0, 6))
        
        # Stack all predictions
        if all_predictions:
            return np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
        else:
            return np.array([]).reshape(0, 6)

    def cleanup(self) -> None:
        """Clean up PyTorch resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
