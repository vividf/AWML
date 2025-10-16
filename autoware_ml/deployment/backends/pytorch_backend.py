"""PyTorch inference backend."""

import logging
import time
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base_backend import BaseBackend


class PyTorchBackend(BaseBackend):
    """
    PyTorch inference backend.

    Runs inference using native PyTorch models.
    Supports two initialization modes:
    1. Direct model instance (for verification)
    2. Model path with config (for evaluation)
    """

    def __init__(self, model: Union[torch.nn.Module, str], model_cfg: Optional[object] = None, device: str = "cpu"):
        """
        Initialize PyTorch backend.

        Args:
            model: PyTorch model instance or path to checkpoint
            model_cfg: Model configuration (required if model is a path)
            device: Device to run inference on
        """
        super().__init__(model_path="", device=device)
        self._torch_device = torch.device(device)
        self.model_cfg = model_cfg
        self._logger = logging.getLogger(__name__)

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


        # TODO(vividf): cleanup this code, too many if-else statements

        # Move input to correct device
        input_tensor = input_tensor.to(self._torch_device)

        # Run inference with timing
        with torch.no_grad():
            start_time = time.perf_counter()
            output = self._model(input_tensor)
            end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Handle different output formats
        # MMDetection models return tuples/lists of DetDataSample objects in test mode
        if isinstance(output, (tuple, list)):
            # For detection models, typically returns list of DetDataSample objects
            if len(output) == 1:
                output = output[0]
            else:
                # Multiple outputs, use the first one (usually predictions)
                output = output[0]

        # MMDetection DetDataSample - extract predictions
        if hasattr(output, "pred_instances"):
            # 2D detection: pred_instances contains bboxes, scores, labels
            pred_instances = output.pred_instances
            # Concatenate all prediction tensors for comparison
            tensors_to_concat = []

            # Extract bboxes
            if hasattr(pred_instances, "bboxes") and pred_instances.bboxes is not None:
                bboxes = pred_instances.bboxes
                # Convert to tensor if needed
                if not isinstance(bboxes, torch.Tensor):
                    if isinstance(bboxes, list):
                        if len(bboxes) > 0:
                            bboxes = (
                                torch.stack(bboxes) if isinstance(bboxes[0], torch.Tensor) else torch.tensor(bboxes)
                            )
                        else:
                            bboxes = torch.zeros((0, 4))
                    elif isinstance(bboxes, np.ndarray):
                        bboxes = torch.from_numpy(bboxes)
                    else:
                        raise ValueError(f"Unexpected bboxes type: {type(bboxes)}")
                tensors_to_concat.append(bboxes)

            # Extract scores
            if hasattr(pred_instances, "scores") and pred_instances.scores is not None:
                scores = pred_instances.scores
                if not isinstance(scores, torch.Tensor):
                    if isinstance(scores, list):
                        scores = torch.tensor(scores) if len(scores) > 0 else torch.zeros(0)
                    elif isinstance(scores, np.ndarray):
                        scores = torch.from_numpy(scores)
                    else:
                        raise ValueError(f"Unexpected scores type: {type(scores)}")
                if scores.ndim == 1:
                    scores = scores.unsqueeze(-1)
                tensors_to_concat.append(scores)

            # Extract labels
            if hasattr(pred_instances, "labels") and pred_instances.labels is not None:
                labels = pred_instances.labels
                if not isinstance(labels, torch.Tensor):
                    if isinstance(labels, list):
                        labels = torch.tensor(labels) if len(labels) > 0 else torch.zeros(0)
                    elif isinstance(labels, np.ndarray):
                        labels = torch.from_numpy(labels)
                    else:
                        raise ValueError(f"Unexpected labels type: {type(labels)}")
                if labels.ndim == 1:
                    labels = labels.unsqueeze(-1)
                tensors_to_concat.append(labels)

            # Concatenate or create empty tensor
            if tensors_to_concat:
                if len(tensors_to_concat) > 1:
                    output = torch.cat(tensors_to_concat, dim=-1)
                else:
                    output = tensors_to_concat[0]
            else:
                # No predictions found, return empty tensor
                output = torch.zeros((0, 6))  # [num_boxes, (x1, y1, x2, y2, score, label)]

        elif hasattr(output, "pred_instances_3d"):
            # 3D detection: pred_instances_3d contains bboxes_3d, scores_3d, labels_3d
            pred_instances_3d = output.pred_instances_3d
            tensors_to_concat = []
            if hasattr(pred_instances_3d, "bboxes_3d") and pred_instances_3d.bboxes_3d is not None:
                bboxes_3d = pred_instances_3d.bboxes_3d.tensor
                tensors_to_concat.append(bboxes_3d)
            if hasattr(pred_instances_3d, "scores_3d") and pred_instances_3d.scores_3d is not None:
                scores_3d = pred_instances_3d.scores_3d
                tensors_to_concat.append(scores_3d.unsqueeze(-1) if scores_3d.ndim == 1 else scores_3d)
            if hasattr(pred_instances_3d, "labels_3d") and pred_instances_3d.labels_3d is not None:
                labels_3d = pred_instances_3d.labels_3d
                tensors_to_concat.append(labels_3d.unsqueeze(-1) if labels_3d.ndim == 1 else labels_3d)

            if tensors_to_concat:
                output = torch.cat(tensors_to_concat, dim=-1) if len(tensors_to_concat) > 1 else tensors_to_concat[0]
            else:
                output = torch.zeros((0, 10))  # Empty 3D predictions

        elif hasattr(output, "output"):
            # Generic output attribute
            output = output.output
        elif isinstance(output, dict) and "output" in output:
            # Dictionary with output key
            output = output["output"]
        elif isinstance(output, list):
            # Raw model outputs (e.g., YOLOX returns list of tensors for multi-scale outputs)
            if len(output) > 0 and all(isinstance(o, torch.Tensor) for o in output):
                # Multi-scale detection outputs - concatenate along the anchor dimension
                # YOLOX typically returns list of [batch, num_anchors, num_classes+5] for each scale
                # Concatenate them for comparison
                output = torch.cat([o.flatten(1) for o in output], dim=1)  # Flatten and concat
            else:
                # Empty list or unexpected format
                output = torch.zeros((1, 0)) if len(output) == 0 else output[0]

        # Special handling for YOLOX models to match ONNX wrapper output format
        # Check if this is a YOLOX model by looking for bbox_head attribute
        self._logger.info(f"Model type: {type(self._model)}")
        self._logger.info(f"Has bbox_head: {hasattr(self._model, 'bbox_head')}")
        if hasattr(self._model, "bbox_head"):
            self._logger.info(f"bbox_head type: {type(self._model.bbox_head)}")
            self._logger.info(f"Has num_classes: {hasattr(self._model.bbox_head, 'num_classes')}")
            if hasattr(self._model.bbox_head, "num_classes"):
                self._logger.info(f"num_classes: {self._model.bbox_head.num_classes}")

        self._logger.info(f"Output type: {type(output)}")
        self._logger.info(f"Has pred_instances: {hasattr(output, 'pred_instances')}")

        # Check if this is a YOLOX model and we need to extract raw head outputs
        is_yolox_model = hasattr(self._model, "bbox_head") and hasattr(self._model.bbox_head, "num_classes")

        # Check if output is DetDataSample (which means we need to extract raw outputs)
        is_detdatasample = hasattr(output, "pred_instances")

        # Check if output is a tensor (which might be processed output that needs raw extraction)
        is_tensor = isinstance(output, torch.Tensor)

        self._logger.info(f"Is YOLOX model: {is_yolox_model}")
        self._logger.info(f"Is DetDataSample: {is_detdatasample}")
        self._logger.info(f"Is tensor: {is_tensor}")

        # For YOLOX models, if we get a tensor output, we need to extract raw head outputs
        # to match the ONNX wrapper format
        if is_yolox_model and (is_detdatasample or is_tensor):
            # This is a YOLOX model, extract raw head outputs to match ONNX wrapper format
            self._logger.info("Detected YOLOX model, extracting raw head outputs for ONNX compatibility")

            # Get raw head outputs by calling the model's head directly
            with torch.no_grad():
                # Extract features
                feat = self._model.extract_feat(input_tensor)
                # Get head outputs: (cls_scores, bbox_preds, objectnesses)
                cls_scores, bbox_preds, objectnesses = self._model.bbox_head(feat)

                # Process each detection level (matching ONNX wrapper logic)
                outputs = []
                for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
                    # Apply sigmoid to objectness and cls_score (NOT to bbox_pred)
                    # This matches ONNX wrapper logic
                    level_output = torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1)
                    outputs.append(level_output)

                # Flatten and concatenate all levels (matching ONNX wrapper logic exactly)
                batch_size = outputs[0].shape[0]
                num_channels = outputs[0].shape[1]
                outputs = torch.cat([x.reshape(batch_size, num_channels, -1) for x in outputs], dim=2).permute(0, 2, 1)

                # Keep the 3D format (1, 18900, 13) to match ONNX/TensorRT format
                output = outputs

                self._logger.info(f"YOLOX raw output shape: {output.shape}")
                self._logger.info(f"Expected format: (1, 18900, 13) - matching ONNX/TensorRT")
        else:
            self._logger.info("Not a YOLOX model or output format not recognized, using standard processing")

        if not isinstance(output, (torch.Tensor, np.ndarray)):
            raise ValueError(f"Unexpected PyTorch output type after extraction: {type(output)}")

        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output_array = output.cpu().numpy()
        else:
            output_array = output

        return output_array, latency_ms

    def cleanup(self) -> None:
        """Clean up PyTorch resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
