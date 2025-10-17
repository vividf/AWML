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

        # Move input to correct device
        input_tensor = input_tensor.to(self._torch_device)

        # Run inference with timing
        with torch.no_grad():
            start_time = time.perf_counter()
            output = self._model(input_tensor)
            end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Process output based on model type and format
        output = self._process_model_output(output, input_tensor)
        
        # Convert to numpy array
        output_array = self._convert_to_numpy(output)
        
        return output_array, latency_ms

    def _process_model_output(self, output: Union[torch.Tensor, object], input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process model output based on its type and format.
        
        Args:
            output: Raw model output
            input_tensor: Input tensor used for inference
            
        Returns:
            Processed tensor output
        """
        print("########################################################")
        print("output", type(output))
        
        # Handle tuple/list outputs
        if isinstance(output, (tuple, list)):
            print("Extract output from tuple/list")
            # print tuple information 
            print("tuple information", len(output), len(output[0]), len(output[1]), len(output[2]))
            
            # 詳細查看每個檢測層的形狀
            if len(output) == 3:  # YOLOX 輸出格式
                cls_scores, bbox_preds, objectnesses = output
                print("\n=== YOLOX 檢測層形狀分析 ===")
                print(f"檢測層數量: {len(cls_scores)}")
                
                for i, (cls_score, bbox_pred, objectness) in enumerate(zip(cls_scores, bbox_preds, objectnesses)):
                    print(f"\n檢測層 {i+1} (P{i+3}層):")
                    print(f"  cls_scores[{i}].shape:   {cls_score.shape}")
                    print(f"  bbox_preds[{i}].shape:   {bbox_pred.shape}")
                    print(f"  objectnesses[{i}].shape: {objectness.shape}")
                    
                    # 計算該層的 anchor 數量
                    if len(cls_score.shape) == 4:  # [batch, channels, height, width]
                        batch_size, channels, height, width = cls_score.shape
                        anchors_per_layer = height * width
                        print(f"  anchor 數量: {anchors_per_layer} ({height}x{width})")
                
                # 計算總 anchor 數量
                total_anchors = sum(cls_score.shape[-2] * cls_score.shape[-1] for cls_score in cls_scores)
                print(f"\n總 anchor 數量: {total_anchors}")
                print(f"預期最終輸出形狀: [batch_size, {total_anchors}, 13]")
                print("其中 13 = bbox_reg(4) + objectness(1) + class_scores(8)")
            
            output = output[0] if len(output) == 1 else output[0]

        # Handle different output formats
        if hasattr(output, "pred_instances"):
            print("Extract 2D predictions")
            output = self._extract_2d_predictions(output)
        elif hasattr(output, "pred_instances_3d"):
            print("Extract 3D predictions")
            output = self._extract_3d_predictions(output)
        elif hasattr(output, "output"):
            print("Extract output")
            output = output.output
        elif isinstance(output, dict) and "output" in output:
            print("Extract output from dict")
            output = output["output"]
        elif isinstance(output, list):
            print("Extract output from list")
            output = self._process_list_output(output)

        # Special handling for YOLOX models
        if self._is_yolox_model() and self._needs_raw_output_extraction(output):
            print("Extract yolox raw output")
            output = self._extract_yolox_raw_output(input_tensor)
        
        return output

    def _extract_2d_predictions(self, output: object) -> torch.Tensor:
        """Extract 2D detection predictions from DetDataSample."""
        pred_instances = output.pred_instances
        tensors_to_concat = []
        
        # Extract bboxes
        if hasattr(pred_instances, "bboxes") and pred_instances.bboxes is not None:
            bboxes = self._convert_to_tensor(pred_instances.bboxes)
            tensors_to_concat.append(bboxes)
        
        # Extract scores
        if hasattr(pred_instances, "scores") and pred_instances.scores is not None:
            scores = self._convert_to_tensor(pred_instances.scores)
            if scores.ndim == 1:
                scores = scores.unsqueeze(-1)
            tensors_to_concat.append(scores)
        
        # Extract labels
        if hasattr(pred_instances, "labels") and pred_instances.labels is not None:
            labels = self._convert_to_tensor(pred_instances.labels)
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            tensors_to_concat.append(labels)
        
        # Concatenate or return empty tensor
        if tensors_to_concat:
            return torch.cat(tensors_to_concat, dim=-1) if len(tensors_to_concat) > 1 else tensors_to_concat[0]
        else:
            return torch.zeros((0, 6))  # [num_boxes, (x1, y1, x2, y2, score, label)]

    def _extract_3d_predictions(self, output: object) -> torch.Tensor:
        """Extract 3D detection predictions from DetDataSample."""
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
            return torch.cat(tensors_to_concat, dim=-1) if len(tensors_to_concat) > 1 else tensors_to_concat[0]
        else:
            return torch.zeros((0, 10))  # Empty 3D predictions

    def _process_list_output(self, output: list) -> torch.Tensor:
        """Process list-type outputs (e.g., multi-scale detection outputs)."""
        if len(output) > 0 and all(isinstance(o, torch.Tensor) for o in output):
            # Multi-scale detection outputs - concatenate along the anchor dimension
            return torch.cat([o.flatten(1) for o in output], dim=1)
        else:
            return torch.zeros((1, 0)) if len(output) == 0 else output[0]

    def _convert_to_tensor(self, data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert various data types to torch.Tensor."""
        if isinstance(data, torch.Tensor):
            return data
        
        if isinstance(data, list):
            if len(data) == 0:
                return torch.zeros(0)
            elif isinstance(data[0], torch.Tensor):
                return torch.stack(data)
            else:
                return torch.tensor(data)
        
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        
        raise ValueError(f"Unexpected data type: {type(data)}")

    def _is_yolox_model(self) -> bool:
        """Check if the model is a YOLOX model."""
        return (hasattr(self._model, "bbox_head") and 
                hasattr(self._model.bbox_head, "num_classes"))

    def _needs_raw_output_extraction(self, output: Union[torch.Tensor, object]) -> bool:
        """Check if raw output extraction is needed for YOLOX models."""
        return (hasattr(output, "pred_instances") or isinstance(output, torch.Tensor))

    def _extract_yolox_raw_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract raw YOLOX head outputs to match ONNX wrapper format."""
        with torch.no_grad():
            # Extract features
            feat = self._model.extract_feat(input_tensor)
            print("feat", len(feat))
            # Get head outputs: (cls_scores, bbox_preds, objectnesses)
            cls_scores, bbox_preds, objectnesses = self._model.bbox_head(feat)
            print("cls_scores", len(cls_scores))
            print("bbox_preds", len(bbox_preds))
            print("objectnesses", len(objectnesses))
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

            return outputs

    def _convert_to_numpy(self, output: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor output to numpy array."""
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif isinstance(output, np.ndarray):
            return output
        else:
            raise ValueError(f"Unexpected output type: {type(output)}")

    def cleanup(self) -> None:
        """Clean up PyTorch resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
