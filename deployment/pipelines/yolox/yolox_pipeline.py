"""
YOLOX Deployment Pipeline Base Class.

This module provides the abstract base class for YOLOX deployment,
defining the unified pipeline that shares preprocessing and postprocessing logic
while allowing backend-specific optimizations for model inference.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from mmdet.models.dense_heads.yolox_head import YOLOXHead
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from deployment.pipelines.base.detection_2d_pipeline import Detection2DPipeline

logger = logging.getLogger(__name__)


class YOLOXDeploymentPipeline(Detection2DPipeline):
    """
    Abstract base class for YOLOX deployment pipeline.

    This class defines the complete inference flow for YOLOX, with:
    - Shared preprocessing (image resize, normalization)
    - Shared postprocessing (bbox decoding, NMS, coordinate transform)
    - Abstract method for backend-specific model inference

    The design eliminates code duplication by centralizing PyTorch processing
    while allowing ONNX/TensorRT backends to optimize the model inference.
    """

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        num_classes: int = 8,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300,
        backend_type: str = "unknown",
    ):
        """
        Initialize YOLOX pipeline.

        Args:
            model: Model object (PyTorch model, ONNX session, TensorRT engine)
            device: Device for inference ('cuda' or 'cpu')
            num_classes: Number of object classes
            class_names: List of class names
            input_size: Model input size (height, width)
            score_threshold: Confidence threshold for filtering detections
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            backend_type: Backend type ('pytorch', 'onnx', 'tensorrt')
        """
        # Default class names for T4Dataset
        if class_names is None:
            class_names = ["unknown", "car", "truck", "bus", "trailer", "motorcycle", "pedestrian", "bicycle"]

        # Initialize parent class
        super().__init__(
            model=model,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            backend_type=backend_type,
        )

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        # Initialize MMDetection YOLOXHead for postprocessing

        # Create a lightweight YOLOXHead instance for postprocessing only
        # We only need prior_generator, _bbox_decode, and _bbox_post_process methods

        # TODO(vividf): get in_channels and strides from model or use defaults
        self.yolox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=128,  # Dummy value, not used for postprocessing
            strides=[8, 16, 32],
            test_cfg=ConfigDict(
                dict(
                    score_thr=score_threshold,
                    nms=dict(type="nms", iou_threshold=nms_threshold),
                    max_per_img=max_detections,
                )
            ),
        )
        self.yolox_head.eval()  # Set to eval mode

    # ========== Preprocessing Methods ==========

    def preprocess(self, input_data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for YOLOX inference.

        This method expects preprocessed tensor from MMDetection pipeline.
        All preprocessing (resize, padding, normalization) should be done
        by MMDetection's test_pipeline before calling this method.

        Args:
            input_data: Preprocessed tensor [1, C, H, W] from MMDetection pipeline
            **kwargs: Additional preprocessing parameters (must include 'img_info' for metadata)

        Returns:
            Tuple of (preprocessed_tensor, preprocessing_metadata)
            - preprocessed_tensor: [1, C, H, W]
            - preprocessing_metadata: Dict with 'scale_factor', 'original_shape', 'input_shape'

        Raises:
            TypeError: If input_data is not a torch.Tensor
            ValueError: If img_info is missing from kwargs
        """
        if not isinstance(input_data, torch.Tensor):
            raise TypeError(
                f"YOLOX pipeline requires preprocessed tensor from MMDetection pipeline. "
                f"Got {type(input_data)}. Please use data_loader.preprocess() first."
            )

        # Extract metadata from kwargs
        metadata = {}

        # Get img_info from kwargs (required)
        img_info = kwargs.get("img_info", {})
        if not img_info:
            raise ValueError(
                "img_info is required for YOLOX preprocessing. "
                "Please provide img_info in kwargs when calling infer()."
            )

        metadata["scale_factor"] = img_info.get("scale_factor", [1.0, 1.0, 1.0, 1.0])
        metadata["original_shape"] = (img_info.get("height", 1080), img_info.get("width", 1440))

        # Tensor from MMDet pipeline is in [0, 255] range
        # YOLOX data_preprocessor does NOT do normalization (no mean/std configured)
        # So we use the tensor as-is to match training behavior
        tensor = input_data
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        # Ensure tensor resides on pipeline device for backend consistency
        if isinstance(self.device, torch.device):
            target_device = self.device
        else:
            target_device = torch.device(self.device)
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=True)

        metadata["input_shape"] = tuple(tensor.shape[2:])

        return tensor, metadata

    # ========== Abstract Method (Backend-specific) ==========

    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> np.ndarray:
        """
        Run YOLOX model inference (backend-specific).

        This method must be implemented by each backend (PyTorch/ONNX/TensorRT)
        to provide optimized model inference.

        Args:
            preprocessed_input: Preprocessed image tensor [1, C, H, W]

        Returns:
            Model output [1, num_predictions, 4+1+num_classes]
            Format: [bbox_reg(4), objectness(1), class_scores(num_classes)]
        """
        raise NotImplementedError

    # ========== Postprocessing Methods ==========

    def postprocess(self, model_output: np.ndarray, metadata: Dict = None) -> List[Dict]:
        """
        Postprocess YOLOX model output to final detections.

        Steps:
        1. Parse model output (bbox, objectness, class scores)
        2. Compute final scores (objectness * class_score)
        3. Filter by score threshold
        4. Apply NMS
        5. Transform coordinates back to original image space
        6. Limit to max_detections

        Args:
            model_output: Raw model output [1, num_predictions, 4+1+num_classes]
            metadata: Preprocessing metadata

        Returns:
            List of detections in format:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'score': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        if metadata is None:
            metadata = {}

        # Model output shape: [1, num_predictions, 4+1+num_classes]
        # Remove batch dimension
        predictions = model_output[0]  # [num_predictions, 4+1+num_classes]

        # Parse predictions (raw YOLOX head outputs)
        bbox_reg = predictions[:, :4]  # [dx, dy, dw, dh] (raw regression relative to stride)
        objectness = predictions[:, 4]  # [num_predictions] - objectness score (sigmoid)
        class_scores = predictions[:, 5:]  # [num_predictions, num_classes] (sigmoid)
        input_h, input_w = metadata.get("input_shape", self.input_size)

        # Convert to torch tensors
        bbox_reg_tensor = torch.from_numpy(bbox_reg).float().to(self.device)
        objectness_tensor = torch.from_numpy(objectness).float().to(self.device)
        class_scores_tensor = torch.from_numpy(class_scores).float().to(self.device)

        # Generate priors using YOLOXHead's prior_generator
        featmap_sizes = [(input_h // 8, input_w // 8), (input_h // 16, input_w // 16), (input_h // 32, input_w // 32)]
        mlvl_priors = self.yolox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=torch.float32,
            device=self.device,
            with_stride=True,  # Returns [center_x, center_y, stride_w, stride_h]
        )
        priors_tensor = torch.cat(mlvl_priors, dim=0)  # [N, 4]

        # Align lengths if any mismatch (safety)
        num = min(len(bbox_reg_tensor), len(priors_tensor))
        bbox_reg_tensor = bbox_reg_tensor[:num]
        objectness_tensor = objectness_tensor[:num]
        class_scores_tensor = class_scores_tensor[:num]
        priors_tensor = priors_tensor[:num]

        # Decode bboxes using YOLOXHead._bbox_decode
        bboxes_tensor = self.yolox_head._bbox_decode(priors_tensor, bbox_reg_tensor.unsqueeze(0))
        bboxes_tensor = bboxes_tensor[0]  # Remove batch dimension: [N, 4]

        # Compute scores and labels (same as YOLOXHead.predict_by_feat)
        max_scores, labels = torch.max(class_scores_tensor, 1)
        final_scores = max_scores * objectness_tensor

        # Filter by score threshold
        valid_mask = final_scores >= self.score_threshold
        bboxes_tensor = bboxes_tensor[valid_mask]
        final_scores = final_scores[valid_mask]
        labels = labels[valid_mask]

        if len(final_scores) == 0:
            return []

        # Create InstanceData for _bbox_post_process
        results = InstanceData(bboxes=bboxes_tensor, scores=final_scores, labels=labels)

        # Prepare img_meta for rescale
        img_meta = {}
        if "scale_factor" in metadata:
            sf = metadata["scale_factor"]
            if isinstance(sf, (list, tuple, np.ndarray)) and len(sf) >= 2:
                img_meta["scale_factor"] = [float(sf[0]), float(sf[1])]
            else:
                raise ValueError(f"Invalid scale_factor format: {sf}")
        else:
            raise ValueError("scale_factor is required in metadata")

        # Use YOLOXHead._bbox_post_process for NMS and rescale
        # Note: YOLOXHead._bbox_post_process does NOT handle max_per_img
        # (unlike base_dense_head), so we need to handle it manually
        processed_results = self.yolox_head._bbox_post_process(
            results=results,
            cfg=self.yolox_head.test_cfg,
            rescale=True,  # Rescale to original image space
            with_nms=True,  # Apply NMS
            img_meta=img_meta,
        )

        # Limit to max detections (YOLOXHead._bbox_post_process doesn't handle max_per_img)
        if len(processed_results.scores) > self.max_detections:
            top_indices = torch.argsort(processed_results.scores, descending=True)[: self.max_detections]
            processed_results = processed_results[top_indices]

        # results
        results = []
        for i in range(len(processed_results.bboxes)):
            results.append(
                {
                    "bbox": processed_results.bboxes[i].cpu().numpy().tolist(),
                    "score": float(processed_results.scores[i].cpu().numpy()),
                    "class_id": int(processed_results.labels[i].cpu().numpy()),
                    "class_name": self.class_names[int(processed_results.labels[i].cpu().numpy())],
                }
            )

        return results

    def _apply_nms(self, bboxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """
        Apply batched NMS (per-class NMS).

        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]

        Returns:
            Indices of boxes to keep
        """
        try:
            # Try to use MMDetection's batched NMS for consistency
            from mmcv.ops import batched_nms

            bboxes_tensor = torch.from_numpy(bboxes).float()
            scores_tensor = torch.from_numpy(scores).float()
            class_ids_tensor = torch.from_numpy(class_ids).long()

            nms_cfg = dict(type="nms", iou_threshold=self.nms_threshold)

            _, keep_indices = batched_nms(bboxes_tensor, scores_tensor, class_ids_tensor, nms_cfg)

            return keep_indices.cpu().numpy()

        except ImportError:
            logger.warning("mmcv.ops.batched_nms not available, using per-class NMS")
            # Fallback to per-class NMS
            return self._per_class_nms(bboxes, scores, class_ids)

    def _per_class_nms(self, bboxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """
        Apply per-class NMS (fallback implementation).

        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]

        Returns:
            Indices of boxes to keep
        """
        keep_indices = []

        # Apply NMS per class
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_bboxes = bboxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]

            # Apply NMS for this class
            keep = self._nms(class_bboxes, class_scores, self.nms_threshold)
            keep_indices.extend(class_indices[keep])

        return np.array(keep_indices)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"input_size={self.input_size}, "
            f"num_classes={self.num_classes}, "
            f"backend={self.backend_type})"
        )
