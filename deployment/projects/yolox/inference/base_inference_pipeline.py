"""YOLOX inference pipeline base (shared preprocess + decode/NMS postprocess).

The ONNX graph emits raw, undecoded head outputs (``[B, anchors, 4+1+num_classes]``); decoding,
score thresholding, NMS, and rescaling back to original image space are done here in Python using
mmdet's ``YOLOXHead`` (prior generation + ``_bbox_decode`` + ``_bbox_post_process``). This is
identical across backends, so it lives once in this base; each backend only implements
``run_model``. Postprocess knobs (classes, strides, thresholds) come from the model config via
:class:`YOLOXDecodeParams` rather than being hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)

# YOLOX FPN strides used to generate priors; overridable from the model config's bbox_head.
_DEFAULT_STRIDES: Tuple[int, ...] = (8, 16, 32)


@dataclass(frozen=True)
class YOLOXDecodeParams:
    """Postprocess parameters for YOLOX decode + NMS, sourced from the model config.

    Replaces the old pipeline's hardcoded class list and thresholds so one bundle decodes any YOLOX
    variant. ``num_classes`` is derived from ``class_names``; the rest come from
    ``model.bbox_head.strides`` and ``model.test_cfg``.
    """

    class_names: Tuple[str, ...]
    strides: Tuple[int, ...] = _DEFAULT_STRIDES
    score_threshold: float = 0.01
    nms_threshold: float = 0.65
    max_detections: int = 300

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @classmethod
    def from_model_cfg(cls, model_cfg: Config, class_names: Sequence[str]) -> "YOLOXDecodeParams":
        """Build decode params from a model config and the resolved class names.

        Reads ``strides`` from ``model.bbox_head`` and ``score_thr`` / ``nms.iou_threshold`` /
        ``max_per_img`` from ``model.test_cfg`` (each with the YOLOX-standard default).
        """
        model = model_cfg.get("model", {}) or {}
        bbox_head = model.get("bbox_head", {}) or {}
        test_cfg = model.get("test_cfg", {}) or {}
        nms_cfg = test_cfg.get("nms", {}) or {}

        strides = tuple(bbox_head.get("strides", _DEFAULT_STRIDES))
        return cls(
            class_names=tuple(class_names),
            strides=strides,
            score_threshold=float(test_cfg.get("score_thr", 0.01)),
            nms_threshold=float(nms_cfg.get("iou_threshold", 0.65)),
            max_detections=int(test_cfg.get("max_per_img", 300)),
        )


class YOLOXInferencePipeline(BaseInferencePipeline):
    """Base YOLOX pipeline: backends override ``run_model``; preprocess/postprocess are shared."""

    def __init__(
        self,
        model: Any,
        backend_type: Backend,
        device: DeviceSpec,
        decode_params: YOLOXDecodeParams,
    ) -> None:
        super().__init__(model=model, backend_type=backend_type, device=device)
        self.decode_params = decode_params

        # A lightweight mmdet YOLOXHead used only for its prior generator + decode/NMS helpers.
        # in_channels/feat_channels are irrelevant to postprocess, so a dummy value is fine.
        from mmdet.models.dense_heads.yolox_head import YOLOXHead

        self._yolox_head = YOLOXHead(
            num_classes=decode_params.num_classes,
            in_channels=128,
            strides=list(decode_params.strides),
            test_cfg=ConfigDict(
                dict(
                    score_thr=decode_params.score_threshold,
                    nms=dict(type="nms", iou_threshold=decode_params.nms_threshold),
                    max_per_img=decode_params.max_detections,
                )
            ),
        )
        self._yolox_head.eval()

    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Return the (already loader-preprocessed) image tensor as float32 on the pipeline device."""
        tensor = input_data
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.to(self.torch_device)

    def postprocess(
        self,
        model_output: np.ndarray,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Decode raw YOLOX outputs into original-space detections.

        Steps: split into (bbox_reg, objectness, class_scores); generate priors for the input
        feature-map sizes; decode boxes; score = objectness * max class prob; threshold; NMS +
        rescale to original image space; cap at ``max_detections``.

        Args:
            model_output: Raw output ``[1, num_anchors, 4+1+num_classes]`` (numpy or tensor).
            metadata: Must carry ``scale_factor`` (for rescale) and ``input_shape`` (H, W) used for
                prior generation.

        Returns:
            List of ``{bbox: [x1, y1, x2, y2], score, class_id, class_name}`` in original image space.
        """
        metadata = dict(metadata or {})
        params = self.decode_params

        predictions = model_output[0] if isinstance(model_output, np.ndarray) else np.asarray(model_output)[0]
        bbox_reg = torch.from_numpy(np.ascontiguousarray(predictions[:, :4])).float().to(self.torch_device)
        objectness = torch.from_numpy(np.ascontiguousarray(predictions[:, 4])).float().to(self.torch_device)
        class_scores = torch.from_numpy(np.ascontiguousarray(predictions[:, 5:])).float().to(self.torch_device)

        input_h, input_w = metadata.get("input_shape", (0, 0))
        if not input_h or not input_w:
            raise ValueError("YOLOX postprocess requires metadata['input_shape'] = (H, W).")

        featmap_sizes = [(input_h // s, input_w // s) for s in params.strides]
        mlvl_priors = self._yolox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=torch.float32,
            device=self.torch_device,
            with_stride=True,
        )
        priors = torch.cat(mlvl_priors, dim=0)

        # Guard against a prior/prediction count mismatch (e.g. odd input sizes).
        num = min(len(bbox_reg), len(priors))
        bbox_reg, objectness, class_scores, priors = bbox_reg[:num], objectness[:num], class_scores[:num], priors[:num]

        bboxes = self._yolox_head._bbox_decode(priors, bbox_reg.unsqueeze(0))[0]
        max_scores, labels = torch.max(class_scores, dim=1)
        final_scores = max_scores * objectness

        keep = final_scores >= params.score_threshold
        bboxes, final_scores, labels = bboxes[keep], final_scores[keep], labels[keep]
        if len(final_scores) == 0:
            return []

        results = InstanceData(bboxes=bboxes, scores=final_scores, labels=labels)
        img_meta = {"scale_factor": self._rescale_factor(metadata)}

        processed = self._yolox_head._bbox_post_process(
            results=results,
            cfg=self._yolox_head.test_cfg,
            rescale=True,
            with_nms=True,
            img_meta=img_meta,
        )

        # YOLOXHead._bbox_post_process does not honour max_per_img; enforce it explicitly.
        if len(processed.scores) > params.max_detections:
            top = torch.argsort(processed.scores, descending=True)[: params.max_detections]
            processed = processed[top]

        detections: List[Dict[str, Any]] = []
        for i in range(len(processed.bboxes)):
            class_id = int(processed.labels[i].item())
            detections.append(
                {
                    "bbox": processed.bboxes[i].cpu().numpy().tolist(),
                    "score": float(processed.scores[i].item()),
                    "class_id": class_id,
                    "class_name": (
                        params.class_names[class_id] if class_id < params.num_classes else f"class_{class_id}"
                    ),
                }
            )
        return detections

    @staticmethod
    def _rescale_factor(metadata: Mapping[str, Any]) -> List[float]:
        """Return ``[w_scale, h_scale]`` for rescaling boxes to original space.

        Raises:
            ValueError: If ``scale_factor`` is missing or malformed.
        """
        sf = metadata.get("scale_factor")
        if isinstance(sf, (list, tuple, np.ndarray)) and len(sf) >= 2:
            return [float(sf[0]), float(sf[1])]
        raise ValueError(f"metadata['scale_factor'] must have >= 2 values, got {sf!r}.")
