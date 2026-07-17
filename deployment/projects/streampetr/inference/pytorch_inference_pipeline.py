"""
StreamPETR PyTorch pipeline (staged to match ONNX/TensorRT).

Runs the same three tracing modules that produced the ONNX graphs
(``export/onnx_models/``), so the PyTorch reference and the exported backends execute the
identical deployed-graph semantics — which is what makes cross-backend verification
meaningful for a temporal model.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.primitives.device import DeviceSpec
from deployment.projects.streampetr.export.onnx_models.encoder_onnx import StreamPETREncoderONNX
from deployment.projects.streampetr.export.onnx_models.position_embedding_onnx import (
    StreamPETRPositionEmbeddingONNX,
)
from deployment.projects.streampetr.export.onnx_models.pts_head_onnx import StreamPETRPtsHeadONNX
from deployment.projects.streampetr.inference.streampetr_inference_pipeline import (
    StreamPETRInferencePipeline,
)

logger = logging.getLogger(__name__)

#: pre_memory input order of the traced head forward (after the data_* tensors).
_HEAD_INPUT_ORDER = (
    "x",
    "pos_embed",
    "cone",
    "data_timestamp",
    "data_ego_pose",
    "data_ego_pose_inv",
    "pre_memory_embedding",
    "pre_memory_reference_point",
    "pre_memory_timestamp",
    "pre_memory_egopose",
    "pre_memory_velo",
)


class StreamPETRPyTorchInferencePipeline(StreamPETRInferencePipeline):
    """PyTorch-based StreamPETR pipeline running the export container modules."""

    def __init__(self, pytorch_model: torch.nn.Module, device: DeviceSpec) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.PYTORCH, device=device)
        # Denoising off, mirroring the exported graph (train-only branch).
        pytorch_model.pts_bbox_head.with_dn = False
        self._encoder = StreamPETREncoderONNX(pytorch_model)
        self._position_embedding = StreamPETRPositionEmbeddingONNX(pytorch_model)
        self._pts_head = StreamPETRPtsHeadONNX(pytorch_model)
        logger.info("PyTorch pipeline initialized (deployed-graph staged inference)")

    @override
    def run_encoder(self, img: torch.Tensor) -> torch.Tensor:
        # extract_img_feat squeezes a batch-1 input in place — hand it a clone.
        with torch.no_grad():
            return self._encoder(self.to_device_tensor(img).clone())

    @override
    def run_position_embedding(
        self,
        img_metas_pad: torch.Tensor,
        img_feats: torch.Tensor,
        intrinsics: torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self._position_embedding(
                self.to_device_tensor(img_metas_pad),
                self.to_device_tensor(img_feats),
                self.to_device_tensor(intrinsics),
                self.to_device_tensor(img2lidar),
            )

    @override
    def run_head(self, head_inputs: Dict[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        args = tuple(self.to_device_tensor(head_inputs[name]) for name in _HEAD_INPUT_ORDER)
        with torch.no_grad():
            return self._pts_head(*args)
