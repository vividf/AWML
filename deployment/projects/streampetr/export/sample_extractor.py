"""StreamPETR export sample extractor.

Unlike the original standalone exporter (which traced every component with ``np.random``
dummies), this extractor loads one **real** clip-start frame and chains the components: it
runs the encoder to produce ``img_feats``, then the position embedding to produce
``pos_embed``/``cone``, and seeds a zeroed memory queue — the state a real sequence starts
from. Every downstream component is therefore traced with the tensors it actually consumes.
"""

from __future__ import annotations

import logging

import torch

from deployment.export.pipelines.sample_extractor import SampleExtractor
from deployment.io.base_data_loader import BaseDataLoader
from deployment.projects.streampetr.export.onnx_models.encoder_onnx import StreamPETREncoderONNX
from deployment.projects.streampetr.export.onnx_models.position_embedding_onnx import (
    StreamPETRPositionEmbeddingONNX,
)
from deployment.projects.streampetr.io.sample_types import StreamPETRExportSample, pad_to_4x4

logger = logging.getLogger(__name__)


class StreamPETRSampleExtractor(SampleExtractor):
    """Extracts a real frame and computes the chained component inputs."""

    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> StreamPETRExportSample:
        """Extract a typed export sample from the model and data loader.

        Args:
            model: StreamPETR (Petr3D) model with ``extract_img_feat`` and ``pts_bbox_head``.
            data_loader: StreamPETR data loader (clip-ordered).
            sample_idx: Index of the frame to trace with.

        Returns:
            Typed :class:`StreamPETRExportSample` for the component builder.
        """
        sample = data_loader.load_sample(sample_idx)
        if not sample["metadata"].get("is_sequence_start", False):
            logger.warning(
                "Export sample %d is not a sequence start; tracing still uses a zeroed memory "
                "queue (the state a clip starts from).",
                sample_idx,
            )

        data = sample["input"]
        img = self._as_float(data["img"], "img", expected_dim=5)
        intrinsics = pad_to_4x4(self._as_float(data["intrinsics"], "intrinsics", expected_dim=4))
        lidar2img = self._as_float(data["lidar2img"], "lidar2img", expected_dim=4)
        img2lidar = torch.inverse(lidar2img)
        data_ego_pose = self._as_float(data["ego_pose"], "ego_pose", expected_dim=3)
        data_ego_pose_inv = self._as_float(data["ego_pose_inv"], "ego_pose_inv", expected_dim=3)
        data_timestamp = data["timestamp"].double().reshape(1)

        pad_h, pad_w = int(img.shape[-2]), int(img.shape[-1])
        img_metas_pad = torch.tensor([pad_h, pad_w, 3], dtype=torch.float32)

        # Trace on CPU/FP32. Deliberately NOT chained: Petr3D overrides ``train()`` without
        # returning ``self``, so ``model.eval()`` evaluates to None (same reason the original
        # exporter called `tm.float(); tm.cpu(); tm.eval()` as statements).
        model.float()
        model.cpu()
        model.eval()

        with torch.no_grad():
            # `extract_img_feat` squeezes a batch-1 input in place, so hand it a clone.
            img_feats = StreamPETREncoderONNX(model)(img.clone())
            pos_embed, cone = StreamPETRPositionEmbeddingONNX(model)(img_metas_pad, img_feats, intrinsics, img2lidar)

        head = model.pts_bbox_head
        memory_len = int(head.memory_len)
        embed_dims = int(head.embed_dims)

        logger.info(
            "Export sample %d: img=%s img_feats=%s pos_embed=%s memory_len=%d",
            sample_idx,
            tuple(img.shape),
            tuple(img_feats.shape),
            tuple(pos_embed.shape),
            memory_len,
        )

        return StreamPETRExportSample(
            img=img,
            intrinsics=intrinsics,
            img2lidar=img2lidar,
            img_metas_pad=img_metas_pad,
            img_feats=img_feats,
            pos_embed=pos_embed,
            cone=cone,
            data_timestamp=data_timestamp,
            data_ego_pose=data_ego_pose,
            data_ego_pose_inv=data_ego_pose_inv,
            memory_embedding=torch.zeros(1, memory_len, embed_dims),
            memory_reference_point=torch.zeros(1, memory_len, 3),
            memory_timestamp=torch.zeros(1, memory_len, 1),
            memory_egopose=torch.zeros(1, memory_len, 4, 4),
            memory_velo=torch.zeros(1, memory_len, 2),
        )

    @staticmethod
    def _as_float(tensor: torch.Tensor, name: str, expected_dim: int) -> torch.Tensor:
        """Validate rank and cast to float32 (dataset tensors may arrive as float64)."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != expected_dim:
            raise ValueError(f"{name} must have {expected_dim} dims, got shape {tuple(tensor.shape)}")
        return tensor.float()
