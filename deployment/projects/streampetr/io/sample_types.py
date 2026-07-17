"""Typed sample payloads for StreamPETR deployment.

``StreamPETRExportSample`` carries every tensor the three export components need, with the
chained intermediates (``img_feats``, ``pos_embed``, ``cone``) computed from a real dataset
frame by :class:`~deployment.projects.streampetr.export.sample_extractor.StreamPETRSampleExtractor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def pad_to_4x4(mats: torch.Tensor) -> torch.Tensor:
    """Embed ``[..., 3, 3]`` matrices into homogeneous ``[..., 4, 4]`` (identity padding).

    The dataset yields 3x3 ``cam2img`` intrinsics; the deployed graph contract is 4x4.
    ``position_embeding`` reads only ``[..., 0, 0]`` / ``[..., 1, 1]``, so this is
    value-equivalent. Shared by the export sample extractor and the inference pipelines.
    """
    if mats.shape[-2:] == (4, 4):
        return mats
    if mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [..., 3, 3] or [..., 4, 4] matrices, got {tuple(mats.shape)}")
    padded = torch.zeros(*mats.shape[:-2], 4, 4, dtype=mats.dtype)
    padded[..., :3, :3] = mats
    padded[..., 3, 3] = 1.0
    return padded


@dataclass(frozen=True)
class StreamPETRExportSample:
    """A real-frame export sample with chained component inputs.

    Attributes:
        img: Multi-view images ``[1, N_cam, 3, H, W]`` (float32).
        intrinsics: Camera intrinsics ``[1, N_cam, 4, 4]`` (float32).
        img2lidar: Inverse of ``lidar2img`` per camera, ``[1, N_cam, 4, 4]`` (float32).
        img_metas_pad: ``[pad_h, pad_w, 3]`` as a float32 tensor of shape ``[3]``.
        img_feats: Encoder output ``[1, N_cam, C, H/stride, W/stride]`` (float32).
        pos_embed: Position embedding ``[1, N_cam*fH*fW, C]`` (float32).
        cone: Camera cone parameters ``[1, N_cam*fH*fW, 8]`` (float32).
        data_timestamp: Frame timestamp ``[1]`` (float64). Traced but unused in the graph
            (the in-graph timestamp arithmetic is commented out upstream); onnxsim prunes it.
        data_ego_pose: lidar2global pose ``[1, 4, 4]`` (float32).
        data_ego_pose_inv: Inverse pose ``[1, 4, 4]`` (float32).
        memory_embedding: Zeroed memory queue ``[1, memory_len, C]`` (float32).
        memory_reference_point: Zeroed ``[1, memory_len, 3]`` (float32).
        memory_timestamp: Zeroed ``[1, memory_len, 1]`` (float32).
        memory_egopose: Zeroed ``[1, memory_len, 4, 4]`` (float32).
        memory_velo: Zeroed ``[1, memory_len, 2]`` (float32).
    """

    img: torch.Tensor
    intrinsics: torch.Tensor
    img2lidar: torch.Tensor
    img_metas_pad: torch.Tensor
    img_feats: torch.Tensor
    pos_embed: torch.Tensor
    cone: torch.Tensor
    data_timestamp: torch.Tensor
    data_ego_pose: torch.Tensor
    data_ego_pose_inv: torch.Tensor
    memory_embedding: torch.Tensor
    memory_reference_point: torch.Tensor
    memory_timestamp: torch.Tensor
    memory_egopose: torch.Tensor
    memory_velo: torch.Tensor
