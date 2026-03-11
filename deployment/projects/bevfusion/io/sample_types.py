"""BEVFusion deployment sample types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class VoxelDict(TypedDict):
    voxels: torch.Tensor
    num_points_per_voxel: torch.Tensor
    coors: torch.Tensor


@dataclass(frozen=True)
class BEVFusionFeatureSample:
    """Typed sample payload for BEVFusion ONNX export.

    Attributes:
        voxels: Raw voxel features [M, max_points, C].
        coors: Voxel coordinates [M, 3] (z, y, x without batch index).
        num_points_per_voxel: Number of points per voxel [M].
        points: Original point cloud tensor [N, point_dim].
    """

    voxels: torch.Tensor
    coors: torch.Tensor
    num_points_per_voxel: torch.Tensor
    points: torch.Tensor
