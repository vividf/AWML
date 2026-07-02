"""BEVFusion sparse ``coors`` layout for deploy / ONNX / TensorRT.

Voxelization (``pts_voxel_layer``) returns indices as ``[x, y, z]`` (see
``dynamic_voxelize_kernel`` in ``bevfusion/ops/voxel``).

Legacy Autoware-compatible ONNX expects graph **inputs** as ``[z, y, x]`` (no batch).
Inside the exported wrapper, indices are flipped to ``[x, y, z]`` and a batch column
is prepended before ``pts_middle_encoder`` (``sparse_shape`` is ``[H, W, D]``).

PyTorch evaluation uses ``[batch, x, y, z]`` directly and does not use this module.
"""

from __future__ import annotations

import torch


def voxel_indices_xyz_to_graph_input_zyx(coors: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` voxel indices ``[x, y, z]`` → graph input ``[z, y, x]``."""
    if coors.ndim != 2 or coors.shape[1] != 3:
        return coors
    return coors.flip(dims=[-1]).contiguous()


def graph_input_zyx_to_model_indices_xyz(coors: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` graph input ``[z, y, x]`` → model indices ``[x, y, z]`` (wrapper flip)."""
    if coors.ndim != 2 or coors.shape[1] != 3:
        return coors
    return coors.flip(dims=[-1]).contiguous()
