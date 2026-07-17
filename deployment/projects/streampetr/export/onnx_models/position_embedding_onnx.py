"""ONNX tracing module for the ``position_embedding`` component (3D position encoding).

Verbatim port of ``TrtPositionEmbeddingContainer`` in
``projects/StreamPETR/deploy/containers.py``.

The traced graph consumes only the camera geometry (``intrinsics``, ``img2lidar``) and the
padded image shape; ``img_feats`` contributes shapes alone, so onnxsim prunes it from the
simplified graph (the deployed reference has 3 inputs).

The original standalone exporter registered ``onnxruntime.tools.pytorch_export_contrib_ops``
before exporting this component. The deployed reference graphs contain no ``com.microsoft``
ops (pure ai.onnx opset 18), so the registration is not reproduced here — registering it
process-wide could perturb the other two components' graphs. If this component's export
fails on a missing symbolic, register the contrib ops immediately before exporting *only*
this component (and export it last).
"""

from __future__ import annotations

from typing import Tuple

import torch


class StreamPETRPositionEmbeddingONNX(torch.nn.Module):
    """Traces ``prepare_location`` + ``StreamPETRHead.position_embeding`` as one graph.

    Inputs: ``img_metas_pad [3]``, ``img_feats [1, N, C, fH, fW]``,
    ``intrinsics [1, N, 4, 4]``, ``img2lidar [1, N, 4, 4]``.
    Outputs: ``pos_embed [1, N*fH*fW, C]``, ``cone [1, N*fH*fW, 8]``.
    """

    def __init__(self, mod: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(
        self,
        img_metas_pad: torch.Tensor,
        img_feats: torch.Tensor,
        intrinsics: torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mod = self.mod
        data = {
            "img_feats": img_feats,
            "intrinsics": intrinsics,
            "img2lidar": img2lidar,
        }
        location = mod.prepare_location(img_metas_pad, **data)
        return mod.pts_bbox_head.position_embeding(data, location, None, img_metas_pad)
