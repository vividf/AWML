"""ONNX tracing module for the ``extract_img_feat`` component (image backbone + neck).

Verbatim port of ``TrtEncoderContainer`` in ``projects/StreamPETR/deploy/containers.py``.
"""

from __future__ import annotations

import torch


class StreamPETREncoderONNX(torch.nn.Module):
    """Traces ``Petr3D.extract_img_feat`` as a standalone graph.

    Input ``img`` is ``[1, N_cam, 3, H, W]``; output ``img_feats`` is
    ``[1, N_cam, C, H/stride, W/stride]``.

    Note: ``extract_img_feat`` squeezes a batch-1 5D input **in place**; callers that reuse
    the input tensor afterwards must pass a clone.
    """

    def __init__(self, mod: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mod = self.mod
        return mod.extract_img_feat(img, 1).squeeze(1)
