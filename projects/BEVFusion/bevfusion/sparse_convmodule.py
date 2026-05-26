# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse conv-bnorm-act stacks for BEVFusion (spconv2).

mmdet3d ``make_sparse_convmodule`` uses ``nn.ReLU`` for ``act``.  With spconv2,
``SparseSequential`` passes ``SparseConvTensor`` between layers, so ``nn.ReLU`` breaks
during ONNX tracing.  Use ``spconv.pytorch.SparseReLU`` instead (same pattern as
``SparseSyncBatchNorm`` in spconv).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.utils import OptConfigType
from torch import nn

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseReLU, SparseSequential
else:
    from mmcv.ops import SparseSequential


def make_sparse_convmodule(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]],
    indice_key: Optional[str] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    conv_type: str = "SubMConv3d",
    norm_cfg: OptConfigType = None,
    order: Tuple[str, ...] = ("conv", "norm", "act"),
) -> SparseSequential:
    """Same API as ``mmdet3d.models.layers.make_sparse_convmodule`` but ``act`` uses ``SparseReLU`` when spconv2 is on."""
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)
    if norm_cfg is None:
        norm_cfg = dict(type="BN1d")

    layers: list = []
    for layer in order:
        if layer == "conv":
            if conv_type not in [
                "SparseInverseConv3d",
                "SparseInverseConv2d",
                "SparseInverseConv1d",
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False,
                    )
                )
        elif layer == "norm":
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            if IS_SPCONV2_AVAILABLE:
                layers.append(SparseReLU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))

    return SparseSequential(*layers)
