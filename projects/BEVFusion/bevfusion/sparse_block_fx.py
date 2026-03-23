# Copyright (c) OpenMMLab. All rights reserved.
"""FX-traceable sparse basic block for spconv INT8 (prepare_fx/convert_fx).

Uses SparseReLU and (out + identity) instead of replace_feature(out.features + identity)
so torch.fx can fuse the residual add+relu. See spconv docs/INT8_GUIDE.md.
"""

from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.utils import OptConfigType
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn


def _sparse_replace_feature_for_fx_impl(out, new_features):
    """Replace sparse features without ``if 'replace_feature' in out.__dir__()`` (that check uses the traced tensor and breaks prepare_fx)."""
    if IS_SPCONV2_AVAILABLE:
        return out.replace_feature(new_features)
    out.features = new_features
    return out


if hasattr(torch, "fx") and hasattr(torch.fx, "wrap"):
    _sparse_replace_feature_for_fx = torch.fx.wrap(_sparse_replace_feature_for_fx_impl)
else:
    _sparse_replace_feature_for_fx = _sparse_replace_feature_for_fx_impl


if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseModule, SparseReLU
else:
    from mmcv.ops import SparseConvTensor, SparseModule

    SparseReLU = nn.ReLU  # fallback


class SparseBasicBlockFX(BasicBlock, SparseModule):
    """Sparse basic block that is torch.fx traceable for spconv INT8.

    Same structure as SparseBasicBlock (conv1, norm1, relu, conv2, norm2, downsample)
    but forward uses SparseReLU and (out + identity) with identity as SparseConvTensor
    so FX can fuse the residual add+relu. Checkpoint from SparseBasicBlock loads.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int]] = 1,
        downsample: nn.Module = None,
        indice_key: Optional[str] = None,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
    ) -> None:
        SparseModule.__init__(self)
        if conv_cfg is None:
            conv_cfg = dict(type="SubMConv3d")
        conv_cfg.setdefault("indice_key", indice_key)
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d")
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        # Use spconv SparseReLU for the mid activation so FX/ONNX/JIT never call nn.ReLU on a
        # SparseConvTensor (BasicBlock's nn.ReLU only accepts Tensor; convert_fx can wire relu wrong).
        if IS_SPCONV2_AVAILABLE:
            inplace = bool(getattr(self.relu, "inplace", True))
            self.relu = SparseReLU(inplace=inplace)
            self.relu_final = SparseReLU(inplace=True)
        else:
            self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = _sparse_replace_feature_for_fx(out, self.norm1(out.features))
        if IS_SPCONV2_AVAILABLE:
            out = self.relu(out)
        else:
            out = _sparse_replace_feature_for_fx(out, self.relu(out.features))

        out = self.conv2(out)
        out = _sparse_replace_feature_for_fx(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu_final(out + identity)
        return out
