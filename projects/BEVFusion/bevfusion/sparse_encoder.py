# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, List, Optional

from mmdet3d.models.layers.sparse_block import SparseBasicBlock
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

import numpy as np
import torch

from .sparse_block_fx import SparseBasicBlockFX
from .sparse_convmodule import make_sparse_convmodule

logger = logging.getLogger(__name__)


def _is_fx_proxy(t) -> bool:
    try:
        from torch.fx import Proxy

        return isinstance(t, Proxy)
    except Exception:
        return False


def _in_torch_fx_prepare_trace() -> bool:
    """Detect torch.fx symbolic_trace used by torch.ao.quantization.prepare_fx."""
    try:
        fn = getattr(torch.fx, "is_symbolic_trace", None)
        if callable(fn) and fn():
            return True
    except Exception:
        pass
    try:
        from torch.fx._symbolic_trace import is_tracing

        if callable(is_tracing) and is_tracing():
            return True
    except Exception:
        pass
    try:
        from torch.fx._symbolic_trace import _is_fx_tracing

        if callable(_is_fx_tracing) and _is_fx_tracing():
            return True
    except Exception:
        pass
    return False


def _conv_out_to_bev(out_tensor, output_channels: Optional[int] = None, target_z: int = 2) -> torch.Tensor:
    """Convert conv_out SparseConvTensor to BEV (N, C*Z, H, W).

    ``spconv.dense()`` returns ``[N, C, *spatial_shape]`` in the same axis order as
    ``SparseConvTensor.spatial_shape``. Depth (collapsed z after ``conv_out``) may be
    the **first** ``(D,H,W)`` **or** **last** ``(H,W,D)`` axis depending on how
    ``sparse_shape`` / voxel indices are ordered. The legacy code assumed ``(H,W,D)``
    and always permuted ``D`` to the channel-flatten axis; if the tensor is actually
    ``[N,C,D,H,W]`` with small ``D`` (e.g. 2) and large ``H,W`` (e.g. 1440), that
    permute yields a bogus channel count (e.g. ``128*1440``) instead of ``256``.

    We pick the **smallest** spatial extent in ``spatial_shape`` as the depth axis to
    merge into channels (after ``conv_out`` it should be ~2; the BEV plane stays ~1440).
    This matches both ``(H,W,D)`` and ``(D,H,W)`` layouts without relying on a fixed permute.

    FX INT8 / ONNX trace sometimes leaves the **full** z grid (e.g. 41) in ``spatial_shape``
    so ``C * Z`` becomes 5248 instead of ``output_channels * target_z`` (256). In that case we
    apply ``adaptive_avg_pool3d`` along Z to ``target_z`` so SECOND's ``in_channels=256`` matches.
    This is an export-time fallback; for best fidelity fix spconv/FX so ``conv_out`` shrinks Z.

    ``output_channels`` defaults to ``out_tensor.features.shape[1]`` so FX ``GraphModule`` calls
    that only pass ``out_tensor`` (single arg) still work.
    """
    if output_channels is None:
        output_channels = int(out_tensor.features.shape[1])
    spatial_shape = [int(s) for s in out_tensor.spatial_shape]
    x = out_tensor.dense()
    n, c = x.shape[0], x.shape[1]
    # Map smallest logical axis -> depth to flatten with C (same as mmdet3d SparseEncoder when D is first).
    zi = min(range(3), key=lambda i: spatial_shape[i])
    order = [j for j in range(3) if j != zi] + [zi]
    perm = (0, 1, 2 + order[0], 2 + order[1], 2 + order[2])
    y = x.permute(*perm).contiguous()
    # y layout: N, C, H, W, Z (Z is smallest axis, typically 2 after conv_out)
    z = y.shape[4]
    h, w = y.shape[2], y.shape[3]
    z_i = int(z) if not isinstance(z, int) else z
    flat_c = int(c) * z_i
    expected = int(output_channels) * int(target_z)
    if int(c) == int(output_channels) and flat_c != expected and z_i > int(target_z):
        logger.warning(
            "BEVFusion _conv_out_to_bev: dense Z=%d gives %d channels (expected %d); "
            "collapsing Z→%d via adaptive_avg_pool3d for ONNX/backbone compatibility.",
            z_i,
            flat_c,
            expected,
            target_z,
        )
        y5 = y.permute(0, 1, 4, 2, 3).contiguous()
        y5 = torch.nn.functional.adaptive_avg_pool3d(y5, (int(target_z), int(h), int(w)))
        return y5.reshape(n, int(c) * int(target_z), int(h), int(w))
    return y.view(n, flat_c, h, w)


# FX trace fails when control flow uses symbolic (traced) variables; dense() or shape/view can trigger that.
# Wrapping makes this a single non-traced call so prepare_fx succeeds.
if hasattr(torch.fx, "wrap"):
    _conv_out_to_bev = torch.fx.wrap(_conv_out_to_bev)


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
        self,
        in_channels,
        aug_features_min_values,
        aug_features_max_values,
        num_aug_features,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        return_middle_feats=False,
    ):
        super(SparseEncoder, self).__init__()
        assert block_type in ["conv_module", "basicblock", "basicblock_fx"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.register_buffer("aug_features_min_values", torch.tensor(aug_features_min_values))
        self.register_buffer("aug_features_max_values", torch.tensor(aug_features_max_values))
        self.num_aug_features = num_aug_features
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        if num_aug_features:
            self.in_channels = in_channels * num_aug_features * 2
            self.register_buffer("exponents", (2 ** torch.arange(0, num_aug_features).float()))

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                self.in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                self.in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """

        if self.num_aug_features:
            num_points = voxel_features.shape[0]
            x = (voxel_features - self.aug_features_min_values.view(1, -1)) / (
                self.aug_features_max_values - self.aug_features_min_values
            ).view(1, -1)
            y = x.reshape(-1, 1) * np.pi * self.exponents.reshape(1, -1)
            y = y.reshape(num_points, -1)
            voxel_features = torch.cat([torch.cos(y), torch.sin(y)], dim=1)

        # INT8 / quantize_per_tensor needs float32 features. Do not branch on is_floating_point() or dtype:
        # under prepare_fx, voxel_features is a Proxy and those checks participate in control flow → TraceError.
        voxel_features = voxel_features.contiguous().to(dtype=torch.float32)
        # int32 + contiguous: spconv implicit_gemm merge_sort assumes int32 indices; non-contiguous → illegal access
        coors = coors.to(dtype=torch.int32, device=voxel_features.device).contiguous()
        # Do not compare symbolic shapes under FX (prepare_fx); see _in_torch_fx_prepare_trace docstring.
        if (
            not torch.jit.is_tracing()
            and not _in_torch_fx_prepare_trace()
            and not _is_fx_proxy(voxel_features)
            and not _is_fx_proxy(coors)
            and voxel_features.shape[0] != coors.shape[0]
        ):
            raise ValueError(f"voxel_features / coors row mismatch: {voxel_features.shape[0]} vs {coors.shape[0]}")
        input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]; use wrapped op so FX does not trace dense()/shape/view (avoids "symbolically traced variables in control flow")
        out = self.conv_out(encode_features[-1])
        spatial_features = _conv_out_to_bev(out, self.output_channels)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = "conv_module",
        conv_cfg: Optional[dict] = None,
    ) -> int:
        """Build encoder layers. Overridden to support basicblock_fx for torch.fx INT8."""
        if conv_cfg is None:
            conv_cfg = dict(type="SubMConv3d")
        assert block_type in ["conv_module", "basicblock", "basicblock_fx"]
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type in ("basicblock", "basicblock_fx"):
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        block_cls = SparseBasicBlockFX if block_type == "basicblock_fx" else SparseBasicBlock
                        blocks_list.append(
                            block_cls(
                                in_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
