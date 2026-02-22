from typing import Optional

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet.models.backbones.resnet import ResNet as _ResNet
from mmengine.logging import print_log
from torch import nn


@MODELS.register_module(force=True)
class BEVResNet(_ResNet):
    """BEV-friendly ResNet backbone for 3D detection.

    This wrapper modifies the stem layer to avoid downsampling at the input,
    making it suitable for BEV (Bird's Eye View) feature extraction where
    we want to preserve spatial resolution.

    Key modifications:
    - deep_stem=True: Uses three 3x3 convs instead of 7x7 (more efficient, better boundary behavior)
    - conv1_stride=1: First conv stride=1 (no downsampling)
    - with_pool=False: Disable maxpool (no downsampling)
    - This ensures output feature maps have sizes: (H, W), (H/2, W/2), (H/4, W/4)
    """

    def __init__(
        self,
        conv1_stride: int = 1,
        with_pool: bool = False,
        pool_stride: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            conv1_stride (int): Stride of the first conv layer in stem.
                Default: 1 (no downsampling).
            with_pool (bool): Whether to use maxpool after conv1.
                Default: False (no maxpool).
            pool_stride (int, optional): Stride of maxpool if with_pool=True.
                Only used when with_pool=True. Default: 1 (no downsampling).
            **kwargs: Other arguments passed to parent ResNet class.
        """
        self.conv1_stride = conv1_stride
        self.with_pool = with_pool
        self.pool_stride = pool_stride if pool_stride is not None else 1

        log_msg = f"BEV-friendly ResNet: conv1_stride={conv1_stride}, " f"with_pool={with_pool}"
        if with_pool:
            log_msg += f", pool_stride={self.pool_stride}"
        print_log(log_msg)

        super(BEVResNet, self).__init__(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Override stem layer to support BEV-friendly configuration."""
        if self.deep_stem:
            # For deep_stem, modify the first conv stride
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=self.conv1_stride,  # Use configurable stride
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg, stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False
                ),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            # Standard stem: modify conv1 stride
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=self.conv1_stride,  # Use configurable stride instead of hardcoded 2
                padding=3,
                bias=False,
            )
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

        # Configure maxpool based on with_pool flag
        if self.with_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.pool_stride, padding=1)  # Use configurable stride
        else:
            # If no pool, create an identity layer to maintain compatibility
            self.maxpool = nn.Identity()
            print_log("MaxPool disabled in ResNet stem (BEV-friendly mode)")
