# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# ------------------------------------------------------------------------
# VoVNet code adapted from projects/StreamPETR/stream_petr/models/backbones/vovnet.py
# BEVVoVNet wrapper added for BEV-friendly stem configuration.
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.logging import print_log
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

VoVNet99_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False,
}

_STAGE_SPECS = {
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 depthwise separable convolution with padding."""
    return [
        (
            "{}_{}/dw_conv3x3".format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
                bias=False,
            ),
        ),
        (
            "{}_{}/pw_conv1x1".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        ),
        ("{}_{}/pw_norm".format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ("{}_{}/pw_relu".format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding."""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding."""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch, "{}_reduction".format(module_name), "0"))
            )
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
        self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False
    ):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                ),
            )


class VoVNet(BaseModule):
    """VoVNet backbone.

    Adapted from projects/StreamPETR/stream_petr/models/backbones/vovnet.py.
    Not registered as a standalone module here to avoid duplicate registration
    with the StreamPETR version.
    """

    def __init__(
        self,
        spec_name,
        input_ch=3,
        out_features=None,
        frozen_stages=-1,
        norm_eval=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(VoVNet, self).__init__(init_cfg)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, " 'please use "init_cfg" instead')
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        stage_specs = _STAGE_SPECS[spec_name]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features

        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stirde = 4
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):
            name = "stage%d" % (i + 2)
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs.append(x)

        return outputs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            m = getattr(self, "stem")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"stage{i+1}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super(VoVNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class BEVVoVNet(VoVNet):
    """BEV-friendly VoVNet backbone for 3D detection.

    Modifies VoVNet's stem to use configurable strides, removing the default
    4x downsampling (stride 2, 1, 2) to preserve spatial resolution for BEV
    feature maps from PointPillarsScatter.

    With default stem_strides=(1, 1, 1) on 1020x1020 BEV input:
        Stem:   128ch @ 1020x1020  (no downsampling)
        Stage2: 256ch @ 1020x1020  (no MaxPool)
        Stage3: 512ch @ 510x510   (MaxPool stride=2)
        Stage4: 768ch @ 255x255   (MaxPool stride=2)

    Using out_features=("stage2", "stage3", "stage4") produces spatial sizes
    [1020, 510, 255] which align with SECONDFPN upsample_strides=[0.5, 1, 2]
    to produce 510x510 output (grid_size / out_size_factor = 1020 / 2).

    Args:
        stem_strides (tuple[int]): Strides for the 3 stem convolutions.
            Original VoVNet uses (2, 1, 2) for 4x total downsampling.
            Default (1, 1, 1) preserves full BEV spatial resolution.
        spec_name (str): VoVNet variant specification name (e.g., "V-99-eSE").
        input_ch (int): Number of input channels. Default: 3.
        out_features (tuple[str]): Names of output feature layers.
        frozen_stages (int): Stages to freeze. -1 means no freezing.
        norm_eval (bool): Whether to set BN layers to eval mode.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self, stem_strides=(1, 1, 1), **kwargs):
        super().__init__(**kwargs)

        spec_name = kwargs["spec_name"]
        stage_specs = _STAGE_SPECS[spec_name]
        stem_ch = stage_specs["stem"]
        depthwise = stage_specs["dw"]
        input_ch = kwargs.get("input_ch", 3)

        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", stem_strides[0])
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", stem_strides[1])
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", stem_strides[2])

        self.stem = nn.Sequential(OrderedDict(stem))

        total_stem_stride = 1
        for s in stem_strides:
            total_stem_stride *= s

        current_stride = total_stem_stride
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        for i in range(1, 4):
            name = f"stage{i + 2}"
            current_stride *= 2
            self._out_feature_strides[name] = current_stride

        print_log(
            f"BEVVoVNet: stem_strides={stem_strides}, "
            f"total_stem_stride={total_stem_stride}, "
            f"feature_strides={self._out_feature_strides}"
        )
