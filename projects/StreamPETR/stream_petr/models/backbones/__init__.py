# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# EVA-ViT pulls in optional training-only deps (fvcore, flash-attn); deployment
# environments that use the VoVNet backbones can miss them.
try:
    from .eva_vit import EVAViT
except ImportError:
    EVAViT = None
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
