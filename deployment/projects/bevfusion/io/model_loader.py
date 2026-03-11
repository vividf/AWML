"""BEVFusion model loading utilities for deployment."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core.device import DeviceSpec


def _register_bevfusion_modules() -> None:
    """Register BEVFusion and SparseConvolution modules into MMDet3D registries."""
    import projects.BEVFusion.bevfusion  # noqa: F401
    import projects.SparseConvolution  # noqa: F401


def build_bevfusion_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build a BEVFusion model from config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to .pth checkpoint file.
        device: Target device.

    Returns:
        Loaded and eval-mode BEVFusion model.
    """
    init_default_scope("mmdet3d")
    _register_bevfusion_modules()

    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)

    torch_device = device.to_torch_device()
    model.to(torch_device)
    load_checkpoint(model, checkpoint_path, map_location=torch_device)
    model.eval()
    model.cfg = model_cfg
    return model
