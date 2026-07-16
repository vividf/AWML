"""Calibration-classifier model loading for deployment.

Loads the mmpretrain ``ImageClassifier`` (5-channel ResNet-18) via ``mmpretrain.apis.get_model``.
The model is exported and run in ``mode='tensor'`` (called with a bare input tensor, no
``data_samples``), so it returns raw class logits ``[1, num_classes]`` — softmax is applied in the
pipeline's postprocess, and the exported ONNX graph therefore emits logits (IdentityWrapper).
"""

from __future__ import annotations

import logging

import torch
from mmengine.config import Config

from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)


def build_calibration_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build the mmpretrain classifier from config, load its checkpoint, and return it in eval mode.

    Args:
        model_cfg: MMEngine model configuration (mmpretrain ``ImageClassifier``).
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Target device specification (typically CPU at load time; the executor moves the
            model onto the eval/verify device later).

    Returns:
        The loaded classifier in eval mode, with ``model.cfg`` set to the config it was built from.
    """
    from mmpretrain.apis import get_model

    model = get_model(model_cfg, checkpoint_path, device=str(device))
    model.eval()
    if getattr(model, "cfg", None) is None:
        model.cfg = model_cfg
    return model
