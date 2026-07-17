"""
StreamPETR model loading utilities.

Builds an export-ready StreamPETR (Petr3D) model from an MMEngine config: registers the
StreamPETR module variants, swaps flash attention for the ONNX-exportable
``PETRMultiheadAttention``, then builds and loads the checkpoint via the shared mmdet3d core.
"""

from __future__ import annotations

import logging

import torch
from mmengine.config import Config

from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)

#: ONNX-exportable attention type substituted for flash attention at export time.
_EXPORT_ATTENTION_TYPE = "PETRMultiheadAttention"


def import_custom_modules() -> None:
    """Register StreamPETR modules (Petr3D, StreamPETRHead, datasets, …) into the registries.

    ``Config.fromfile`` already honors the model config's ``custom_imports``; this explicit
    import keeps the loader safe when a caller hands us an already-parsed config.
    """
    import projects.StreamPETR.stream_petr  # noqa: F401


def create_export_model_cfg(model_cfg: Config) -> Config:
    """Create a model config whose decoder attention is ONNX-exportable.

    Flash attention has no ONNX export path, so both decoder attention layers are swapped to
    ``PETRMultiheadAttention`` — same surgery as the original
    ``projects/StreamPETR/deploy/torch2onnx.py``.

    Args:
        model_cfg: Original MMEngine model configuration.

    Returns:
        New config whose ``model`` subtree builds the deployment export graph.
    """
    export_model_cfg = model_cfg.copy()
    attn_cfgs = export_model_cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs
    for index, attn_cfg in enumerate(attn_cfgs):
        original = attn_cfg["type"]
        if original != _EXPORT_ATTENTION_TYPE:
            attn_cfg["type"] = _EXPORT_ATTENTION_TYPE
            logger.info("Decoder attn_cfgs[%d]: %s -> %s (ONNX export)", index, original, _EXPORT_ATTENTION_TYPE)
    return export_model_cfg


def build_streampetr_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build a StreamPETR model from config and load checkpoint weights (for export + reference eval).

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.

    Returns:
        The loaded model in eval mode; the export config it was built from is available as ``model.cfg``.
    """
    import_custom_modules()
    export_model_cfg = create_export_model_cfg(model_cfg)
    return build_mmdet3d_model(export_model_cfg, checkpoint_path, device)
