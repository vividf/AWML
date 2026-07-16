"""YOLOX model loading for deployment.

Loads an mmdet YOLOX detector for export + reference evaluation. Two YOLOX-specific concerns live
here (everything else is the shared runtime's job):

- **Custom-module registration** — the model config declares its project modules (backbones,
  necks, datasets) under ``custom_imports``; importing them registers the classes so
  ``init_detector`` can build the graph. Reading them from the config (instead of hardcoding a
  variant's package) keeps one bundle able to deploy any YOLOX model.
- **ReLU6 → ReLU** — YOLOX-opt variants train with ReLU6 activations, which export to ONNX as
  ``Clip(0, 6)`` and degrade on some TensorRT paths. Swapping them for ReLU before export *and* on
  the reference model keeps PyTorch / ONNX / TensorRT numerically aligned.
"""

from __future__ import annotations

import logging

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)


def import_custom_modules(model_cfg: Config) -> None:
    """Import a model config's ``custom_imports`` so its registry entries are available.

    Mirrors the repo's ``__import__`` registration idiom (see the old ``preprocessing_builder``):
    each listed module is imported for its ``@MODELS.register_module()`` / transform / dataset
    side effects. Variant-agnostic — the module list comes from the config, not this bundle.

    Args:
        model_cfg: MMEngine model config, optionally carrying a ``custom_imports`` dict
            (``{"imports": [...], "allow_failed_imports": bool}``).

    Raises:
        ImportError: If a required module cannot be imported and ``allow_failed_imports`` is False.
    """
    custom_imports = model_cfg.get("custom_imports")
    if not custom_imports:
        return
    if isinstance(custom_imports, dict):
        imports = custom_imports.get("imports", [])
        allow_failed = bool(custom_imports.get("allow_failed_imports", False))
    else:
        imports, allow_failed = list(custom_imports), False

    for module_path in imports:
        try:
            __import__(module_path)
        except ImportError:
            logger.warning("Failed to import custom module '%s' declared in model config.", module_path)
            if not allow_failed:
                raise


def _replace_relu6_with_relu(module: torch.nn.Module) -> None:
    """Recursively replace every ``nn.ReLU6`` with ``nn.ReLU`` in place (ONNX-friendly)."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU6):
            setattr(module, name, torch.nn.ReLU(inplace=getattr(child, "inplace", False)))
        else:
            _replace_relu6_with_relu(child)


def build_yolox_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build an mmdet YOLOX model from config, load its checkpoint, and return it in eval mode.

    Args:
        model_cfg: MMEngine model configuration (its ``custom_imports`` are processed first).
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Target device specification (typically CPU at load time; the executor moves the
            model onto the eval/verify device later).

    Returns:
        The loaded detector in eval mode, with ReLU6 activations swapped for ReLU and ``model.cfg``
        set to the config it was built from.
    """
    from mmdet.apis import init_detector

    init_default_scope("mmdet")
    import_custom_modules(model_cfg)

    model = init_detector(model_cfg, checkpoint_path, device=str(device))
    _replace_relu6_with_relu(model)
    model.eval()
    if getattr(model, "cfg", None) is None:
        model.cfg = model_cfg
    return model
