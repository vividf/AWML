# Copyright (c) OpenMMLab. All rights reserved.
"""Shared PTQ-producer helpers.

The one home for the producer-side plumbing that was previously copied verbatim between
``projects/centerpoint/quantization/quantize.py`` and ``projects/bevfusion_l/quantization/quantize.py``
(calibration-dataloader seeding/shuffle, checkpoint + calibration-cache save, quantization-library
logging init). Each project's ``quantize.py`` keeps only its model-specific steps; this module owns
the identical mechanics — imitating the single-impl pattern of
``deployment/quantization/sparse`` + its project re-export.

Everything here is deliberately torch/mmengine-lazy: importing this module is cheap and safe on
hosts without the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def init_quant_logging() -> None:
    """Silence pytorch-quantization's absl logging (verbose INFO chatter) down to ERROR.

    No-op when absl is not installed. Call once at producer-CLI start.
    """
    try:
        from absl import logging as quant_logging

        quant_logging.set_verbosity(quant_logging.ERROR)
    except ImportError:
        pass


def build_calib_dataloader(
    cfg,
    *,
    batch_size: int,
    seed: Optional[int] = None,
    shuffle: bool = False,
    max_num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    """Build the calibration dataloader from ``cfg.val_dataloader`` with PTQ overrides.

    Mutates ``cfg.val_dataloader`` (best-effort — only when it is a plain dict), then builds the
    dataloader via mmengine's ``Runner.build_dataloader``. The overrides:

    - ``batch_size`` is always set (larger batches reduce calibration seed sensitivity).
    - ``seed`` (when given) seeds ``random`` / ``numpy`` / ``torch`` / ``torch.cuda`` so a shuffled
      calibration order is reproducible.
    - ``shuffle`` (when True) removes any configured ``sampler`` (mutually exclusive) and sets
      ``shuffle=True``.
    - ``max_num_workers`` / ``persistent_workers`` are optional per-project dataloader caps
      (BEVFusion caps workers at 4 and disables persistent workers; CenterPoint leaves both alone).

    Args:
        cfg: MMEngine config whose ``val_dataloader`` drives calibration.
        batch_size: Calibration batch size.
        seed: Optional RNG seed for reproducible shuffling.
        shuffle: Shuffle the calibration data.
        max_num_workers: Optional upper bound on ``num_workers``.
        persistent_workers: Optional override for ``persistent_workers``.

    Returns:
        The built calibration dataloader.
    """
    import torch
    from mmengine.runner import Runner

    if isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader["batch_size"] = batch_size

        if max_num_workers is not None:
            cfg.val_dataloader["num_workers"] = min(
                cfg.val_dataloader.get("num_workers", max_num_workers), max_num_workers
            )
        if persistent_workers is not None:
            cfg.val_dataloader["persistent_workers"] = persistent_workers

        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if shuffle:
            # Remove existing sampler to allow shuffle (they are mutually exclusive).
            if "sampler" in cfg.val_dataloader:
                del cfg.val_dataloader["sampler"]
            cfg.val_dataloader["shuffle"] = True

    return Runner.build_dataloader(cfg.val_dataloader)


def save_ptq_checkpoint(model, output: str, calibrator=None) -> Tuple[Path, Optional[Path]]:
    """Save the PTQ checkpoint (``{"state_dict": ...}``) and, when given, the calibration cache.

    The calibration cache is written next to the checkpoint with a ``.calib`` suffix. A failed
    cache save raises — the ``.calib`` file is what lets QAT skip recalibration, so losing it
    silently is worse than failing loud.

    Args:
        model: The calibrated, quantized model.
        output: Output checkpoint path.
        calibrator: Optional ``CalibrationManager``; when given, its cache is saved too.

    Returns:
        ``(checkpoint_path, calib_cache_path)`` — the cache path is None without a calibrator.
    """
    import torch

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)

    calib_path: Optional[Path] = None
    if calibrator is not None:
        calib_path = output_path.with_suffix(".calib")
        calibrator.save_calib_cache(str(calib_path))

    return output_path, calib_path
