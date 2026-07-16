# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion calibration forward — the one home for the voxel-dtype-normalizing forward.

Shared by the PTQ producer (``quantize.py``) and the QAT hook (``qat_hook.py``): some
dataloader/preprocessor paths provide integer voxel features / points, which must be coerced to
float32 before ``test_step`` so dense Q/DQ calibration sees floating-point activations. Hoisted to
module level so both consumers import it instead of copying it (spec_qat.md §4 WP4.1).
"""

from __future__ import annotations


def force_float_voxel_inputs(batch):
    """Best-effort dtype normalization before ``test_step`` during calibration.

    Integer voxel features or points are coerced to float32 where needed for dense Q/DQ
    calibration; anything else passes through untouched.
    """
    import torch

    if not isinstance(batch, dict):
        return batch
    inputs = batch.get("inputs", None)
    if not isinstance(inputs, dict):
        return batch

    vox = inputs.get("voxels", None)
    if isinstance(vox, dict):
        v = vox.get("voxels", None)
        if isinstance(v, torch.Tensor) and not v.is_floating_point():
            vox["voxels"] = v.to(dtype=torch.float32).contiguous()

    points = inputs.get("points", None)
    if isinstance(points, (list, tuple)):
        normalized = []
        changed = False
        for p in points:
            if isinstance(p, torch.Tensor) and not p.is_floating_point():
                normalized.append(p.to(dtype=torch.float32))
                changed = True
            else:
                normalized.append(p)
        if changed:
            inputs["points"] = type(points)(normalized) if isinstance(points, tuple) else normalized
    return batch


def calibration_forward(model, batch):
    """Calibration ``forward_fn(model, batch)``: dtype-normalize, then run the model."""
    batch = force_float_voxel_inputs(batch)
    if hasattr(model, "test_step"):
        return model.test_step(batch)
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        return model(*batch)
    return model(batch)
