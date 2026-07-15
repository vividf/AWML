# Copyright (c) OpenMMLab. All rights reserved.
"""Build the CenterPoint :class:`QuantizationPlan` from a :class:`QuantizationConfig`.

The single place that maps CenterPoint's quantization config onto schemes. The deploy loader, the
PTQ producer, and the QAT hook all call it, so they compose an *identical* quantized module tree and
the PTQ ``state_dict`` and the deploy ``load_state_dict`` line up by construction — mirroring
BEVFusion's :func:`~deployment.projects.bevfusion_l.quantization.plan.build_bevfusion_plan`.
"""

from __future__ import annotations

from deployment.config.schema import QuantizationConfig
from deployment.quantization.schemes.base import QuantizationPlan

from .schemes import CenterPointDenseScheme


def build_centerpoint_plan(config: QuantizationConfig) -> QuantizationPlan:
    """Compose the dense scheme for CenterPoint from a typed config.

    Args:
        config: Parsed deploy ``quantization`` block.

    Returns:
        A :class:`QuantizationPlan` (currently a single :class:`CenterPointDenseScheme`).
    """
    plan = QuantizationPlan()
    plan.add(
        CenterPointDenseScheme(
            keep_fp16=config.keep_fp16,
            disable_recipes=config.disable_recipes,
            fuse_bn=config.fuse_bn,
        )
    )
    return plan
