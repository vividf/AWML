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
            quant_backbone=config.quant_backbone,
            quant_neck=config.quant_neck,
            quant_head=config.quant_head,
            quant_voxel_encoder=config.quant_voxel_encoder,
            quant_add=config.quant_add,
            quant_linear_backbone=config.quant_linear_backbone,
            quant_ese_mul_identity=config.quant_ese_mul_identity,
            quant_ese_pool_input=config.quant_ese_pool_input,
            quant_maxpool_input=config.quant_maxpool_input,
            sensitive_layers=config.resolved_sensitive_layers(),
            fuse_bn=config.fuse_bn,
        )
    )
    return plan
