# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion quantization: composition of framework schemes for this model.

The generic sparse spconv-INT8 primitives now live in the framework
(:mod:`deployment.quantization.sparse`); this package only declares how BEVFusion composes them:
the sparse scheme (:class:`SpconvInt8Scheme`) and the plan builder (:func:`build_bevfusion_plan`).
The offline PTQ producer CLI is :mod:`deployment.projects.bevfusion_l.quantization.quantize`.
"""

from .plan import build_bevfusion_plan
from .schemes import SpconvInt8Scheme

__all__ = [
    "SpconvInt8Scheme",
    "build_bevfusion_plan",
]
