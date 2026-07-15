# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion quantization: composition of framework schemes for this model.

The dense tower is quantized via the framework's generic Q/DQ schemes; the sparse encoder deploys in
FP16 and only needs SparseConv+BN folding (:class:`SpconvBnFuseScheme`). The plan builder
(:func:`build_bevfusion_plan`) composes them, and the offline PTQ producer CLI is
:mod:`deployment.projects.bevfusion_l.quantization.quantize`.
"""

from .plan import build_bevfusion_plan
from .schemes import SpconvBnFuseScheme

__all__ = [
    "SpconvBnFuseScheme",
    "build_bevfusion_plan",
]
