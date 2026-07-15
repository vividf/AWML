# Copyright (c) OpenMMLab. All rights reserved.
"""Quantization schemes: the uniform seam between deployment stages and quantization.

See :mod:`deployment.quantization.schemes.base` for the design. Project-specific schemes
(e.g. BEVFusion's sparse ``SpconvBnFuseScheme``) live in the project bundle and subclass
:class:`QuantizationScheme`; the framework ships the interface, the plan, and the generic
dense Q/DQ scheme reusable by any Conv2d-based tower.
"""

from .base import QuantizationPlan, QuantizationScheme
from .dense_qdq import DenseQDQScheme

__all__ = [
    "QuantizationScheme",
    "QuantizationPlan",
    "DenseQDQScheme",
]
