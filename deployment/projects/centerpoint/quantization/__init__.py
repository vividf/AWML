# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint-specific quantization.

Model-specific composition on top of the model-agnostic :mod:`deployment.quantization` framework:

* :mod:`.quant_model` applies the generic engine + recipes to CenterPoint's named components.
* :mod:`.schemes` / :mod:`.plan` wrap that composition as a :class:`QuantizationScheme` /
  :func:`build_centerpoint_plan`, so the deploy loader, the PTQ producer, and the QAT hook all
  build an identical quantized module tree from one place.
* :mod:`.qat_hook` is the MMEngine QAT training hook.
* :mod:`.quantize` is the offline PTQ / QAT producer CLI.
"""

from .plan import build_centerpoint_plan
from .schemes import CenterPointDenseScheme

__all__ = [
    "CenterPointDenseScheme",
    "build_centerpoint_plan",
]
