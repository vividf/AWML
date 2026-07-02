# Copyright (c) OpenMMLab. All rights reserved.
"""Model-architecture-specific quantization recipes.

The generic engine in :mod:`deployment.quantization.core` inserts Q/DQ into any Conv2d/Linear tower.
This package holds the parts that must know a specific backbone's block structure: the forward
hooks that reposition Q/DQ for TensorRT-friendly fusion (:mod:`.forward_hooks`) and the functions
that walk a model to attach quantizers + install those hooks (:mod:`.attach`).
"""

from .attach import (
    attach_ese_mul_identity_quantizer,
    attach_ese_pool_input_quantizer,
    attach_maxpool_input_quantizer,
    attach_quant_add,
)
from .forward_hooks import (
    BasicBlockForwardHook,
    ConvNeXtBlockForwardHook,
    OSAModuleForwardHook,
    QuantBeforePool,
    SparseBasicBlockForwardHook,
    eSEModuleForwardHook,
)

__all__ = [
    "attach_quant_add",
    "attach_ese_mul_identity_quantizer",
    "attach_ese_pool_input_quantizer",
    "attach_maxpool_input_quantizer",
    "QuantBeforePool",
    "BasicBlockForwardHook",
    "SparseBasicBlockForwardHook",
    "ConvNeXtBlockForwardHook",
    "OSAModuleForwardHook",
    "eSEModuleForwardHook",
]
