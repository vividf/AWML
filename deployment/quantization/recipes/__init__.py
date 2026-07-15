# Copyright (c) OpenMMLab. All rights reserved.
"""Model-architecture-specific quantization recipes.

The generic engine in :mod:`deployment.quantization.core` inserts Q/DQ into any Conv2d/Linear tower.
This package holds the parts that must know a specific backbone's block structure: the forward
hooks that reposition Q/DQ for TensorRT-friendly fusion (:mod:`.forward_hooks`) and the functions
that walk a model to attach quantizers + install those hooks (:mod:`.attach`).

No re-exports on purpose: every consumer imports from the concrete submodule
(``recipes.attach`` / ``recipes.forward_hooks``), which is also the only place these names are
maintained.
"""
