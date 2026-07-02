# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint-specific quantization.

Model-specific composition on top of the model-agnostic :mod:`deployment.quantization` framework:
``quant_model`` (:mod:`.quant_model`) applies the generic engine + recipes to CenterPoint's named
components, the PTQ pipeline lives in :mod:`.ptq`, the QAT training hook in :mod:`.qat_hook`, and the
offline producer CLI is :mod:`.quantize`.
"""
