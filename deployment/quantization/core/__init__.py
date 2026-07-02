# Copyright (c) OpenMMLab. All rights reserved.
"""Model-agnostic quantization engine.

The generic building blocks that know only about ``nn.Conv2d`` / ``nn.Linear`` and the
pytorch-quantization toolkit — not about any particular model: quantized module subclasses
(:mod:`.modules`), BN fusion (:mod:`.fusion`), calibration (:mod:`.calibration`), the Q/DQ
module-replacement engine (:mod:`.replace`), and quantizer utilities (:mod:`.utils`).

Architecture-specific placement lives in :mod:`deployment.quantization.recipes`; the deployment
seam lives in :mod:`deployment.quantization.schemes`.
"""
