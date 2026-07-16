# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion QAT hook — thin registration of the shared :class:`QATHookBase`.

All training-loop logic lives in :mod:`deployment.quantization.qat_hook`; this subclass supplies
only the two project seams:

- the shared BEVFusion plan (dense Q/DQ + sparse SparseConv-BN fold — the same plan the PTQ
  producer and deploy loader build, so the QAT tree is identical by construction). The sparse
  encoder carries no fake-quant and deploys FP16; its weights still fine-tune during QAT
  (spec_qat.md §D7).
- the voxel-dtype-normalizing calibration forward shared with the PTQ producer.
"""

from mmengine.registry import HOOKS

from deployment.quantization.qat_hook import QATHookBase

from .calibration import calibration_forward
from .plan import build_bevfusion_plan


@HOOKS.register_module()
class BEVFusionQATHook(QATHookBase):
    """QAT hook for BEVFusion-L (see :class:`QATHookBase` for behavior and args)."""

    build_plan = staticmethod(build_bevfusion_plan)
    calib_forward_fn = staticmethod(calibration_forward)
