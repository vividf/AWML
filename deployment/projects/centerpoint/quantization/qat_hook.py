# Copyright (c) OpenMMLab. All rights reserved.
"""CenterPoint QAT hook — thin registration of the shared :class:`QATHookBase`.

All training-loop logic lives in :mod:`deployment.quantization.qat_hook`; this subclass supplies
only the project seam: the shared CenterPoint plan (the same one the PTQ producer and deploy
loader build, so the QAT tree is identical by construction). CenterPoint's calibration uses the
default ``model.test_step`` forward, so no ``calib_forward_fn`` is needed.

Registered under the historical name ``QATHook`` (existing configs / CLI wiring keep working).
"""

from mmengine.registry import HOOKS

from deployment.quantization.qat_hook import QATHookBase

from .plan import build_centerpoint_plan


@HOOKS.register_module()
class QATHook(QATHookBase):
    """QAT hook for CenterPoint (see :class:`QATHookBase` for behavior and args)."""

    build_plan = staticmethod(build_centerpoint_plan)
