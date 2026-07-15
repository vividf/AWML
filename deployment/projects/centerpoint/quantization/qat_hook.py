# Copyright (c) OpenMMLab. All rights reserved.
"""QAT training hook for CenterPoint with MMEngine."""

from typing import List, Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class QATHook(Hook):
    """
    Hook for Quantization-Aware Training (QAT) with CenterPoint.

    This hook integrates QAT into the MMEngine training loop by:
    1. Inserting Q/DQ nodes before training starts
    2. Fusing BatchNorm layers (optional)
    3. Calibrating quantizers at a specified epoch
    4. Disabling quantization for sensitive layers

    The hook should be added to the config's custom_hooks list:

    ```python
    custom_hooks = [
        dict(
            type='QATHook',
            calibration_batches=100,
            calibration_epoch=0,
            freeze_bn=True,
            keep_fp16=['pts_voxel_encoder', 'pts_backbone.stem'],
        ),
    ]
    ```

    Args:
        calibration_batches: Number of batches for initial calibration.
            If None or <= 0, calibrate on all available training batches.
        calibration_epoch: Epoch at which to run calibration (default: 0)
        freeze_bn: Whether to fuse and freeze BatchNorm layers
        keep_fp16: Glob patterns (subtree match) to keep in FP16 — the deploy config's ``keep_fp16``.
            The hook builds the SAME plan the PTQ producer / deploy loader build, so the quantized tree
            is identical, and disables any quantizers left inside these subtrees after calibration.
        disable_recipes: Recipe names to skip ("add" / "ese" / "maxpool") — the deploy config's
            ``disable_recipes``.
        amax_method: Method for computing amax ('mse', 'entropy', 'percentile', 'max')
        calib_cache_path: Optional path to a PTQ calibration cache (.calib). If provided,
            QATHook will load amax values from this cache and skip the initial calibration.

    Example:
        >>> # In config file
        >>> custom_hooks = [
        ...     dict(
        ...         type='QATHook',
        ...         calibration_batches=100,
        ...         calibration_epoch=0,
        ...         freeze_bn=True,
        ...     ),
        ... ]
    """

    priority = "NORMAL"

    def __init__(
        self,
        calibration_batches: Optional[int] = 100,
        calibration_epoch: int = 0,
        freeze_bn: bool = True,
        keep_fp16: Optional[List[str]] = None,
        disable_recipes: Optional[List[str]] = None,
        amax_method: str = "mse",
        calib_cache_path: Optional[str] = None,
    ):
        self.calibration_batches = calibration_batches
        self.calibration_epoch = calibration_epoch
        self.freeze_bn = freeze_bn
        self.keep_fp16: List[str] = list(keep_fp16 or [])
        self.disable_recipes: List[str] = list(disable_recipes or [])
        self.amax_method = amax_method
        self.calib_cache_path = calib_cache_path

        # State flags
        self._quantized = False
        self._calibrated = False

    def before_train(self, runner) -> None:
        """
        Insert Q/DQ nodes before training starts (optional BN fusion first).

        Args:
            runner: MMEngine runner instance
        """
        from deployment.config.schema import QuantizationConfig

        from .plan import build_centerpoint_plan

        model = runner.model

        # Handle DataParallel/DistributedDataParallel wrappers
        if hasattr(model, "module"):
            model = model.module

        runner.logger.info("QATHook: Initializing quantization...")

        # Fuse BN + insert Q/DQ via the SAME shared plan the PTQ / deploy paths use, so the QAT
        # module tree is identical to what will be deployed. ``sensitive_layers`` arrives already
        # resolved from the deploy config.
        runner.logger.info("QATHook: Fusing BatchNorm + inserting Q/DQ via shared CenterPoint plan...")
        config = QuantizationConfig(
            fuse_bn=self.freeze_bn,
            keep_fp16=tuple(self.keep_fp16),
            disable_recipes=tuple(self.disable_recipes),
        )
        build_centerpoint_plan(config).prepare(model)
        model.train()

        self._quantized = True
        runner.logger.info("QATHook: Quantization modules inserted")

    def before_train_epoch(self, runner) -> None:
        """
        Calibrate quantizers at the specified epoch.

        Args:
            runner: MMEngine runner instance
        """
        if not self._quantized:
            runner.logger.warning("QATHook: Model not quantized, skipping calibration")
            return

        if runner.epoch == self.calibration_epoch and not self._calibrated:
            from deployment.quantization.core.calibration import CalibrationManager
            from deployment.quantization.core.utils import disable_quantizers_in

            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            dataloader = runner.train_dataloader

            # Resolve "all batches" mode
            if self.calibration_batches is None or int(self.calibration_batches) <= 0:
                try:
                    effective_batches = len(dataloader)
                except Exception:
                    effective_batches = 100
            else:
                effective_batches = int(self.calibration_batches)

            # If calibration cache is provided, load it and skip calibration
            if self.calib_cache_path:
                runner.logger.info(f"QATHook: Loading calibration cache: {self.calib_cache_path}")
                calibrator = CalibrationManager(model)
                calibrator.load_calib_cache(self.calib_cache_path)
            else:
                runner.logger.info(f"QATHook: Starting calibration with {effective_batches} batches...")

                # Run calibration
                calibrator = CalibrationManager(model)
                calibrator.calibrate(
                    dataloader,
                    num_batches=effective_batches,
                    method=self.amax_method,
                )

            # Disable quantizers left in FP16 by keep_fp16 (recipes may have attached quantizers inside
            # those subtrees; the same expansion the plan used gives the concrete module names).
            from deployment.quantization import expand_keep_fp16

            skip_names = expand_keep_fp16(model, self.keep_fp16, log=False)
            if skip_names:
                runner.logger.info(f"QATHook: Disabling quantizers in {len(skip_names)} keep_fp16 modules...")
                disable_quantizers_in(model, skip_names)

            self._calibrated = True
            runner.logger.info("QATHook: Calibration complete")

    def after_train(self, runner) -> None:
        """
        Log quantization status after training.

        Args:
            runner: MMEngine runner instance
        """
        if self._quantized:
            from deployment.quantization.core.utils import count_quantizers

            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            counts = count_quantizers(model)
            runner.logger.info(
                f"QATHook: Training complete. "
                f"Quantizers: {counts['enabled']} enabled, "
                f"{counts['disabled']} disabled, "
                f"{counts['total']} total"
            )
