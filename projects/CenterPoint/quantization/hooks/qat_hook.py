# Copyright (c) OpenMMLab. All rights reserved.
"""QAT training hook for CenterPoint with MMEngine."""

from typing import List, Optional, Set

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
            sensitive_layers=['pts_backbone.blocks.0.0'],
        ),
    ]
    ```

    Args:
        calibration_batches: Number of batches for initial calibration
        calibration_epoch: Epoch at which to run calibration (default: 0)
        freeze_bn: Whether to fuse and freeze BatchNorm layers
        sensitive_layers: List of layer names to skip quantization
        amax_method: Method for computing amax ('mse', 'entropy', 'percentile', 'max')
        quant_backbone: Whether to quantize pts_backbone
        quant_neck: Whether to quantize pts_neck
        quant_head: Whether to quantize pts_bbox_head
        quant_voxel_encoder: Whether to quantize pts_voxel_encoder

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
        calibration_batches: int = 100,
        calibration_epoch: int = 0,
        freeze_bn: bool = True,
        sensitive_layers: Optional[List[str]] = None,
        amax_method: str = "mse",
        quant_backbone: bool = True,
        quant_neck: bool = True,
        quant_head: bool = True,
        quant_voxel_encoder: bool = True,
    ):
        self.calibration_batches = calibration_batches
        self.calibration_epoch = calibration_epoch
        self.freeze_bn = freeze_bn
        self.sensitive_layers: Set[str] = set(sensitive_layers or [])
        self.amax_method = amax_method
        self.quant_backbone = quant_backbone
        self.quant_neck = quant_neck
        self.quant_head = quant_head
        self.quant_voxel_encoder = quant_voxel_encoder

        # State flags
        self._quantized = False
        self._calibrated = False

    def before_train(self, runner) -> None:
        """
        Insert Q/DQ nodes before training starts.

        This method:
        1. Optionally fuses BatchNorm layers
        2. Inserts quantization modules into the model

        Args:
            runner: MMEngine runner instance
        """
        from projects.CenterPoint.quantization.fusion import fuse_model_bn
        from projects.CenterPoint.quantization.replace import (
            quant_conv_module,
            quant_linear_module,
        )

        model = runner.model

        # Handle DataParallel/DistributedDataParallel wrappers
        if hasattr(model, "module"):
            model = model.module

        runner.logger.info("QATHook: Initializing quantization...")

        # Step 1: Fuse BatchNorm
        if self.freeze_bn:
            runner.logger.info("QATHook: Fusing BatchNorm layers...")
            model.eval()
            fuse_model_bn(model)
            model.train()

        # Step 2: Insert Q/DQ nodes
        runner.logger.info("QATHook: Inserting Q/DQ nodes...")

        if self.quant_backbone and hasattr(model, "pts_backbone"):
            quant_conv_module(model.pts_backbone, self.sensitive_layers, "pts_backbone")
            runner.logger.info("  - Quantized pts_backbone")

        if self.quant_neck and hasattr(model, "pts_neck"):
            quant_conv_module(model.pts_neck, self.sensitive_layers, "pts_neck")
            runner.logger.info("  - Quantized pts_neck")

        if self.quant_head and hasattr(model, "pts_bbox_head"):
            quant_conv_module(model.pts_bbox_head, self.sensitive_layers, "pts_bbox_head")
            runner.logger.info("  - Quantized pts_bbox_head")

        if self.quant_voxel_encoder and hasattr(model, "pts_voxel_encoder"):
            quant_linear_module(model.pts_voxel_encoder, self.sensitive_layers, "pts_voxel_encoder")
            runner.logger.info("  - Quantized pts_voxel_encoder")

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
            from projects.CenterPoint.quantization.calibration import CalibrationManager
            from projects.CenterPoint.quantization.utils import disable_quantization

            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            dataloader = runner.train_dataloader

            runner.logger.info(f"QATHook: Starting calibration with {self.calibration_batches} batches...")

            # Run calibration
            calibrator = CalibrationManager(model)
            calibrator.calibrate(
                dataloader,
                num_batches=self.calibration_batches,
                method=self.amax_method,
            )

            # Disable sensitive layers
            if self.sensitive_layers:
                runner.logger.info(f"QATHook: Disabling {len(self.sensitive_layers)} sensitive layers...")
                for layer_name in self.sensitive_layers:
                    try:
                        layer = dict(model.named_modules())[layer_name]
                        disable_quantization(layer).apply()
                        runner.logger.info(f"  - Disabled: {layer_name}")
                    except KeyError:
                        runner.logger.warning(f"  - Layer not found: {layer_name}")

            self._calibrated = True
            runner.logger.info("QATHook: Calibration complete")

    def after_train(self, runner) -> None:
        """
        Log quantization status after training.

        Args:
            runner: MMEngine runner instance
        """
        if self._quantized:
            from projects.CenterPoint.quantization.utils import count_quantizers

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
