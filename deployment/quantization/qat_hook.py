# Copyright (c) OpenMMLab. All rights reserved.
"""Shared QAT training hook (mmengine) — the one home for the QAT training-loop logic.

``QATHookBase`` holds everything project-agnostic about QAT: plan-prepare before training,
epoch-0 calibration (or ``.calib`` cache load), keep_fp16 quantizer disable, and the end-of-training
status log. Each project registers a thin subclass next to its plan supplying exactly the two seams
that differ (same shape as ``SampleExtractor`` / ``ComponentBuilder`` on the export side):

- :attr:`build_plan` — the project's ``build_<model>_plan`` (the sacred invariant: the SAME plan the
  PTQ producer and deploy loader build, so the QAT tree is identical by construction).
- :attr:`calib_forward_fn` — optional ``fn(model, batch)`` for models whose calibration forward
  needs project-specific input massaging (BEVFusion's voxel-dtype normalization); ``None`` falls
  back to ``CalibrationManager``'s default ``model.test_step`` path.

The QAT method itself is frozen-amax STE fine-tuning: calibrated scales stay fixed buffers and only
the weights train — the production method in both CUDA-CenterPoint and modelopt (spec_qat.md §0/§2).
There is deliberately no learnable-amax machinery here.

This module imports mmengine at import time and is therefore NOT re-exported from
``deployment.quantization`` (which stays import-cheap on hosts without the training stack); training
configs reach it via ``custom_imports`` on the concrete subclass module.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from mmengine.hooks import Hook

from deployment.config.schema import QuantizationConfig


class QATHookBase(Hook):
    """Project-agnostic QAT hook body. Do not register this class — register a project subclass.

    Subclass contract::

        @HOOKS.register_module()
        class MyProjectQATHook(QATHookBase):
            build_plan = staticmethod(build_myproject_plan)
            calib_forward_fn = staticmethod(my_forward)   # optional

    Args:
        calibration_batches: Number of batches for the epoch-0 calibration.
            None or <= 0 → calibrate on all available training batches.
        calibration_epoch: Epoch at which to run calibration (default: 0).
        fuse_bn: Fold BatchNorm before inserting Q/DQ — must mirror the deploy config's
            ``fuse_bn`` (the producer CLI passes it through; a mismatch would desynchronize
            the QAT tree from the deployed tree).
        keep_fp16: The deploy config's ``keep_fp16`` glob patterns (subtree match). The hook
            builds the SAME plan the PTQ producer / deploy loader build, then disables any
            quantizers the recipes left inside these subtrees after calibration.
        disable_recipes: The deploy config's ``disable_recipes`` ("add" / "ese" / "maxpool").
        amax_method: Method for computing amax ("mse", "entropy", "percentile", "max").
        calib_cache_path: Optional PTQ calibration cache (``.calib``). When given, amax values
            are loaded from it and the epoch-0 calibration pass is skipped.
    """

    priority = "NORMAL"

    #: Project seam 1 — ``build_plan(config: QuantizationConfig) -> QuantizationPlan``.
    build_plan: Callable = None  # type: ignore[assignment]

    #: Project seam 2 — optional ``calib_forward_fn(model, batch)`` used during calibration.
    calib_forward_fn: Optional[Callable] = None

    def __init__(
        self,
        calibration_batches: Optional[int] = 100,
        calibration_epoch: int = 0,
        fuse_bn: bool = True,
        keep_fp16: Optional[List[str]] = None,
        disable_recipes: Optional[List[str]] = None,
        amax_method: str = "mse",
        calib_cache_path: Optional[str] = None,
    ):
        if type(self).build_plan is None:
            raise TypeError(
                f"{type(self).__name__} must set `build_plan` to the project's build_<model>_plan "
                "(register a project subclass; QATHookBase itself is not usable)."
            )
        self.calibration_batches = calibration_batches
        self.calibration_epoch = calibration_epoch
        self.fuse_bn = fuse_bn
        self.keep_fp16: List[str] = list(keep_fp16 or [])
        self.disable_recipes: List[str] = list(disable_recipes or [])
        self.amax_method = amax_method
        self.calib_cache_path = calib_cache_path

        # State flags
        self._quantized = False
        self._calibrated = False

    @staticmethod
    def _unwrap(model):
        """Unwrap DataParallel / DistributedDataParallel."""
        return model.module if hasattr(model, "module") else model

    def before_train(self, runner) -> None:
        """Fuse BN + insert Q/DQ via the shared project plan before training starts."""
        from mmengine.dist import get_world_size

        # v1 hard boundary (spec_qat.md §6 R2): mmengine wraps the model in DDP at Runner init,
        # BEFORE this hook mutates the tree — module replacement inside a wrapped DDP model can
        # desynchronize reducer buckets. Fail loud instead of training wrong.
        if get_world_size() > 1:
            raise RuntimeError(
                "QAT supports single-GPU training only (v1): the hook mutates the module tree "
                "after the DDP wrap, which desynchronizes DDP reducer buckets. Run on one GPU "
                "(spec_qat.md §6 R2)."
            )

        model = self._unwrap(runner.model)

        runner.logger.info("QATHook: Initializing quantization...")

        # Fuse BN + insert Q/DQ via the SAME shared plan the PTQ / deploy paths use, so the QAT
        # module tree is identical to what will be deployed.
        runner.logger.info("QATHook: Fusing BatchNorm + inserting Q/DQ via the shared project plan...")
        config = QuantizationConfig(
            fuse_bn=self.fuse_bn,
            keep_fp16=tuple(self.keep_fp16),
            disable_recipes=tuple(self.disable_recipes),
        )
        type(self).build_plan(config).prepare(model)
        model.train()

        self._quantized = True
        runner.logger.info("QATHook: Quantization modules inserted")

    def before_train_epoch(self, runner) -> None:
        """Calibrate quantizers at the configured epoch (or load the ``.calib`` cache)."""
        if not self._quantized:
            runner.logger.warning("QATHook: Model not quantized, skipping calibration")
            return

        if runner.epoch == self.calibration_epoch and not self._calibrated:
            from deployment.quantization.core.calibration import CalibrationManager
            from deployment.quantization.core.utils import disable_quantizers_in

            model = self._unwrap(runner.model)

            # If calibration cache is provided, load it and skip calibration
            if self.calib_cache_path:
                runner.logger.info(f"QATHook: Loading calibration cache: {self.calib_cache_path}")
                calibrator = CalibrationManager(model)
                calibrator.load_calib_cache(self.calib_cache_path)
            else:
                # Calibrate on a CLEAN (val / test-pipeline, un-augmented) dataloader, not the
                # augmented train loader. Rationale (spec_qat.md §D8): this is exactly what PTQ
                # calibrates on, so the QAT amax matches the proven-good PTQ amax; and it avoids
                # train augmentation (GT-paste / rot / flip) feeding degenerate inputs through
                # ``test_step`` that can poison a histogram with Inf and yield NaN fake-quant.
                dataloader, source = self._calibration_dataloader(runner)

                if self.calibration_batches is None or int(self.calibration_batches) <= 0:
                    try:
                        effective_batches = len(dataloader)
                    except Exception:
                        effective_batches = 100
                else:
                    effective_batches = int(self.calibration_batches)

                runner.logger.info(
                    f"QATHook: Calibrating on the {source} dataloader with {effective_batches} batches..."
                )
                calibrator = CalibrationManager(model)
                calibrator.calibrate(
                    dataloader,
                    num_batches=effective_batches,
                    method=self.amax_method,
                    forward_fn=type(self).calib_forward_fn,
                )

            # Disable quantizers left in FP16 by keep_fp16 (recipes may have attached quantizers inside
            # those subtrees; the same expansion the plan used gives the concrete module names).
            from deployment.quantization import expand_keep_fp16

            skip_names = expand_keep_fp16(model, self.keep_fp16, log=False)
            if skip_names:
                runner.logger.info(f"QATHook: Disabling quantizers in {len(skip_names)} keep_fp16 modules...")
                disable_quantizers_in(model, skip_names)

            # Fail loud (with layer names) on unhealthy amax instead of NaN-ing deep inside the loss
            # (e.g. the Hungarian matcher's ``linear_sum_assignment``).
            self._assert_amax_healthy(model, runner.logger)

            self._calibrated = True
            runner.logger.info("QATHook: Calibration complete")

    def _calibration_dataloader(self, runner):
        """Return ``(dataloader, source_label)`` for calibration — prefer the clean val loader.

        Falls back to the train loader (with a warning) when no val dataloader is configured.
        """
        val_dl = getattr(runner, "val_dataloader", None)
        if val_dl is not None:
            try:
                if isinstance(val_dl, dict):
                    from mmengine.runner import Runner

                    return Runner.build_dataloader(val_dl), "val"
                return val_dl, "val"
            except Exception as e:  # noqa: BLE001 — any build failure falls back to the train loader
                runner.logger.warning(
                    f"QATHook: could not use the val dataloader for calibration ({e}); "
                    "falling back to the train dataloader (augmented — amax may drift from PTQ)."
                )
        else:
            runner.logger.warning(
                "QATHook: no val dataloader configured; calibrating on the (augmented) train "
                "dataloader. Prefer reusing the PTQ .calib cache for amax parity (spec_qat.md §D8)."
            )
        return runner.train_dataloader, "train"

    @staticmethod
    def _assert_amax_healthy(model, logger) -> None:
        """Guard against calibration producing NaN/Inf/None/non-positive ``_amax``.

        Only enabled quantizers matter (a disabled one never fake-quants). NaN/Inf/None indicate a
        poisoned or missed calibration → raise with the offending layer names. A finite but
        non-positive amax (a genuinely dead / all-zero channel) is clamped to a small epsilon and
        warned, since that channel quantizes to ~0 either way.
        """
        import torch

        from deployment.quantization.core.utils import get_tensor_quantizer_cls

        tq_cls = get_tensor_quantizer_cls()
        if tq_cls is None:
            return

        fatal, clamped = [], []
        for name, mod in model.named_modules():
            if not isinstance(mod, tq_cls) or getattr(mod, "_disabled", False):
                continue
            amax = getattr(mod, "_amax", None)
            if amax is None:
                fatal.append((name, "amax=None (never calibrated)"))
            elif not torch.isfinite(amax).all():
                fatal.append((name, "amax has NaN/Inf (poisoned calibration input)"))
            elif float(amax.min()) <= 0.0:
                mod._amax = amax.clamp(min=1e-8)
                clamped.append(name)

        if clamped:
            logger.warning(
                f"QATHook: clamped non-positive amax to 1e-8 in {len(clamped)} quantizer(s) "
                f"(dead/all-zero channels): {clamped[:10]}{' ...' if len(clamped) > 10 else ''}"
            )
        if fatal:
            preview = "\n  ".join(f"{n}: {r}" for n, r in fatal[:20])
            raise RuntimeError(
                f"QAT calibration produced {len(fatal)} unhealthy quantizer amax value(s); "
                "fake-quant would emit NaN/Inf and crash the loss (e.g. the Hungarian matcher). "
                f"First offenders:\n  {preview}\n"
                "Fixes: reuse the PTQ .calib cache (--ptq-calib-cache <model>_ptq.calib), or "
                "calibrate on clean val data (the default), or add the layer to keep_fp16."
            )

    def after_train(self, runner) -> None:
        """Log quantization status after training."""
        if self._quantized:
            from deployment.quantization.core.utils import count_quantizers

            counts = count_quantizers(self._unwrap(runner.model))
            runner.logger.info(
                f"QATHook: Training complete. "
                f"Quantizers: {counts['enabled']} enabled, "
                f"{counts['disabled']} disabled, "
                f"{counts['total']} total"
            )
