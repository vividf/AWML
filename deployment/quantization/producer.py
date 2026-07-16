# Copyright (c) OpenMMLab. All rights reserved.
"""Shared PTQ/QAT-producer helpers.

The one home for the producer-side plumbing that was previously copied verbatim between
``projects/centerpoint/quantization/quantize.py`` and ``projects/bevfusion_l/quantization/quantize.py``
(calibration-dataloader seeding/shuffle, checkpoint + calibration-cache save, quantization-library
logging init) — plus the QAT training driver (``run_qat_training``) and QAT checkpoint packaging
(``save_qat_checkpoint``), which the two project ``run_qat``s share so they differ only in project
constants (spec_qat.md §4 WP2/WP4). Each project's ``quantize.py`` keeps only its model-specific
steps; this module owns the identical mechanics — imitating the single-impl pattern of
``deployment/quantization/sparse`` + its project re-export.

Everything here is deliberately torch/mmengine-lazy: importing this module is cheap and safe on
hosts without the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple


def init_quant_logging() -> None:
    """Silence pytorch-quantization's absl logging (verbose INFO chatter) down to ERROR.

    No-op when absl is not installed. Call once at producer-CLI start.
    """
    try:
        from absl import logging as quant_logging

        quant_logging.set_verbosity(quant_logging.ERROR)
    except ImportError:
        pass


def build_calib_dataloader(
    cfg,
    *,
    batch_size: int,
    seed: Optional[int] = None,
    shuffle: bool = False,
    max_num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    """Build the calibration dataloader from ``cfg.val_dataloader`` with PTQ overrides.

    Mutates ``cfg.val_dataloader`` (best-effort — only when it is a plain dict), then builds the
    dataloader via mmengine's ``Runner.build_dataloader``. The overrides:

    - ``batch_size`` is always set (larger batches reduce calibration seed sensitivity).
    - ``seed`` (when given) seeds ``random`` / ``numpy`` / ``torch`` / ``torch.cuda`` so a shuffled
      calibration order is reproducible.
    - ``shuffle`` (when True) removes any configured ``sampler`` (mutually exclusive) and sets
      ``shuffle=True``.
    - ``max_num_workers`` / ``persistent_workers`` are optional per-project dataloader caps
      (BEVFusion caps workers at 4 and disables persistent workers; CenterPoint leaves both alone).

    Args:
        cfg: MMEngine config whose ``val_dataloader`` drives calibration.
        batch_size: Calibration batch size.
        seed: Optional RNG seed for reproducible shuffling.
        shuffle: Shuffle the calibration data.
        max_num_workers: Optional upper bound on ``num_workers``.
        persistent_workers: Optional override for ``persistent_workers``.

    Returns:
        The built calibration dataloader.
    """
    import torch
    from mmengine.runner import Runner

    if isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader["batch_size"] = batch_size

        if max_num_workers is not None:
            cfg.val_dataloader["num_workers"] = min(
                cfg.val_dataloader.get("num_workers", max_num_workers), max_num_workers
            )
        if persistent_workers is not None:
            cfg.val_dataloader["persistent_workers"] = persistent_workers

        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if shuffle:
            # Remove existing sampler to allow shuffle (they are mutually exclusive).
            if "sampler" in cfg.val_dataloader:
                del cfg.val_dataloader["sampler"]
            cfg.val_dataloader["shuffle"] = True

    return Runner.build_dataloader(cfg.val_dataloader)


def save_ptq_checkpoint(model, output: str, calibrator=None) -> Tuple[Path, Optional[Path]]:
    """Save the PTQ checkpoint (``{"state_dict": ...}``) and, when given, the calibration cache.

    The calibration cache is written next to the checkpoint with a ``.calib`` suffix. A failed
    cache save raises — the ``.calib`` file is what lets QAT skip recalibration, so losing it
    silently is worse than failing loud.

    Args:
        model: The calibrated, quantized model.
        output: Output checkpoint path.
        calibrator: Optional ``CalibrationManager``; when given, its cache is saved too.

    Returns:
        ``(checkpoint_path, calib_cache_path)`` — the cache path is None without a calibrator.
    """
    import torch

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)

    calib_path: Optional[Path] = None
    if calibrator is not None:
        calib_path = output_path.with_suffix(".calib")
        calibrator.save_calib_cache(str(calib_path))

    return output_path, calib_path


def save_qat_checkpoint(model, output: str, *, work_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """Package a finished QAT run into the PTQ-shaped artifact: ``{"state_dict"}`` + sibling ``.calib``.

    A raw mmengine work-dir checkpoint (optimizer state, schedulers, message hub) "happens to load"
    through the deploy loader, but that is not a contract — this emits exactly what
    :func:`save_ptq_checkpoint` emits, so PTQ and QAT checkpoints are interchangeable deploy-side
    (spec_qat.md §D4, the ``mto.save`` analogue).

    When ``work_dir`` contains a ``best_*.pth`` (mmengine ``CheckpointHook`` with ``save_best``),
    its weights are loaded (strict) before packaging — "best" is measured on the quantized model
    during the QAT val loop, i.e. in the deployed numeric regime. Otherwise the model's in-memory
    (last-epoch) weights are packaged.

    Args:
        model: The quantized model at the end of ``runner.train()`` (unwrapped).
        output: Output checkpoint path (e.g. ``..._qat.pth``).
        work_dir: The QAT training work dir, searched for a ``best_*.pth``.

    Returns:
        ``(checkpoint_path, calib_cache_path)``.
    """
    import torch

    from deployment.quantization.core.calibration import CalibrationManager

    if work_dir:
        best_ckpts = list(Path(work_dir).glob("best_*.pth"))
        if best_ckpts:
            best = max(best_ckpts, key=lambda p: p.stat().st_mtime)
            print(f"  Packaging best QAT checkpoint: {best}")
            ckpt = torch.load(str(best), map_location="cpu")
            model.load_state_dict(ckpt.get("state_dict", ckpt))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)

    # The amax buffers are already in the state_dict; the sibling ``.calib`` keeps the PTQ/QAT
    # artifact contract uniform and lets a later QAT run skip recalibration.
    calib_path = output_path.with_suffix(".calib")
    CalibrationManager(model).save_calib_cache(str(calib_path))

    return output_path, calib_path


def _pick_setting(cli_value, block_value, mode_name, flag, block_path, default=None):
    """CLI-wins merge for one producer setting: CLI flag → config block → recipe default → error."""
    if cli_value is not None:
        return cli_value
    if block_value is not None:
        return block_value
    if default is not None:
        return default
    raise SystemExit(f"{mode_name} needs {flag} (or {block_path} in the deploy config) — neither was given.")


def resolve_qat_settings(args, config, deploy_checkpoint_path: Optional[str]) -> dict:
    """Merge producer-CLI flags over the deploy config's ``quantization.qat`` block (CLI wins).

    Shared by both projects' ``quantize.py qat`` commands. Expects the argparse namespace to carry
    ``config`` / ``checkpoint`` / ``epochs`` / ``lr`` / ``calibrate_samples`` / ``ptq_calib_cache`` /
    ``work_dir`` / ``output`` (all defaulting to None).

    Reference recipe defaults (epochs=10, lr=1e-4) apply only when there is no ``qat`` block at all —
    a block must state epochs/lr explicitly (:class:`~deployment.config.schema.QATConfig` enforces it).

    Returns:
        Dict with keys ``train_cfg``, ``checkpoint``, ``epochs``, ``lr``, ``calibrate_samples``,
        ``calib_cache``, ``work_dir``, ``output``.

    Raises:
        SystemExit: When a required setting is available from neither CLI nor config.
    """
    qat = config.qat

    def _pick(cli_value, block_value, flag, block_key, default=None):
        return _pick_setting(cli_value, block_value, "QAT", flag, f"quantization.qat.{block_key}", default)

    return dict(
        train_cfg=_pick(args.config, qat.train_cfg if qat else None, "--config", "train_cfg"),
        checkpoint=_pick(args.checkpoint, qat.checkpoint if qat else None, "--checkpoint", "checkpoint"),
        epochs=_pick(args.epochs, qat.epochs if qat else None, "--epochs", "epochs", default=10),
        lr=_pick(args.lr, qat.lr if qat else None, "--lr", "lr", default=1e-4),
        calibrate_samples=(
            args.calibrate_samples if args.calibrate_samples is not None else (qat.calibrate_samples if qat else None)
        ),
        calib_cache=args.ptq_calib_cache or (qat.calib_cache if qat else None),
        work_dir=args.work_dir or (qat.work_dir if qat else None),
        output=_pick_setting(args.output, deploy_checkpoint_path, "QAT", "--output", "the top-level checkpoint_path"),
    )


def resolve_ptq_settings(
    args,
    config,
    deploy_checkpoint_path: Optional[str],
    deploy_model_cfg: Optional[str] = None,
    *,
    default_calibrate_samples: int,
) -> dict:
    """Merge producer-CLI flags over the deploy config's ``quantization.ptq`` block (CLI wins).

    Shared by both projects' ``quantize.py ptq`` commands — the exact sibling of
    :func:`resolve_qat_settings`, so PTQ and QAT runs are config-driven the same way. Expects the
    argparse namespace to carry ``config`` / ``checkpoint`` / ``calibrate_samples`` / ``batch_size`` /
    ``calib_seed`` / ``calib_shuffle`` / ``output`` (all defaulting to None).

    ``default_calibrate_samples`` is the project's reference default (CenterPoint 100, BEVFusion 256 —
    the historical CLI defaults) and applies only when there is no ``ptq`` block at all — a block must
    state ``calibrate_samples`` explicitly (:class:`~deployment.config.schema.PTQConfig` enforces it).

    The model config and the output are the deploy config's top-level artifact-manifest keys:
    ``model_cfg`` (PTQ calibrates against the same model config the artifact deploys with) and
    ``checkpoint_path`` (the producer emits exactly the artifact the deploy config expects) —
    ``--config`` / ``--output`` override them.

    Returns:
        Dict with keys ``model_cfg``, ``checkpoint``, ``calibrate_samples``, ``batch_size``,
        ``calib_seed``, ``calib_shuffle``, ``output``.

    Raises:
        SystemExit: When a required setting is available from neither CLI nor config.
    """
    ptq = config.ptq

    def _pick(cli_value, block_value, flag, block_key, default=None):
        return _pick_setting(cli_value, block_value, "PTQ", flag, f"quantization.ptq.{block_key}", default)

    return dict(
        model_cfg=_pick_setting(args.config, deploy_model_cfg, "PTQ", "--config", "the top-level model_cfg"),
        checkpoint=_pick(args.checkpoint, ptq.checkpoint if ptq else None, "--checkpoint", "checkpoint"),
        calibrate_samples=_pick(
            args.calibrate_samples,
            ptq.calibrate_samples if ptq else None,
            "--calibrate-samples",
            "calibrate_samples",
            default=default_calibrate_samples,
        ),
        batch_size=_pick(args.batch_size, ptq.batch_size if ptq else None, "--batch-size", "batch_size", default=1),
        # None is a valid resolved value for both (no seeding / no shuffle), so no _pick here.
        calib_seed=args.calib_seed if args.calib_seed is not None else (ptq.calib_seed if ptq else None),
        calib_shuffle=(
            args.calib_shuffle if args.calib_shuffle is not None else bool(ptq.calib_shuffle if ptq else False)
        ),
        output=_pick_setting(args.output, deploy_checkpoint_path, "PTQ", "--output", "the top-level checkpoint_path"),
    )


def run_qat_training(
    *,
    train_cfg_path: str,
    checkpoint: str,
    hook_import: str,
    hook_type: str,
    quant_config,
    epochs: int,
    lr: float,
    output: str,
    batch_size: int = 1,
    calibration_batches: int = 0,
    calib_cache: Optional[str] = None,
    work_dir: Optional[str] = None,
    extra_imports: Sequence[str] = (),
) -> Tuple[Path, Path]:
    """Drive one QAT fine-tune run: config surgery → hook injection → train → package.

    The shared body of both projects' ``quantize.py qat`` commands. Project identity enters only
    through ``hook_import`` / ``hook_type`` (the registered :class:`~deployment.quantization.qat_hook.
    QATHookBase` subclass) and ``extra_imports`` (model-registry imports the train config may need).
    Placement (``keep_fp16`` / ``disable_recipes`` / ``fuse_bn``) flows from ``quant_config`` — the
    deploy config stays the single source of truth, so the QAT tree matches PTQ and deploy by
    construction.

    Hard policies (spec_qat.md §D5, §6 R3/R4):

    - **AMP is always off** — ``AmpOptimWrapper`` is downgraded to ``OptimWrapper``. Both references
      run QAT in fp32; fake-quant under autocast is a numerics trap.
    - **Resume is refused** — ``resume`` restores into the *unquantized* tree before the hook
      mutates it. Parked for v2 (hook ``before_run`` prepare); failing loud beats failing deep
      inside ``load_state_dict``.
    - **EMA hooks are stripped** — neither reference QAT uses EMA, and an EMA copy of a mutated
      tree is untested surface.

    Args:
        train_cfg_path: The mm training config to fine-tune with.
        checkpoint: FP training checkpoint to initialize from (``load_from``).
        hook_import: Dotted module path of the project's QAT hook (added to ``custom_imports``).
        hook_type: Registered hook type name (e.g. ``"QATHook"``, ``"BEVFusionQATHook"``).
        quant_config: The deploy config's parsed :class:`~deployment.config.schema.QuantizationConfig`.
        epochs: Fine-tune epochs (reference: ~10% of original training).
        lr: Fine-tune learning rate (reference: 1e-4).
        output: Path for the packaged ``{"state_dict"}`` checkpoint (sibling ``.calib`` is emitted).
        batch_size: Train/val dataloader batch size override.
        calibration_batches: Epoch-0 calibration batches; ``0`` → all training batches.
        calib_cache: Optional ``.calib`` cache — skips the epoch-0 calibration pass.
        work_dir: Training work dir (default: ``<output parent>/qat_training``).
        extra_imports: Additional ``custom_imports`` modules (project model registries).

    Returns:
        ``(checkpoint_path, calib_cache_path)`` from :func:`save_qat_checkpoint`.
    """
    from mmengine.config import Config

    cfg = Config.fromfile(train_cfg_path)

    if bool(cfg.get("resume", False)):
        raise RuntimeError(
            "Resuming a QAT run is not supported (v1): mmengine restores the checkpoint into the "
            "unquantized tree before the QAT hook mutates it. Start a fresh QAT run from an FP "
            "checkpoint, or reuse a .calib cache to skip recalibration (spec_qat.md §6 R3)."
        )

    # custom_imports: hook module + project model registries.
    if not hasattr(cfg, "custom_imports"):
        cfg.custom_imports = dict(imports=[], allow_failed_imports=False)
    if "imports" not in cfg.custom_imports:
        cfg.custom_imports["imports"] = []
    for module_path in (*extra_imports, hook_import):
        if module_path not in cfg.custom_imports["imports"]:
            cfg.custom_imports["imports"].append(module_path)

    # AMP always off (spec_qat.md §D5): both references run QAT in fp32.
    if cfg.optim_wrapper.get("type") == "AmpOptimWrapper":
        print("QAT: AmpOptimWrapper detected — forcing OptimWrapper (QAT runs in fp32; spec_qat.md D5).")
        cfg.optim_wrapper.type = "OptimWrapper"
        cfg.optim_wrapper.pop("dtype", None)
        cfg.optim_wrapper.pop("loss_scale", None)

    # Training overrides.
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.train_cfg.max_epochs = epochs
    if isinstance(getattr(cfg, "train_dataloader", None), dict):
        cfg.train_dataloader["batch_size"] = batch_size
    if isinstance(getattr(cfg, "val_dataloader", None), dict):
        cfg.val_dataloader["batch_size"] = batch_size

    cfg.work_dir = work_dir or str(Path(output).parent / "qat_training")

    # Strip EMA hooks (spec_qat.md §6 R4): the references have none, and an EMA copy of the
    # hook-mutated tree is untested surface.
    if hasattr(cfg, "custom_hooks") and cfg.custom_hooks:
        kept = [h for h in cfg.custom_hooks if "EMA" not in str(h.get("type", ""))]
        if len(kept) != len(cfg.custom_hooks):
            print("QAT: stripped EMA hook(s) from custom_hooks (spec_qat.md R4).")
        cfg.custom_hooks = kept
    else:
        cfg.custom_hooks = []

    cfg.custom_hooks.append(
        dict(
            type=hook_type,
            calibration_batches=calibration_batches,
            calibration_epoch=0,
            fuse_bn=quant_config.fuse_bn,
            keep_fp16=list(quant_config.keep_fp16),
            disable_recipes=list(quant_config.disable_recipes),
            calib_cache_path=calib_cache,
        )
    )

    cfg.load_from = checkpoint

    print("\nQAT training configuration prepared.")
    print(f"Work directory: {cfg.work_dir}")
    if calib_cache:
        print(f"Using calibration cache: {calib_cache} (skips the epoch-0 calibration pass)")
    print("\nStarting QAT training...")
    print("=" * 80)

    # Import custom modules before building the runner (populates registries).
    for module_path in cfg.custom_imports["imports"]:
        try:
            __import__(module_path)
            print(f"  Imported: {module_path}")
        except ImportError as e:
            if not cfg.custom_imports.get("allow_failed_imports", False):
                raise ImportError(f"Failed to import module '{module_path}'. Error: {e}") from e
            print(f"  Warning: Failed to import {module_path}: {e}")

    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()

    model = runner.model
    if hasattr(model, "module"):
        model = model.module

    print("\nPackaging QAT checkpoint (PTQ-shaped artifact)...")
    return save_qat_checkpoint(model, output, work_dir=cfg.work_dir)
