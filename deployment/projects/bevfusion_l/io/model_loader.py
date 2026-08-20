"""BEVFusion model loading utilities for deployment.

Supports optional dense quantization (backbone, neck, head via modelopt TensorQuantizer
Q/DQ). The sparse encoder (``pts_middle_encoder``) always deploys in FP16 and is only SparseConv+BN
folded.

The plain (non-quantized) path stays on the shared :func:`build_mmdet3d_model` core plus the
refactor's ``fuse_spconv_bn_in_encoder`` fold; the quantized path builds the model, inserts the
dense Q/DQ tree via the shared :func:`build_bevfusion_plan`, and loads a PTQ/FP32 checkpoint.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.config.schema import QuantizationConfig
from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion_l.export.spconv_bn_fusion import fuse_spconv_bn_in_encoder
from deployment.quantization import get_tensor_quantizer_cls, move_quantizer_amax_to_device

logger = logging.getLogger(__name__)


def _require_lidar_only_bevfusion(model: torch.nn.Module) -> None:
    """Assert the loaded checkpoint is a LiDAR-only BEVFusion model.

    The ``bevfusion_l`` bundle only deploys the LiDAR path (voxels -> sparse encoder -> dense head);
    it has no camera/fusion export. A camera (``bevfusion_c``) or fusion (``bevfusion_cl``)
    checkpoint would trace a graph this bundle cannot serve, so fail loud once here at load with a
    clear message rather than deep inside ONNX export. ``pts_middle_encoder`` is the sparse encoder
    every export path and the PyTorch backend depend on, so its absence is also caught here.
    """
    if getattr(model, "fusion_layer", None) is not None:
        raise RuntimeError(
            "bevfusion_l deploys LiDAR-only BEVFusion, but the loaded checkpoint has a fusion_layer. "
            "Use a LiDAR-only checkpoint (a camera/fusion model needs a dedicated bevfusion_c / "
            "bevfusion_cl bundle)."
        )
    if getattr(model, "img_backbone", None) is not None:
        raise RuntimeError(
            "bevfusion_l deploys LiDAR-only BEVFusion, but the loaded checkpoint has an img_backbone. "
            "Use a LiDAR-only checkpoint (a camera/fusion model needs a dedicated bevfusion_c / "
            "bevfusion_cl bundle)."
        )
    if getattr(model, "pts_middle_encoder", None) is None:
        raise RuntimeError(
            "bevfusion_l requires a sparse pts_middle_encoder (LiDAR BEVFusion), but the loaded "
            "checkpoint has none."
        )


def _strip_module_prefix_state_dict(state_dict: dict, model: torch.nn.Module) -> dict:
    """If checkpoint keys use ``module.`` prefix but the model does not, strip the prefix."""
    if not state_dict:
        return state_dict
    model_keys = set(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not ckpt_keys:
        return state_dict
    prefixed = sum(1 for k in ckpt_keys if k.startswith("module."))
    model_prefixed = any(k.startswith("module.") for k in model_keys)
    if prefixed == len(ckpt_keys) and not model_prefixed:
        out = {k[len("module.") :]: v for k, v in state_dict.items()}
        logger.info("Stripped 'module.' prefix from %d checkpoint keys for load_state_dict", len(out))
        return out
    return state_dict


def _log_load_state_dict_result(result) -> None:
    """Log missing/unexpected keys from ``load_state_dict``, split by sparse vs. dense."""

    def _split(keys):
        sparse = [k for k in keys if k.startswith("pts_middle_encoder")]
        other = [k for k in keys if not k.startswith("pts_middle_encoder")]
        return sparse, other

    logger.info(
        "[load-state-dict] missing=%d, unexpected=%d",
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    for kind, keys in (("missing", result.missing_keys), ("unexpected", result.unexpected_keys)):
        if not keys:
            continue
        sparse, other = _split(keys)
        logger.info("[load-state-dict] sparse %s=%d, other %s=%d", kind, len(sparse), kind, len(other))
        if sparse:
            logger.info("[load-state-dict] sparse %s sample: %s", kind, sparse[:10])
        if other:
            logger.info("[load-state-dict] other %s sample: %s", kind, other[:10])


def build_bevfusion_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    quantization: Optional[QuantizationConfig] = None,
    *,
    fuse_spconv_bn: bool = False,
) -> torch.nn.Module:
    """Build a BEVFusion model from config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to .pth checkpoint file.
        device: Target device.
        quantization: Typed ``quantization`` section (parsed once by ``BaseDeploymentConfig``);
            ``None`` or ``enabled=False`` takes the plain FP path.
        fuse_spconv_bn: If True and ``quantization`` is not enabled, fuse each
            ``SparseConvolution`` + ``BatchNorm1d`` pair in ``pts_middle_encoder`` after load
            (eval-mode Conv-BN fold, a graph optimization for the sparse ONNX export).
            Ignored when ``quantization.enabled`` is True (PTQ already fuses sparse BN).

    Returns:
        Loaded and eval-mode BEVFusion model.

    Raises:
        RuntimeError: If the checkpoint is not a LiDAR-only BEVFusion model (see
            :func:`_require_lidar_only_bevfusion`).
    """
    # Imported for their side effect: registering BEVFusion and the deploy-only SparseConvolution
    # classes into the MMDet3D registries so ``MODELS.build`` resolves them for export/inference.
    # Deliberately lazy (inside the builder, not at module level): the SparseConvolution fork is
    # inference-only (its forward raises in training mode), and this module sits on the QAT hook's
    # import chain — QAT training must build the model on the stock spconv classes instead.
    import projects.BEVFusion.bevfusion  # noqa: F401
    import projects.SparseConvolution  # noqa: F401

    if quantization is not None and quantization.enabled:
        # Quantized path: build + insert Q/DQ + load PTQ/FP32 ourselves (the shared
        # build_mmdet3d_model cannot express the pre-load quant-tree insertion).
        init_default_scope("mmdet3d")
        torch_device = device.to_torch_device()
        model = MODELS.build(copy.deepcopy(model_cfg.model))
        model.to(torch_device)

        try:
            model = _load_with_quantization(model, checkpoint_path, torch_device, quantization)
        except Exception as e:
            logger.exception("Quantization pipeline failed (full traceback above). Summary: %s", e)
            logger.error(f"Quantization failed: {e}. See message below for whether a safe FP32 fallback is possible.")
            if quantization.ptq_checkpoint:
                raise RuntimeError(
                    "PTQ checkpoint load failed; cannot fall back to plain FP32 load_checkpoint. "
                    "Typical cause: the deploy PTQ load order must match "
                    "bevfusion_l/quantization/quantize.py (sparse BN fuse, then dense BN fuse + Q/DQ "
                    "insert, then load_state_dict). Fix the error above or set quantization.enabled=False "
                    "and use an FP32 checkpoint."
                ) from e
            # Non-PTQ quant failure: rebuild clean model and load FP32 checkpoint.
            logger.info("Falling back to FP32: rebuilding model from config and load_checkpoint.")
            model = MODELS.build(copy.deepcopy(model_cfg.model))
            model.to(torch_device)
            load_checkpoint(model, checkpoint_path, map_location=torch_device)

        model.eval()
        model.cfg = model_cfg
        _require_lidar_only_bevfusion(model)
        return model

    # Plain (FP32/FP16) path: shared build core + optional SparseConv-BN fold.
    model = build_mmdet3d_model(model_cfg, checkpoint_path, device)
    _require_lidar_only_bevfusion(model)

    if fuse_spconv_bn:
        encoder = getattr(model, "pts_middle_encoder", None)
        if encoder is not None:
            count = fuse_spconv_bn_in_encoder(encoder)
            logger.info("Fused %d SparseConv-BN pair(s) in pts_middle_encoder", count)

    return model


def _load_with_quantization(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    config: QuantizationConfig,
) -> torch.nn.Module:
    """Load a quantized BEVFusion model via the shared :class:`QuantizationPlan`.

    The plan (built by :func:`build_bevfusion_plan`) owns *how* each tower is quantized; this
    function only orchestrates *loading* (state_dict prep, weight-layout permutation, device
    placement, inference-mode toggles). The SAME plan is used by the PTQ producer, so the
    quantized module tree is identical on both sides and the PTQ ``state_dict`` lines up.

    Two modes:

    A) PTQ checkpoint (``ptq_checkpoint=True``): build the quantized tree via ``plan.prepare`` and
       then ``load_state_dict`` (the checkpoint carries calibrated ``_amax``).
    B) FP32 checkpoint: ``load_state_dict`` first, then ``plan.prepare`` inserts uncalibrated Q/DQ
       (dense only; would need runtime calibration).
    """
    from deployment.projects.bevfusion_l.quantization.plan import build_bevfusion_plan

    is_ptq = config.ptq_checkpoint

    if is_ptq:
        logger.info("Loading PTQ checkpoint (pre-calibrated Q/DQ nodes)...")

        # Rebuild the quantized module tree BEFORE load_state_dict, using the shared plan
        # (dense Q/DQ + BN fuse, sparse BN fuse). Identical to the PTQ producer.
        build_bevfusion_plan(config).prepare(model)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _strip_module_prefix_state_dict(state_dict, model)

        result = model.load_state_dict(state_dict, strict=False)
        _log_load_state_dict_result(result)

        num_amax = sum(1 for k in state_dict if "_amax" in k)
        logger.info(f"PTQ state_dict contains {num_amax} amax entries, {len(state_dict)} total keys")

        move_quantizer_amax_to_device(model, device)

        tensor_quantizer_cls = get_tensor_quantizer_cls()
        if tensor_quantizer_cls:
            loaded = 0
            for name, mod in model.named_modules():
                if isinstance(mod, tensor_quantizer_cls) and hasattr(mod, "_amax") and mod._amax is not None:
                    loaded += 1
            logger.info(f"PTQ checkpoint loaded: {loaded} quantizers have calibrated amax values")

        _set_tensor_quantizers_inference_mode(model)

    else:
        load_checkpoint(model, checkpoint_path, map_location=device)
        model.eval()
        logger.info("Applying dense quantization (uncalibrated) to BEVFusion model...")
        build_bevfusion_plan(config).prepare(model)

    logger.info("Quantization applied successfully")
    return model


def _set_tensor_quantizers_inference_mode(model: torch.nn.Module) -> int:
    """Match ``CalibrationManager._disable_calibration_mode`` after PTQ ``load_state_dict``.

    Newly inserted ``TensorQuantizer`` modules may still have calibration defaults
    (fake-quant off, stats on). That yields a different dense branch than the PTQ
    script after ``calibrator.calibrate`` and can drive mAP/near-zero despite a
    valid ``state_dict``.

    BEVFusion-only for now: the CenterPoint loader does not run this toggle (it instead validates
    amax positivity — ``_validate_quantizer_amax``). TODO(Docker): decide whether both loaders
    should run both steps and, if so, share this via ``deployment.quantization.core.utils``
    (spec.md §5.2 4B.3).
    """
    tensor_quantizer_cls = get_tensor_quantizer_cls()
    if tensor_quantizer_cls is None:
        return 0
    n = 0
    for module in model.modules():
        if not isinstance(module, tensor_quantizer_cls):
            continue
        try:
            if getattr(module, "_calibrator", None) is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            n += 1
        except Exception as ex:
            logger.debug("TensorQuantizer inference mode skip: %s", ex)
    if n:
        logger.info("Set %d TensorQuantizer modules to inference mode (post PTQ load)", n)
    return n
