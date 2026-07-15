"""BEVFusion model loading utilities for deployment.

Supports optional quantization:
- Dense parts (backbone, neck, head): pytorch_quantization (TensorQuantizer Q/DQ)
- Sparse encoder (pts_middle_encoder): NVIDIA TensorQuantizer Sparse INT8 (including ``conv_out``).

The plain (non-quantized) path stays on the shared :func:`build_mmdet3d_model` core plus the
refactor's ``fuse_spconv_bn_in_encoder`` fold; the quantized path builds the model, inserts the
Q/DQ tree via the shared :func:`build_bevfusion_plan`, and loads a PTQ/FP32 checkpoint.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

# Imported for their side effect: registering BEVFusion and SparseConvolution modules into the
# MMDet3D registries so ``MODELS.build`` can resolve them during export.
import projects.BEVFusion.bevfusion  # noqa: F401
import projects.SparseConvolution  # noqa: F401
from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion_l.export.spconv_bn_fusion import fuse_spconv_bn_in_encoder

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


def _permute_sparse_encoder_weights_to_match_model(
    state_dict: dict,
    model: torch.nn.Module,
) -> None:
    """Permute pts_middle_encoder 5D weights from (C_in, C_out, K, K, K) to (C_out, K, K, K, C_in) when checkpoint and model shapes differ.

    PTQ may save sparse conv in one layout; the built model expects KRSC (out, k, k, k, in). Mutates state_dict in place.
    """
    model_sd = model.state_dict()
    for key in list(state_dict.keys()):
        if not key.startswith("pts_middle_encoder") or not key.endswith(".weight"):
            continue
        if key not in model_sd:
            continue
        v = state_dict[key]
        m = model_sd[key]
        if v.dim() != 5 or m.dim() != 5 or v.shape == m.shape:
            continue
        # Checkpoint (C_in, C_out, Kz, Ky, Kx) -> (C_out, Kz, Ky, Kx, C_in) to match KRSC
        perm = v.permute(1, 2, 3, 4, 0)
        if perm.shape == m.shape:
            state_dict[key] = perm
            logger.info(
                "Permuted sparse encoder weight layout for %s: %s -> %s", key, tuple(v.shape), tuple(perm.shape)
            )
        else:
            logger.warning(
                "Cannot fix pts_middle_encoder weight %s: ckpt %s perm-> %s, model %s",
                key,
                tuple(v.shape),
                tuple(perm.shape),
                tuple(m.shape),
            )


def verify_spconv_int8_encoder(model: torch.nn.Module) -> Dict[str, Any]:
    """Summarize sparse encoder INT8 readiness (NVIDIA ``TensorQuantizer`` on SparseConvolution)."""
    enc = getattr(model, "pts_middle_encoder", None)
    n_quant_convs = 0
    if enc is not None:
        for _name, mod in enc.named_modules():
            if hasattr(mod, "_input_quantizer") and hasattr(mod, "_weight_quantizer"):
                n_quant_convs += 1
    return {
        "is_int8": n_quant_convs > 0,
        "encoder_type": type(enc).__name__ if enc is not None else "None",
        "nvidia_quantized_sparse_conv_count": n_quant_convs,
        "total_param_count": sum(p.numel() for p in enc.parameters()) if enc is not None else 0,
    }


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


def _verify_spconv_scale_buffers(model: torch.nn.Module, ckpt_state_dict: dict) -> None:
    """Verify that spconv INT8 quantization params were loaded from checkpoint.

    Primarily for NVIDIA checkpoints (_amax keys). Legacy checkpoints may still show scale/zero_point keys.
    """
    enc = getattr(model, "pts_middle_encoder", None)
    if enc is None:
        logger.warning("[spconv-quant-check] NO pts_middle_encoder found!")
        return

    ckpt_sparse_keys = [k for k in ckpt_state_dict if k.startswith("pts_middle_encoder.")]
    ckpt_amax_keys = [k for k in ckpt_sparse_keys if "_amax" in k]
    ckpt_scale_keys = [k for k in ckpt_sparse_keys if "scale" in k or "zero_point" in k]

    if ckpt_amax_keys:
        logger.info("[spconv-quant-check] NVIDIA approach: checkpoint has %d _amax keys", len(ckpt_amax_keys))
        sample_keys = ckpt_amax_keys[:5]
    elif ckpt_scale_keys:
        logger.info("[spconv-quant-check] legacy approach: checkpoint has %d scale/zp keys", len(ckpt_scale_keys))
        sample_keys = ckpt_scale_keys[:5]
    else:
        logger.warning(
            "[spconv-quant-check] no _amax or scale/zp keys in checkpoint (has %d pts_middle_encoder keys total)",
            len(ckpt_sparse_keys),
        )
        sample_keys = []
    for k in sample_keys:
        v = ckpt_state_dict[k]
        logger.info("  %s shape=%s first3=%s", k, tuple(v.shape), v.flatten().tolist()[:3])

    if ckpt_scale_keys:
        logger.info("[spconv-scale-check] ckpt scale/zp keys sample: %s", ckpt_scale_keys[:5])

    model_all_keys = [f"pts_middle_encoder.{k}" for k in dict(enc.named_parameters())]
    model_all_keys += [f"pts_middle_encoder.{k}" for k in dict(enc.named_buffers())]
    model_scale_keys = [k for k in model_all_keys if "scale" in k or "zero_point" in k]
    if model_scale_keys:
        logger.info("[spconv-scale-check] model scale/zp keys sample: %s", model_scale_keys[:5])


def _import_tensor_quantizer():
    """Lazily import TensorQuantizer from pytorch_quantization.

    Importing pytorch_quantization pulls in ``absl.logging``, which hijacks the root
    logger: it installs its own handler (only WARNING+ reaches stderr) and can raise the
    root level. The net effect is that every log record emitted afterwards — the rest of
    ONNX/TensorRT export AND the entire evaluation phase — silently disappears from both
    console and the deployment log file.

    The absl hijack can also be triggered *earlier* on this code path — the transitive
    ``from deployment.quantization import ...`` in the dense-quant load imports
    pytorch_quantization before this function is ever called — so we cannot rely on a
    snapshot taken here. Instead we delegate to ``restore_deployment_logging``, which
    re-asserts the canonical logging config captured by the CLI at ``setup_logging`` time
    (a no-op when the CLI did not configure logging, e.g. unit tests).
    """
    try:
        from pytorch_quantization.nn import TensorQuantizer

        return TensorQuantizer
    except ImportError:
        return None
    finally:
        try:
            from deployment.cli.args import restore_deployment_logging

            restore_deployment_logging()
        except Exception:
            pass


def build_bevfusion_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    quantization: Optional[dict] = None,
    *,
    fuse_spconv_bn: bool = False,
) -> torch.nn.Module:
    """Build a BEVFusion model from config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to .pth checkpoint file.
        device: Target device.
        quantization: Optional quantization config dict with keys:
            - enabled: bool
            - fuse_bn: bool (fuse BatchNorm for dense parts)
            - quant_backbone, quant_neck, quant_head: bool
            - quant_add: bool (quantize residual add)
            - sensitive_layers: list of layer name prefixes to skip
            - spconv_int8: bool (use spconv INT8 for sparse encoder)
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
    if quantization and quantization.get("enabled", False):
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
            if quantization.get("ptq_checkpoint"):
                raise RuntimeError(
                    "PTQ checkpoint load failed; cannot fall back to plain FP32 load_checkpoint. "
                    "Typical causes: (1) deploy PTQ load order must match "
                    "bevfusion_l/quantization/quantize.py (dense BN fuse + Q/DQ insert, then NVIDIA "
                    "sparse quantizer tree, then load_state_dict). (2) ``spconv_int8_fp16_layers`` in "
                    "deploy must match the PTQ run. Fix the error above or set quantization.enabled=False "
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
    quantization: dict,
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
    from deployment.config.schema import QuantizationConfig
    from deployment.projects.bevfusion_l.quantization.plan import build_bevfusion_plan

    config = QuantizationConfig.from_dict(quantization)
    is_ptq = config.ptq_checkpoint
    spconv_int8 = config.spconv_int8

    if is_ptq:
        logger.info("Loading PTQ checkpoint (pre-calibrated Q/DQ nodes)...")

        # Rebuild the quantized module tree BEFORE load_state_dict, using the shared plan
        # (dense Q/DQ + BN fuse, sparse BN fuse / NVIDIA quantizers). Identical to the PTQ producer.
        build_bevfusion_plan(config, include_sparse=True).prepare(model)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _strip_module_prefix_state_dict(state_dict, model)

        # PTQ checkpoint may have sparse conv weights in (C_in, C_out, K, K, K);
        # runtime model expects (C_out, K, K, K, C_in). Permute to match.
        if spconv_int8:
            _permute_sparse_encoder_weights_to_match_model(state_dict, model)

        result = model.load_state_dict(state_dict, strict=False)
        _log_load_state_dict_result(result)

        if spconv_int8:
            _verify_spconv_scale_buffers(model, state_dict)
            miss_sparse_bn = any(k.startswith("pts_middle_encoder") and ".bn" in k for k in result.missing_keys)
            unexp_legacy_obs = any(
                k.startswith("pts_middle_encoder")
                and ("_scale_" in k or "_zero_point_" in k or k.endswith("_scale_0") or k.endswith("_zero_point_0"))
                for k in result.unexpected_keys
            )
            if miss_sparse_bn and unexp_legacy_obs:
                logger.error(
                    "PTQ sparse tower key mismatch: checkpoint looks like a legacy observer-style sparse tower "
                    "(scale/zero_point observer keys) but the loaded model expects BN/Conv weights. "
                    "Regenerate the PTQ .pth with the current NVIDIA TensorQuantizer pipeline in "
                    "bevfusion_l/quantization/quantize.py (sparse _amax keys)."
                )

        num_amax = sum(1 for k in state_dict if "_amax" in k)
        logger.info(f"PTQ state_dict contains {num_amax} amax entries, {len(state_dict)} total keys")

        _move_quantizer_amax_to_device(model, device)

        tensor_quantizer_cls = _import_tensor_quantizer()
        if tensor_quantizer_cls:
            loaded = 0
            for name, mod in model.named_modules():
                if isinstance(mod, tensor_quantizer_cls) and hasattr(mod, "_amax") and mod._amax is not None:
                    loaded += 1
            logger.info(f"PTQ checkpoint loaded: {loaded} quantizers have calibrated amax values")

        _set_tensor_quantizers_inference_mode(model)

        if spconv_int8:
            from deployment.quantization.sparse.spconv_add_patch import (
                ensure_spconv_quantize_per_tensor_float_activations,
            )

            ensure_spconv_quantize_per_tensor_float_activations()

            info = verify_spconv_int8_encoder(model)
            nqc = int(info.get("nvidia_quantized_sparse_conv_count", 0))
            if info.get("is_int8"):
                logger.info(
                    "Spconv INT8 encoder: %d SparseConvolution module(s) with NVIDIA TensorQuantizer",
                    nqc,
                )
            else:
                logger.warning(
                    "Spconv INT8: no SparseConvolution modules with _input_quantizer/_weight_quantizer "
                    "found after load — sparse PTQ keys may not have applied. Check missing_keys and "
                    "spconv_int8_fp16_layers alignment with the PTQ run."
                )

    else:
        load_checkpoint(model, checkpoint_path, map_location=device)
        model.eval()
        logger.info("Applying dense quantization (uncalibrated) to BEVFusion model...")
        build_bevfusion_plan(config, include_sparse=False).prepare(model)

    logger.info("Quantization applied successfully")
    return model


def _move_quantizer_amax_to_device(model: torch.nn.Module, device: torch.device) -> None:
    """Move all TensorQuantizer amax values to the target device."""
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    moved_count = 0
    for _name, module in model.named_modules():
        if isinstance(module, tensor_quantizer_cls):
            if hasattr(module, "_amax") and module._amax is not None:
                if module._amax.device != device:
                    module._amax = module._amax.to(device)
                    moved_count += 1

    if moved_count > 0:
        logger.info(f"Moved {moved_count} quantizer amax tensors to {device}")


def _set_tensor_quantizers_inference_mode(model: torch.nn.Module) -> int:
    """Match ``CalibrationManager._disable_calibration_mode`` after PTQ ``load_state_dict``.

    Newly inserted ``TensorQuantizer`` modules may still have calibration defaults
    (fake-quant off, stats on). That yields a different dense branch than the PTQ
    script after ``calibrator.calibrate`` and can drive mAP/near-zero despite a
    valid ``state_dict``.
    """
    tensor_quantizer_cls = _import_tensor_quantizer()
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


def setup_quantization_for_onnx_export() -> None:
    """Configure pytorch-quantization for ONNX export (Q/DQ nodes)."""
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    tensor_quantizer_cls.use_fb_fake_quant = True
    logger.info("Enabled use_fb_fake_quant for ONNX export")
