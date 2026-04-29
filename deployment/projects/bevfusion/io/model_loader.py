"""BEVFusion model loading utilities for deployment.

Supports optional quantization:
- Dense parts (backbone, neck, head): pytorch_quantization (TensorQuantizer Q/DQ)
- Sparse encoder (pts_middle_encoder): NVIDIA TensorQuantizer INT8 Path B (including ``conv_out``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Set

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core.device import DeviceSpec

logger = logging.getLogger(__name__)


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


def _fuse_spconv_bn(model: torch.nn.Module) -> None:
    """Fuse BatchNorm into sparse convolutions in pts_middle_encoder."""
    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        return

    try:
        from deployment.projects.bevfusion.quantization.spconv_int8 import (
            _fuse_spconv_bn_in_encoder,
        )

        count = _fuse_spconv_bn_in_encoder(sparse_encoder)
        logger.info(f"Fused {count} SparseConv-BN pairs in pts_middle_encoder")
    except ImportError:
        logger.warning("spconv_int8 module not available; skipping sparse BN fusion")


def _prepare_encoder_for_nvidia_int8(
    model: torch.nn.Module,
    exclude_patterns: Optional[list] = None,
) -> None:
    """Add NVIDIA ``TensorQuantizer`` to sparse encoder so PTQ checkpoint ``_amax`` keys load.

    The NVIDIA approach (adapted from CUDA-BEVFusion) stores per-module ``_amax``
    values in the checkpoint.  At evaluation time we recreate the same module
    structure by adding ``_input_quantizer`` and ``_weight_quantizer`` submodules
    to each ``SparseConvolution``, then ``load_state_dict`` fills in the calibrated
    ``_amax``.  No FX tracing or graph transformation needed.

    ``exclude_patterns`` (from ``spconv_int8_fp16_layers`` in deploy_cfg) MUST
    match exactly what was passed to ``apply_nvidia_spconv_int8`` during PTQ,
    otherwise the module tree will have more/fewer quantizer submodules than
    the checkpoint ``state_dict`` expects → ``load_state_dict`` will emit
    noisy missing/unexpected keys.
    """
    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        return
    try:
        from deployment.projects.bevfusion.quantization.spconv_int8 import (
            apply_nvidia_spconv_int8,
        )
    except ImportError:
        logger.warning("spconv_int8 not available; cannot add NVIDIA quantizers")
        return

    sparse_encoder.eval()
    apply_nvidia_spconv_int8(
        sparse_encoder,
        exclude_patterns=list(exclude_patterns or []),
    )
    # PTQ saves these Path-B buffers; register so load_state_dict(strict=False) loads them instead
    # of reporting unexpected_keys (and so inspection / future export see checkpoint values).
    if not hasattr(sparse_encoder, "_pathb_sparse_tail_absmax"):
        sparse_encoder.register_buffer(
            "_pathb_sparse_tail_absmax",
            torch.tensor(0.0, dtype=torch.float32),
        )
    if not hasattr(sparse_encoder, "_pathb_last_int8_conv_output_absmax"):
        sparse_encoder.register_buffer(
            "_pathb_last_int8_conv_output_absmax",
            torch.tensor(0.0, dtype=torch.float32),
        )
    logger.info("Added NVIDIA TensorQuantizer to pts_middle_encoder (amax loaded from checkpoint)")


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
    out: Dict[str, Any] = {
        "is_int8": n_quant_convs > 0,
        "encoder_type": type(enc).__name__ if enc is not None else "None",
        "is_graph_module": False,
        "nvidia_quantized_sparse_conv_count": n_quant_convs,
        "quantized_param_count": 0,
        "total_param_count": sum(p.numel() for p in enc.parameters()) if enc is not None else 0,
        "quantized_module_types": set(),
        "quant_activation_buffer_keys": 0,
    }
    return out


def _verify_spconv_scale_buffers(model: torch.nn.Module, ckpt_state_dict: dict) -> None:
    """Verify that spconv INT8 quantization params were loaded from checkpoint.

    Primarily for NVIDIA checkpoints (_amax keys). Legacy FX checkpoints may still show scale/zero_point keys.
    """
    enc = getattr(model, "pts_middle_encoder", None)
    if enc is None:
        print("[spconv-quant-check] NO pts_middle_encoder found!")
        return

    ckpt_sparse_keys = [k for k in ckpt_state_dict if k.startswith("pts_middle_encoder.")]
    ckpt_amax_keys = [k for k in ckpt_sparse_keys if "_amax" in k]
    ckpt_scale_keys = [k for k in ckpt_sparse_keys if "scale" in k or "zero_point" in k]

    if ckpt_amax_keys:
        print(f"[spconv-quant-check] NVIDIA approach: checkpoint has {len(ckpt_amax_keys)} _amax keys")
        for k in ckpt_amax_keys[:5]:
            v = ckpt_state_dict[k]
            t = v.flatten().tolist()[:3]
            print(f"  {k} shape={tuple(v.shape)} first3={t}")
    elif ckpt_scale_keys:
        print(f"[spconv-quant-check] FX approach: checkpoint has {len(ckpt_scale_keys)} scale/zp keys")
        for k in ckpt_scale_keys[:5]:
            v = ckpt_state_dict[k]
            t = v.flatten().tolist()[:3]
            print(f"  {k} shape={tuple(v.shape)} first3={t}")
    else:
        print(
            f"[spconv-quant-check] WARNING: no _amax or scale/zp keys in checkpoint "
            f"(has {len(ckpt_sparse_keys)} pts_middle_encoder keys total)"
        )
    if ckpt_scale_keys:
        print(f"[spconv-scale-check] ckpt scale/zp keys sample: {ckpt_scale_keys[:5]}")

    model_all_keys = [f"pts_middle_encoder.{k}" for k in dict(enc.named_parameters())]
    model_all_keys += [f"pts_middle_encoder.{k}" for k in dict(enc.named_buffers())]
    model_scale_keys = [k for k in model_all_keys if "scale" in k or "zero_point" in k]
    if model_scale_keys:
        print(f"[spconv-scale-check] model scale/zp keys sample: {model_scale_keys[:5]}")


def _register_bevfusion_modules() -> None:
    """Register BEVFusion and SparseConvolution modules into MMDet3D registries."""
    import projects.BEVFusion.bevfusion  # noqa: F401
    import projects.SparseConvolution  # noqa: F401


def _import_tensor_quantizer():
    """Lazily import TensorQuantizer from pytorch_quantization."""
    try:
        from pytorch_quantization.nn import TensorQuantizer

        return TensorQuantizer
    except ImportError:
        return None


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
            SparseConvolution + BatchNorm1d pair in ``pts_middle_encoder`` after
            ``load_checkpoint`` (same ``fuse_spconv_bn_eval`` path as PTQ load).
            Ignored when ``quantization.enabled`` is True (PTQ already fuses sparse BN).

    Returns:
        Loaded and eval-mode BEVFusion model.
    """
    init_default_scope("mmdet3d")
    _register_bevfusion_modules()

    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)

    torch_device = device.to_torch_device()
    model.to(torch_device)

    if quantization and quantization.get("enabled", False):
        try:
            model = _load_with_quantization(model, checkpoint_path, torch_device, quantization)
        except Exception as e:
            logger.exception(
                "Quantization pipeline failed (full traceback above). Summary: %s",
                e,
            )
            logger.error(
                f"Quantization failed: {e}. " "See message below for whether a safe FP32 fallback is possible."
            )
            if quantization.get("ptq_checkpoint"):
                raise RuntimeError(
                    "PTQ checkpoint load failed; cannot fall back to plain FP32 load_checkpoint. "
                    "Typical causes: (1) deploy PTQ load order must match bevfusion_quantization.py "
                    "(dense BN fuse + Q/DQ insert, then NVIDIA sparse quantizer tree, then load_state_dict). "
                    "(2) ``spconv_int8_fp16_layers`` in deploy must match the PTQ run. "
                    "Fix the error above or set quantization.enabled=False and use an FP32 checkpoint."
                ) from e
            # Non-PTQ quant failure: rebuild clean model and load FP32 checkpoint (mmengine + spconv BN hooks).
            logger.info("Falling back to FP32: rebuilding model from config and load_checkpoint.")
            model_config = copy.deepcopy(model_cfg.model)
            model = MODELS.build(model_config)
            model.to(torch_device)
            load_checkpoint(model, checkpoint_path, map_location=torch_device)
    else:
        load_checkpoint(model, checkpoint_path, map_location=torch_device)
        if fuse_spconv_bn:
            _fuse_spconv_bn(model)

    model.eval()
    model.cfg = model_cfg
    return model


def _load_with_quantization(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    quantization: dict,
) -> torch.nn.Module:
    """Load model with dense quantization applied.

    Supports two modes:
    A) PTQ checkpoint (quantization.ptq_checkpoint=True):
       1. Fuse BatchNorm for dense parts
       2. Insert Q/DQ nodes (to recreate quantized model structure)
       3. Load PTQ checkpoint (state_dict contains calibrated _amax values)

    B) FP32 checkpoint (default):
       1. Load FP32 checkpoint
       2. Fuse BatchNorm for dense parts
       3. Insert Q/DQ nodes (uncalibrated - need runtime calibration)

    Spconv INT8 is applied separately by the runner (needs calibration data).
    """
    is_ptq = quantization.get("ptq_checkpoint", False)

    fuse_bn = quantization.get("fuse_bn", True)
    quant_backbone = quantization.get("quant_backbone", True)
    quant_neck = quantization.get("quant_neck", True)
    quant_head = quantization.get("quant_head", True)
    quant_add = quantization.get("quant_add", False)
    sensitive_layers = set(quantization.get("sensitive_layers", []) or [])

    if is_ptq:
        logger.info("Loading PTQ checkpoint (pre-calibrated Q/DQ nodes)...")

        # Match bevfusion_quantization.run_ptq: fuse → dense Q/DQ insert → NVIDIA sparse quantizers → load.

        if fuse_bn:
            _fuse_dense_bn(model)
            _fuse_spconv_bn(model)

        if quant_backbone or quant_neck or quant_head:
            try:
                _apply_dense_quantization(
                    model,
                    quant_backbone=quant_backbone,
                    quant_neck=quant_neck,
                    quant_head=quant_head,
                    quant_add=quant_add,
                    skip_names=sensitive_layers,
                )
            except Exception:
                logger.exception("PTQ load: dense Q/DQ insertion (_apply_dense_quantization) failed")
                raise

        # Add NVIDIA TensorQuantizer to sparse encoder so PTQ _amax keys load correctly.
        # ``spconv_int8_fp16_layers`` is hoisted into ``quantization`` by runner.py (from the
        # deploy_cfg top-level key). Must match what PTQ used, or state_dict keys won't align.
        if quantization.get("spconv_int8", False):
            try:
                _prepare_encoder_for_nvidia_int8(
                    model,
                    exclude_patterns=quantization.get("spconv_int8_fp16_layers", []) or [],
                )
            except Exception:
                logger.exception("PTQ load: NVIDIA sparse encoder quantizer setup failed")
                raise

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _strip_module_prefix_state_dict(state_dict, model)

        # PTQ checkpoint may have sparse conv weights in (C_in, C_out, K, K, K); FX-converted model expects (C_out, K, K, K, C_in). Permute to match.
        if quantization.get("spconv_int8", False):
            _permute_sparse_encoder_weights_to_match_model(state_dict, model)

        result = model.load_state_dict(state_dict, strict=False)

        print(f"[load-state-dict] missing={len(result.missing_keys)}, " f"unexpected={len(result.unexpected_keys)}")
        if result.missing_keys:
            sparse_miss = [k for k in result.missing_keys if k.startswith("pts_middle_encoder")]
            other_miss = [k for k in result.missing_keys if not k.startswith("pts_middle_encoder")]
            print(f"[load-state-dict] sparse missing={len(sparse_miss)}, other missing={len(other_miss)}")
            if sparse_miss:
                print(f"[load-state-dict] sparse missing sample: {sparse_miss[:10]}")
            if other_miss:
                print(f"[load-state-dict] other missing sample: {other_miss[:10]}")
        if result.unexpected_keys:
            sparse_unexp = [k for k in result.unexpected_keys if k.startswith("pts_middle_encoder")]
            other_unexp = [k for k in result.unexpected_keys if not k.startswith("pts_middle_encoder")]
            print(f"[load-state-dict] sparse unexpected={len(sparse_unexp)}, other unexpected={len(other_unexp)}")
            if sparse_unexp:
                print(f"[load-state-dict] sparse unexpected sample: {sparse_unexp[:10]}")
            if other_unexp:
                print(f"[load-state-dict] other unexpected sample: {other_unexp[:10]}")

        if quantization.get("spconv_int8", False):
            _verify_spconv_scale_buffers(model, state_dict)
            miss_sparse_bn = any(k.startswith("pts_middle_encoder") and ".bn" in k for k in result.missing_keys)
            unexp_fx_obs = any(
                k.startswith("pts_middle_encoder")
                and ("_scale_" in k or "_zero_point_" in k or k.endswith("_scale_0") or k.endswith("_zero_point_0"))
                for k in result.unexpected_keys
            )
            if miss_sparse_bn and unexp_fx_obs:
                logger.error(
                    "PTQ sparse tower key mismatch: checkpoint looks like a legacy FX-quantized sparse tower "
                    "(scale/zero_point observer keys) but the loaded model expects BN/Conv weights. "
                    "Regenerate the PTQ .pth with the current NVIDIA TensorQuantizer pipeline in "
                    "bevfusion_quantization.py (sparse _amax keys, no prepare_fx)."
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

        if quantization.get("spconv_int8", False):
            from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
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

        logger.info("Applying dense quantization to BEVFusion model...")

        if fuse_bn:
            _fuse_dense_bn(model)

        if quant_backbone or quant_neck or quant_head:
            _apply_dense_quantization(
                model,
                quant_backbone=quant_backbone,
                quant_neck=quant_neck,
                quant_head=quant_head,
                quant_add=quant_add,
                skip_names=sensitive_layers,
            )

    logger.info("Dense quantization applied successfully")
    return model


def _fuse_dense_bn(model: torch.nn.Module) -> None:
    """Fuse BatchNorm in dense parts only (backbone, neck, head).

    We skip the sparse encoder (pts_middle_encoder) since spconv BN fusion
    is handled by the spconv FX quantization pipeline.
    """
    try:
        from deployment.quantization import fuse_model_bn
    except ImportError:
        logger.warning("deployment.quantization.fuse_model_bn not available; " "trying standalone BN fusion...")
        _fuse_dense_bn_standalone(model)
        return

    logger.info("Fusing BatchNorm for dense parts...")

    for submodule_name in ["pts_backbone", "pts_neck", "bbox_head"]:
        submodule = getattr(model, submodule_name, None)
        if submodule is not None:
            submodule.eval()
            fuse_model_bn(submodule)
            logger.info(f"  Fused BN in {submodule_name}")


def _apply_dense_quantization(
    model: torch.nn.Module,
    quant_backbone: bool = True,
    quant_neck: bool = True,
    quant_head: bool = True,
    quant_add: bool = False,
    skip_names: Optional[Set[str]] = None,
) -> None:
    """Apply pytorch_quantization to dense parts of BEVFusion.

    Uses the same quant_conv_module / quant_model pattern as CenterPoint.
    Requires NVIDIA pytorch-quantization package.
    """
    skip_names = skip_names or set()

    logger.info(
        "Dense quantization flags: backbone=%s, neck=%s, head=%s, add=%s",
        quant_backbone,
        quant_neck,
        quant_head,
        quant_add,
    )

    try:
        from deployment.quantization import quant_conv_module
        from deployment.quantization.replace import attach_quant_add

        if quant_backbone and hasattr(model, "pts_backbone"):
            quant_conv_module(model.pts_backbone, skip_names, "pts_backbone")
            logger.info("  Quantized pts_backbone (Conv2d -> QuantConv2d)")

        if quant_neck and hasattr(model, "pts_neck"):
            quant_conv_module(model.pts_neck, skip_names, "pts_neck")
            logger.info("  Quantized pts_neck (Conv2d -> QuantConv2d)")

        if quant_head and hasattr(model, "bbox_head"):
            quant_conv_module(model.bbox_head, skip_names, "bbox_head")
            logger.info("  Quantized bbox_head (Conv2d -> QuantConv2d)")

        if quant_add:
            attach_quant_add(model)
            logger.info("  Attached residual quantizers")

    except (ImportError, Exception) as e:
        if "pytorch-quantization" in str(e) or "pytorch_quantization" in str(e):
            logger.warning(
                "pytorch_quantization not installed. Skipping dense Conv2d quantization. "
                "Dense parts will run in FP32. Install with: pip install pytorch-quantization "
                "--extra-index-url https://pypi.ngc.nvidia.com"
            )
        else:
            raise


def _fuse_dense_bn_standalone(model: torch.nn.Module) -> None:
    """Standalone BN fusion that doesn't require pytorch_quantization.

    Uses torch.nn.utils.fusion if available, otherwise skips.
    """
    try:
        from torch.ao.nn.utils import fuse as torch_fuse
    except ImportError:
        pass

    import torch.nn as nn

    def _fuse_conv_bn_eval(conv, bn):
        """Fuse conv+bn in eval mode."""
        assert not conv.training and not bn.training
        is_transposed = isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))

        if conv.bias is None:
            conv_bias = torch.zeros_like(bn.running_mean)
        else:
            conv_bias = conv.bias

        bn_weight = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
        bn_bias = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)

        bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
        scale = bn_weight * bn_var_rsqrt

        if is_transposed:
            shape = [1, -1] + [1] * (conv.weight.ndim - 2)
        else:
            shape = [-1] + [1] * (conv.weight.ndim - 1)

        conv.weight = nn.Parameter((conv.weight * scale.reshape(shape)).contiguous())
        conv.bias = nn.Parameter(((conv_bias - bn.running_mean) * scale + bn_bias).contiguous())

    def _fuse_module(module):
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]
            if left_mod is None or right_mod is None:
                continue

            is_conv = isinstance(left_mod, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d))
            is_bn = isinstance(right_mod, (nn.BatchNorm1d, nn.BatchNorm2d))

            if is_conv and is_bn:
                _fuse_conv_bn_eval(left_mod, right_mod)
                setattr(module, right_name, nn.Identity())

        for child_name, child_mod in children:
            if child_mod is not None:
                _fuse_module(child_mod)

    fused = 0
    for submodule_name in ["pts_backbone", "pts_neck", "bbox_head"]:
        submodule = getattr(model, submodule_name, None)
        if submodule is not None:
            submodule.eval()
            _fuse_module(submodule)
            logger.info(f"  Fused BN in {submodule_name} (standalone)")
            fused += 1

    if fused > 0:
        logger.info(f"Standalone BN fusion done for {fused} submodules")


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
