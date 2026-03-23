"""BEVFusion model loading utilities for deployment.

Supports optional quantization:
- Dense parts (backbone, neck, head): pytorch_quantization (TensorQuantizer Q/DQ)
- Sparse encoder (pts_middle_encoder): spconv INT8 (cumm kernels via FX graph mode)
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


def _replace_encoder_with_fx_converted_structure(model: torch.nn.Module, device: torch.device) -> None:
    """Replace pts_middle_encoder with FX-converted structure so PTQ state_dict loads.

    PTQ with spconv_int8 saves the encoder after prepare_fx → calibrate → convert_fx.
    When loading, we need the same GraphModule structure; we run prepare_fx and
    convert_fx without calibration, then load_state_dict fills in weights/scales.
    """
    sparse_encoder = getattr(model, "pts_middle_encoder", None)
    if sparse_encoder is None:
        return
    try:
        from deployment.projects.bevfusion.quantization.spconv_int8 import (
            apply_spconv_int8_quantization,
            convert_spconv_int8,
        )
    except ImportError:
        logger.warning("spconv_int8 not available; cannot recreate FX encoder structure")
        return

    in_channels = getattr(sparse_encoder, "in_channels", 5)
    sparse_encoder.eval()
    prepared = apply_spconv_int8_quantization(sparse_encoder, device, in_channels=in_channels)
    converted = convert_spconv_int8(prepared)
    model.pts_middle_encoder = converted
    logger.info("Replaced pts_middle_encoder with FX-converted structure for PTQ load")


def _permute_sparse_encoder_weights_to_match_model(
    state_dict: dict,
    model: torch.nn.Module,
) -> None:
    """Permute pts_middle_encoder 5D weights from (C_in, C_out, K, K, K) to (C_out, K, K, K, C_in) when checkpoint and model shapes differ.

    PTQ may save sparse conv in one layout; the FX-converted model expects KRSC (out, k, k, k, in). Mutates state_dict in place.
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
    """Verify that pts_middle_encoder is the FX-converted INT8 quantized module.

    Returns a dict with:
        - is_int8: True if encoder looks like FX quantized (GraphModule + qint8 params)
        - encoder_type: type name of pts_middle_encoder
        - is_graph_module: True if encoder is torch.fx.GraphModule
        - quantized_param_count: number of parameters with dtype qint8
        - total_param_count: total parameters in encoder
        - quantized_module_types: set of module class names that look quantized
    """
    enc = getattr(model, "pts_middle_encoder", None)
    out: Dict[str, Any] = {
        "is_int8": False,
        "encoder_type": type(enc).__name__ if enc is not None else "None",
        "is_graph_module": False,
        "quantized_param_count": 0,
        "total_param_count": 0,
        "quantized_module_types": set(),
    }
    if enc is None:
        return out

    out["is_graph_module"] = type(enc).__name__ == "GraphModule"
    if hasattr(torch.fx, "GraphModule") and isinstance(enc, torch.fx.GraphModule):
        out["is_graph_module"] = True

    quant_types: Set[str] = set()
    for _name, mod in enc.named_modules():
        mod_str = type(mod).__module__ + "." + type(mod).__name__
        if "quantized" in mod_str or "quantization" in mod_str:
            quant_types.add(mod_str)

    out["total_param_count"] = sum(p.numel() for p in enc.parameters())
    out["quantized_param_count"] = sum(
        p.numel() for p in enc.parameters() if getattr(p, "is_quantized", False) or p.dtype == torch.qint8
    )
    out["quantized_module_types"] = quant_types
    out["is_int8"] = out["is_graph_module"] and (out["quantized_param_count"] > 0 or len(quant_types) > 0)
    return out


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
            logger.error(f"Quantization failed: {e}. Falling back to FP32 model (loading only matching weights).")
            ckpt = torch.load(checkpoint_path, map_location=torch_device)
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict = _strip_module_prefix_state_dict(state_dict, model)
            model_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            # Always try sparse weight layout fix when shapes differ (PTQ / checkpoint may use ICOK vs KRSC).
            _permute_sparse_encoder_weights_to_match_model(filtered, model)
            model.load_state_dict(filtered, strict=False)
            skipped = len(state_dict) - len(filtered)
            if skipped:
                logger.warning(
                    f"Skipped {skipped} checkpoint keys (quantizer/sparse INT8) that do not match FP32 model."
                )
    else:
        load_checkpoint(model, checkpoint_path, map_location=torch_device)

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

        if fuse_bn:
            _fuse_dense_bn(model)
            _fuse_spconv_bn(model)

        # PTQ with spconv_int8 saves pts_middle_encoder as FX-converted GraphModule.
        # Recreate the same structure (prepare_fx + convert_fx, no calibration) so load_state_dict matches.
        if quantization.get("spconv_int8", False):
            _replace_encoder_with_fx_converted_structure(model, device)

        if quant_backbone or quant_neck or quant_head:
            _apply_dense_quantization(
                model,
                quant_backbone=quant_backbone,
                quant_neck=quant_neck,
                quant_head=quant_head,
                quant_add=quant_add,
                skip_names=sensitive_layers,
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _strip_module_prefix_state_dict(state_dict, model)

        # PTQ checkpoint may have sparse conv weights in (C_in, C_out, K, K, K); FX-converted model expects (C_out, K, K, K, C_in). Permute to match.
        if quantization.get("spconv_int8", False):
            _permute_sparse_encoder_weights_to_match_model(state_dict, model)

        result = model.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            logger.warning(f"PTQ load: {len(result.missing_keys)} missing keys (first 10): {result.missing_keys[:10]}")
        if result.unexpected_keys:
            logger.warning(
                f"PTQ load: {len(result.unexpected_keys)} unexpected keys (first 10): {result.unexpected_keys[:10]}"
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

        if quantization.get("spconv_int8", False):
            from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
                ensure_spconv_quantized_add_sparse_support,
                retarget_graphmodule_quantized_add_calls,
            )

            ensure_spconv_quantized_add_sparse_support()
            retarget_graphmodule_quantized_add_calls(model)

            info = verify_spconv_int8_encoder(model)
            if info["is_int8"]:
                logger.info(
                    f"Spconv INT8 encoder verified: GraphModule with {info['quantized_param_count']} qint8 params "
                    f"(total {info['total_param_count']}), {len(info['quantized_module_types'])} quantized module types"
                )
            else:
                qtypes = info.get("quantized_module_types") or set()
                sample = sorted(qtypes)[:5]
                logger.warning(
                    f"Spconv INT8 encoder verification failed: encoder_type={info['encoder_type']} "
                    f"is_graph_module={info['is_graph_module']} "
                    f"quantized_params={info['quantized_param_count']} "
                    f"quantized_module_types={len(qtypes)} sample={sample}. "
                    "Sparse encoder may not be running INT8 (qint8 param count can be 0 if weights live in quantized submodules)."
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


def setup_quantization_for_onnx_export() -> None:
    """Configure pytorch-quantization for ONNX export (Q/DQ nodes)."""
    tensor_quantizer_cls = _import_tensor_quantizer()
    if tensor_quantizer_cls is None:
        return

    tensor_quantizer_cls.use_fb_fake_quant = True
    logger.info("Enabled use_fb_fake_quant for ONNX export")
