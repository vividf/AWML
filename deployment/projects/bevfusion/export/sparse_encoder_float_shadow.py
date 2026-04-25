"""FP32 sparse-encoder shadow for ``torch.onnx.export``.

**FX GraphModule:** ``convert_fx`` + spconv INT8 yields ``aten::_empty_affine_quantized``; the
exporter swaps in a fused FP32 ``BEVFusionSparseEncoder`` and copies float weights from the GM.

**NVIDIA TensorQuantizer (scheme A):** PTQ loads a plain ``BEVFusionSparseEncoder`` with
``_input_quantizer`` / ``_weight_quantizer`` on sparse convs — not a ``GraphModule``. Without shadow,
``torch.onnx.export`` traces Q/DQ into sparse ONNX. When ``encoder_has_nvidia_tensor_quantizers``
holds, we use the same shadow rebuild and copy **float** weights from the live encoder's
``state_dict`` (quantizer buffers are skipped), producing a Q/DQ-free sparse ONNX.

See ``docs/11_int8_pathb_autoware_plugin.md`` (scheme A) and ``docs/12_int8_sparse_pipeline_ptq_onnx_trt.md``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Attributes copied from ``BEVFusionSparseEncoder`` onto the FX root; required to rebuild FP32 shadow.
SPARSE_ENCODER_SHADOW_ATTRS: tuple[str, ...] = (
    "sparse_shape",
    "in_channels",
    "base_channels",
    "output_channels",
    "encoder_channels",
    "encoder_paddings",
    "num_aug_features",
    "aug_features_min_values",
    "aug_features_max_values",
)


def has_sparse_encoder_shadow_attributes(module: nn.Module) -> bool:
    """True if ``module`` carries the config fields needed by ``build_float_sparse_encoder_shadow``."""
    return all(hasattr(module, name) for name in SPARSE_ENCODER_SHADOW_ATTRS)


def encoder_has_nvidia_tensor_quantizers(encoder: nn.Module) -> bool:
    """True if sparse tower uses NVIDIA-style quantizers on conv modules (PTQ deploy path).

    ``apply_nvidia_spconv_int8`` adds ``_input_quantizer`` and ``_weight_quantizer`` to each
    quantized ``SparseConvolution``. Used to decide ONNX FP32 shadow (scheme A).
    """
    for m in encoder.modules():
        if m is encoder:
            continue
        if getattr(m, "_input_quantizer", None) is None:
            continue
        if getattr(m, "_weight_quantizer", None) is None:
            continue
        return True
    return False


def copy_sparse_encoder_public_attrs(src: nn.Module, dst: nn.Module) -> None:
    """Copy BEVFusion sparse encoder config fields from ``src`` onto ``dst`` (e.g. FX root after convert_fx).

    ``convert_fx`` / ``remove_conv_add_dq`` often return a ``GraphModule`` that **drops** the original
    ``sparse_shape``, ``encoder_channels``, etc. ONNX shadow detection then fails and export still
    traces quantized ops. Call this on the converted root using the pre-``prepare_fx`` encoder as
    ``src``.
    """
    for name in SPARSE_ENCODER_SHADOW_ATTRS:
        if not hasattr(src, name):
            continue
        val = getattr(src, name)
        if torch.is_tensor(val):
            setattr(dst, name, val.detach().clone())
        else:
            setattr(dst, name, copy.deepcopy(val))
    for name in ("norm_cfg", "block_type", "order", "return_middle_feats"):
        if hasattr(src, name):
            setattr(dst, name, copy.deepcopy(getattr(src, name)))


def encoder_cfg_overrides_from_bevfusion_model(model: Optional[nn.Module]) -> Dict[str, Any]:
    """Build shadow-attribute overrides from ``model.cfg.model['pts_middle_encoder']`` (MMEngine config)."""
    if model is None:
        return {}
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return {}
    model_dict = getattr(cfg, "model", None)
    if model_dict is None:
        return {}
    try:
        enc = model_dict.get("pts_middle_encoder") if hasattr(model_dict, "get") else None
    except Exception:
        return {}
    if enc is None:
        return {}
    if isinstance(enc, Mapping) and not isinstance(enc, dict):
        try:
            enc = dict(enc)
        except Exception:
            return {}
    if not isinstance(enc, dict):
        return {}
    out: Dict[str, Any] = {}
    for name in SPARSE_ENCODER_SHADOW_ATTRS:
        if name in enc:
            out[name] = copy.deepcopy(enc[name])
    for name in ("norm_cfg", "block_type", "order", "return_middle_feats"):
        if name in enc:
            out[name] = copy.deepcopy(enc[name])
    return out


def resolve_sparse_onnx_shadow(
    pts_middle_encoder: Optional[nn.Module],
    bevfusion: Optional[nn.Module] = None,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """Pick a ``GraphModule`` under ``pts_middle_encoder`` and optional config overrides for shadow rebuild.

    Returns:
        (graph_module_or_none, cfg_overrides). When the FX root lacks attributes (common after
        ``convert_fx``), ``cfg_overrides`` is filled from ``bevfusion.cfg`` so
        ``build_float_sparse_encoder_shadow`` can still run.
    """
    if pts_middle_encoder is None:
        return None, {}
    gm_cls = getattr(torch.fx, "GraphModule", None)
    if gm_cls is None:
        return None, {}

    for m in pts_middle_encoder.modules():
        if isinstance(m, gm_cls) and has_sparse_encoder_shadow_attributes(m):
            return m, {}

    first_gm: Optional[nn.Module] = None
    for m in pts_middle_encoder.modules():
        if isinstance(m, gm_cls):
            first_gm = m
            break
    if first_gm is None:
        # Scheme A: NVIDIA TensorQuantizer on a normal BEVFusionSparseEncoder (no nested GraphModule).
        if encoder_has_nvidia_tensor_quantizers(pts_middle_encoder):
            overrides_nv = encoder_cfg_overrides_from_bevfusion_model(bevfusion)
            if has_sparse_encoder_shadow_attributes(pts_middle_encoder):
                if overrides_nv:
                    logger.info(
                        "Sparse ONNX shadow (NVIDIA TensorQuantizer path): encoder has shadow attrs; "
                        "%d optional cfg key(s) from model.cfg.",
                        len(overrides_nv),
                    )
                return pts_middle_encoder, overrides_nv
            can_nv = all(
                hasattr(pts_middle_encoder, name) or (name in overrides_nv) for name in SPARSE_ENCODER_SHADOW_ATTRS
            )
            if can_nv:
                if overrides_nv:
                    logger.info(
                        "Sparse ONNX shadow (NVIDIA TensorQuantizer path): merging %d key(s) from "
                        "model.cfg pts_middle_encoder.",
                        len(overrides_nv),
                    )
                return pts_middle_encoder, overrides_nv
            logger.warning(
                "pts_middle_encoder has NVIDIA TensorQuantizers but lacks SPARSE_ENCODER_SHADOW_ATTRS "
                "and model.cfg.model.pts_middle_encoder is incomplete; ONNX FP32 shadow skipped "
                "(sparse ONNX may contain Q/DQ)."
            )
        return None, {}

    overrides = encoder_cfg_overrides_from_bevfusion_model(bevfusion)
    can_fill = all(hasattr(first_gm, name) or (name in overrides) for name in SPARSE_ENCODER_SHADOW_ATTRS)
    if not can_fill:
        logger.warning(
            "FX GraphModule under pts_middle_encoder lacks shadow attrs and model.cfg has "
            "incomplete pts_middle_encoder; ONNX float shadow cannot run. Missing will block export."
        )
        return None, {}

    if overrides:
        logger.info(
            "Sparse ONNX shadow: GraphModule missing some attrs; merging %d keys from model.cfg "
            "pts_middle_encoder.",
            len(overrides),
        )
    return first_gm, overrides


def find_graphmodule_for_sparse_onnx_shadow(root: Optional[nn.Module]) -> Optional[nn.Module]:
    """Backward-compatible: GraphModule only, no ``model.cfg`` merge (prefer ``resolve_sparse_onnx_shadow``)."""
    gm, _ = resolve_sparse_onnx_shadow(root, None)
    return gm


def is_spconv_fx_graphmodule_encoder(module: Optional[nn.Module]) -> bool:
    """True if ``module`` is an FX GraphModule (typical after ``convert_fx`` on the sparse tower).

    Do not require ``type(module).__name__ == "GraphModule"``: ``convert_fx`` / subclasses use
    different runtime class names while still inheriting ``torch.fx.GraphModule``. A too-strict
    check skips the FP32 shadow and ``torch.onnx.export`` then hits
    ``aten::_empty_affine_quantized`` (same class of issue Lidar AI Solution avoids by using
    ``exptool`` on a non-quantized graph instead of the standard ONNX exporter on Q tensors).
    """
    if module is None:
        return False
    gm_cls = getattr(torch.fx, "GraphModule", None)
    return gm_cls is not None and isinstance(module, gm_cls)


def build_float_sparse_encoder_shadow(
    gm: nn.Module,
    device: torch.device,
    *,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Construct a fused FP32 ``BEVFusionSparseEncoder`` and load weights from ``gm`` state_dict.

    ``gm`` may be an FX ``GraphModule`` **or** a ``BEVFusionSparseEncoder`` with NVIDIA
    ``TensorQuantizer`` children; only floating conv/BN parameters are copied, not ``_amax``.

    ``cfg_overrides`` supplies fields missing on the source module (from ``model.cfg``), e.g. after
    ``convert_fx`` stripped ``sparse_shape`` / ``encoder_channels``.
    """
    from mmengine.registry import MODELS, init_default_scope

    import projects.BEVFusion.bevfusion  # noqa: F401 — register BEVFusionSparseEncoder

    init_default_scope("mmdet3d")

    def _pick(name: str) -> Any:
        if cfg_overrides is not None and name in cfg_overrides:
            return cfg_overrides[name]
        return getattr(gm, name, None)

    missing = [r for r in SPARSE_ENCODER_SHADOW_ATTRS if _pick(r) is None]
    if missing:
        raise RuntimeError(
            "Cannot rebuild FP32 sparse encoder for ONNX: GraphModule + overrides missing: "
            f"{missing}. Ensure the FP32 shadow encoder defines these (match training sparse_encoder), or pass "
            f"a BEVFusion model with model.cfg.model.pts_middle_encoder. See "
            f"docs/5_bevfusion_onnx_trt_spconv_int8.md (bevfusion project)."
        )

    def _buf_to_list(buf: torch.Tensor) -> list:
        return buf.detach().cpu().flatten().tolist()

    def _aug_to_list(val: Any) -> list:
        if isinstance(val, torch.Tensor):
            return _buf_to_list(val)
        if isinstance(val, (list, tuple)):
            return list(val)
        raise TypeError(f"aug_features_* must be tensor or list, got {type(val)!r}")

    default_norm = dict(type="BN1d", eps=1e-3, momentum=0.01)
    nc = _pick("norm_cfg")
    norm_cfg = copy.deepcopy(nc if nc is not None else default_norm)

    block_type = _pick("block_type")
    if block_type is None:
        block_type = "basicblock"
    order_val = _pick("order")
    order = tuple(order_val) if order_val is not None else ("conv", "norm", "act")

    enc_channels = _pick("encoder_channels")
    if isinstance(enc_channels, torch.Tensor):
        raise TypeError("encoder_channels must be nested tuples, not Tensor")
    enc_paddings = _pick("encoder_paddings")
    sparse_shape = _pick("sparse_shape")
    sparse_shape = list(sparse_shape) if not isinstance(sparse_shape, list) else list(sparse_shape)

    ret_mid = _pick("return_middle_feats")
    return_middle_feats = bool(ret_mid) if ret_mid is not None else False

    enc_cfg: Dict[str, Any] = dict(
        type="BEVFusionSparseEncoder",
        in_channels=int(_pick("in_channels")),
        aug_features_min_values=_aug_to_list(_pick("aug_features_min_values")),
        aug_features_max_values=_aug_to_list(_pick("aug_features_max_values")),
        num_aug_features=int(_pick("num_aug_features")),
        sparse_shape=sparse_shape,
        order=order,
        norm_cfg=norm_cfg,
        base_channels=int(_pick("base_channels")),
        output_channels=int(_pick("output_channels")),
        encoder_channels=enc_channels,
        encoder_paddings=enc_paddings,
        block_type=block_type,
        return_middle_feats=return_middle_feats,
    )

    enc: nn.Module = MODELS.build(enc_cfg)
    enc.to(device)
    enc.eval()

    from deployment.projects.bevfusion.quantization.spconv_int8 import _fuse_spconv_bn_in_encoder

    _fuse_spconv_bn_in_encoder(enc)

    gm_sd = gm.state_dict()
    enc_sd = enc.state_dict()

    def _align_5d_spconv_weight_to_krsc(v: torch.Tensor, target: torch.Size) -> Optional[torch.Tensor]:
        """FX / PTQ checkpoints often store 5D sparse conv as (C_in, C_out, Kz, Ky, Kx); MMDet encoder uses KRSC (C_out, Kz, Ky, Kx, C_in)."""
        if v.dim() != 5 or len(target) != 5:
            return None
        if v.shape == target:
            return v
        # Explicit ICOC -> KRSC when channel/spatial layout matches (robust across spconv / FX versions).
        if (
            v.shape[0] == target[4]
            and v.shape[1] == target[0]
            and v.shape[2] == target[1]
            and v.shape[3] == target[2]
            and v.shape[4] == target[3]
        ):
            return v.permute(1, 2, 3, 4, 0).contiguous()
        perm = v.permute(1, 2, 3, 4, 0).contiguous()
        if perm.shape == target:
            return perm
        perm2 = v.permute(4, 0, 1, 2, 3).contiguous()
        if perm2.shape == target:
            return perm2
        return None

    def _fx_flat_state_key(key: str) -> str:
        """FX / PTQ sometimes uses underscore keys (e.g. ``encoder_layers_encoder_layer1_0_conv1``)."""
        return key.replace(".", "_")

    def _gm_value_for_key(key: str) -> Optional[torch.Tensor]:
        flat = _fx_flat_state_key(key)
        for cand in (
            key,
            f"module.{key}",
            f"pts_middle_encoder.{key}",
            flat,
            f"module.{flat}",
            f"pts_middle_encoder.{flat}",
        ):
            if cand in gm_sd:
                return gm_sd[cand]  # type: ignore[return-value]
        if key.startswith("module.") and key[len("module.") :] in gm_sd:
            return gm_sd[key[len("module.") :]]  # type: ignore[return-value]
        return None

    # Copy with ``Tensor.copy_`` instead of ``load_state_dict``: spconv registers
    # ``load_state_dict`` pre-hooks that permute *disk* layouts when SPCONV_SAVED_WEIGHT_LAYOUT
    # is set; mutating the same dict we validated can also desync shapes vs. plain Parameters.
    n_copied = 0
    with torch.no_grad():
        for k, t in enc_sd.items():
            v = _gm_value_for_key(k)
            if v is None or not torch.is_tensor(v):
                continue
            if getattr(v, "is_quantized", False):
                try:
                    v = v.dequantize()
                except Exception:
                    continue
            elif v.dtype in (torch.qint8, torch.quint8):
                deq_fn = getattr(v, "dequantize", None)
                if callable(deq_fn):
                    v = deq_fn()
                else:
                    continue

            v = v.detach()
            if v.dim() == 5 and t.dim() == 5 and v.shape != t.shape:
                aligned = _align_5d_spconv_weight_to_krsc(v, t.shape)
                if aligned is not None:
                    logger.info(
                        "Float shadow ICOC->KRSC %s: %s -> %s",
                        k,
                        tuple(v.shape),
                        tuple(aligned.shape),
                    )
                    v = aligned
                else:
                    logger.debug("Float shadow skip 5D %s: gm %s vs enc %s", k, tuple(v.shape), tuple(t.shape))
                    continue
            elif v.shape != t.shape:
                continue

            if v.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
                w = v.to(device=t.device, dtype=t.dtype, non_blocking=False).contiguous()
            elif v.dtype in (torch.int32, torch.int64, torch.bool):
                w = v.to(device=t.device, non_blocking=False).contiguous()
            else:
                continue

            if w.shape != t.shape:
                raise RuntimeError(
                    f"Float shadow internal error: key {k!r} tensor shape {tuple(w.shape)} vs encoder "
                    f"{tuple(t.shape)} after layout fix."
                )

            parent_path, dot, leaf = k.rpartition(".")
            if not dot:
                continue
            try:
                sub = enc.get_submodule(parent_path)
            except AttributeError:
                logger.debug("Float shadow: no submodule for state key %s", k)
                continue
            dst = getattr(sub, leaf, None)
            if dst is None or not torch.is_tensor(dst):
                continue
            dst.copy_(w)
            n_copied += 1

    logger.info(
        "Sparse ONNX float shadow: copied %d / %d state entries from source encoder via in-place copy "
        "(bypasses spconv load_state_dict hooks).",
        n_copied,
        len(enc_sd),
    )

    return enc
