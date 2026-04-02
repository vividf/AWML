"""Spconv INT8 quantization for BEVFusion sparse encoder.

Two approaches are implemented:

**NVIDIA approach (default, recommended)** — adapted from CUDA-BEVFusion:
    Uses ``pytorch_quantization.TensorQuantizer`` with histogram collection +
    MSE-based amax selection.  Each SparseConvolution gets ``_input_quantizer``
    and ``_weight_quantizer`` submodules whose ``forward`` applies fake-quantise
    on ``input.features`` and ``self.weight``.  No FX tracing needed.

    Flow: add_quantizers → calibrate (histogram) → compute_amax(method=mse) → save.

**FX approach (legacy)** — spconv ``prepare_fx`` / ``convert_fx`` / ``transform_qdq``:
    Requires FX-traceable graph, ``SparseBasicBlockFX``, and ``SPCONV_FX_TRACE_MODE``.
    Known issue: ``non_traceable_module_classes`` + spconv graph transforms cause
    peak-clipping that destroys detection confidence (mAP ≈ 0).
"""

from __future__ import annotations

import contextlib
import logging
import math
import types
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def install_spconv_quantize_per_tensor_float_input_guard() -> None:
    """Force floating-point activations before ``torch.quantize_per_tensor``.

    ``convert_fx`` GraphModule forward **binds the spconv ``quantize_per_tensor`` function
    object** at export time. Monkey-patching ``spconv.pytorch.quantization.core.quantize_per_tensor``
    **afterwards does not change** that closed-over reference—the graph still runs the old
    callable, which then calls ``torch.quantize_per_tensor(int_tensor)`` and raises
    ``RuntimeError: Quantize only works on Float Tensor, got Int``.

    Patching ``torch.quantize_per_tensor`` fixes every caller (including the captured spconv
    wrapper) because ``torch`` is resolved at call time inside that wrapper's bytecode.

    We still patch spconv's module attribute so dynamic imports see a guarded wrapper.
    Idempotent.
    """
    if getattr(torch, "_awml_quantize_per_tensor_float_guard", False):
        return

    _orig_torch_qpt = torch.quantize_per_tensor

    def torch_quantize_per_tensor_float_input(*args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            t = args[0]
            if not getattr(t, "is_quantized", False) and not t.is_floating_point():
                args = (t.float(),) + args[1:]
        return _orig_torch_qpt(*args, **kwargs)

    torch.quantize_per_tensor = torch_quantize_per_tensor_float_input
    torch._awml_quantize_per_tensor_float_guard = True

    try:
        import spconv.pytorch.quantization.core as spconv_qcore
        from spconv.pytorch.core import SparseConvTensor
    except Exception as e:
        logger.debug("spconv quantize module guard skipped: %s", e)
        logger.info("Installed torch.quantize_per_tensor float-input guard (FX INT8 sparse).")
        return

    if getattr(spconv_qcore, "_awml_quantize_input_coercion_installed", False):
        logger.info("Installed torch.quantize_per_tensor float-input guard (FX INT8 sparse).")
        return

    _orig_spconv = spconv_qcore.quantize_per_tensor

    def _coerce(ten):
        if isinstance(ten, torch.Tensor):
            return ten.float() if not ten.is_floating_point() else ten
        if isinstance(ten, SparseConvTensor):
            f = ten.features
            if isinstance(f, torch.Tensor) and not f.is_floating_point():
                return ten.replace_feature(f.float())
            return ten
        if isinstance(ten, (list, tuple)):
            ctor = type(ten)
            return ctor(_coerce(v) for v in ten)
        return ten

    def spconv_quantize_per_tensor_guarded(ten, scale, zero_point, dtype):
        return _orig_spconv(_coerce(ten), scale, zero_point, dtype)

    spconv_qcore.quantize_per_tensor = spconv_quantize_per_tensor_guarded
    spconv_qcore._awml_quantize_input_coercion_installed = True
    logger.info("Installed torch.quantize_per_tensor + spconv.core quantize_per_tensor float-input guards.")


def _fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse BatchNorm into sparse convolutions inside the given sparse encoder.

    Used by PTQ (before prepare_fx) and by deployment model_loader so that
    state_dict keys match. Returns the number of fused Conv-BN pairs.
    """
    try:
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError:
        logger.warning("spconv quantization utils not available")
        return 0

    from spconv.pytorch.conv import SparseConvolution

    sparse_encoder.eval()
    fused_count = 0

    for module in sparse_encoder.modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]
            if isinstance(left_mod, SparseConvolution) and isinstance(right_mod, torch.nn.BatchNorm1d):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, torch.nn.Identity())
                fused_count += 1

    return fused_count


def _sparse_basic_block_to_fx(block: nn.Module) -> nn.Module:
    """Build SparseBasicBlockFX with same spconv indice_key/stride/downsample; copy conv/norm weights."""
    from mmdet3d.models.layers.sparse_block import SparseBasicBlock

    from projects.BEVFusion.bevfusion.sparse_block_fx import SparseBasicBlockFX

    if not isinstance(block, SparseBasicBlock):
        raise TypeError(f"expected SparseBasicBlock, got {type(block)!r}")

    inplanes = block.conv1.in_channels
    planes = block.conv1.out_channels
    stride = block.conv1.stride
    if isinstance(stride, (tuple, list)):
        stride = tuple(int(s) for s in stride) if len(stride) > 1 else int(stride[0])
    else:
        stride = int(stride)
    downsample = block.downsample
    indice_key = getattr(block.conv1, "indice_key", None)
    device = next(block.parameters()).device

    fx = SparseBasicBlockFX(
        inplanes,
        planes,
        stride=stride,
        downsample=downsample,
        indice_key=indice_key,
        conv_cfg=None,
        norm_cfg=None,
    ).to(device)
    fx.load_state_dict(block.state_dict(), strict=False)
    return fx


def upgrade_pts_middle_encoder_basicblocks_to_fx(sparse_encoder: nn.Module) -> int:
    """Replace ``SparseBasicBlock`` with ``SparseBasicBlockFX`` under ``pts_middle_encoder``.

    PTQ spconv checkpoints are usually produced with ``block_type=basicblock_fx``; FX graphs name
    activations (e.g. ``relu_final_scale``). Rebuilding from a ``basicblock`` config yields
    different Q/DQ parameter names and many missing/unexpected keys. Call this **before**
    ``prepare_fx`` / ``convert_fx`` when loading such checkpoints.

    Returns:
        Number of blocks replaced.
    """
    from mmdet3d.models.layers.sparse_block import SparseBasicBlock

    replaced = 0

    def walk(m: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(m._modules.items()):
            if child is None:
                continue
            if isinstance(child, SparseBasicBlock):
                m._modules[name] = _sparse_basic_block_to_fx(child)
                replaced += 1
            else:
                walk(child)

    walk(sparse_encoder)
    if replaced:
        logger.info(
            "Upgraded %d SparseBasicBlock -> SparseBasicBlockFX before spconv prepare_fx (PTQ key alignment)",
            replaced,
        )
    return replaced


def _get_spconv_quantization_imports():
    """Lazily import spconv quantization utilities."""
    from spconv.pytorch.quantization import (
        get_default_spconv_qconfig_mapping,
        get_spconv_backend_config,
        get_spconv_convert_custom_config,
        get_spconv_prepare_custom_config,
        prepare_spconv_torch_inference,
        remove_conv_add_dq,
        transform_qdq,
    )
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    return {
        "prepare_fx": prepare_fx,
        "convert_fx": convert_fx,
        "get_default_spconv_qconfig_mapping": get_default_spconv_qconfig_mapping,
        "get_spconv_backend_config": get_spconv_backend_config,
        "get_spconv_convert_custom_config": get_spconv_convert_custom_config,
        "get_spconv_prepare_custom_config": get_spconv_prepare_custom_config,
        "prepare_spconv_torch_inference": prepare_spconv_torch_inference,
        "remove_conv_add_dq": remove_conv_add_dq,
        "transform_qdq": transform_qdq,
    }


def bevfusion_spconv_qconfig_mapping(is_qat: bool = False):
    """Spconv QConfigMapping for BEVFusion sparse encoder.

    Key change from spconv default: uses ``SparseMinMaxObserver`` for activations
    instead of ``SparseHistogramObserver``.

    HistogramObserver minimises KL-divergence for the *bulk* of the activation
    distribution.  In sparse 3D detection, >96 % of BEV cells are zero and <0.1 %
    carry the high-magnitude object peaks.  Histogram calibration clips at ~2.5
    while actual peaks reach 8+, destroying detection confidence.

    MinMaxObserver preserves the full [min, max] range.  The lower precision in the
    near-zero background is irrelevant because the dense backbone/neck/head (all
    FP32) absorb those values anyway.

    ``conv_out`` is excluded from INT8 (FP32 last stage).
    """
    from torch.ao.quantization.observer import default_per_channel_weight_observer
    from torch.ao.quantization.qconfig import QConfig

    imports = _get_spconv_quantization_imports()
    qm = imports["get_default_spconv_qconfig_mapping"](is_qat=is_qat)

    if not is_qat:
        try:
            from spconv.pytorch.quantization.fake_q import SparseMinMaxObserver

            minmax_qconfig = QConfig(
                activation=SparseMinMaxObserver.with_args(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    eps=2 ** -12,
                ),
                weight=default_per_channel_weight_observer,
            )
            qm = qm.set_global(minmax_qconfig)
            logger.info(
                "BEVFusion spconv: using SparseMinMaxObserver (preserves outlier "
                "peaks critical for 3D object detection)"
            )
        except ImportError:
            logger.warning(
                "SparseMinMaxObserver not available in spconv; falling back to "
                "default SparseHistogramObserver"
            )

    for name in ("conv_out", "conv_out.0", "conv_out.1", "conv_out.2"):
        qm = qm.set_module_name(name, None)
    return qm


def _sparse_conv_weight_to_fp32_if_needed(t: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(t, torch.Tensor):
        return None
    if getattr(t, "is_quantized", False):
        try:
            return t.dequantize()
        except Exception:
            return None
    if t.dtype in (torch.qint8, torch.quint8):
        return None
    return t


def replace_bevfusion_sparse_encoder_conv_out_with_native_fp32(encoder: nn.Module) -> bool:
    """Swap ``encoder.conv_out`` for a fresh FP32 ``SparseSequential`` and load (dequantized) weights.

    Ensures the strided last block uses the same spconv geometry as training regardless of FX/INT8
    artifacts. Safe to call after ``load_state_dict`` (deploy) or at end of ``convert_spconv_int8`` (PTQ).
    """
    old = getattr(encoder, "conv_out", None)
    if old is None:
        return False
    try:
        first = old[0]
        in_ch = int(first.in_channels)
        out_ch = int(getattr(encoder, "output_channels", getattr(first, "out_channels", 0)))
    except Exception as e:
        logger.debug("replace conv_out: cannot read channels: %s", e)
        return False
    if out_ch <= 0:
        return False

    norm_cfg = getattr(encoder, "norm_cfg", None)
    if not isinstance(norm_cfg, dict):
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

    try:
        from projects.BEVFusion.bevfusion.sparse_convmodule import make_sparse_convmodule
    except Exception as e:
        logger.warning("replace conv_out: make_sparse_convmodule import failed: %s", e)
        return False

    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    new_conv = make_sparse_convmodule(
        in_ch,
        out_ch,
        kernel_size=(1, 1, 3),
        stride=(1, 1, 2),
        norm_cfg=norm_cfg,
        padding=0,
        indice_key="spconv_down2",
        conv_type="SparseConv3d",
    )
    new_conv.to(device)
    new_conv.eval()

    old_sd = old.state_dict()
    new_sd = new_conv.state_dict()
    load_dict: Dict[str, torch.Tensor] = {}

    def _maybe_align_5d(src: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
        if src.dim() != 5 or len(target_shape) != 5 or src.shape == target_shape:
            return src if src.shape == target_shape else None
        if (
            src.shape[0] == target_shape[4]
            and src.shape[1] == target_shape[0]
            and src.shape[2] == target_shape[1]
            and src.shape[3] == target_shape[2]
            and src.shape[4] == target_shape[3]
        ):
            return src.permute(1, 2, 3, 4, 0).contiguous()
        perm = src.permute(1, 2, 3, 4, 0).contiguous()
        if perm.shape == target_shape:
            return perm
        return None

    for k, target in new_sd.items():
        if k not in old_sd:
            continue
        v = old_sd[k]
        v = _sparse_conv_weight_to_fp32_if_needed(v)
        if v is None or not isinstance(v, torch.Tensor):
            continue
        if v.shape != target.shape and v.dim() == 5:
            aligned = _maybe_align_5d(v, target.shape)
            if aligned is not None:
                v = aligned
        if v.shape == target.shape and v.dtype == target.dtype:
            load_dict[k] = v

    inc = new_conv.load_state_dict(load_dict, strict=False)
    miss = getattr(inc, "missing_keys", ())
    unexp = getattr(inc, "unexpected_keys", ())
    if miss:
        logger.warning(
            "replace conv_out: %d missing keys after partial load (first 5): %s",
            len(miss),
            list(miss)[:5],
        )
    if unexp:
        logger.debug("replace conv_out: unexpected keys: %s", list(unexp)[:5])

    encoder.conv_out = new_conv
    logger.info(
        "Replaced pts_middle_encoder.conv_out with native FP32 SparseSequential "
        "(%d / %d tensors copied).",
        len(load_dict),
        len(new_sd),
    )
    return True


def _ensure_torch_device(device: Union[torch.device, str]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Expected torch.device or str for device, got {type(device)!r}")


def _create_example_inputs(
    model: nn.Module,
    device: torch.device,
    in_channels: int = 5,
    num_voxels: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Create example inputs for FX tracing of the sparse encoder.

    Args:
        model: The sparse encoder module.
        device: Target device.
        in_channels: Number of input channels for voxel features.
        num_voxels: Number of example voxels.

    Returns:
        Tuple of (voxel_features, coors, batch_size).
    """
    dev = _ensure_torch_device(device)
    # Match BEVFusion 120m lidar grid (see default_lidar_intensity_120m.grid_size).
    sparse_shape = getattr(model, "sparse_shape", [1440, 1440, 41])

    voxel_features = torch.randn((num_voxels, in_channels), device=dev)
    coors = torch.zeros((num_voxels, 4), dtype=torch.int32, device=dev)
    for i in range(num_voxels):
        coors[i, 0] = 0
        coors[i, 1] = i % sparse_shape[0]
        coors[i, 2] = i % sparse_shape[1]
        coors[i, 3] = i % sparse_shape[2]

    return voxel_features, coors, 1


def _enable_spconv_fx_trace_mode() -> None:
    """Spconv requires FX trace mode during prepare_fx (see spconv example/mnist, SPCONV_FX_TRACE_MODE).

    Disables strict SparseConvTensor __init__ checks and avoids trace failures from symbolic tensors.
    Must update both ``spconv.constants`` (source of truth) and ``spconv.pytorch.core`` (imported name).
    """
    try:
        import spconv.constants as spconv_constants
        import spconv.pytorch.core as spconv_core

        spconv_constants.SPCONV_FX_TRACE_MODE = True
        spconv_core.SPCONV_FX_TRACE_MODE = True
    except Exception:
        pass


@contextlib.contextmanager
def _sparse_basic_block_skip_dim_assert_for_fx() -> Iterator[None]:
    """Patch mmdet3d ``SparseBasicBlock.forward`` only for the ``prepare_fx`` call.

    Upstream forward does ``assert x.features.dim() == 2``. Under symbolic trace that
    touches ``Proxy.__bool__`` and raises ``TraceError``. ``is_tracing()`` is not
    reliable when execution goes through spconv ``SparseSequential`` wrappers, so while
    this context is active we omit the assert entirely (original forward is restored
    right after ``prepare_fx``).

    Rest of forward matches OpenMMLab mmdet3d ``sparse_block.SparseBasicBlock`` (norm1/norm2).
    """
    try:
        import mmdet3d.models.layers.sparse_block as sb_mod
    except ImportError:
        yield
        return

    cls = sb_mod.SparseBasicBlock
    replace_feature = sb_mod.replace_feature
    orig_forward = cls.forward

    def forward_patched(self, x):
        identity = x.features
        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))
        return out

    cls.forward = forward_patched
    try:
        yield
    finally:
        cls.forward = orig_forward


def _disable_spconv_fx_trace_mode() -> None:
    """Turn off spconv FX trace mode (both modules that cache the flag).

    Deployment sets ``SPCONV_FX_TRACE_MODE=1`` early for INT8 ONNX/spconv. That global relaxed mode can
    interact badly with ``pytorch_quantization`` while inserting TensorQuantizers (torch.fx Proxy +
    control-flow errors). Disable **only** for dense Q/DQ insertion; ``apply_spconv_int8_quantization``
    calls ``_enable_spconv_fx_trace_mode()`` again before ``prepare_fx``.
    """
    try:
        import spconv.constants as spconv_constants
        import spconv.pytorch.core as spconv_core

        spconv_constants.SPCONV_FX_TRACE_MODE = False
        spconv_core.SPCONV_FX_TRACE_MODE = False
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NVIDIA pytorch_quantization approach (adapted from CUDA-BEVFusion)
# ---------------------------------------------------------------------------

def _get_sparse_conv_types() -> tuple:
    """Return a tuple of SparseConvolution classes to quantize."""
    from spconv.pytorch.conv import SparseConvolution as SpconvSparseConvolution
    conv_types: list = [SpconvSparseConvolution]
    try:
        from projects.SparseConvolution.sparse_conv import SparseConvolution as CustomSparseConvolution
        conv_types.append(CustomSparseConvolution)
    except ImportError:
        pass
    return tuple(conv_types)


def _nvidia_quantized_forward(self, input):
    """Quantized forward that wraps input.features and weight with TensorQuantizer."""
    if input is not None and hasattr(input, "features"):
        input = input.replace_feature(self._input_quantizer(input.features))
    if self.weight is None:
        return self._original_forward(input)
    quant_weight = self._weight_quantizer(self.weight)
    # PTQ / inference runs under ``torch.no_grad()``: keep **full-precision** values in
    # ``self.weight.data`` across forwards.  Old behavior replaced ``self.weight`` with
    # ``Parameter(quant_weight)`` every call → checkpoint / ONNX carried **fake-quant-baked**
    # weights while ``_weight_quantizer._amax`` was calibrated for **unbaked** distributions;
    # TRT ImplicitGemmInt8 then re-quantized baked weights with those scales → systematic
    # drift and huge ``lidar_bev`` / NaNs in the dense head.
    if not torch.is_grad_enabled():
        orig = self.weight.data.clone()
        try:
            self.weight.data.copy_(quant_weight)
            return self._original_forward(input)
        finally:
            self.weight.data.copy_(orig)
    self.weight = Parameter(quant_weight)
    return self._original_forward(input)


def apply_nvidia_spconv_int8(
    sparse_encoder: nn.Module,
    exclude_conv_out: bool = True,
) -> nn.Module:
    """Add NVIDIA ``TensorQuantizer`` to every ``SparseConvolution`` in *sparse_encoder*.

    Follows the CUDA-BEVFusion pattern: histogram collection for activations,
    per-**output**-channel weight scales for 5-D sparse weights.

    Sparse conv weight layout is ``[C_out, k1, k2, k3, C_in]``.  Use ``axis=(0)``
    so ``_weight_quantizer._amax`` is one value per **output** channel (length
    ``C_out``).  ``axis=(4)`` would calibrate along **input** channels only
    (length ``C_in``), which breaks Path B / ``ImplicitGemmInt8`` which expects
    ``channel_scale`` shaped ``[C_out]``.

    ``conv_out`` is excluded by default (stays FP32).

    Returns the same *sparse_encoder* (modified in-place).
    """
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor

    input_desc = QuantDescriptor(num_bits=8, calib_method="histogram")
    weight_desc = QuantDescriptor(num_bits=8, axis=(0))
    conv_types = _get_sparse_conv_types()

    count = 0
    for name, module in list(sparse_encoder.named_modules()):
        if not isinstance(module, conv_types):
            continue
        if exclude_conv_out and "conv_out" in name:
            logger.info("  Skipping conv_out from NVIDIA quantization: %s", name)
            continue

        iq = quant_nn.TensorQuantizer(input_desc)
        wq = quant_nn.TensorQuantizer(weight_desc)
        if isinstance(getattr(iq, "_calibrator", None), calib.HistogramCalibrator):
            iq._calibrator._torch_hist = True
        if isinstance(getattr(wq, "_calibrator", None), calib.HistogramCalibrator):
            wq._calibrator._torch_hist = True

        module.register_module("_input_quantizer", iq)
        module.register_module("_weight_quantizer", wq)

        module._original_forward = module.forward
        module.forward = types.MethodType(_nvidia_quantized_forward, module)

        count += 1
        if count <= 3:
            logger.info("  [nvidia-quant] %s: added TensorQuantizer (in=%d, out=%d)",
                        name, module.in_channels, module.out_channels)

    logger.info("Applied NVIDIA TensorQuantizer to %d sparse conv modules", count)
    return sparse_encoder


def _collect_pathb_sparse_tail_absmax_before_conv_out(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Record max |features| at the FP32 ``conv_out`` boundary (CUDA-BEVFusion / libspconv semantics).

    CUDA-BEVFusion ONNX stores per-layer ``input_dynamic_range`` and sets ``conv_out`` to
    ``output_precision=fp16`` while the backbone stays int8: the **last int8 conv's output scale**
    must match the **real tensor** entering ``conv_out``. That tensor is not covered by a
    TensorQuantizer in AWML (``conv_out`` is excluded from INT8), so Path B ONNX transform
    reads ``pts_middle_encoder._pathb_sparse_tail_absmax`` saved here.

    Must run **after** ``enable_quant`` so the max matches inference fake-quant behavior.
    """
    conv_out = getattr(encoder, "conv_out", None)
    if conv_out is None or not calibration_data:
        return

    device = next(encoder.parameters()).device
    mx = torch.zeros((), device=device, dtype=torch.float32)

    def _pre_hook(_mod: nn.Module, inp: Tuple[object, ...]) -> None:
        nonlocal mx
        st = inp[0]
        feats = getattr(st, "features", None)
        if feats is None or feats.numel() == 0:
            return
        v = feats.detach().abs().max().to(device=mx.device, dtype=torch.float32)
        mx.copy_(torch.maximum(mx, v))

    hook = conv_out.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            for voxel_features, coors, batch_size in calibration_data:
                encoder(voxel_features, coors, batch_size)
    finally:
        hook.remove()

    encoder.register_buffer("_pathb_sparse_tail_absmax", mx.detach().clone())
    val = float(mx.detach().cpu().item())
    if val <= 0.0 or not math.isfinite(val):
        raise RuntimeError(
            "[nvidia-calib] Path-B: _pathb_sparse_tail_absmax is invalid (0 or non-finite). "
            "Encoder forwards failed, or conv_out pre-hook never saw non-empty features."
        )
    print(
        f"[nvidia-calib] Path-B: max |sparse features| before conv_out = {val:.6f} "
        f"(saved as pts_middle_encoder._pathb_sparse_tail_absmax for sparse_int8_onnx_transform)"
    )


def _module_from_pts_stem(encoder: nn.Module, stem: str) -> nn.Module:
    """Resolve ``stem`` (checkpoint-relative under ``pts_middle_encoder``) to a submodule."""
    cur: nn.Module = encoder
    for tok in stem.split("."):
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _nvidia_quantized_sparse_conv_stems(encoder: nn.Module) -> List[str]:
    """Module names (relative to *encoder*) of SparseConvs that have NVIDIA input quantizers."""
    conv_types = _get_sparse_conv_types()
    stems: List[str] = []
    for name, m in encoder.named_modules():
        if not name or "conv_out" in name:
            continue
        if not isinstance(m, conv_types):
            continue
        if getattr(m, "_input_quantizer", None) is None:
            continue
        stems.append(name)
    return stems


def _collect_pathb_last_int8_conv_output_absmax(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Record max |features| at the **output of the last INT8 SparseConv** (pre tail BN/ReLU).

    ``_pathb_sparse_tail_absmax`` measures the tensor **entering** ``conv_out``, i.e. after the
    final encoder block's norm/residual/ReLU.  ``ImplicitGemmInt8``'s ``output_scale`` must match
    the **linear conv output** of the last quantized sparse conv (what the ONNX node emits before
    downstream FP ops).  Using the conv_out-boundary value there **over-``output_scale``s** and
    blows up ``lidar_bev`` in TensorRT while PyTorch fake-quant stays sane.

    Saved as ``pts_middle_encoder._pathb_last_int8_conv_output_absmax`` for
    ``sparse_int8_onnx_transform`` (preferred terminal scale).
    """
    stems = _nvidia_quantized_sparse_conv_stems(encoder)
    if not stems or not calibration_data:
        return
    try:
        from deployment.projects.bevfusion.export.sparse_int8_onnx_transform import (
            _topologically_sorted_sparse_stems,
        )
    except Exception as e:
        raise ImportError(
            "Path-B: failed to import _topologically_sorted_sparse_stems for last-int8 absmax"
        ) from e

    topo = _topologically_sorted_sparse_stems(stems)
    last_stem = topo[-1]
    try:
        mod = _module_from_pts_stem(encoder, last_stem)
    except Exception as e:
        raise RuntimeError(
            f"Path-B: could not resolve stem {last_stem!r} for last-int8 absmax"
        ) from e

    device = next(encoder.parameters()).device
    mx = torch.zeros((), device=device, dtype=torch.float32)

    def _hook(_m: nn.Module, _inp: Tuple[object, ...], out: object) -> None:
        nonlocal mx
        feats = getattr(out, "features", None)
        if feats is None or feats.numel() == 0:
            return
        v = feats.detach().abs().max().to(device=mx.device, dtype=torch.float32)
        mx.copy_(torch.maximum(mx, v))

    hook_h = mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for voxel_features, coors, batch_size in calibration_data:
                encoder(voxel_features, coors, batch_size)
    finally:
        hook_h.remove()

    encoder.register_buffer("_pathb_last_int8_conv_output_absmax", mx.detach().clone())
    val = float(mx.detach().cpu().item())
    if val <= 0.0 or not math.isfinite(val):
        raise RuntimeError(
            "[nvidia-calib] Path-B: _pathb_last_int8_conv_output_absmax is invalid (0 or non-finite). "
            "Encoder forwards failed, or last INT8 sparse conv never produced non-empty features."
        )
    print(
        f"[nvidia-calib] Path-B: max |features| after last INT8 sparse conv ({last_stem}) = "
        f"{val:.6f} (saved as pts_middle_encoder._pathb_last_int8_conv_output_absmax; "
        "preferred for ONNX terminal output_scale)"
    )


def calibrate_spconv_nvidia(
    encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Calibrate sparse encoder with histogram + MSE (CUDA-BEVFusion approach).

    1. Enable calibration mode on all TensorQuantizers (collect histograms, no fake-quant).
    2. Run calibration data through the encoder.
    3. compute_amax(method=mse) for all quantizers.
    4. Re-enable fake-quantization.
    5. Collect ``_pathb_sparse_tail_absmax`` at the conv_out boundary for Path B TRT export.
    """
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn

    encoder.eval()
    device = next(encoder.parameters()).device
    n_samples = len(calibration_data)

    n_quantizers = 0
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            n_quantizers += 1
            if mod._calibrator is not None:
                mod.disable_quant()
                mod.enable_calib()
            else:
                mod.disable()
    print(f"[nvidia-calib] {n_quantizers} TensorQuantizers in calibration mode")
    print(f"[nvidia-calib] Collecting histograms over {n_samples} samples...")

    with torch.no_grad():
        pbar = tqdm(calibration_data, total=n_samples, desc="NVIDIA calib", leave=True)
        for i, (voxel_features, coors, batch_size) in enumerate(pbar):
            try:
                encoder(voxel_features, coors, batch_size)
            except Exception as e:
                raise RuntimeError(
                    f"[nvidia-calib] sample {i + 1}/{n_samples} failed during histogram collection"
                ) from e

    print("[nvidia-calib] Computing amax (method=mse) from histograms...")
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            if mod._calibrator is not None:
                if isinstance(mod._calibrator, calib.MaxCalibrator):
                    mod.load_calib_amax(strict=False)
                else:
                    mod.load_calib_amax(strict=False, method="mse")
                if mod._amax is not None:
                    mod._amax = mod._amax.to(device)

    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            if mod._calibrator is not None:
                mod.enable_quant()
                mod.disable_calib()
            else:
                mod.enable()

    _collect_pathb_sparse_tail_absmax_before_conv_out(encoder, calibration_data)
    _collect_pathb_last_int8_conv_output_absmax(encoder, calibration_data)
    _report_nvidia_quantizer_stats(encoder)


def _report_nvidia_quantizer_stats(encoder: nn.Module) -> None:
    """Print amax summary for all NVIDIA TensorQuantizers after calibration."""
    from pytorch_quantization import nn as quant_nn

    count = 0
    for name, mod in encoder.named_modules():
        if isinstance(mod, quant_nn.TensorQuantizer):
            amax = getattr(mod, "_amax", None)
            if amax is not None:
                if amax.numel() <= 3:
                    vals = amax.flatten().tolist()
                    print(f"  [nvidia-amax] {name}: amax={vals}")
                else:
                    print(f"  [nvidia-amax] {name}: amax shape={tuple(amax.shape)}, "
                          f"min={amax.min():.4f}, max={amax.max():.4f}, mean={amax.mean():.4f}")
                count += 1
    print(f"[nvidia-calib] {count} quantizers with calibrated amax")


# ---------------------------------------------------------------------------
# FX approach (legacy — see docstring at top of file)
# ---------------------------------------------------------------------------

def apply_spconv_int8_quantization(
    sparse_encoder: nn.Module,
    device: torch.device,
    in_channels: int = 5,
) -> nn.Module:
    """Apply spconv INT8 quantization to the sparse encoder using FX graph mode.

    This performs: prepare_fx → returns prepared model ready for calibration.
    After calibration, call convert_spconv_int8() to finalize.

    Args:
        sparse_encoder: The BEVFusionSparseEncoder module.
        device: Target device.
        in_channels: Number of voxel feature channels.

    Returns:
        Prepared sparse encoder (with observers inserted, ready for calibration).
    """
    _enable_spconv_fx_trace_mode()
    from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
        ensure_spconv_quantize_per_tensor_float_activations,
    )

    ensure_spconv_quantize_per_tensor_float_activations()

    imports = _get_spconv_quantization_imports()

    imports["prepare_spconv_torch_inference"](with_linear=False)

    qconfig_mapping = bevfusion_spconv_qconfig_mapping(is_qat=False)
    logger.info("BEVFusion spconv FX: conv_out block excluded from INT8 (FP32 last stage).")
    backend_config = imports["get_spconv_backend_config"]()
    prepare_custom_config = imports["get_spconv_prepare_custom_config"]()
    # The model uses custom SparseConv3d/SubMConv3d from projects/SparseConvolution
    # which are NOT in spconv's DEFAULT_SPARSE_CONV_TYPES. Without this, FX traces
    # inside their _conv_forward, causing spatial_shape bugs during symbolic tracing.
    try:
        from projects.SparseConvolution.sparse_conv import SparseConv3d as CustomSparseConv3d
        from projects.SparseConvolution.sparse_conv import SubMConv3d as CustomSubMConv3d
        from projects.SparseConvolution.sparse_conv import SparseConvolution as CustomSparseConvolution

        for cls in (CustomSparseConv3d, CustomSubMConv3d, CustomSparseConvolution):
            if cls not in prepare_custom_config.non_traceable_module_classes:
                prepare_custom_config.non_traceable_module_classes.append(cls)
        logger.info("Added custom SparseConv types to non_traceable_module_classes for prepare_fx")
    except ImportError:
        logger.warning("Could not import custom SparseConv types; FX may trace inside them")

    example_inputs = _create_example_inputs(sparse_encoder, device, in_channels=in_channels)

    sparse_encoder.eval()
    logger.info("Running prepare_fx on sparse encoder for INT8 quantization...")
    with _sparse_basic_block_skip_dim_assert_for_fx():
        prepared = imports["prepare_fx"](
            sparse_encoder,
            qconfig_mapping,
            example_inputs,
            backend_config=backend_config,
            prepare_custom_config=prepare_custom_config,
        )

    logger.info("Sparse encoder prepared for INT8 calibration")
    return prepared


def calibrate_spconv_model(
    prepared_encoder: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> None:
    """Run calibration data through the prepared sparse encoder.

    Full-scene voxels are used (no subsampling). OOM is no longer expected because
    custom SparseConv types are registered as non-traceable leaf modules in prepare_fx,
    so observers are only placed at module boundaries — not inside _conv_forward.

    Args:
        prepared_encoder: Prepared (with observers) sparse encoder.
        calibration_data: List of (voxel_features, coors, batch_size) tuples.
    """
    prepared_encoder.eval()
    n_samples = len(calibration_data)
    logger.info("Calibrating sparse encoder with %d samples (full voxels, no cap)", n_samples)

    total_voxels = 0
    with torch.no_grad():
        pbar = tqdm(calibration_data, total=n_samples, desc="Calibrating spconv", leave=True)
        for i, (voxel_features, coors, batch_size) in enumerate(pbar):
            n_vox = int(voxel_features.shape[0])
            total_voxels += n_vox
            if i < 5:
                logger.info("  [calib] Sample %d/%d: %d voxels", i + 1, n_samples, n_vox)
            try:
                prepared_encoder(voxel_features, coors, batch_size)
            except Exception as e:
                pbar.write(f"  Warning: spconv calib sample {i + 1}/{n_samples} failed: {e}")

    logger.info(
        "Calibration complete: %d samples, total_voxels=%d, avg=%.0f",
        n_samples, total_voxels, total_voxels / max(n_samples, 1),
    )

    _report_observer_stats(prepared_encoder)


def _report_observer_stats(model: nn.Module) -> None:
    """Print observer statistics after calibration to verify observers were activated."""
    from torch.ao.quantization.observer import ObserverBase

    total = 0
    calibrated = 0
    uncalibrated_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, ObserverBase):
            total += 1
            try:
                scale, zp = mod.calculate_qparams()
                s_val = float(scale.flatten()[0]) if scale.numel() > 0 else 1.0
                if s_val != 1.0:
                    calibrated += 1
                    obs_min = getattr(mod, "min_val", None)
                    obs_max = getattr(mod, "max_val", None)
                    extra = ""
                    if obs_min is not None and obs_max is not None:
                        mn = float(obs_min) if obs_min.numel() == 1 else float(obs_min.min())
                        mx = float(obs_max) if obs_max.numel() == 1 else float(obs_max.max())
                        extra = f", observed_range=[{mn:.4f}, {mx:.4f}]"
                    print(f"  [observer] {name}: scale={s_val:.6f}, "
                          f"zp={int(zp.flatten()[0])}{extra}")
                else:
                    uncalibrated_names.append(name)
            except Exception:
                uncalibrated_names.append(name)

    print(f"[observer-summary] {calibrated}/{total} observers calibrated "
          f"(observer type: {type(mod).__name__})")
    if uncalibrated_names:
        print(f"[observer-summary] {len(uncalibrated_names)} UNCALIBRATED observers "
              f"(default scale=1.0): {uncalibrated_names[:10]}")


def convert_spconv_int8(
    prepared_encoder: nn.Module,
    *,
    attr_source: Optional[nn.Module] = None,
) -> nn.Module:
    """Convert a calibrated prepared model to quantized INT8.

    Args:
        prepared_encoder: Calibrated prepared sparse encoder.
        attr_source: Module to copy ``sparse_shape`` / ``encoder_channels`` / … onto the FX root
            after conversion (the pre-``prepare_fx`` ``BEVFusionSparseEncoder``). ``convert_fx`` often
            drops these attributes; without them, ONNX export cannot swap in an FP32 shadow encoder.

    Returns:
        Quantized sparse encoder using cumm INT8 kernels.
    """
    from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
        ensure_spconv_quantize_per_tensor_float_activations,
        ensure_spconv_quantized_add_sparse_support,
    )

    ensure_spconv_quantize_per_tensor_float_activations()
    ensure_spconv_quantized_add_sparse_support()

    imports = _get_spconv_quantization_imports()

    backend_config = imports["get_spconv_backend_config"]()
    convert_custom_config = imports["get_spconv_convert_custom_config"]()

    logger.info("Converting sparse encoder to INT8...")
    converted = imports["convert_fx"](
        prepared_encoder,
        convert_custom_config=convert_custom_config,
        backend_config=backend_config,
    )

    logger.info("Applying transform_qdq...")
    converted = imports["transform_qdq"](converted)

    logger.info("Applying remove_conv_add_dq...")
    converted = imports["remove_conv_add_dq"](converted)

    # Keep SPCONV_FX_TRACE_MODE=True: the FX GraphModule was traced with relaxed
    # SparseConvTensor __init__; its compiled graph includes dequantize ops that
    # produce float32 indices. Disabling the mode triggers "only support int32".

    try:
        from deployment.projects.bevfusion.export.sparse_encoder_float_shadow import (
            copy_sparse_encoder_public_attrs,
        )

        src = attr_source if attr_source is not None else prepared_encoder
        copy_sparse_encoder_public_attrs(src, converted)
    except Exception as e:
        logger.warning("Could not copy sparse encoder public attrs onto FX root: %s", e)

    logger.info("Sparse encoder INT8 conversion complete")
    return converted
