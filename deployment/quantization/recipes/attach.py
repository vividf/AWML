# Copyright (c) OpenMMLab. All rights reserved.
"""Attach quantizers and swap in the architecture-specific forward hooks.

These functions walk a model, insert :class:`~pytorch_quantization.nn.TensorQuantizer` modules at
TensorRT-friendly locations, and replace matched blocks' ``forward`` with the hooks in
:mod:`.forward_hooks`. They compose the generic engine
(:mod:`deployment.quantization.core`) with per-architecture placement, and are consumed by project
quant recipes (e.g. CenterPoint ``quant_model``) and
:class:`~deployment.quantization.schemes.dense_qdq.DenseQDQScheme`.

``attach_quant_add`` dispatches across every supported residual block type (ResNet ``BasicBlock``,
sparse ``SparseBasicBlock``, ``ConvNeXtBlock``, VoVNet ``_OSA_module``), so it deliberately spans
architectures rather than living in a single per-architecture file.
"""

import logging
from typing import Optional, Set

import torch.nn as nn

from deployment.quantization.core.availability import require_pytorch_quantization_installed
from deployment.quantization.core.descriptors import default_input_desc
from deployment.quantization.core.modules import QuantConv2d
from deployment.quantization.core.replace import ensure_quant_descriptors_initialized

from .forward_hooks import (
    BasicBlockForwardHook,
    ConvNeXtBlockForwardHook,
    OSAModuleForwardHook,
    QuantBeforePool,
    SparseBasicBlockForwardHook,
    eSEModuleForwardHook,
)

logger = logging.getLogger(__name__)


def _input_quant_desc():
    """Return the shared INT8 activation descriptor used by conv input quantizers.

    Reuses ``QuantConv2d.default_quant_desc_input`` so the residual / eSE / pool quantizers share
    the *exact* descriptor the conv layers use (calibration consistency), and guarantees a histogram
    ``calib_method``. Centralizes a block that was copy-pasted ~6 times across this module.
    """
    ensure_quant_descriptors_initialized()
    desc = QuantConv2d.default_quant_desc_input
    if desc is None:
        desc = default_input_desc()
        QuantConv2d.default_quant_desc_input = desc
    if not getattr(desc, "calib_method", None):
        desc.calib_method = "histogram"
    return desc


def _new_input_quantizer():
    """Create a fresh ``TensorQuantizer`` on the shared conv-input descriptor."""
    from pytorch_quantization.nn import TensorQuantizer

    return TensorQuantizer(_input_quant_desc())


def _install_forward_hook(module: nn.Module, hook_cls) -> None:
    """Replace ``module.forward`` with ``hook_cls(module)``, saving the original once (idempotent)."""
    if isinstance(module.forward, hook_cls):
        return
    if not hasattr(module, "_original_forward"):
        module._original_forward = module.forward
    module.forward = hook_cls(module)


# Residual-block class names attach_quant_add matches (exact or substring — subclassed block names
# like "MyConvNeXtBlock" still match). The full supported set; this deliberately spans architectures.
_RESIDUAL_BLOCK_CLASSES = {"BasicBlock", "SparseBasicBlock", "ConvNeXtBlock", "_OSA_module"}


def attach_quant_add(model: nn.Module):
    """
    Attach residual_quantizer to modules that perform residual add and replace their forward methods.

    This follows the same approach as lidar-ai-solution (CUDA-BEVFusion):
    - Only quantize the identity branch (residual connection), not the conv path output
    - This enables TensorRT to fuse Conv+Add operations, reducing reformat operations
    - The residual_quantizer uses the same quant descriptor as conv layers for consistency

    Args:
        model: Model whose residual blocks (see ``_RESIDUAL_BLOCK_CLASSES``) get the recipe.
    """
    require_pytorch_quantization_installed("residual quantization")
    ensure_quant_descriptors_initialized()

    target_class_names = _RESIDUAL_BLOCK_CLASSES

    attached_count = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in target_class_names or any(t in cls_name for t in target_class_names):
            # _OSA_module: attach concat_input_quantizers for branch inputs only (main path no Q/DQ, like ResNet Add)
            if cls_name == "_OSA_module":
                n_branch_inputs = len(module.layers)  # skip connections: x + layer0..layer(n-2); main = layer(n-1) out
                if (
                    not hasattr(module, "concat_input_quantizers")
                    or len(module.concat_input_quantizers) != n_branch_inputs
                ):
                    concat_quantizers = nn.ModuleList([_new_input_quantizer() for _ in range(n_branch_inputs)])
                    module.add_module("concat_input_quantizers", concat_quantizers)
                # When identity=True we reuse concat_input_quantizers[0] as single Q for block input (no extra module)
                if not getattr(module, "identity", False):
                    _install_forward_hook(module, OSAModuleForwardHook)
                    continue
            # Attach residual_quantizer if not already present. Aligned with lidar-ai-solution:
            # with a downsample, create a fresh quantizer; otherwise reuse an existing branch input
            # quantizer (shares calibration data). Reused quantizers are assigned as plain attributes
            # (not add_module) because a TensorQuantizer cannot be a submodule of two parents; the
            # forward hook still calls it so ONNX export traces the Q/DQ.
            if not hasattr(module, "residual_quantizer"):
                if hasattr(module, "downsample") and module.downsample is not None:
                    module.add_module("residual_quantizer", _new_input_quantizer())
                    attached_count += 1
                elif hasattr(module, "conv1") and hasattr(module.conv1, "_input_quantizer"):
                    module.residual_quantizer = module.conv1._input_quantizer
                    attached_count += 1
                elif hasattr(module, "depthwise_conv") and hasattr(module.depthwise_conv, "_input_quantizer"):
                    # ConvNeXtBlock: reuse depthwise_conv._input_quantizer
                    module.residual_quantizer = module.depthwise_conv._input_quantizer
                    attached_count += 1
                elif (
                    cls_name == "_OSA_module"
                    and hasattr(module, "concat")
                    and len(module.concat) > 0
                    and hasattr(module.concat[0], "_input_quantizer")
                ):
                    # VoVNet _OSA_module: reuse concat's first conv (QuantConv2d) input quantizer
                    module.residual_quantizer = module.concat[0]._input_quantizer
                    attached_count += 1
                else:
                    module.add_module("residual_quantizer", _new_input_quantizer())
                    attached_count += 1

            # Replace forward with the block-specific hook (quantizes only the residual branch).
            if "ConvNeXtBlock" in cls_name:
                _install_forward_hook(module, ConvNeXtBlockForwardHook)
            elif cls_name == "_OSA_module":
                _install_forward_hook(module, OSAModuleForwardHook)
            elif "Sparse" in cls_name:
                _install_forward_hook(module, SparseBasicBlockForwardHook)
            else:
                _install_forward_hook(module, BasicBlockForwardHook)

    if attached_count > 0:
        logger.info("Attached residual_quantizer to %d residual blocks", attached_count)


def attach_ese_quantizers(model: nn.Module) -> int:
    """
    Set up the single-Q-at-input eSE recipe on every ``eSEModule`` (one call, no ordering contract).

    Per module: attach ``pool_input_quantizer`` — the ONE Q/DQ at the eSE input, whose output ``qx``
    is shared by the pooling branch (``avg_pool → fc → hsigmoid``) *and* the ``Mul`` bypass — plus
    ``mul_gate_quantizer`` for the gate operand, then install :class:`eSEModuleForwardHook` once.
    Result: both ``Mul`` operands are INT8 with a single FP32→INT8 reformat at the eSE input.

    (Replaces the old order-dependent ``attach_ese_pool_input_quantizer`` →
    ``attach_ese_mul_identity_quantizer`` pair. The legacy two-Q path — a separate
    ``mul_identity_quantizer``, i.e. a second reformat with the pool branch left unquantized — was
    deleted in Goal 2; no shipping config used it.)

    Returns:
        Number of eSEModules set up.
    """
    require_pytorch_quantization_installed("eSE quantization")
    ensure_quant_descriptors_initialized()
    count = 0
    for _name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        if getattr(module, "pool_input_quantizer", None) is None:
            module.add_module("pool_input_quantizer", _new_input_quantizer())
        if getattr(module, "mul_gate_quantizer", None) is None:
            module.add_module("mul_gate_quantizer", _new_input_quantizer())
        _install_forward_hook(module, eSEModuleForwardHook)
        count += 1
    if count > 0:
        logger.info("Attached single-Q eSE quantizers (pool_input + mul_gate) to %d eSEModules", count)
    return count


def attach_maxpool_input_quantizer(
    model: nn.Module,
    skip_names: Optional[Set[str]] = None,
) -> int:
    """
    Replace nn.MaxPool2d modules with QuantBeforePool(quantizer, pool) so QDQ is applied before MaxPool.

    VoVNet _OSA_stage uses "Pooling" (MaxPool2d) before the first OSA block in stage3/stage4.
    This adds QDQ on the pool input so the MaxPool layer has quantized input in the ONNX graph.

    Returns:
        Number of MaxPool2d modules replaced with QuantBeforePool.
    """
    require_pytorch_quantization_installed("MaxPool input quantization")
    ensure_quant_descriptors_initialized()
    skip_names = skip_names or set()
    name_to_module = dict(model.named_modules())
    to_replace = []  # (parent_module, child_name, pool_module)

    for name, module in model.named_modules():
        if not isinstance(module, nn.MaxPool2d):
            continue
        if isinstance(module, QuantBeforePool):
            continue
        if any(name.startswith(s) for s in skip_names):
            continue
        parts = name.split(".")
        if not parts:
            continue
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = name_to_module.get(parent_name) if parent_name else model
        if parent is None:
            continue
        to_replace.append((parent, child_name, module))

    count = 0
    for parent, child_name, pool_module in to_replace:
        setattr(parent, child_name, QuantBeforePool(_new_input_quantizer(), pool_module))
        count += 1

    if count > 0:
        logger.info("Attached QDQ before %d MaxPool2d modules", count)
    return count
