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


def attach_quant_add(model: nn.Module, target_class_names: Optional[Set[str]] = None):
    """
    Attach residual_quantizer to modules that perform residual add and replace their forward methods.

    This follows the same approach as lidar-ai-solution (CUDA-BEVFusion):
    - Only quantize the identity branch (residual connection), not the conv path output
    - This enables TensorRT to fuse Conv+Add operations, reducing reformat operations
    - The residual_quantizer uses the same quant descriptor as conv layers for consistency

    Args:
        model: CenterPoint model
        target_class_names: Optional set of class name strings to match
                            (e.g., {"SparseBasicBlock", "BasicBlock"}). If None,
                            will match class names containing "BasicBlock".
    """
    require_pytorch_quantization_installed("residual quantization")
    ensure_quant_descriptors_initialized()

    target_class_names = target_class_names or {"BasicBlock", "SparseBasicBlock", "ConvNeXtBlock", "_OSA_module"}

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


def attach_ese_mul_identity_quantizer(model: nn.Module) -> int:
    """
    Attach mul_gate_quantizer to eSEModule so gate path has Q-DQ before Mul.
    When pool_input_quantizer is already present, do NOT add mul_identity_quantizer:
    bypass path uses pool_input_quantizer output (single Q at eSE input → one reformat).
    When pool_input_quantizer is absent, add both mul_identity_quantizer and mul_gate_quantizer.

    Returns:
        Number of eSEModules that got (mul_gate_quantizer and optionally mul_identity_quantizer) attached.
    """
    require_pytorch_quantization_installed("eSE Mul quantization")
    ensure_quant_descriptors_initialized()
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        # Already has pool_input_quantizer → single Q at input; only ensure mul_gate_quantizer (no mul_identity)
        if hasattr(module, "pool_input_quantizer") and module.pool_input_quantizer is not None:
            if not hasattr(module, "mul_gate_quantizer") or module.mul_gate_quantizer is None:
                module.add_module("mul_gate_quantizer", _new_input_quantizer())
            _install_forward_hook(module, eSEModuleForwardHook)
            count += 1
            continue
        # No pool_input_quantizer: attach both mul_identity and mul_gate (legacy two-Q path)
        if hasattr(module, "mul_identity_quantizer") and module.mul_identity_quantizer is not None:
            _install_forward_hook(module, eSEModuleForwardHook)
            if not hasattr(module, "mul_gate_quantizer") or module.mul_gate_quantizer is None:
                module.add_module("mul_gate_quantizer", _new_input_quantizer())
            count += 1
            continue
        if hasattr(module, "fc") and hasattr(module.fc, "_input_quantizer") and module.fc._input_quantizer is not None:
            module.mul_identity_quantizer = module.fc._input_quantizer
        else:
            module.add_module("mul_identity_quantizer", _new_input_quantizer())
        module.add_module("mul_gate_quantizer", _new_input_quantizer())
        _install_forward_hook(module, eSEModuleForwardHook)
        count += 1
    if count > 0:
        logger.info(
            "Attached eSE Mul quantizers to %d eSEModules "
            "(single Q at input when pool_input present, else identity+gate Q-DQ)",
            count,
        )
    return count


def attach_ese_pool_input_quantizer(model: nn.Module) -> int:
    """
    Attach pool_input_quantizer to eSEModule so that QDQ is applied before avg_pool.

    eSE: input -> [optional QDQ] -> avg_pool -> fc -> hsigmoid; identity -> [optional QDQ] -> Mul.
    This adds QDQ on the pooling branch input so the pooling layer has quantized input.

    Returns:
        Number of eSEModules that got pool_input_quantizer attached.
    """
    require_pytorch_quantization_installed("eSE pool input quantization")
    ensure_quant_descriptors_initialized()
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        if not (hasattr(module, "pool_input_quantizer") and module.pool_input_quantizer is not None):
            module.add_module("pool_input_quantizer", _new_input_quantizer())
        _install_forward_hook(module, eSEModuleForwardHook)
        count += 1
    if count > 0:
        logger.info("Attached pool_input_quantizer to %d eSEModules (QDQ before pooling)", count)
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
