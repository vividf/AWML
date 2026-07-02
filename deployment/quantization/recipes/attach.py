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

from deployment.quantization.core.availability import pytorch_quantization_install_hint
from deployment.quantization.core.modules import QuantConv2d
from deployment.quantization.core.replace import _ensure_quant_descriptors_initialized

from .forward_hooks import (
    BasicBlockForwardHook,
    ConvNeXtBlockForwardHook,
    OSAModuleForwardHook,
    QuantBeforePool,
    SparseBasicBlockForwardHook,
    eSEModuleForwardHook,
)


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
    try:
        from pytorch_quantization import tensor_quant
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError(pytorch_quantization_install_hint("residual quantization"))

    # Ensure quantization descriptors are initialized
    _ensure_quant_descriptors_initialized()

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
                    quant_desc = QuantConv2d.default_quant_desc_input
                    if quant_desc is None:
                        quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
                    else:
                        if not hasattr(quant_desc, "calib_method") or quant_desc.calib_method is None:
                            quant_desc.calib_method = "histogram"
                    concat_quantizers = nn.ModuleList([TensorQuantizer(quant_desc) for _ in range(n_branch_inputs)])
                    module.add_module("concat_input_quantizers", concat_quantizers)
                # When identity=True we reuse concat_input_quantizers[0] as single Q for block input (no extra module)
                if not getattr(module, "identity", False):
                    if not isinstance(module.forward, OSAModuleForwardHook):
                        if not hasattr(module, "_original_forward"):
                            module._original_forward = module.forward
                        module.forward = OSAModuleForwardHook(module)
                    continue
            # Attach residual_quantizer if not already present
            # Aligned with lidar-ai-solution:
            # - If downsample exists: create new TensorQuantizer
            # - If no downsample: reuse conv1._input_quantizer (shares calibration data)
            if not hasattr(module, "residual_quantizer"):
                if hasattr(module, "downsample") and module.downsample is not None:
                    # Has downsample: create new quantizer
                    quant_desc = QuantConv2d.default_quant_desc_input
                    if quant_desc is None:
                        # Fallback to default if not initialized
                        quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
                    else:
                        # Ensure calib_method is set for calibration
                        if not hasattr(quant_desc, "calib_method") or quant_desc.calib_method is None:
                            quant_desc.calib_method = "histogram"
                    residual_quantizer = TensorQuantizer(quant_desc)
                    # Register as submodule so PyTorch ONNX export can trace it
                    module.add_module("residual_quantizer", residual_quantizer)
                    attached_count += 1
                elif hasattr(module, "conv1") and hasattr(module.conv1, "_input_quantizer"):
                    # No downsample: reuse conv1._input_quantizer (same as lidar-ai-solution)
                    # Note: We cannot use add_module() here because conv1._input_quantizer is already
                    # a submodule of conv1. PyTorch doesn't allow a module to be a submodule of multiple parents.
                    # However, ONNX export should still trace the call if we access it correctly.
                    # We'll just assign it as an attribute, and the forward hook will call it.
                    # The key is that TensorQuantizer.use_fb_fake_quant and _enable_onnx_export must be set.
                    residual_quantizer = module.conv1._input_quantizer
                    # Assign as attribute (not submodule) - ONNX export will trace the call
                    # IMPORTANT: Even though it's a reference, ONNX export should trace it when called
                    # in the forward hook. The quantizer's forward method will be called, and if
                    # _enable_onnx_export is True, it will export as QDQ nodes.
                    module.residual_quantizer = residual_quantizer
                    attached_count += 1
                elif hasattr(module, "depthwise_conv") and hasattr(module.depthwise_conv, "_input_quantizer"):
                    # ConvNeXtBlock: reuse depthwise_conv._input_quantizer
                    residual_quantizer = module.depthwise_conv._input_quantizer
                    module.residual_quantizer = residual_quantizer
                    attached_count += 1
                elif (
                    cls_name == "_OSA_module"
                    and hasattr(module, "concat")
                    and len(module.concat) > 0
                    and hasattr(module.concat[0], "_input_quantizer")
                ):
                    # VoVNet _OSA_module: reuse concat's first conv (QuantConv2d) input quantizer
                    residual_quantizer = module.concat[0]._input_quantizer
                    module.residual_quantizer = residual_quantizer
                    attached_count += 1
                else:
                    # Fallback: create new quantizer
                    quant_desc = QuantConv2d.default_quant_desc_input
                    if quant_desc is None:
                        quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
                    else:
                        # Ensure calib_method is set for calibration
                        if not hasattr(quant_desc, "calib_method") or quant_desc.calib_method is None:
                            quant_desc.calib_method = "histogram"
                    residual_quantizer = TensorQuantizer(quant_desc)
                    # Register as submodule so PyTorch ONNX export can trace it
                    module.add_module("residual_quantizer", residual_quantizer)
                    attached_count += 1

            # Replace forward method with hook that uses residual_quantizer
            if "ConvNeXtBlock" in cls_name:
                if not isinstance(module.forward, ConvNeXtBlockForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = ConvNeXtBlockForwardHook(module)
            elif cls_name == "_OSA_module":
                if not isinstance(module.forward, OSAModuleForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = OSAModuleForwardHook(module)
            elif "Sparse" in cls_name:
                # SparseBasicBlock: use SparseBasicBlockForwardHook
                if not isinstance(module.forward, SparseBasicBlockForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = SparseBasicBlockForwardHook(module)
            else:
                # BasicBlock: use BasicBlockForwardHook
                if not isinstance(module.forward, BasicBlockForwardHook):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = BasicBlockForwardHook(module)

    if attached_count > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Attached residual_quantizer to {attached_count} residual blocks")


def attach_ese_mul_identity_quantizer(model: nn.Module) -> int:
    """
    Attach mul_gate_quantizer to eSEModule so gate path has Q-DQ before Mul.
    When pool_input_quantizer is already present, do NOT add mul_identity_quantizer:
    bypass path uses pool_input_quantizer output (single Q at eSE input → one reformat).
    When pool_input_quantizer is absent, add both mul_identity_quantizer and mul_gate_quantizer.

    Returns:
        Number of eSEModules that got (mul_gate_quantizer and optionally mul_identity_quantizer) attached.
    """
    try:
        from pytorch_quantization import tensor_quant
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError(pytorch_quantization_install_hint("eSE Mul quantization"))

    _ensure_quant_descriptors_initialized()
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        # Already has pool_input_quantizer → single Q at input; only ensure mul_gate_quantizer (no mul_identity)
        if hasattr(module, "pool_input_quantizer") and module.pool_input_quantizer is not None:
            if not hasattr(module, "mul_gate_quantizer") or module.mul_gate_quantizer is None:
                quant_desc = QuantConv2d.default_quant_desc_input
                if quant_desc is None:
                    quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
                elif not getattr(quant_desc, "calib_method", None):
                    quant_desc.calib_method = "histogram"
                module.add_module("mul_gate_quantizer", TensorQuantizer(quant_desc))
            if not isinstance(module.forward, eSEModuleForwardHook):
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                module.forward = eSEModuleForwardHook(module)
            count += 1
            continue
        # No pool_input_quantizer: attach both mul_identity and mul_gate (legacy two-Q path)
        if hasattr(module, "mul_identity_quantizer") and module.mul_identity_quantizer is not None:
            if not isinstance(module.forward, eSEModuleForwardHook):
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                module.forward = eSEModuleForwardHook(module)
            if not hasattr(module, "mul_gate_quantizer") or module.mul_gate_quantizer is None:
                quant_desc = QuantConv2d.default_quant_desc_input
                if quant_desc is None:
                    quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
                elif not getattr(quant_desc, "calib_method", None):
                    quant_desc.calib_method = "histogram"
                module.add_module("mul_gate_quantizer", TensorQuantizer(quant_desc))
            count += 1
            continue
        quant_desc = QuantConv2d.default_quant_desc_input
        if quant_desc is None:
            quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
        elif not getattr(quant_desc, "calib_method", None):
            quant_desc.calib_method = "histogram"
        if hasattr(module, "fc") and hasattr(module.fc, "_input_quantizer") and module.fc._input_quantizer is not None:
            module.mul_identity_quantizer = module.fc._input_quantizer
        else:
            q = TensorQuantizer(quant_desc)
            module.add_module("mul_identity_quantizer", q)
        module.add_module("mul_gate_quantizer", TensorQuantizer(quant_desc))
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        module.forward = eSEModuleForwardHook(module)
        count += 1
    if count > 0:
        logger = logging.getLogger(__name__)
        logger.info(
            f"Attached eSE Mul quantizers to {count} eSEModules "
            "(single Q at input when pool_input present, else identity+gate Q-DQ)"
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
    try:
        from pytorch_quantization import tensor_quant
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError(pytorch_quantization_install_hint("eSE pool input quantization"))

    _ensure_quant_descriptors_initialized()
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        if hasattr(module, "pool_input_quantizer") and module.pool_input_quantizer is not None:
            if not isinstance(module.forward, eSEModuleForwardHook):
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                module.forward = eSEModuleForwardHook(module)
            count += 1
            continue
        quant_desc = QuantConv2d.default_quant_desc_input
        if quant_desc is None:
            quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
        elif not getattr(quant_desc, "calib_method", None):
            quant_desc.calib_method = "histogram"
        q = TensorQuantizer(quant_desc)
        module.add_module("pool_input_quantizer", q)
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        module.forward = eSEModuleForwardHook(module)
        count += 1
    if count > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Attached pool_input_quantizer to {count} eSEModules (QDQ before pooling)")
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
    try:
        from pytorch_quantization import tensor_quant
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError(pytorch_quantization_install_hint("MaxPool input quantization"))

    _ensure_quant_descriptors_initialized()
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

    quant_desc = QuantConv2d.default_quant_desc_input
    if quant_desc is None:
        quant_desc = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
    elif not getattr(quant_desc, "calib_method", None):
        quant_desc.calib_method = "histogram"

    count = 0
    for parent, child_name, pool_module in to_replace:
        q = TensorQuantizer(quant_desc)
        wrapper = QuantBeforePool(q, pool_module)
        setattr(parent, child_name, wrapper)
        count += 1

    if count > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Attached QDQ before {count} MaxPool2d modules")
    return count
