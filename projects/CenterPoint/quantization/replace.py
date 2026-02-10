"""Module replacement functions for quantization."""

from typing import Optional, Set, Type

import torch.nn as nn
import torch.utils.checkpoint as cp

from .modules import QuantAdd, QuantConv2d, QuantConvTranspose2d, QuantLinear

# Flag to track if quantization descriptors have been initialized
_quant_descriptors_initialized = False


def _ensure_quant_descriptors_initialized():
    """
    Ensure that default quantization descriptors are initialized.

    This must be called before using transfer_to_quantization since the
    default_quant_desc_* class attributes are only set in __init__.
    """
    global _quant_descriptors_initialized
    if _quant_descriptors_initialized:
        return

    try:
        from pytorch_quantization import tensor_quant
    except ImportError:
        raise ImportError(
            "pytorch-quantization is required for quantization support. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )

    # Initialize QuantConv2d descriptors
    if QuantConv2d.default_quant_desc_input is None:
        QuantConv2d.default_quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
    if QuantConv2d.default_quant_desc_weight is None:
        QuantConv2d.default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    # Initialize QuantConvTranspose2d descriptors
    if QuantConvTranspose2d.default_quant_desc_input is None:
        QuantConvTranspose2d.default_quant_desc_input = tensor_quant.QuantDescriptor(
            num_bits=8, calib_method="histogram"
        )
    if QuantConvTranspose2d.default_quant_desc_weight is None:
        # Use per-tensor weight quantization for TensorRT compatibility.
        QuantConvTranspose2d.default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    # Guard rail: ConvTranspose2d INT8 in TensorRT is often fragile with per-channel
    # weight quantization. If someone sets it back to a per-channel descriptor, it can
    # break TRT build with "vol == 1 failed" or "Could not find any implementation".
    # We force per-tensor weights unless users explicitly opt out by passing a custom
    # descriptor into `init_quantizer` after replacement.
    try:
        qdw = QuantConvTranspose2d.default_quant_desc_weight
        if getattr(qdw, "axis", None) not in (None, (), []):
            QuantConvTranspose2d.default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    except Exception:
        # Be conservative: never fail descriptor initialization due to this guard.
        pass

    # Initialize QuantLinear descriptors
    if QuantLinear.default_quant_desc_input is None:
        QuantLinear.default_quant_desc_input = tensor_quant.QuantDescriptor(num_bits=8, calib_method="histogram")
    if QuantLinear.default_quant_desc_weight is None:
        # Per-row quantization for Linear layers (per output channel)
        QuantLinear.default_quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=8, axis=(0,))

    _quant_descriptors_initialized = True


def transfer_to_quantization(nn_instance: nn.Module, quant_module: Type) -> nn.Module:
    """
    Transfer weights and attributes from original module to quantized version.

    This function creates a new quantized module instance and copies all
    attributes from the original module, then initializes the quantizers.

    Args:
        nn_instance: Original PyTorch module (Conv2d, Linear, etc.)
        quant_module: Quantized module class (QuantConv2d, QuantLinear, etc.)

    Returns:
        Quantized module with copied weights and initialized quantizers
    """
    # Ensure quantization descriptors are initialized
    _ensure_quant_descriptors_initialized()

    # Create new instance without calling __init__
    quant_instance = quant_module.__new__(quant_module)

    # Copy all attributes from original module
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    # Initialize quantizers
    quant_instance.init_quantizer(
        quant_module.default_quant_desc_input,
        quant_module.default_quant_desc_weight,
    )

    return quant_instance


def quant_conv_module(model: nn.Module, skip_names: Optional[Set[str]] = None, prefix: str = ""):
    """
    Replace all Conv2d and ConvTranspose2d modules with quantized versions.

    This function recursively traverses the model and replaces all Conv2d
    and ConvTranspose2d modules with QuantConv2d and QuantConvTranspose2d
    respectively, except for modules whose names are in skip_names.

    Args:
        model: PyTorch model to modify
        skip_names: Set of module names to skip (full path from model root)
        prefix: Current prefix for module naming (used in recursion)

    Example:
        >>> model = CenterPoint(...)
        >>> quant_conv_module(model.pts_backbone)
        >>> quant_conv_module(model.pts_neck)
        >>> quant_conv_module(model.pts_bbox_head)
    """
    skip_names = skip_names or set()

    # Check if model is None or not a valid nn.Module
    if model is None or not isinstance(model, nn.Module):
        return

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in skip list
        # (This enables skipping containers like 'pts_backbone.blocks.0')
        if full_name in skip_names:
            continue

        # Recursively process submodules (only if submodule is not None)
        if submodule is not None:
            quant_conv_module(submodule, skip_names, full_name)

        # Replace Conv2d with QuantConv2d
        if isinstance(submodule, nn.Conv2d) and not isinstance(submodule, QuantConv2d):
            model._modules[name] = transfer_to_quantization(submodule, QuantConv2d)

        # Replace ConvTranspose2d with QuantConvTranspose2d
        elif isinstance(submodule, nn.ConvTranspose2d) and not isinstance(submodule, QuantConvTranspose2d):
            model._modules[name] = transfer_to_quantization(submodule, QuantConvTranspose2d)


def quant_linear_module(model: nn.Module, skip_names: Optional[Set[str]] = None, prefix: str = ""):
    """
    Replace all Linear modules with quantized versions.

    This function recursively traverses the model and replaces all Linear
    modules with QuantLinear, except for modules whose names are in skip_names.

    Args:
        model: PyTorch model to modify
        skip_names: Set of module names to skip (full path from model root)
        prefix: Current prefix for module naming (used in recursion)

    Example:
        >>> model = CenterPoint(...)
        >>> quant_linear_module(model.pts_voxel_encoder)
    """
    skip_names = skip_names or set()

    # Check if model is None or not a valid nn.Module
    if model is None or not isinstance(model, nn.Module):
        return

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in skip list
        if full_name in skip_names:
            continue

        # Recursively process submodules (only if submodule is not None)
        if submodule is not None:
            quant_linear_module(submodule, skip_names, full_name)

        # Replace Linear with QuantLinear
        if isinstance(submodule, nn.Linear) and not isinstance(submodule, QuantLinear):
            model._modules[name] = transfer_to_quantization(submodule, QuantLinear)


def quant_model(
    model: nn.Module,
    quant_backbone: bool = True,
    quant_neck: bool = True,
    quant_head: bool = True,
    quant_voxel_encoder: bool = True,
    quant_add: bool = False,
    quant_linear_backbone: bool = False,
    skip_names: Optional[Set[str]] = None,
):
    """
    Apply quantization to CenterPoint model components.

    This is a convenience function that applies quantization to specified
    components of a CenterPoint model.

    Args:
        model: CenterPoint model
        quant_backbone: Whether to quantize pts_backbone
        quant_neck: Whether to quantize pts_neck
        quant_head: Whether to quantize pts_bbox_head
        quant_voxel_encoder: Whether to quantize pts_voxel_encoder
        quant_linear_backbone: Whether to quantize Linear layers in pts_backbone
        skip_names: Set of module names to skip

    Example:
        >>> model = CenterPoint(...)
        >>> quant_model(model, skip_names={'pts_backbone.blocks.0'})
        >>> # If you also want to attach residual_quantizer to residual blocks:
        >>> quant_model(model, quant_add=True)
    """
    skip_names = skip_names or set()

    if quant_backbone and hasattr(model, "pts_backbone"):
        quant_conv_module(model.pts_backbone, skip_names, "pts_backbone")
        if quant_linear_backbone:
            quant_linear_module(model.pts_backbone, skip_names, "pts_backbone")

    if quant_neck and hasattr(model, "pts_neck"):
        quant_conv_module(model.pts_neck, skip_names, "pts_neck")

    if quant_head and hasattr(model, "pts_bbox_head"):
        quant_conv_module(model.pts_bbox_head, skip_names, "pts_bbox_head")

    if quant_voxel_encoder and hasattr(model, "pts_voxel_encoder"):
        quant_linear_module(model.pts_voxel_encoder, skip_names, "pts_voxel_encoder")

    if quant_add:
        attach_quant_add(model)


class BasicBlockForwardHook:
    """
    Forward hook for BasicBlock to use residual_quantizer for residual connections.

    This hook replaces the forward method of BasicBlock to quantize only the identity
    branch (residual connection), not the conv path output. This enables TensorRT to
    fuse Conv+Add operations, reducing reformat operations.

    According to TensorRT best practices:
    - Only quantize the residual branch (identity), not the conv path output
    - This allows TensorRT to fuse Conv+Add into a single kernel
    - The conv path output (after norm2) should remain unquantized until after Add
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection."""
        self = self.obj

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Quantize only the identity branch (residual connection), not the conv path output
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            # Directly call residual_quantizer as a module
            # This is critical for ONNX export to trace the quantizer call
            # The quantizer must be registered as a submodule or accessible as an attribute
            identity = self.residual_quantizer(identity)

        out = out + identity
        out = self.relu(out)
        return out


class SparseBasicBlockForwardHook:
    """
    Forward hook for SparseBasicBlock to use residual_quantizer for residual connections.

    This hook replaces the forward method of SparseBasicBlock to quantize only the identity
    branch (residual connection), not the conv path output. This enables TensorRT to
    fuse Conv+Add operations, reducing reformat operations.

    SparseBasicBlock works with SparseConvTensor which requires replace_feature.
    According to TensorRT best practices:
    - Only quantize the residual branch (identity), not the conv path output
    - This allows TensorRT to fuse Conv+Add into a single kernel
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection for sparse tensors."""
        self = self.obj

        identity = x
        out = self.conv1(x)

        # Handle ReLU (may be fused in conv1)
        if hasattr(self, "relu") and not getattr(self.conv1, "act_type", None):
            out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Quantize only the identity branch (residual connection), not the conv path output
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            # Directly call residual_quantizer as a module
            # This is critical for ONNX export to trace the quantizer call
            identity = identity.replace_feature(self.residual_quantizer(identity.features))

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out


class ConvNeXtBlockForwardHook:
    """
    Forward hook for ConvNeXtBlock to use residual_quantizer on the identity branch.

    Mirrors mmpretrain ConvNeXtBlock.forward, but quantizes the shortcut before add.
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection for ConvNeXtBlock."""
        self = self.obj

        def _inner_forward(x):
            identity = x

            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format="channel_last")
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format="channel_last")
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format="channel_first")
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format="channel_first")
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            # Quantize only the identity branch (residual connection).
            if hasattr(self, "residual_quantizer"):
                identity = self.residual_quantizer(identity)

            x = identity + self.drop_path(x)
            return x

        if getattr(self, "with_cp", False) and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


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
        raise ImportError(
            "pytorch-quantization is required for residual quantization. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )

    # Ensure quantization descriptors are initialized
    _ensure_quant_descriptors_initialized()

    target_class_names = target_class_names or {"BasicBlock", "SparseBasicBlock", "ConvNeXtBlock"}

    attached_count = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in target_class_names or any(name in cls_name for name in target_class_names):
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
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Attached residual_quantizer to {attached_count} residual blocks")
