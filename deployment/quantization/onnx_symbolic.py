"""
ONNX symbolic registration for quantized modules.

This module registers ONNX symbolic functions for QuantAdd and other quantized
modules to ensure proper ONNX export and TensorRT fusion support.
"""

import logging

try:
    import torch
    import torch.onnx.symbolic_helper as sym_help
    from torch.onnx import register_custom_op_symbolic

    TORCH_ONNX_AVAILABLE = True
except ImportError:
    TORCH_ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


def register_quant_add_symbolic():
    """
    Register ONNX symbolic function for QuantAdd.

    This ensures QuantAdd is exported as a standard ONNX Add node,
    which TensorRT can properly fuse without reformat operations.
    """
    if not TORCH_ONNX_AVAILABLE:
        logger.warning("torch.onnx not available, skipping QuantAdd symbolic registration")
        return

    try:
        from deployment.quantization.modules.quant_add import QuantAdd

        def quant_add_symbolic(g, x, y, *args, **kwargs):
            """
            ONNX symbolic function for QuantAdd.

            Exports QuantAdd as a standard ONNX Add node.
            The quantization is handled by the Q/DQ nodes from TensorQuantizer,
            so we just need to export the add operation itself.
            """
            return g.op("Add", x, y)

        # Register symbolic function for QuantAdd.forward
        # Note: We need to register it for the module's forward method
        # PyTorch will look for symbolic functions in the format:
        # <module_class>.<method_name>
        try:
            # Try to register using the module path
            register_custom_op_symbolic(
                "deployment.quantization.modules.quant_add::QuantAdd.forward",
                quant_add_symbolic,
                opset_version=13,
            )
        except Exception:
            # Fallback: register using torch.onnx.symbolic_registry
            # This is a more direct approach
            try:
                import torch.onnx.symbolic_registry as sym_registry

                # Register for opset 13 and above
                for opset_version in range(13, 18):
                    try:
                        sym_registry.register_op(
                            f"deployment.quantization.modules.quant_add::QuantAdd",
                            quant_add_symbolic,
                            "",
                            opset_version,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        # Alternative: Use monkey patching to add symbolic method to QuantAdd class
        # This is more reliable as it directly adds the method to the class
        if hasattr(torch.onnx, "register_custom_op_symbolic"):
            # Register using the class directly
            QuantAdd.symbolic = staticmethod(quant_add_symbolic)
            logger.info("Registered QuantAdd ONNX symbolic function via class method")

    except ImportError as e:
        logger.warning(f"Could not import QuantAdd: {e}, skipping symbolic registration")
    except Exception as e:
        logger.warning(f"Failed to register QuantAdd symbolic: {e}")


def register_all_quantization_symbolics():
    """Register all quantization-related ONNX symbolic functions."""
    register_quant_add_symbolic()


# Auto-register on import
if TORCH_ONNX_AVAILABLE:
    register_all_quantization_symbolics()
