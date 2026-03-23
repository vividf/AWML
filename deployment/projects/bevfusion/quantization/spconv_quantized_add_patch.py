"""Patch spconv FX ``quantized_add`` for SparseConvTensor residuals.

``transform_qdq`` wires ``torch.ops.quantized.add`` to spconv's ``quantized_add``.
Stock spconv only handles dense QTensor; sparse BasicBlock FX uses ``out + identity``
which stays as ``SparseConvTensor`` through the graph, so ONNX / eager fails on
``x.shape``. We wrap the stock implementation to run on ``.features`` and
``replace_feature``.

Also syncs ``spconv.pytorch.quantization.graph.quantized_add`` when that module
keeps a stale import alias (pip spconv). Retarget existing ``GraphModule`` nodes
after load so checkpoints from older runs pick up the patched function.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def ensure_spconv_quantized_add_sparse_support() -> None:
    """Idempotently wrap ``spconv.pytorch.quantization.core.quantized_add``."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import spconv.pytorch.quantization.core as core
        import spconv.pytorch.quantization.graph as qgraph
    except ImportError:
        logger.warning("spconv quantization not importable; skip quantized_add sparse patch")
        return

    dense_impl = core.quantized_add

    def quantized_add_sparse_aware(x, y, scale, zero_point):
        from spconv.pytorch.core import SparseConvTensor

        if isinstance(x, SparseConvTensor) and isinstance(y, SparseConvTensor):
            return x.replace_feature(dense_impl(x.features, y.features, scale, zero_point))
        if isinstance(x, SparseConvTensor):
            return x.replace_feature(dense_impl(x.features, y, scale, zero_point))
        if isinstance(y, SparseConvTensor):
            return y.replace_feature(dense_impl(x, y.features, scale, zero_point))
        return dense_impl(x, y, scale, zero_point)

    core.quantized_add = quantized_add_sparse_aware
    # Older spconv: graph.py did ``from ...core import quantized_add`` (stale binding).
    if hasattr(qgraph, "quantized_add"):
        qgraph.quantized_add = quantized_add_sparse_aware

    _PATCHED = True
    logger.debug("Patched spconv quantization quantized_add for SparseConvTensor operands")


def retarget_graphmodule_quantized_add_calls(root: torch.nn.Module) -> int:
    """Point FX ``call_function`` quantized_add nodes at current ``core.quantized_add``.

    Returns the number of nodes updated.
    """
    import torch.fx

    try:
        import spconv.pytorch.quantization.core as core
    except ImportError:
        return 0

    target_fn = core.quantized_add
    updated = 0

    for mod in root.modules():
        graph = getattr(mod, "graph", None)
        if graph is None or not isinstance(graph, torch.fx.Graph):
            continue
        local = 0
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            fn = node.target
            if not callable(fn):
                continue
            if getattr(fn, "__name__", None) != "quantized_add":
                continue
            mod_name = getattr(fn, "__module__", "") or ""
            if "spconv" not in mod_name or "quantization" not in mod_name:
                continue
            if fn is target_fn:
                continue
            node.target = target_fn
            local += 1
        if local:
            try:
                mod.recompile()
                mod.graph.lint()
            except Exception as e:
                logger.warning("retarget GraphModule lint/recompile: %s", e)
            updated += local

    if updated:
        logger.info("Retargeted %d FX quantized_add nodes to patched spconv core.quantized_add", updated)
    return updated
