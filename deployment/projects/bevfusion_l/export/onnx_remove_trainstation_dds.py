"""Remove data-dependent-shape (``trainStation``) boundaries from the sparse ONNX graph.

The 4 down-sampling ``GetIndicePairsImplicitGemm`` nodes (``subm=0``) declare a TensorRT *size
tensor* for their output extent, forcing a ``DeviceToShapeHostCopy`` sync that splits the sparse
engine into ``[trainStationN]`` segments and idles the GPU between them. Their outputs — the
rulebook — depend only on voxel geometry, so they can be computed before the engine runs.

This pass performs the export-side half: it deletes those 4 nodes and promotes their consumed
outputs to **graph inputs** with an ordinary dynamic dim, so TensorRT resolves every shape from
``setInputShape`` before ``enqueueV3`` and the size tensors (hence the trainStations) disappear.
The 17 submanifold nodes stay in the graph — they run on the (now graph-input) down-sampled
coordinates and introduce no data-dependent shape. The ``ImplicitGemm`` conv nodes are untouched;
they already derive their output extent from the ``pair_mask`` input dim.

The runtime must supply the 4 rulebooks. Their names, geometry and precompute live in
:mod:`deployment.projects.bevfusion_l.io.sparse_rulebook_inputs`; :func:`embed_rulebook_stages_metadata`
additionally records the geometry read off the graph into the model's ``metadata_props`` so the
on-board runtime needs neither a sidecar file nor hard-coded stage constants.

Registered as a ``post_transforms`` entry on the ``bevfusion_sparse`` component (gated by the
deploy-config flag ``spconv_remove_trainstation``); see ``export/component_builder.py``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Sequence, Tuple

import onnx
from onnx import TensorProto, helper

from deployment.projects.bevfusion_l.io.sparse_rulebook_inputs import (
    DOWNSAMPLE_STAGES,
    RULEBOOK_SLOTS,
    downsample_spatial_shapes,
    rulebook_input_base,
    rulebook_input_name,
)

logger = logging.getLogger(__name__)

_GET_INDICE_PAIRS_OP = "GetIndicePairsImplicitGemm"

# metadata_props key holding the down-sample stage geometry as a compact JSON list. The Autoware
# runtime reads it back with a minimal protobuf field scanner (``load_stages_from_onnx``).
RULEBOOK_STAGES_METADATA_KEY = "rulebook_stages"


def _attr(node: onnx.NodeProto, name: str) -> Any:
    """Return an int / int-list attribute of ``node``, or ``None`` when absent."""
    for attribute in node.attribute:
        if attribute.name == name:
            return list(attribute.ints) if attribute.ints else attribute.i
    return None


def _attr_ints(node: onnx.NodeProto, name: str) -> List[int]:
    """Return an attribute as a list of ints (scalar attributes become a 1-element list)."""
    value = _attr(node, name)
    return [int(v) for v in value] if isinstance(value, list) else [int(value)]


_ENCODER_LAYER_TOKEN = re.compile(r"encoder_layer(\d+)$")


def _stage_tag(node_name: str) -> str:
    """Short, stable tag for the promoted input names, e.g. ``encoder_layer1`` -> ``l1``.

    Falls back to ``out`` for ``conv_out`` and to a sanitized node name otherwise. The scope token
    must end in the layer index (``encoder_layer1``, not a container like ``encoder_layers``), so a
    differently nested encoder yields a distinct fallback tag rather than a silent collision.
    """
    if "conv_out" in node_name:
        return "out"
    for token in node_name.split("/"):
        match = _ENCODER_LAYER_TOKEN.match(token)
        if match:
            return "l" + match.group(1)
    return node_name.strip("/").replace("/", "_")


def _rulebook_input(name: str, dims: Sequence[Any]) -> onnx.ValueInfoProto:
    """Declare an INT32 graph input; ``dims`` entries are ints (fixed) or strs (symbolic)."""
    return helper.make_tensor_value_info(name, TensorProto.INT32, list(dims))


def _warn_on_runtime_table_mismatch(stages_meta: List[Dict[str, Any]]) -> None:
    """Warn if the graph's down-sample stages differ from the AWML-eval stage table.

    The exported ONNX is self-describing either way (see :func:`embed_rulebook_stages_metadata`),
    so a mismatch does not invalidate it for the on-board runtime — but AWML's own TensorRT
    evaluation precomputes rulebooks from
    :data:`~deployment.projects.bevfusion_l.io.sparse_rulebook_inputs.DOWNSAMPLE_STAGES`, and would
    silently produce wrong results for this engine. Warn loudly rather than fail the export.
    """
    graph_geometry = [
        (m["onnx_base"].rsplit("/", 1)[-1], tuple(m["ksize"]), tuple(m["stride"]), tuple(m["padding"]))
        for m in stages_meta
    ]
    table_geometry = [(s.tag, s.ksize, s.stride, s.padding) for s in DOWNSAMPLE_STAGES]
    if graph_geometry != table_geometry:
        logger.warning(
            "Graph down-sample stages %s do not match sparse_rulebook_inputs.DOWNSAMPLE_STAGES %s. "
            "The exported ONNX is still valid (stage geometry is embedded in metadata_props), but "
            "AWML TensorRT evaluation of this engine would precompute the wrong rulebooks.",
            graph_geometry,
            table_geometry,
        )
        return

    expected_spatial = downsample_spatial_shapes(stages_meta[0]["spatial_shape"])
    graph_spatial = [tuple(m["spatial_shape"]) for m in stages_meta]
    if graph_spatial != expected_spatial:
        logger.warning(
            "Graph down-sample spatial shapes %s do not match the cascade derived from sparse_shape "
            "%s (%s); AWML TensorRT evaluation of this engine would precompute wrong rulebooks.",
            graph_spatial,
            stages_meta[0]["spatial_shape"],
            expected_spatial,
        )


def remove_trainstation_dds(
    model: onnx.ModelProto, *, size_dim_prefix: str = "sp_dds"
) -> Tuple[onnx.ModelProto, List[str], List[Dict[str, Any]]]:
    """Delete the down-sampling GetIndicePairs nodes and expose their outputs as graph inputs.

    The promoted inputs, per down-sample stage (short tag ``l1``/``l2``/``l3``/``out``, with a
    per-stage symbolic dim ``N``):

      * ``rulebook/<tag>/out_indices``  : ``[N, 4]``   (node output 0)
      * ``rulebook/<tag>/pair_fwd``     : ``[KV, N]``  (node output 1)
      * ``rulebook/<tag>/pair_mask``    : ``[N, 1]``   (node output 2)
      * ``rulebook/<tag>/mask_argsort`` : ``[N]``      (node output 3)

    Node output 4 (the ``num_act_out`` scalar) is dropped — it has no consumer in the graph. Every
    consumer edge is rewritten from the removed node's intermediate tensor name to the clean
    ``rulebook/<tag>/<slot>`` name, so the graph stays valid and the promoted inputs read as one
    group rather than as outputs of a node that is gone.

    Args:
        model: The exported sparse-encoder ONNX model, modified in place.
        size_dim_prefix: Prefix for the per-stage symbolic dim names.

    Returns:
        ``(model, new_input_names, stages_meta)``. ``stages_meta`` is one dict per removed node, in
        graph order, with keys ``onnx_base``, ``ksize``, ``stride``, ``padding``, ``dilation``,
        ``spatial_shape`` — the fields the Autoware runtime's ``SparseDownsampleStage`` expects.
        Both lists are empty when the graph has no down-sampling nodes (already stripped, or a
        submanifold-only graph).
    """
    graph = model.graph

    downsample_nodes = [n for n in graph.node if n.op_type == _GET_INDICE_PAIRS_OP and _attr(n, "subm") == 0]
    if not downsample_nodes:
        logger.warning("No down-sampling %s nodes found; nothing to remove.", _GET_INDICE_PAIRS_OP)
        return model, [], []

    new_inputs: List[onnx.ValueInfoProto] = []
    new_input_names: List[str] = []
    stages_meta: List[Dict[str, Any]] = []
    renames: Dict[str, str] = {}  # removed node's output tensor -> promoted graph-input name
    seen_tags: set = set()

    for node in downsample_nodes:
        ksize = _attr_ints(node, "ksize")
        kernel_volume = math.prod(ksize)
        tag = _stage_tag(node.name)
        if tag in seen_tags:
            # Two stages resolving to the same tag would declare the same graph input twice; the
            # ONNX checker would only report a cryptic SSA violation much later.
            raise ValueError(
                f"Down-sample node {node.name!r} resolves to stage tag {tag!r}, which is already "
                "taken. The encoder's node naming is not what _stage_tag() expects."
            )
        seen_tags.add(tag)
        size_dim = f"{size_dim_prefix}_{tag}_n"

        stages_meta.append(
            {
                "onnx_base": rulebook_input_base(tag),
                "ksize": ksize,
                "stride": _attr_ints(node, "stride"),
                "padding": _attr_ints(node, "padding"),
                "dilation": _attr_ints(node, "dilation"),
                "spatial_shape": _attr_ints(node, "spatial_shape"),
            }
        )

        dims_per_slot = {
            "out_indices": [size_dim, 4],
            "pair_fwd": [kernel_volume, size_dim],
            "pair_mask": [size_dim, 1],
            "mask_argsort": [size_dim],
        }
        for slot_index, slot in enumerate(RULEBOOK_SLOTS):
            promoted_name = rulebook_input_name(tag, slot)
            renames[node.output[slot_index]] = promoted_name
            new_inputs.append(_rulebook_input(promoted_name, dims_per_slot[slot]))
            new_input_names.append(promoted_name)

        logger.info(
            "trainStation DDS removal: promoting %s outputs to graph inputs rulebook/%s/* (KV=%d, dim=%s)",
            tag,
            tag,
            kernel_volume,
            size_dim,
        )

    for node in downsample_nodes:
        graph.node.remove(node)

    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in renames:
                node.input[i] = renames[input_name]

    # The removed nodes' outputs are graph inputs now, typed via make_tensor_value_info; leaving
    # their old value_info entries behind would declare the same tensors twice.
    for stale in [vi for vi in graph.value_info if vi.name in renames]:
        graph.value_info.remove(stale)

    graph.input.extend(new_inputs)

    _warn_on_runtime_table_mismatch(stages_meta)

    logger.info(
        "Removed %d down-sampling %s node(s); added %d rulebook graph input(s).",
        len(downsample_nodes),
        _GET_INDICE_PAIRS_OP,
        len(new_inputs),
    )
    return model, new_input_names, stages_meta


def embed_rulebook_stages_metadata(model: onnx.ModelProto, stages_meta: List[Dict[str, Any]]) -> onnx.ModelProto:
    """Record ``stages_meta`` in the model's ``metadata_props`` as compact JSON.

    Makes the stripped graph self-describing: the Autoware runtime reads the stage geometry back
    from the ONNX itself, so it needs neither a sidecar file nor hard-coded stage constants.
    Idempotent — any previous entry under the same key is replaced. A no-op for an empty
    ``stages_meta`` (nothing was removed).
    """
    if not stages_meta:
        return model
    for existing in [prop for prop in model.metadata_props if prop.key == RULEBOOK_STAGES_METADATA_KEY]:
        model.metadata_props.remove(existing)
    entry = model.metadata_props.add()
    entry.key = RULEBOOK_STAGES_METADATA_KEY
    entry.value = json.dumps(stages_meta, separators=(",", ":"))
    logger.info("Embedded %d rulebook stage(s) in ONNX metadata_props.", len(stages_meta))
    return model
