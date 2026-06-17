"""Remove data-dependent-shape (DDS) / ``trainStation`` boundaries from the
BEVFusion sparse ONNX graph.

Background
----------
The sparse middle-encoder ONNX contains 21 ``GetIndicePairsImplicitGemm``
nodes.  The 4 **down-sampling** ones (``subm=0``) declare a *size tensor* for
their output extent (the active-voxel count after stride-2 pooling is data
dependent).  TensorRT must read that count back to the host
(``DeviceToShapeHostCopy``) before sizing downstream tensors, which splits the
engine into ``[trainStationN]`` segments and stalls the GPU (~30% idle, see
``BEVFusion_spconv_DDS_optimization.md``).

The 17 **submanifold** nodes (``subm=1``) keep the active-voxel set unchanged,
so their output extent equals their input extent — no size tensor, no DDS.

Optimization (Route A)
----------------------
The rulebook (``out_indices``/``pair_fwd``/``pair_mask``/``mask_argsort``)
depends only on voxel geometry, not feature values, so it can be precomputed in
preprocessing and fed in as graph inputs.  This pass performs the export-side
half: it deletes the 4 down-sampling ``GetIndicePairsImplicitGemm`` nodes and
promotes their consumed outputs to **graph inputs** with a normal dynamic dim.
TensorRT then resolves every shape from ``setInputShape`` before ``enqueueV3``
— the size tensors (and thus the trainStations) disappear.

The submanifold ``GetIndicePairsImplicitGemm`` nodes are kept in the graph: they
run on the (now graph-input) down-sampled coordinates and add no DDS.  The
``ImplicitGemm`` conv nodes are untouched — they already derive their output
extent from the ``pair_mask`` input dim.

The runtime (autoware_bevfusion) must compute these 4 rulebooks from the voxel
coordinates and bind them to the new inputs; see §7.2 of the design doc.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)

_GET_INDICE_PAIRS_OP = "GetIndicePairsImplicitGemm"

# Clean, hierarchical names for the promoted rulebook graph inputs. The leading
# ``rulebook/<stage>/`` segment makes Netron group all 16 inputs into one
# collapsible ``rulebook`` box (with ``l1``/``l2``/``l3``/``out`` sub-boxes)
# instead of 16 dangling ``…GetIndicePairsImplicitGemm_output_N`` tensors that
# look like outputs of a node no longer in the graph. The slot order is the
# fixed output order of ``GetIndicePairsImplicitGemm`` (see sparse_functional.py):
#   0=out_indices, 1=pair_fwd, 2=pair_mask, 3=mask_argsort (out[4]=num_act_out unused).
_INPUT_NAMESPACE = "rulebook"
_SLOTS = ("out_indices", "pair_fwd", "pair_mask", "mask_argsort")


def rulebook_input_name(stage_tag: str, slot: str) -> str:
    """Canonical graph-input name for a rulebook tensor, e.g. ``rulebook/l1/pair_fwd``.

    This is the single source of truth for the name shared by the export side and
    both runtimes (AWML eval ``sparse_rulebook_precompute`` and autoware_bevfusion).
    """
    return f"{_INPUT_NAMESPACE}/{stage_tag}/{slot}"


def _attr(node: onnx.NodeProto, name: str):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            return a.i
    return None


def _stage_tag(node_name: str) -> str:
    """Short, stable tag for the new input names, e.g. ``encoder_layer1`` -> ``l1``.

    Falls back to ``out`` for ``conv_out`` and to a sanitized node name otherwise.
    """
    if "conv_out" in node_name:
        return "out"
    for token in node_name.split("/"):
        if token.startswith("encoder_layer"):
            return "l" + token[len("encoder_layer") :]
    return node_name.strip("/").replace("/", "_")


def _make_input(name: str, dims) -> onnx.ValueInfoProto:
    """Create an INT32 graph input. ``dims`` entries: int -> fixed, str -> symbolic."""
    return helper.make_tensor_value_info(name, TensorProto.INT32, list(dims))


def remove_trainstation_dds(
    model: onnx.ModelProto, *, size_dim_prefix: str = "sp_dds"
) -> Tuple[onnx.ModelProto, List[str]]:
    """Delete down-sampling GetIndicePairs nodes and expose their outputs as inputs.

    Returns the modified model and the ordered list of new graph-input names. The
    new inputs (per down-sampling stage with short tag ``l1``/``l2``/``l3``/``out``
    and shared symbolic dim ``N``):

      * ``rulebook/<tag>/out_indices``  : ``[N, 4]``   (out[0])
      * ``rulebook/<tag>/pair_fwd``     : ``[KV, N]``  (out[1])
      * ``rulebook/<tag>/pair_mask``    : ``[N, 1]``   (out[2])
      * ``rulebook/<tag>/mask_argsort`` : ``[N]``      (out[3])

    The original outputs of the removed nodes are intermediate tensors named after
    the (now-gone) node path (``…/GetIndicePairsImplicitGemm_output_N``); every
    consumer edge is rewritten to the clean ``rulebook/<tag>/<slot>`` name so the
    graph stays valid and Netron groups the inputs under one ``rulebook`` box.

    out[4] (``num_act_out`` scalar) is dropped — it has no consumer in the graph.
    """
    g = model.graph

    downs = [n for n in g.node if n.op_type == _GET_INDICE_PAIRS_OP and _attr(n, "subm") == 0]
    if not downs:
        logger.warning("No down-sampling %s nodes found; nothing to do.", _GET_INDICE_PAIRS_OP)
        return model, []

    new_inputs: List[onnx.ValueInfoProto] = []
    new_input_names: List[str] = []
    rename: Dict[str, str] = {}  # old intermediate tensor name -> new graph-input name

    for node in downs:
        ksize = _attr(node, "ksize")
        ksize = ksize if isinstance(ksize, list) else [ksize]
        kernel_volume = 1
        for k in ksize:
            kernel_volume *= int(k)

        tag = _stage_tag(node.name)
        size_dim = f"{size_dim_prefix}_{tag}_n"

        # out[0..3] are consumed; out[4] (scalar num_act_out) is not.
        dims_per_slot = {
            "out_indices": [size_dim, 4],
            "pair_fwd": [kernel_volume, size_dim],
            "pair_mask": [size_dim, 1],
            "mask_argsort": [size_dim],
        }
        for slot_idx, slot in enumerate(_SLOTS):
            old_name = node.output[slot_idx]
            new_name = rulebook_input_name(tag, slot)
            rename[old_name] = new_name
            new_inputs.append(_make_input(new_name, dims_per_slot[slot]))
            new_input_names.append(new_name)

        logger.info(
            "trainStation DDS removal: promoting %s (KV=%d, dim=%s) outputs to graph inputs %s",
            tag,
            kernel_volume,
            size_dim,
            rulebook_input_name(tag, "*"),
        )

    for node in downs:
        g.node.remove(node)

    # Rewrite every consumer edge from the old intermediate name to the new input name.
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in rename:
                n.input[i] = rename[inp]

    # Drop now-orphaned value_info for the old intermediate tensors (they are graph
    # inputs now, typed via make_tensor_value_info — stale entries would duplicate them).
    stale_vi = [vi for vi in g.value_info if vi.name in rename]
    for vi in stale_vi:
        g.value_info.remove(vi)

    g.input.extend(new_inputs)

    logger.info(
        "Removed %d down-sampling %s nodes; added %d graph inputs (rulebook/*).",
        len(downs),
        _GET_INDICE_PAIRS_OP,
        len(new_inputs),
    )
    return model, new_input_names


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    parser.add_argument("--prefix", default="sp_dds")
    parser.add_argument("--no-check", action="store_true", help="skip onnx.checker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = onnx.load(args.input_onnx)
    model, names = remove_trainstation_dds(model, size_dim_prefix=args.prefix)

    if not args.no_check:
        onnx.checker.check_model(model)
        try:
            from onnx import shape_inference

            shape_inference.infer_shapes(model, strict_mode=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("shape_inference reported: %s", e)

    onnx.save(model, args.output_onnx)
    print(f"Wrote {args.output_onnx}")
    print(f"New graph inputs ({len(names)}):")
    for n in names:
        print("  " + n)


if __name__ == "__main__":
    _main()
