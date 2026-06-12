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
from typing import List, Tuple

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)

_GET_INDICE_PAIRS_OP = "GetIndicePairsImplicitGemm"


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
    new inputs (per down-sampling stage ``i``, with shared symbolic dim ``N_i``):

      * ``<prefix>_out_indices``  : ``[N_i, 4]``   (out[0])
      * ``<prefix>_pair_fwd``     : ``[KV, N_i]``  (out[1])
      * ``<prefix>_pair_mask``    : ``[N_i, 1]``   (out[2])
      * ``<prefix>_mask_argsort`` : ``[N_i]``      (out[3])

    out[4] (``num_act_out`` scalar) is dropped — it has no consumer in the graph.
    """
    g = model.graph

    downs = [n for n in g.node if n.op_type == _GET_INDICE_PAIRS_OP and _attr(n, "subm") == 0]
    if not downs:
        logger.warning("No down-sampling %s nodes found; nothing to do.", _GET_INDICE_PAIRS_OP)
        return model, []

    new_inputs: List[onnx.ValueInfoProto] = []
    new_input_names: List[str] = []

    for node in downs:
        ksize = _attr(node, "ksize")
        ksize = ksize if isinstance(ksize, list) else [ksize]
        kernel_volume = 1
        for k in ksize:
            kernel_volume *= int(k)

        tag = _stage_tag(node.name)
        size_dim = f"{size_dim_prefix}_{tag}_n"

        # out[0..3] are consumed; out[4] (scalar num_act_out) is not.
        specs = [
            (node.output[0], [size_dim, 4]),  # out_indices
            (node.output[1], [kernel_volume, size_dim]),  # pair_fwd
            (node.output[2], [size_dim, 1]),  # pair_mask
            (node.output[3], [size_dim]),  # mask_argsort
        ]
        for tensor_name, dims in specs:
            new_inputs.append(_make_input(tensor_name, dims))
            new_input_names.append(tensor_name)

        logger.info(
            "trainStation DDS removal: promoting %s (KV=%d, dim=%s) outputs to graph inputs",
            tag,
            kernel_volume,
            size_dim,
        )

    for node in downs:
        g.node.remove(node)

    g.input.extend(new_inputs)

    logger.info(
        "Removed %d down-sampling %s nodes; added %d graph inputs.",
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
