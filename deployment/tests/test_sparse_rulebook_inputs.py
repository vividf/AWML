"""Tests for the trainStation/DDS removal contract (``spconv_remove_trainstation``).

Covers the three halves of the feature that can be checked without CUDA, spconv or a checkpoint:

- the rulebook graph-input **naming contract** shared by the export transform, the deploy-config
  TensorRT-profile derivation and the runtimes,
- the down-sample **stage geometry** (the spatial-shape cascade AWML's TensorRT evaluation relies
  on, locked to the values read off the real 120m BEVFusion-L sparse ONNX),
- the ONNX **graph surgery** itself, on a synthetic graph mirroring the real node layout.

``compute_rulebook_inputs`` is not covered here: it needs a CUDA spconv build (Docker + GPU).
"""

from __future__ import annotations

import json

import onnx
import pytest
from onnx import TensorProto, helper

from deployment.config.schema import ComponentsConfig
from deployment.projects.bevfusion_l.config.component_layout import (
    add_merged_component,
    add_rulebook_input_profiles,
)
from deployment.projects.bevfusion_l.export.onnx_remove_trainstation_dds import (
    RULEBOOK_STAGES_METADATA_KEY,
    _stage_tag,
    embed_rulebook_stages_metadata,
    remove_trainstation_dds,
)
from deployment.projects.bevfusion_l.io.sparse_rulebook_inputs import (
    DOWNSAMPLE_STAGES,
    RULEBOOK_SLOTS,
    downsample_spatial_shapes,
    parse_rulebook_input_name,
    rulebook_input_name,
    split_rulebook_inputs,
)

# Spatial shapes of the 4 down-sample stages in the deployed 120m BEVFusion-L sparse ONNX
# (sparse_shape == grid_size == [1440, 1440, 41]).
_EXPECTED_120M_SPATIAL_SHAPES = [
    (1440, 1440, 41),
    (720, 720, 21),
    (360, 360, 11),
    (180, 180, 5),
]


# -----------------------------------------------------------------------------
# Naming contract
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("slot_index,slot", list(enumerate(RULEBOOK_SLOTS)))
def test_rulebook_name_round_trips(slot_index: int, slot: str) -> None:
    """Every slot name parses back to its stage tag and its fixed node-output index."""
    assert parse_rulebook_input_name(rulebook_input_name("l1", slot)) == ("l1", slot_index)


def test_rulebook_name_parse_is_prefix_agnostic() -> None:
    """The merged full-graph ONNX namespaces sparse tensors, so parsing must ignore the prefix."""
    assert parse_rulebook_input_name("sparse/rulebook/l2/pair_fwd") == ("l2", 1)


@pytest.mark.parametrize("name", ["voxels", "coors", "num_points_per_voxel", "rulebook/l1/not_a_slot"])
def test_non_rulebook_names_do_not_parse(name: str) -> None:
    """Declared model inputs (and malformed slots) must not be mistaken for rulebook inputs."""
    assert parse_rulebook_input_name(name) is None


def test_rulebook_input_name_rejects_unknown_slot() -> None:
    with pytest.raises(ValueError, match="Unknown rulebook slot"):
        rulebook_input_name("l1", "num_act_out")


def test_split_rulebook_inputs_partitions_and_preserves_order() -> None:
    """A stripped engine's inputs split into the declared voxel inputs and the promoted rulebooks."""
    names = ["voxels", "rulebook/l1/out_indices", "coors", "rulebook/out/mask_argsort"]
    assert split_rulebook_inputs(names) == (
        ["voxels", "coors"],
        ["rulebook/l1/out_indices", "rulebook/out/mask_argsort"],
    )


def test_split_rulebook_inputs_on_stock_engine() -> None:
    """An engine from an unmodified graph reports no rulebook inputs."""
    assert split_rulebook_inputs(["voxels", "coors"]) == (["voxels", "coors"], [])


# -----------------------------------------------------------------------------
# Stage geometry
# -----------------------------------------------------------------------------


def test_downsample_spatial_shapes_matches_deployed_120m_graph() -> None:
    """The cascade derived from sparse_shape reproduces the real graph's spatial shapes."""
    assert downsample_spatial_shapes([1440, 1440, 41]) == _EXPECTED_120M_SPATIAL_SHAPES


def test_downsample_spatial_shapes_requires_3d() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        downsample_spatial_shapes([1440, 1440])


def test_downsample_stage_kernel_volumes() -> None:
    """``pair_fwd``'s leading dim: 3x3x3 for the encoder stages, 1x1x3 for conv_out."""
    assert [stage.kernel_volume for stage in DOWNSAMPLE_STAGES] == [27, 27, 27, 3]


# -----------------------------------------------------------------------------
# Deploy-config TensorRT profile derivation
# -----------------------------------------------------------------------------


def _split_components_raw() -> dict:
    """Minimal split sparse+dense component config (only what the profile derivation reads)."""
    return dict(
        bevfusion_sparse=dict(
            onnx_file="bevfusion_sparse.onnx",
            engine_file="bevfusion_sparse.engine",
            io=dict(
                inputs=[dict(name="voxels"), dict(name="coors", dtype="int32")],
                outputs=[dict(name="lidar_bev")],
            ),
            tensorrt_profile=dict(
                coors=dict(min_shape=[1, 3], opt_shape=[64000, 3], max_shape=[256000, 3]),
            ),
        ),
        bevfusion_dense=dict(
            onnx_file="bevfusion_dense.onnx",
            engine_file="bevfusion_dense.engine",
            io=dict(inputs=[dict(name="lidar_bev")], outputs=[dict(name="bbox_pred")]),
        ),
    )


def test_add_rulebook_input_profiles_derives_all_slots_from_coors() -> None:
    """16 entries (4 stages x 4 slots), bounded by the voxel-count envelope declared for coors."""
    components = add_rulebook_input_profiles(ComponentsConfig.from_dict(_split_components_raw()))
    profile = components.get_component("bevfusion_sparse").tensorrt_profile

    expected_names = {rulebook_input_name(stage.tag, slot) for stage in DOWNSAMPLE_STAGES for slot in RULEBOOK_SLOTS}
    assert expected_names <= set(profile)
    assert len(profile) == len(expected_names) + 1  # + the original coors entry

    out_indices = profile[rulebook_input_name("l1", "out_indices")]
    assert (out_indices.min_shape, out_indices.opt_shape, out_indices.max_shape) == ((1, 4), (64000, 4), (256000, 4))

    pair_fwd = profile[rulebook_input_name("out", "pair_fwd")]  # conv_out: KV = 1*1*3
    assert (pair_fwd.min_shape, pair_fwd.opt_shape, pair_fwd.max_shape) == ((3, 1), (3, 64000), (3, 256000))

    mask_argsort = profile[rulebook_input_name("l3", "mask_argsort")]
    assert (mask_argsort.min_shape, mask_argsort.opt_shape, mask_argsort.max_shape) == ((1,), (64000,), (256000,))


def test_add_rulebook_input_profiles_requires_coors_profile() -> None:
    """Without a coors envelope there is nothing to bound the rulebook shapes with — fail loud."""
    raw = _split_components_raw()
    raw["bevfusion_sparse"]["tensorrt_profile"] = {}
    with pytest.raises(KeyError, match="coors"):
        add_rulebook_input_profiles(ComponentsConfig.from_dict(raw))


def test_merged_component_inherits_rulebook_profiles() -> None:
    """The merged full graph keeps the sparse inputs, so it needs the same rulebook profiles."""
    components = add_rulebook_input_profiles(ComponentsConfig.from_dict(_split_components_raw()))
    components = add_merged_component(deploy_cfg={"bevfusion_merge": dict(enabled=True)}, components_cfg=components)

    merged_profile = components.get_component("bevfusion_merged").tensorrt_profile
    assert rulebook_input_name("l1", "pair_mask") in merged_profile


# -----------------------------------------------------------------------------
# ONNX graph surgery
# -----------------------------------------------------------------------------


def _get_indice_pairs_node(name: str, indices_input: str, *, subm: int, stage_index: int) -> onnx.NodeProto:
    """A ``GetIndicePairsImplicitGemm`` node with the real graph's attributes for one stage."""
    stage = DOWNSAMPLE_STAGES[stage_index]
    return helper.make_node(
        "GetIndicePairsImplicitGemm",
        inputs=[indices_input],
        outputs=[f"{name}_output_{i}" for i in range(5)],
        name=name,
        domain="autoware",
        subm=subm,
        ksize=list(stage.ksize),
        stride=list(stage.stride) if not subm else [1, 1, 1],
        padding=list(stage.padding),
        dilation=list(stage.dilation),
        spatial_shape=list(_EXPECTED_120M_SPATIAL_SHAPES[stage_index]),
    )


def _downsample_node_name(stage_tag: str) -> str:
    """The node scope the real 120m export produces for one down-sample stage."""
    if stage_tag == "out":
        return "/pts_middle_encoder/conv_out/conv_out.0/GetIndicePairsImplicitGemm"
    index = stage_tag[1:]
    return (
        f"/pts_middle_encoder/encoder_layer{index}/encoder_layer{index}.2/"
        f"encoder_layer{index}.2.0/GetIndicePairsImplicitGemm"
    )


def _synthetic_sparse_graph() -> onnx.ModelProto:
    """A graph mirroring the real sparse encoder's down-sample chain.

    4 down-sample nodes (one per stage, each consuming the previous stage's ``out_indices``) plus
    one submanifold node that must survive, and an Identity consuming a promoted tensor so the
    consumer-edge rewrite is observable.
    """
    nodes = []
    indices = "coors"
    for stage_index, stage in enumerate(DOWNSAMPLE_STAGES):
        node_name = _downsample_node_name(stage.tag)
        nodes.append(_get_indice_pairs_node(node_name, indices, subm=0, stage_index=stage_index))
        indices = f"{node_name}_output_0"

    # A submanifold node running on the first stage's down-sampled coordinates (must be kept).
    subm_name = "/pts_middle_encoder/encoder_layer2/encoder_layer2.0/GetIndicePairsImplicitGemm"
    nodes.append(_get_indice_pairs_node(subm_name, f"{_downsample_node_name('l1')}_output_0", subm=1, stage_index=1))
    nodes.append(helper.make_node("Identity", inputs=[f"{subm_name}_output_0"], outputs=["out_coors"], name="/tail"))

    graph = helper.make_graph(
        nodes=nodes,
        name="synthetic_sparse",
        inputs=[helper.make_tensor_value_info("coors", TensorProto.INT32, ["voxels_num", 4])],
        outputs=[helper.make_tensor_value_info("out_coors", TensorProto.INT32, ["n", 4])],
    )
    # Both opsets the real sparse export declares; the plugin ops live in the ``autoware`` domain.
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 17), helper.make_operatorsetid("autoware", 1)]
    )


def test_remove_trainstation_dds_promotes_rulebooks_to_graph_inputs() -> None:
    """The 4 down-sample nodes go away; 16 typed graph inputs replace their consumed outputs."""
    model, promoted, stages_meta = remove_trainstation_dds(_synthetic_sparse_graph())

    remaining = [n for n in model.graph.node if n.op_type == "GetIndicePairsImplicitGemm"]
    assert len(remaining) == 1, "the submanifold node must be kept"

    assert promoted == [rulebook_input_name(stage.tag, slot) for stage in DOWNSAMPLE_STAGES for slot in RULEBOOK_SLOTS]
    graph_inputs = {i.name for i in model.graph.input}
    assert set(promoted) <= graph_inputs
    assert "coors" in graph_inputs

    # Every consumer edge now points at the promoted name, not the removed node's output.
    assert remaining[0].input == [rulebook_input_name("l1", "out_indices")]

    assert [meta["onnx_base"] for meta in stages_meta] == [f"rulebook/{s.tag}" for s in DOWNSAMPLE_STAGES]
    assert [tuple(meta["spatial_shape"]) for meta in stages_meta] == _EXPECTED_120M_SPATIAL_SHAPES


def test_remove_trainstation_dds_produces_a_valid_graph() -> None:
    """The surgery must leave a graph the ONNX checker accepts (what the export transform asserts)."""
    model, _, _ = remove_trainstation_dds(_synthetic_sparse_graph())
    onnx.checker.check_model(model)


def test_promoted_input_shapes() -> None:
    """Promoted inputs are INT32 with the slot's rank and the stage's kernel volume on pair_fwd."""
    model, _, _ = remove_trainstation_dds(_synthetic_sparse_graph())
    by_name = {i.name: i for i in model.graph.input}

    def dims(name: str):
        return [d.dim_param or d.dim_value for d in by_name[name].type.tensor_type.shape.dim]

    assert by_name[rulebook_input_name("l1", "out_indices")].type.tensor_type.elem_type == TensorProto.INT32
    assert dims(rulebook_input_name("l1", "out_indices")) == ["sp_dds_l1_n", 4]
    assert dims(rulebook_input_name("l1", "pair_fwd")) == [27, "sp_dds_l1_n"]
    assert dims(rulebook_input_name("out", "pair_fwd")) == [3, "sp_dds_out_n"]
    assert dims(rulebook_input_name("l2", "pair_mask")) == ["sp_dds_l2_n", 1]
    assert dims(rulebook_input_name("l3", "mask_argsort")) == ["sp_dds_l3_n"]


def test_remove_trainstation_dds_is_a_noop_on_a_stripped_graph() -> None:
    """Re-running the transform (or running it on a submanifold-only graph) changes nothing."""
    model, _, _ = remove_trainstation_dds(_synthetic_sparse_graph())
    node_count = len(model.graph.node)

    model, promoted, stages_meta = remove_trainstation_dds(model)
    assert (promoted, stages_meta) == ([], [])
    assert len(model.graph.node) == node_count


def test_embed_rulebook_stages_metadata_is_idempotent() -> None:
    """The stage geometry lands in metadata_props as JSON, replaced (not appended) on re-embed."""
    model, _, stages_meta = remove_trainstation_dds(_synthetic_sparse_graph())
    model = embed_rulebook_stages_metadata(model, stages_meta)
    model = embed_rulebook_stages_metadata(model, stages_meta)

    entries = [p for p in model.metadata_props if p.key == RULEBOOK_STAGES_METADATA_KEY]
    assert len(entries) == 1
    assert json.loads(entries[0].value) == stages_meta


def test_embed_rulebook_stages_metadata_skips_empty() -> None:
    """Nothing removed means nothing to describe — no metadata entry is added."""
    model = embed_rulebook_stages_metadata(_synthetic_sparse_graph(), [])
    assert not [p for p in model.metadata_props if p.key == RULEBOOK_STAGES_METADATA_KEY]


@pytest.mark.parametrize(
    "node_name,expected_tag",
    [
        ("/pts_middle_encoder/encoder_layer1/encoder_layer1.2/encoder_layer1.2.0/GetIndicePairsImplicitGemm", "l1"),
        ("/pts_middle_encoder/encoder_layer3/encoder_layer3.2/encoder_layer3.2.0/GetIndicePairsImplicitGemm", "l3"),
        ("/pts_middle_encoder/conv_out/conv_out.0/GetIndicePairsImplicitGemm", "out"),
        # A container scope (``encoder_layers``) carries no index, so it must not be read as one:
        # collapsing several stages onto one tag would declare the same graph input twice.
        (
            "/pts_middle_encoder/encoder_layers/GetIndicePairsImplicitGemm",
            "pts_middle_encoder_encoder_layers_GetIndicePairsImplicitGemm",
        ),
    ],
)
def test_stage_tag_derivation(node_name: str, expected_tag: str) -> None:
    """Stage tags come from the node scope; only a token ending in the layer index counts."""
    assert _stage_tag(node_name) == expected_tag


def test_colliding_stage_tags_fail_loud() -> None:
    """Two stages resolving to one tag is rejected up front, not as a downstream SSA violation."""
    nodes = [
        _get_indice_pairs_node(
            f"/pts_middle_encoder/conv_out/dup{i}/GetIndicePairsImplicitGemm", "coors", subm=0, stage_index=3
        )
        for i in range(2)
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="colliding",
        inputs=[helper.make_tensor_value_info("coors", TensorProto.INT32, ["voxels_num", 4])],
        outputs=[helper.make_tensor_value_info(f"{nodes[1].name}_output_0", TensorProto.INT32, ["n", 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    with pytest.raises(ValueError, match="already"):
        remove_trainstation_dds(model)
