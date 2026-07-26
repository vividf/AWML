"""Rulebook graph-input contract for the trainStation-stripped sparse graph.

Background
----------
The sparse middle-encoder ONNX contains 21 ``GetIndicePairsImplicitGemm`` nodes. The 4
**down-sampling** ones (``subm=0``) declare a *size tensor* for their output extent (the active
voxel count after stride-2 pooling is data dependent). TensorRT must read that count back to the
host (``DeviceToShapeHostCopy``) before it can size downstream tensors, which splits the engine
into ``[trainStationN]`` segments and stalls the GPU. The 17 submanifold nodes (``subm=1``) keep
the active-voxel set unchanged, so they declare no size tensor and cause no such split.

The rulebook (``out_indices`` / ``pair_fwd`` / ``pair_mask`` / ``mask_argsort``) depends only on
voxel *geometry*, never on feature values, so it can be computed before the engine runs and fed in
as ordinary graph inputs. ``deployment.projects.bevfusion_l.export.onnx_remove_trainstation_dds``
performs the export-side half (delete the 4 nodes, promote their outputs to graph inputs); this
module owns the other half:

- The **naming contract** (:func:`rulebook_input_name` / :func:`parse_rulebook_input_name` /
  :func:`split_rulebook_inputs`) — the single source of truth for the promoted input names, shared
  by the export transform, the deploy-config TensorRT-profile derivation, and the runtimes.
- The **stage geometry** (:data:`DOWNSAMPLE_STAGES` + :func:`downsample_spatial_shapes`) of the
  BEVFusion sparse encoder's 4 down-sampling convolutions.
- The **runtime precompute** (:func:`compute_rulebook_inputs`) that produces the arrays to bind.

Two runtimes consume the promoted inputs, and they get the stage geometry from different places:

- **autoware_bevfusion (on board)** reads it from the ONNX ``metadata_props["rulebook_stages"]``
  entry the export transform embeds, so that graph is self-describing for *any* encoder.
- **AWML TensorRT evaluation** only ever sees the built ``.engine``, which carries no such
  metadata, so it uses :data:`DOWNSAMPLE_STAGES` below — a mirror of the BEVFusion sparse encoder's
  architecture, resolved against the model's own ``sparse_shape`` so it holds for any grid size.
  The export transform compares the graph against this table and warns on a mismatch.

Note on latency measurement: with the rulebooks precomputed here, that work moves *out* of the
engine and is therefore excluded from the ``sparse_ms`` CUDA-event window. A trainStation-stripped
engine is only comparable to a baseline engine if the precompute cost is accounted for separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from deployment.projects.bevfusion_l.io.voxel_inputs import graph_input_zyx_to_model_indices_xyz

logger = logging.getLogger(__name__)

# Graph-input naming contract: ``[<prefix>/]rulebook/<stage_tag>/<slot>``.
#
# The leading ``rulebook/<stage>/`` segment makes Netron group all 16 promoted inputs into one
# collapsible ``rulebook`` box instead of showing 16 dangling tensors named after a node that is no
# longer in the graph. Parsing is prefix-agnostic because ``onnx.compose`` prepends ``sparse/`` to
# every sparse tensor when the split graphs are merged into the single full-graph ONNX.
RULEBOOK_NAMESPACE = "rulebook"

# Slot order is the fixed output order of ``GetIndicePairsImplicitGemm`` (see
# ``projects/SparseConvolution/sparse_functional.py``):
#   0=out_indices, 1=pair_fwd, 2=pair_mask, 3=mask_argsort   (out[4]=num_act_out has no consumer)
RULEBOOK_SLOTS: Tuple[str, ...] = ("out_indices", "pair_fwd", "pair_mask", "mask_argsort")


@dataclass(frozen=True)
class RulebookStage:
    """Kernel geometry of one down-sampling sparse convolution.

    ``tag`` is the short, stable name used in the promoted graph-input names (``rulebook/l1/...``).
    Spatial shapes are deliberately absent: they follow from the model's ``sparse_shape`` via
    :func:`downsample_spatial_shapes`, so this table stays independent of the detection range.
    """

    tag: str
    ksize: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    padding: Tuple[int, int, int]
    dilation: Tuple[int, int, int] = (1, 1, 1)

    @property
    def kernel_volume(self) -> int:
        """Product of the kernel dims — the leading (KV) dim of ``pair_fwd``."""
        return int(np.prod(self.ksize))

    def output_spatial_shape(self, spatial_shape: Sequence[int]) -> Tuple[int, ...]:
        """Spatial shape after this convolution (standard conv output-extent arithmetic)."""
        return tuple(
            (int(dim) + 2 * p - d * (k - 1) - 1) // s + 1
            for dim, k, s, p, d in zip(spatial_shape, self.ksize, self.stride, self.padding, self.dilation)
        )


# The BEVFusion sparse encoder's down-sampling convolutions, in graph order: the first conv of
# encoder stages 2-4 (stride 2 in x/y/z) plus ``conv_out`` (stride 2 in z only). Axis order is the
# encoder's (H, W, D) == (x, y, z), matching ``BEVFusionSparseEncoder.sparse_shape``.
DOWNSAMPLE_STAGES: Tuple[RulebookStage, ...] = (
    RulebookStage(tag="l1", ksize=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
    RulebookStage(tag="l2", ksize=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
    RulebookStage(tag="l3", ksize=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 0)),
    RulebookStage(tag="out", ksize=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0)),
)


def rulebook_input_base(stage_tag: str) -> str:
    """Common prefix of one stage's rulebook inputs, e.g. ``rulebook/l1``.

    This is the ``onnx_base`` the Autoware runtime's ``SparseDownsampleStage`` binds against.
    """
    return f"{RULEBOOK_NAMESPACE}/{stage_tag}"


def rulebook_input_name(stage_tag: str, slot: str) -> str:
    """Canonical graph-input name for one rulebook tensor, e.g. ``rulebook/l1/pair_fwd``."""
    if slot not in RULEBOOK_SLOTS:
        raise ValueError(f"Unknown rulebook slot {slot!r}; expected one of {list(RULEBOOK_SLOTS)}")
    return f"{rulebook_input_base(stage_tag)}/{slot}"


def parse_rulebook_input_name(name: str) -> Optional[Tuple[str, int]]:
    """``[…/]rulebook/<tag>/<slot>`` -> ``(tag, slot_index)``, or ``None`` if not a rulebook input.

    Prefix-agnostic: the merged full-graph ONNX may carry a ``sparse/`` namespace prefix.
    """
    parts = name.split("/")
    try:
        i = parts.index(RULEBOOK_NAMESPACE)
        tag, slot = parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        return None
    if slot not in RULEBOOK_SLOTS:
        return None
    return tag, RULEBOOK_SLOTS.index(slot)


def split_rulebook_inputs(input_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Partition engine input names into ``(regular_inputs, rulebook_inputs)``, preserving order.

    Lets a runtime bind the model's declared inputs through their normal (strict) mapping and treat
    the promoted rulebook inputs separately, without either path having to tolerate the other's
    names. An engine built from an unmodified graph yields an empty rulebook list.
    """
    regular: List[str] = []
    rulebook: List[str] = []
    for name in input_names:
        (rulebook if parse_rulebook_input_name(name) is not None else regular).append(name)
    return regular, rulebook


def downsample_spatial_shapes(sparse_shape: Sequence[int]) -> List[Tuple[int, ...]]:
    """Input spatial shape of each stage in :data:`DOWNSAMPLE_STAGES`, cascaded from ``sparse_shape``.

    Args:
        sparse_shape: The encoder's ``sparse_shape`` (== the first down-sample stage's input extent).

    Returns:
        One spatial shape per stage, in stage order (the first entry is ``sparse_shape`` itself).

    Raises:
        ValueError: If ``sparse_shape`` is not 3-dimensional.
    """
    if len(sparse_shape) != 3:
        raise ValueError(f"sparse_shape must be 3D (x, y, z), got {list(sparse_shape)}")
    shapes: List[Tuple[int, ...]] = []
    current = tuple(int(dim) for dim in sparse_shape)
    for stage in DOWNSAMPLE_STAGES:
        shapes.append(current)
        current = stage.output_spatial_shape(current)
    return shapes


def sparse_shape_from_model(model: torch.nn.Module) -> List[int]:
    """Read ``sparse_shape`` off the model's sparse middle encoder, failing loud if absent."""
    encoder = getattr(model, "pts_middle_encoder", None)
    sparse_shape = getattr(encoder, "sparse_shape", None)
    if sparse_shape is None:
        raise AttributeError(
            "Cannot resolve pts_middle_encoder.sparse_shape from the reference model; it is required "
            "to precompute the rulebook inputs of a trainStation-stripped sparse engine."
        )
    return [int(dim) for dim in sparse_shape]


def compute_rulebook_inputs(
    *,
    coors_zyx: np.ndarray,
    rulebook_names: Sequence[str],
    sparse_shape: Sequence[int],
    device: Union[str, torch.device] = "cuda:0",
) -> Dict[str, np.ndarray]:
    """Compute the rulebook arrays a trainStation-stripped sparse engine expects as graph inputs.

    Reproduces exactly what the removed in-graph nodes used to compute, by cascading
    ``GetIndicePairsImplicitGemm`` over the down-sample stages: the submanifold layers between them
    do not change coordinates, so each stage's input coordinates are the previous stage's
    ``out_indices``. ``do_sort`` is the process-global set from ``deploy_cfg.spconv_do_sort`` (see
    ``BEVFusionDeploymentRunner``), so the precomputed rulebooks match the deployed engine's plugin
    configuration by construction.

    This is also the Python reference for the autoware_bevfusion CUDA implementation.

    Args:
        coors_zyx: ``[N, 3]`` int32 graph-input coordinates ``[z, y, x]`` — the same array bound to
            the engine's ``coors`` input.
        rulebook_names: The engine's rulebook input names (from :func:`split_rulebook_inputs`).
        sparse_shape: The encoder's ``sparse_shape`` (see :func:`sparse_shape_from_model`).
        device: CUDA device to run the rulebook generation on.

    Returns:
        ``{input_name: int32 ndarray}``, one entry per name in ``rulebook_names``.

    Raises:
        KeyError: If a requested name refers to a stage this module's table does not describe —
            the engine was built from an encoder whose down-sample layout differs from
            :data:`DOWNSAMPLE_STAGES`.
    """
    # Imported lazily: importing projects.SparseConvolution force-registers the deploy-only
    # sparse conv classes into MODELS, which must not happen as a side effect of importing this
    # module (the same reason BEVFusionDeploymentRunner defers its set_do_sort import).
    from spconv.core import ConvAlgo
    from spconv.tools import CUDAKernelTimer

    from projects.SparseConvolution.sparse_functional import GetIndicePairsImplicitGemm

    if not rulebook_names:
        return {}

    spatial_shapes = downsample_spatial_shapes(sparse_shape)

    # Same flip the exported sparse wrapper applies before pts_middle_encoder (see io/voxel_inputs.py),
    # then prepend the batch column: [N, 4] (batch, x, y, z) is what GetIndicePairsImplicitGemm consumes.
    coors = torch.as_tensor(coors_zyx, dtype=torch.int32, device=device)
    coords_xyz = graph_input_zyx_to_model_indices_xyz(coors)
    batch = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=device)
    current_indices = torch.cat([batch, coords_xyz], dim=1).contiguous()

    timer = CUDAKernelTimer(False)
    by_tag: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for stage, spatial_shape in zip(DOWNSAMPLE_STAGES, spatial_shapes):
            out_indices, pair_fwd, pair_mask, mask_argsort, _ = GetIndicePairsImplicitGemm.apply(
                current_indices,
                1,  # batch_size
                list(spatial_shape),
                ConvAlgo(1),  # MaskImplicitGemm
                list(stage.ksize),
                list(stage.stride),
                list(stage.padding),
                list(stage.dilation),
                [0, 0, 0],  # out_padding
                False,  # subm
                False,  # transpose
                False,  # is_train
                None,  # alloc
                timer,
            )
            num_active = int(out_indices.shape[0])
            by_tag[stage.tag] = [
                out_indices.to(torch.int32).contiguous(),  # slot 0: [n, 4]
                pair_fwd.to(torch.int32).contiguous(),  # slot 1: [KV, n]
                pair_mask.reshape(num_active, 1).to(torch.int32).contiguous(),  # slot 2: [n, 1]
                mask_argsort.reshape(num_active).to(torch.int32).contiguous(),  # slot 3: [n]
            ]
            current_indices = by_tag[stage.tag][0]

    logger.debug(
        "Precomputed rulebooks (active voxels per stage): %s",
        {tag: int(tensors[0].shape[0]) for tag, tensors in by_tag.items()},
    )

    arrays: Dict[str, np.ndarray] = {}
    for name in rulebook_names:
        parsed = parse_rulebook_input_name(name)
        if parsed is None:
            raise KeyError(f"{name!r} is not a rulebook graph input")
        tag, slot_index = parsed
        if tag not in by_tag:
            raise KeyError(
                f"Rulebook input {name!r} refers to down-sample stage {tag!r}, which is not in "
                f"DOWNSAMPLE_STAGES ({[s.tag for s in DOWNSAMPLE_STAGES]}). The engine was built "
                "from a sparse encoder this module does not describe."
            )
        arrays[name] = by_tag[tag][slot_index].detach().cpu().numpy()
    return arrays
