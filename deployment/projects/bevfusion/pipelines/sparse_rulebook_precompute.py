"""Runtime precompute of the 4 down-sample spconv rulebooks for the
trainStation-stripped sparse engine (Route A).

When ``spconv_remove_trainstation`` is enabled, the sparse ONNX has its 4
down-sample ``GetIndicePairsImplicitGemm`` nodes replaced by graph inputs
(see ``export/sparse_trainstation_transform.py``). At inference time the runtime
must supply those rulebooks. This module reproduces what the in-graph plugin used
to compute, by cascading ``GetIndicePairsImplicitGemm`` over the 4 down-sample
stages (submanifold layers in between do not change coordinates, so each
down-sample's input coords are the previous down-sample's ``out_indices``).

This is the Python reference for the autoware_bevfusion CUDA runtime (Slice 2).
Validated to produce ``lidar_bev`` numerically equal to the baseline engine.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

# Per down-sample stage: (short tag matching the rulebook graph-input names, kernel attrs).
# The export side (export/sparse_trainstation_transform.py) names the promoted inputs
# ``rulebook/<tag>/<slot>`` with tag in {l1,l2,l3,out}; we match on that tag.
# spatial_shape cascade is model-specific (sparse_shape [1440,1440,41], stride-2 xy down-samples
# + conv_out stride (1,1,2)); matches the BEVFusion-L 120m config used for the trainStation deploy.
_DOWNSAMPLE_STAGES = [
    dict(
        tag="l1",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        spatial=[1440, 1440, 41],
    ),
    dict(
        tag="l2",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        spatial=[720, 720, 21],
    ),
    dict(
        tag="l3",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 0],
        dilation=[1, 1, 1],
        spatial=[360, 360, 11],
    ),
    dict(
        tag="out",
        ksize=[1, 1, 3],
        stride=[1, 1, 2],
        padding=[0, 0, 0],
        dilation=[1, 1, 1],
        spatial=[180, 180, 5],
    ),
]

# Graph-input naming contract shared with export/sparse_trainstation_transform.py:
# ``[<prefix>/]rulebook/<tag>/<slot>`` (a ``sparse/`` prefix may be present in the
# merged single-file ONNX; matching is prefix-agnostic). Slot order = the fixed
# GetIndicePairsImplicitGemm output order.
_RULEBOOK_NAMESPACE = "rulebook"
_SLOTS = ("out_indices", "pair_fwd", "pair_mask", "mask_argsort")


def _parse_rulebook_name(name: str):
    """``…/rulebook/<tag>/<slot>`` -> (tag, slot_index), or None if not a rulebook input."""
    parts = name.split("/")
    try:
        i = parts.index(_RULEBOOK_NAMESPACE)
        tag, slot = parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        return None
    if slot not in _SLOTS:
        return None
    return tag, _SLOTS.index(slot)


def has_rulebook_inputs(input_names: List[str]) -> bool:
    return any(_parse_rulebook_name(n) is not None for n in input_names)


def _zyx_to_xyz(coors: torch.Tensor) -> torch.Tensor:
    return coors.flip(dims=[-1]).contiguous()


def compute_rulebook_inputs(
    coors_zyx: np.ndarray, input_names: List[str], device: str = "cuda:0"
) -> Dict[str, np.ndarray]:
    """Compute the rulebook graph-input arrays for a trainStation-stripped sparse engine.

    Args:
        coors_zyx: ``[N,3]`` int32 graph-input coordinates ``[z,y,x]`` (same array fed to
            the engine ``coors`` input).
        input_names: engine input tensor names (only those containing the rulebook marker
            are produced).
    Returns:
        dict mapping each rulebook input name -> int32 numpy array, ready for binding.
    """
    from spconv.core import ConvAlgo
    from spconv.tools import CUDAKernelTimer

    from projects.SparseConvolution.sparse_functional import GetIndicePairsImplicitGemm

    rb_names = [n for n in input_names if _parse_rulebook_name(n) is not None]
    if not rb_names:
        return {}

    coors = torch.as_tensor(coors_zyx, dtype=torch.int32, device=device)
    coords_xyz = _zyx_to_xyz(coors)
    batch = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=device)
    cur = torch.cat([batch, coords_xyz], dim=1).contiguous().to(torch.int32)  # [N,4] (b,x,y,z)

    timer = CUDAKernelTimer(False)
    by_tag: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for d in _DOWNSAMPLE_STAGES:
            out_inds, pair_fwd, pair_mask, mask_argsort, _ = GetIndicePairsImplicitGemm.apply(
                cur,
                1,
                d["spatial"],
                ConvAlgo(1),
                d["ksize"],
                d["stride"],
                d["padding"],
                d["dilation"],
                [0, 0, 0],
                False,
                False,
                False,
                None,
                timer,
            )
            n = int(out_inds.shape[0])
            by_tag[d["tag"]] = [
                out_inds.to(torch.int32).contiguous(),  # output_0 [n,4]
                pair_fwd.to(torch.int32).contiguous(),  # output_1 [KV,n]
                pair_mask.reshape(n, 1).to(torch.int32).contiguous(),  # output_2 [n,1]
                mask_argsort.reshape(n).to(torch.int32).contiguous(),  # output_3 [n]
            ]
            cur = out_inds.to(torch.int32).contiguous()

    out: Dict[str, np.ndarray] = {}
    for name in rb_names:
        tag, slot_idx = _parse_rulebook_name(name)
        if tag not in by_tag:
            raise KeyError(f"Could not match rulebook input to a down-sample stage: {name}")
        out[name] = by_tag[tag][slot_idx].detach().cpu().numpy()
    return out
