"""BEVFusion deploy-config component-layout helpers.

BEVFusion can be deployed in two component layouts:

- **split**: separate ``bevfusion_sparse`` (spconv) + ``bevfusion_dense`` ONNX/TensorRT graphs.
- **merged**: a single ``bevfusion_merged`` graph (sparse inputs → dense outputs).

These helpers query the layout (:func:`is_split_components`, :func:`has_component`) and derive the
component config that follows mechanically from a deploy config's declared intent:

- ``bevfusion_merge`` -> the merged ``bevfusion_merged`` component, built from the split pair
  (:func:`merge_requested`, :func:`add_merged_component`).
- ``spconv_remove_trainstation`` -> TensorRT profile entries for the rulebook graph inputs the
  export transform promotes (:func:`add_rulebook_input_profiles`).

Deriving these here keeps deploy configs declaring *what* they want instead of restating the
mechanical consequences. They operate purely on deploy-config structures, so they live beside the
deployment config rather than in ``io``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Tuple

from deployment.config.schema import ComponentCfg, ComponentIO, ComponentsConfig
from deployment.export.exporters.configs import TensorRTProfileConfig
from deployment.projects.bevfusion_l.io.sparse_rulebook_inputs import DOWNSAMPLE_STAGES, rulebook_input_name
from deployment.projects.bevfusion_l.io.voxel_inputs import COORS_INPUT


def is_split_components(components_cfg: ComponentsConfig) -> bool:
    """True when the deploy config uses split sparse + dense graphs (not a single merged graph)."""
    names = set(components_cfg.component_names())
    return "bevfusion_sparse" in names and "bevfusion_dense" in names


def has_component(components_cfg: ComponentsConfig, component_name: str) -> bool:
    """Return True if the component exists in the layout."""
    try:
        components_cfg.get_component(component_name)
        return True
    except KeyError:
        return False


def merge_requested(deploy_cfg: Mapping[str, Any]) -> bool:
    """Return True when the deploy config requests the split->merged graph merge.

    The single canonical key is ``bevfusion_merge`` (a dict with ``enabled`` / ``onnx_file`` /
    ``engine_file``, or a plain bool).
    """
    merge_raw = deploy_cfg.get("bevfusion_merge", False)
    if isinstance(merge_raw, Mapping):
        return bool(merge_raw.get("enabled", False))
    return bool(merge_raw)


def add_merged_component(
    *,
    deploy_cfg: Mapping[str, Any],
    components_cfg: ComponentsConfig,
) -> ComponentsConfig:
    """Add a merged ``bevfusion_merged`` component while keeping the split components.

    When ``bevfusion_merge`` is enabled and the layout is split, this derives
    ``bevfusion_merged`` by reusing:

    - the split sparse input schema / TensorRT profile, and
    - the split dense output schema.

    If the layout is not split, merge is not requested, or ``bevfusion_merged`` already exists,
    the config is returned unchanged.
    """
    if not is_split_components(components_cfg):
        return components_cfg
    if not merge_requested(deploy_cfg):
        return components_cfg
    if has_component(components_cfg, "bevfusion_merged"):
        return components_cfg

    sparse_cfg = components_cfg.get_component("bevfusion_sparse")
    dense_cfg = components_cfg.get_component("bevfusion_dense")

    merge_raw = deploy_cfg.get("bevfusion_merge", {})
    merge_cfg = merge_raw if isinstance(merge_raw, Mapping) else {}
    onnx_file = str(merge_cfg.get("onnx_file", "bevfusion_lidar.onnx"))
    engine_file = str(merge_cfg.get("engine_file", "bevfusion_lidar.engine"))

    # The merged graph reuses the split sparse inputs / TensorRT profile and the split dense
    # outputs, so build it directly from the already-typed components (no raw-dict round-trip).
    merged = ComponentCfg(
        name="bevfusion_merged",
        onnx_file=onnx_file,
        engine_file=engine_file,
        io=ComponentIO(
            inputs=list(sparse_cfg.io.inputs),
            outputs=list(dense_cfg.io.outputs),
            dynamic_axes=dict(sparse_cfg.io.dynamic_axes),
        ),
        tensorrt_profile=dict(sparse_cfg.tensorrt_profile),
    )
    return components_cfg.with_component(merged)


def add_rulebook_input_profiles(components_cfg: ComponentsConfig) -> ComponentsConfig:
    """Add ``bevfusion_sparse`` TensorRT profile entries for the promoted rulebook graph inputs.

    With ``spconv_remove_trainstation``, the export transform replaces the 4 down-sampling
    ``GetIndicePairsImplicitGemm`` nodes with 16 rulebook graph inputs *after* ``torch.onnx.export``
    (see ``export/onnx_remove_trainstation_dds.py``). They are therefore absent from the component's
    declared ``io.inputs``, but TensorRT still needs a profile for each of them — so the entries are
    derived here rather than restated in every deploy config.

    Each stage's active-voxel count ``N`` is bounded by the voxel-count envelope the config already
    declares for ``coors`` (down-sampling can only reduce the count), and ``pair_fwd``'s leading dim
    is the stage's kernel volume.

    Args:
        components_cfg: Layout containing ``bevfusion_sparse`` with a ``coors`` profile.

    Returns:
        A new ``ComponentsConfig`` whose ``bevfusion_sparse`` profile also covers the rulebook
        inputs. Call before :func:`add_merged_component` so the merged graph inherits them.

    Raises:
        KeyError: If ``bevfusion_sparse`` has no ``coors`` TensorRT profile to take ``N`` from.
        ValueError: If that profile is missing any of its min/opt/max shapes.
    """
    sparse_cfg = components_cfg.get_component("bevfusion_sparse")
    coors_profile = sparse_cfg.tensorrt_profile.get(COORS_INPUT)
    if coors_profile is None:
        raise KeyError(
            f"spconv_remove_trainstation requires a bevfusion_sparse tensorrt_profile['{COORS_INPUT}'] "
            "to bound the rulebook inputs' active-voxel count."
        )
    shapes = (coors_profile.min_shape, coors_profile.opt_shape, coors_profile.max_shape)
    if not all(shapes):
        raise ValueError(
            f"bevfusion_sparse tensorrt_profile['{COORS_INPUT}'] must declare min/opt/max shapes for "
            "spconv_remove_trainstation to bound the rulebook inputs."
        )
    voxel_counts = tuple(shape[0] for shape in shapes)

    def _profile(shape_of_n: Callable[[int], Tuple[int, ...]]) -> TensorRTProfileConfig:
        """Profile for a rulebook tensor whose shape follows from the active-voxel count ``n``."""
        min_shape, opt_shape, max_shape = (shape_of_n(n) for n in voxel_counts)
        return TensorRTProfileConfig(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape)

    profile: Dict[str, TensorRTProfileConfig] = dict(sparse_cfg.tensorrt_profile)
    for stage in DOWNSAMPLE_STAGES:
        kernel_volume = stage.kernel_volume
        profile[rulebook_input_name(stage.tag, "out_indices")] = _profile(lambda n: (n, 4))
        profile[rulebook_input_name(stage.tag, "pair_fwd")] = _profile(lambda n: (kernel_volume, n))
        profile[rulebook_input_name(stage.tag, "pair_mask")] = _profile(lambda n: (n, 1))
        profile[rulebook_input_name(stage.tag, "mask_argsort")] = _profile(lambda n: (n,))

    return components_cfg.with_component(replace(sparse_cfg, tensorrt_profile=profile))
