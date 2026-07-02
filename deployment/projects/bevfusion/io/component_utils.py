"""Helpers for BEVFusion deploy ``components`` layout."""

from __future__ import annotations

from typing import Any, Mapping

from deployment.config.schema import ComponentsConfig


def is_split_bevfusion_components(components_cfg: ComponentsConfig) -> bool:
    """True when deploy config uses sparse + dense ONNX/TRT (route 1), not a single main_body."""
    names = set(components_cfg.component_names())
    return "bevfusion_sparse" in names and "bevfusion_dense" in names


def should_merge_split_bevfusion(deploy_cfg: Mapping[str, Any]) -> bool:
    """Return True when deploy config requests split->single export/eval merge."""
    merge_raw = deploy_cfg.get("bevfusion_merge", deploy_cfg.get("merge_bevfusion", deploy_cfg.get("merge", False)))
    if isinstance(merge_raw, Mapping):
        return bool(merge_raw.get("enabled", False))
    return bool(merge_raw)


def has_component(components_cfg: ComponentsConfig, component_name: str) -> bool:
    """Return True if the component exists."""
    try:
        components_cfg.get_component(component_name)
        return True
    except KeyError:
        return False


def maybe_add_merged_main_body_component(
    *,
    deploy_cfg: Mapping[str, Any],
    components_cfg: ComponentsConfig,
) -> ComponentsConfig:
    """Optionally add merged main_body component while keeping split components.

    When ``bevfusion_merge`` is enabled and components are split, this function adds
    ``bevfusion_main_body`` by reusing:
    - split sparse input schema / TensorRT profile
    - split dense output schema
    """
    if not is_split_bevfusion_components(components_cfg):
        return components_cfg
    if not should_merge_split_bevfusion(deploy_cfg):
        return components_cfg
    if has_component(components_cfg, "bevfusion_main_body"):
        return components_cfg

    sparse_cfg = components_cfg.get_component("bevfusion_sparse")
    dense_cfg = components_cfg.get_component("bevfusion_dense")

    merge_raw = deploy_cfg.get("bevfusion_merge", deploy_cfg.get("merge_bevfusion", deploy_cfg.get("merge", {})))
    merge_cfg = merge_raw if isinstance(merge_raw, Mapping) else {}
    onnx_file = str(merge_cfg.get("onnx_file", "bevfusion_lidar.onnx"))
    engine_file = str(merge_cfg.get("engine_file", "bevfusion_lidar.engine"))

    merged_main_body = {
        "bevfusion_main_body": {
            "onnx_file": onnx_file,
            "engine_file": engine_file,
            "io": {
                "inputs": [{"name": inp.name, "dtype": inp.dtype} for inp in sparse_cfg.io.inputs],
                "outputs": [{"name": out.name, "dtype": out.dtype} for out in dense_cfg.io.outputs],
                "dynamic_axes": dict(sparse_cfg.io.dynamic_axes),
            },
            "tensorrt_profile": {
                name: {
                    "min_shape": list(profile.min_shape),
                    "opt_shape": list(profile.opt_shape),
                    "max_shape": list(profile.max_shape),
                }
                for name, profile in sparse_cfg.tensorrt_profile.items()
            },
        }
    }

    raw_components = {}
    for name, comp in components_cfg.items():
        raw_components[name] = {
            "onnx_file": comp.onnx_file,
            "engine_file": comp.engine_file,
            "io": {
                "inputs": [{"name": inp.name, "dtype": inp.dtype} for inp in comp.io.inputs],
                "outputs": [{"name": out.name, "dtype": out.dtype} for out in comp.io.outputs],
                "dynamic_axes": dict(comp.io.dynamic_axes),
            },
            "tensorrt_profile": {
                k: {
                    "min_shape": list(v.min_shape),
                    "opt_shape": list(v.opt_shape),
                    "max_shape": list(v.max_shape),
                }
                for k, v in comp.tensorrt_profile.items()
            },
        }
    raw_components.update(merged_main_body)
    return ComponentsConfig.from_dict(raw_components)
