"""Helpers for BEVFusion deploy ``components`` layout."""

from __future__ import annotations

from deployment.configs.schema import ComponentsConfig


def is_split_bevfusion_components(components_cfg: ComponentsConfig) -> bool:
    """True when deploy config uses sparse + dense ONNX/TRT (route 1), not a single main_body."""
    names = set(components_cfg.component_names())
    return "bevfusion_sparse" in names and "bevfusion_dense" in names
