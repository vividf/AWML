"""
CenterPoint-specific deploy config: builds generic ComponentsCfg + OnnxConfig
and validates required component names (voxel_encoder, backbone_head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from deployment.core.config.base_config import ComponentsCfg, OnnxConfig

# CenterPoint required component names
REQUIRED_COMPONENTS = ("voxel_encoder", "backbone_head")


@dataclass(frozen=True)
class CenterPointDeployConfig:
    """Typed CenterPoint deploy config: components + onnx_config. Build once, use everywhere."""

    components: ComponentsCfg
    onnx_config: OnnxConfig

    @classmethod
    def from_dict(cls, deploy_cfg: Mapping[str, Any]) -> CenterPointDeployConfig:
        """Build from full deploy_cfg. Validates CenterPoint required components."""
        if "components" not in deploy_cfg:
            raise KeyError("deploy_cfg must define 'components' for CenterPoint deployment.")
        if "onnx_config" not in deploy_cfg:
            raise KeyError("deploy_cfg must define 'onnx_config' for CenterPoint export.")
        components = ComponentsCfg.from_dict(deploy_cfg["components"])
        for name in REQUIRED_COMPONENTS:
            components.get_component(name)  # raise if missing
        return cls(
            components=components,
            onnx_config=OnnxConfig.from_dict(deploy_cfg["onnx_config"]),
        )
