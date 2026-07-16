"""YOLOX-specific deployment config.

Extends :class:`~deployment.config.base.BaseDeploymentConfig` to enforce the YOLOX component
layout. YOLOX exports as a single whole-model component, so the only project-specific rule is that
the deploy config declares a ``model`` component; everything else (classes, thresholds, input size)
is read from the model config at runtime, so one deploy config shape serves any YOLOX variant.
"""

from __future__ import annotations

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig


class YOLOXDeploymentConfig(BaseDeploymentConfig):
    """Deployment config for YOLOX (single ``model`` component)."""

    #: The single component YOLOX exports its whole-model ONNX/TensorRT under.
    _REQUIRED_COMPONENTS = ("model",)

    def __init__(self, deploy_cfg: Config) -> None:
        super().__init__(deploy_cfg)
        self._validate_components()

    def _validate_components(self) -> None:
        """Fail early if the deploy config is missing the required ``model`` component."""
        for component_name in self._REQUIRED_COMPONENTS:
            self.components_cfg.get_component(component_name)
