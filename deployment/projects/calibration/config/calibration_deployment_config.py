"""Calibration-classifier-specific deployment config.

Extends :class:`~deployment.config.base.BaseDeploymentConfig` with the class names (the classifier
model config records ``num_classes`` but not label strings) and enforces the single-component
layout. ``class_names`` is the label-index order used by the metrics interface and result labelling
(index 0 miscalibrated, 1 calibrated).
"""

from __future__ import annotations

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig


class CalibrationDeploymentConfig(BaseDeploymentConfig):
    """Deployment config for the calibration classifier (single ``model`` component)."""

    #: The single component the classifier exports its whole-model ONNX/TensorRT under.
    _REQUIRED_COMPONENTS = ("model",)
    #: Default label-index order when the deploy config does not set ``class_names``.
    _DEFAULT_CLASS_NAMES = ("miscalibrated", "calibrated")

    def __init__(self, deploy_cfg: Config) -> None:
        super().__init__(deploy_cfg)
        class_names = deploy_cfg.get("class_names", self._DEFAULT_CLASS_NAMES)
        self.class_names = list(class_names)
        self._validate_components()

    def _validate_components(self) -> None:
        """Fail early if the deploy config is missing the required ``model`` component."""
        for component_name in self._REQUIRED_COMPONENTS:
            self.components_cfg.get_component(component_name)
