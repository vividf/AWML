from __future__ import annotations

import logging
from typing import Optional, Type

from deployment.core import BaseDeploymentConfig
from deployment.exporters.common.configs import TensorRTExportConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.common.tensorrt_exporter import TensorRTExporter


class ExporterFactory:
    """
    Factory class for instantiating exporters using deployment configs.

    Uses config.resolve_component(component) so single-component auto-resolves when only one is defined.
    """

    @staticmethod
    def create_onnx_exporter(
        config: BaseDeploymentConfig,
        wrapper_cls: Type[BaseModelWrapper],
        logger: logging.Logger,
        component: Optional[str] = None,
    ) -> ONNXExporter:
        """Build an ONNX exporter for the given component (None = auto when only one)."""
        return ONNXExporter(
            config=config.get_onnx_settings(component),
            model_wrapper=wrapper_cls,
            logger=logger,
        )

    @staticmethod
    def create_tensorrt_exporter(
        config: BaseDeploymentConfig,
        logger: logging.Logger,
        config_override: Optional[TensorRTExportConfig] = None,
        component: Optional[str] = None,
    ) -> TensorRTExporter:
        """
        Build a TensorRT exporter. Uses config.get_tensorrt_settings(component) when config_override is None.
        component=None auto-resolves when only one component is defined.
        """
        trt_config = config_override if config_override is not None else config.get_tensorrt_settings(component)

        return TensorRTExporter(
            config=trt_config,
            logger=logger,
        )
