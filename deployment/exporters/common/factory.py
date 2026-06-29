from __future__ import annotations

from typing import Type

from deployment.configs.base import BaseDeploymentConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.common.tensorrt_exporter import TensorRTExporter


class ExporterFactory:
    """Factory class for instantiating exporters using deployment configs."""

    @staticmethod
    def create_onnx_exporter(
        config: BaseDeploymentConfig,
        wrapper_cls: Type[BaseModelWrapper],
        component_name: str,
    ) -> ONNXExporter:
        """Build an ONNX exporter for the given component."""
        return ONNXExporter(
            config=config.get_onnx_settings(component_name),
            model_wrapper=wrapper_cls,
        )

    @staticmethod
    def create_tensorrt_exporter(
        config: BaseDeploymentConfig,
        component_name: str,
    ) -> TensorRTExporter:
        """Build a TensorRT exporter for the given component."""
        return TensorRTExporter(
            config=config.get_tensorrt_settings(component_name),
        )
