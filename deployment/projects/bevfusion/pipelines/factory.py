"""BEVFusion Pipeline Factory."""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from typing_extensions import override

from deployment.configs import ComponentsConfig
from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_factory import BasePipelineFactory
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.registry import pipeline_registry
from deployment.projects.bevfusion.pipelines.onnx import BEVFusionONNXPipeline
from deployment.projects.bevfusion.pipelines.pytorch import BEVFusionPyTorchPipeline
from deployment.projects.bevfusion.pipelines.tensorrt import BEVFusionTensorRTPipeline

logger = logging.getLogger(__name__)


@pipeline_registry.register
class BEVFusionPipelineFactory(BasePipelineFactory):
    """Pipeline factory for BEVFusion across supported backends."""

    @classmethod
    @override
    def get_project_name(cls) -> str:
        return "bevfusion"

    @classmethod
    @override
    def create_pipeline(
        cls,
        model_spec: ModelSpec,
        pytorch_model: torch.nn.Module,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
        tensorrt_plugin_libraries: Iterable[str] = (),
    ) -> BaseDeploymentPipeline:
        device = device or model_spec.device
        backend = model_spec.backend

        cls._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info(f"Creating BEVFusion PyTorch pipeline on {device}")
            return BEVFusionPyTorchPipeline(pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info(f"Creating BEVFusion ONNX pipeline from {model_spec.path} on {device}")
            return BEVFusionONNXPipeline(
                pytorch_model,
                onnx_dir=model_spec.path,
                device=device,
                components_cfg=components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info(f"Creating BEVFusion TensorRT pipeline from {model_spec.path} on {device}")
            plugin_libs = tuple(tensorrt_plugin_libraries)
            return BEVFusionTensorRTPipeline(
                pytorch_model,
                tensorrt_dir=model_spec.path,
                device=device,
                components_cfg=components_cfg,
                plugin_libraries=plugin_libs,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")
