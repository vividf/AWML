"""BEVFusion-specific deployment runner."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config

from deployment.configs import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.core.device import DeviceSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.bevfusion.eval.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.export.onnx_export_pipeline import BEVFusionONNXExportPipeline
from deployment.projects.bevfusion.export.tensorrt_export_pipeline import BEVFusionTensorRTExportPipeline
from deployment.projects.bevfusion.io.model_loader import build_bevfusion_model
from deployment.runtime.runner import BaseDeploymentRunner


class BEVFusionDeploymentRunner(BaseDeploymentRunner):
    """BEVFusion deployment runner."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BEVFusionEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        module: str = "main_body",
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        self._module = module

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

        if self._onnx_pipeline is None:
            self._onnx_pipeline = BEVFusionONNXExportPipeline(
                module=module,
                logger=self.logger,
            )

        if self._tensorrt_pipeline is None:
            self._tensorrt_pipeline = BEVFusionTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                components_cfg=config.components_cfg,
                logger=self.logger,
            )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        cuda_device = self.config.devices.cuda
        if cuda_device is None:
            raise RuntimeError(
                "BEVFusion requires a CUDA device for sparse convolution. " "Set devices.cuda in deploy config."
            )
        model = build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
        )
        self.evaluator.set_pytorch_model(model)
        return model
