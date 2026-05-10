"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config

from deployment.configs.base import BaseDeploymentConfig
from deployment.core.backend import Backend
from deployment.core.contexts import CenterPointExportContext, ExportContext
from deployment.core.device import DeviceSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.centerpoint.eval.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.export.component_builder import CenterPointComponentBuilder
from deployment.projects.centerpoint.export.onnx_export_pipeline import CenterPointONNXExportPipeline
from deployment.projects.centerpoint.export.tensorrt_export_pipeline import CenterPointTensorRTExportPipeline
from deployment.projects.centerpoint.io.model_loader import build_centerpoint_onnx_model
from deployment.projects.centerpoint.io.sample_adapter import CenterPointSampleAdapter
from deployment.runtime.runner import BaseDeploymentRunner


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """CenterPoint deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Attributes:
        model_cfg: Training MMEngine config (from checkpoint experiment file); not replaced after load.
        export_model_cfg: Set in ``load_pytorch_model`` to the deployment export MMEngine config.
        evaluator: CenterPoint evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: CenterPointEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize CenterPoint deployment runner.

        Args:
            data_loader: Data loader for loading samples.
            evaluator: Evaluator for computing metrics.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            logger: Logger instance.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """
        sample_adapter = CenterPointSampleAdapter(
            logger=logger,
        )
        component_builder = CenterPointComponentBuilder(
            components_cfg=config.components_cfg,
            logger=logger,
        )

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
            self._onnx_pipeline = CenterPointONNXExportPipeline(
                exporter_factory=ExporterFactory,
                sample_adapter=sample_adapter,
                component_builder=component_builder,
                logger=self.logger,
            )

        if self._tensorrt_pipeline is None:
            self._tensorrt_pipeline = CenterPointTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                components_cfg=config.components_cfg,
                logger=self.logger,
            )

        self.export_model_cfg: Optional[Config] = None

    def _resolve_export_load_device(self) -> DeviceSpec:
        """Pick CPU vs CUDA for building/loading the export PyTorch model.

        Prefer the deploy config CUDA device when ``torch.cuda.is_available()`` so large PTQ
        ``torch.load`` + tensors land on GPU VRAM instead of spiking host RAM (common silent OOM
        / SIGKILL in small containers when loading was hard-coded to CPU).
        """
        devices = self.config.devices
        if torch.cuda.is_available() and devices.cuda is not None:
            return devices.cuda
        return DeviceSpec.from_value("cpu")

    def _is_pytorch_evaluation_enabled(self) -> bool:
        """Check whether evaluation config enables the PyTorch backend."""
        evaluation_cfg = self.config.evaluation_config
        if not evaluation_cfg.enabled:
            return False
        backends_cfg = self.config.evaluation_backends
        pytorch_backend_cfg = backends_cfg.get("pytorch") or backends_cfg.get(Backend.PYTORCH) or {}
        return bool(pytorch_backend_cfg.get("enabled", False))

    def _should_use_quantized_checkpoint(self, quant_cfg: Optional[dict]) -> bool:
        """Decide whether to build model with quantized checkpoint transformations."""
        if not quant_cfg or not quant_cfg.get("enabled", False):
            return False

        export_cfg = self.config.export_config
        needs_quantized_for_export = export_cfg.should_export_onnx or export_cfg.should_export_tensorrt
        needs_quantized_for_eval_pytorch = self._is_pytorch_evaluation_enabled()
        return needs_quantized_for_export or needs_quantized_for_eval_pytorch

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context with additional parameters.

        Returns:
            Loaded PyTorch model.
        """
        rot_y_axis_reference = self._extract_rot_y_axis_reference(context)

        quant_cfg = self.config.deploy_cfg.get("quantization")
        export_cfg = self.config.export_config
        should_apply_quantized_load = self._should_use_quantized_checkpoint(quant_cfg)
        effective_quant_cfg = quant_cfg if should_apply_quantized_load else None
        if quant_cfg and quant_cfg.get("enabled", False) and not should_apply_quantized_load:
            self.logger.info(
                "Quantization is enabled in config, but export mode is '%s'. "
                "Skipping quantized checkpoint load for evaluation-only run.",
                export_cfg.mode.value,
            )

        load_device = self._resolve_export_load_device()
        self.logger.info("PyTorch export load device: %s", load_device)

        model, export_model_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=load_device,
            rot_y_axis_reference=rot_y_axis_reference,
            quantization=effective_quant_cfg,
        )

        self.export_model_cfg = export_model_cfg
        self._setup_evaluator(model, export_model_cfg)
        return model

    def _extract_rot_y_axis_reference(self, context: ExportContext) -> bool:
        """Extract rot_y_axis_reference from export context.

        Args:
            context: Export context (typed or dict-like).

        Returns:
            Boolean value for rot_y_axis_reference.
        """
        if isinstance(context, CenterPointExportContext):
            return context.rot_y_axis_reference
        if "rot_y_axis_reference" not in context.extra:
            raise KeyError(
                "CenterPoint export requires 'rot_y_axis_reference' in context. "
                "Use CenterPointExportContext or pass it in ExportContext.extra."
            )
        return bool(context.extra["rot_y_axis_reference"])

    def _setup_evaluator(self, model: torch.nn.Module, export_model_cfg: Config) -> None:
        """Wire evaluator to the loaded model and its export-time MMEngine config.

        Args:
            model: Loaded PyTorch model.
            export_model_cfg: Config from ``build_centerpoint_onnx_model``; matches ``model.cfg``.
        """
        self.evaluator.set_pytorch_model(model)
        self.evaluator.set_export_model_cfg(export_model_cfg)
        self.logger.info("Updated evaluator with export_model_cfg")
