"""BEVFusion-specific deployment runner."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config

from deployment.configs.base import BaseDeploymentConfig
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

logger = logging.getLogger(__name__)


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
        self._data_loader = data_loader

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

        quantization = self.config.deploy_cfg.get("quantization", None)
        if quantization is not None:
            # Copy so we can safely mutate without touching the deploy_cfg singleton.
            quantization = dict(quantization)
            # Hoist top-level ``spconv_int8_fp16_layers`` (list of substring patterns matched on
            # sparse-conv module names) into the quantization dict. This is how model_loader
            # learns WHICH sparse convs to keep in FP16 — those modules get NO
            # _input_quantizer/_weight_quantizer, so the reloaded module tree matches the PTQ
            # checkpoint exactly (PTQ ran with the same exclusion).
            fp16_layers = self.config.deploy_cfg.get("spconv_int8_fp16_layers", None)
            if fp16_layers is not None:
                try:
                    quantization["spconv_int8_fp16_layers"] = list(fp16_layers)
                except TypeError:
                    quantization["spconv_int8_fp16_layers"] = []
        if quantization and quantization.get("enabled", False):
            logger.info("=" * 60)
            logger.info("BEVFusion INT8 Quantization Enabled")
            logger.info("  Dense (backbone/neck/head): pytorch_quantization")
            if quantization.get("spconv_int8", False):
                logger.info("  Sparse encoder: spconv INT8 (cumm kernels)")
            fp16_layers_ = quantization.get("spconv_int8_fp16_layers") or []
            if fp16_layers_:
                logger.info(
                    "  Sparse FP16 keep-list (spconv_int8_fp16_layers): %s",
                    fp16_layers_,
                )
            logger.info("=" * 60)

        fuse_spconv_bn = bool(self.config.deploy_cfg.get("fuse_spconv_bn", False))
        model = build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
            quantization=quantization,
            fuse_spconv_bn=fuse_spconv_bn,
        )

        if quantization and quantization.get("enabled") and quantization.get("spconv_int8"):
            model = self._apply_spconv_int8(model, quantization, cuda_device)

        self.evaluator.set_pytorch_model(model)
        return model

    def _apply_spconv_int8(
        self,
        model: torch.nn.Module,
        quantization: dict,
        device: DeviceSpec,
    ) -> torch.nn.Module:
        """Sparse INT8 is applied at PTQ time and recreated in ``model_loader`` for PTQ checkpoints.

        Runtime FX ``prepare_fx`` / ``convert_fx`` is not supported here.
        """
        sparse_encoder = getattr(model, "pts_middle_encoder", None)
        if sparse_encoder is None:
            logger.warning("No pts_middle_encoder found; skipping spconv INT8")
            return model

        has_nvidia_quantizers = any(hasattr(m, "_input_quantizer") for m in sparse_encoder.modules())
        if has_nvidia_quantizers:
            logger.info(
                "pts_middle_encoder has NVIDIA TensorQuantizer modules (PTQ checkpoint load); "
                "no runner-side sparse calibration."
            )
            return model

        logger.warning(
            "spconv_int8 is enabled but pts_middle_encoder has no NVIDIA TensorQuantizers — "
            "use a PTQ .pth from bevfusion_quantization.py with spconv INT8 calibration, "
            "or disable spconv_int8 in deploy."
        )
        return model
