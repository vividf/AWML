"""BEVFusion-specific deployment runner."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.evaluation.backend_executor import BackendExecutor
from deployment.export.contexts import ExportContext
from deployment.io.base_data_loader import BaseDataLoader
from deployment.projects.bevfusion.evaluation.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.export.onnx_export_pipeline import BEVFusionONNXExportPipeline
from deployment.projects.bevfusion.export.tensorrt_export_pipeline import BEVFusionTensorRTExportPipeline
from deployment.projects.bevfusion.io.model_loader import build_bevfusion_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class BEVFusionDeploymentRunner(BaseDeploymentRunner):
    """BEVFusion deployment runner.

    Constructs BEVFusion's model-specific ONNX/TensorRT export pipelines and injects them
    into the project-agnostic ``BaseDeploymentRunner`` via its ``onnx_pipeline`` /
    ``tensorrt_pipeline`` override hooks (BEVFusion needs wrapper modules, TopK constant
    folding, coordinate flips, and split→merge ONNX composition that the generic whole-model
    export cannot express).

    BEVFusion-only deploy-config flags (``quantization``, ``fuse_spconv_bn``,
    ``spconv_int8_fp16_layers``) are read from the raw ``deploy_cfg`` passed in by the
    entrypoint, since ``BaseDeploymentConfig`` only surfaces typed sections.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BEVFusionEvaluator,
        executor: BackendExecutor,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        deploy_cfg: Config,
        module: str = "main_body",
        plugin_libraries: Tuple[str, ...] = (),
        onnx_pipeline: Optional[BEVFusionONNXExportPipeline] = None,
        tensorrt_pipeline: Optional[BEVFusionTensorRTExportPipeline] = None,
    ) -> None:
        self._module = module
        self._deploy_cfg = deploy_cfg

        # Construct the model-specific pipelines BEFORE super().__init__, because the base
        # runner forwards them straight to the ExportOrchestrator (there is no post-init slot).
        if onnx_pipeline is None:
            onnx_pipeline = BEVFusionONNXExportPipeline(module=module)
        if tensorrt_pipeline is None:
            tensorrt_pipeline = BEVFusionTensorRTExportPipeline(
                components_cfg=config.components_cfg,
                plugin_libraries=tuple(plugin_libraries),
            )

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load (optionally quantized) BEVFusion model onto the CUDA device for export.

        The base runner forwards the returned model to ``executor.set_pytorch_model`` after
        export, so PyTorch/ONNX/TensorRT evaluation all reuse this reference.
        """
        cuda_device = self.config.device_config.cuda
        if cuda_device is None:
            raise RuntimeError(
                "BEVFusion requires a CUDA device for sparse convolution. Set devices.cuda in deploy config."
            )

        quantization = self._deploy_cfg.get("quantization", None)
        if quantization is not None:
            # Copy so we can mutate without touching the deploy_cfg singleton.
            quantization = dict(quantization)
            # Hoist top-level ``spconv_int8_fp16_layers`` (substring patterns matched on sparse-conv
            # module names) so model_loader knows which sparse convs to keep FP16 — those get no
            # quantizers, so the reloaded module tree matches the PTQ checkpoint exactly.
            fp16_layers = self._deploy_cfg.get("spconv_int8_fp16_layers", None)
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
                logger.info("  Sparse FP16 keep-list (spconv_int8_fp16_layers): %s", fp16_layers_)
            logger.info("=" * 60)

        fuse_spconv_bn = bool(self._deploy_cfg.get("fuse_spconv_bn", False))
        model = build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
            quantization=quantization,
            fuse_spconv_bn=fuse_spconv_bn,
        )

        if quantization and quantization.get("enabled") and quantization.get("spconv_int8"):
            model = self._apply_spconv_int8(model, quantization)

        return model

    def _apply_spconv_int8(self, model: torch.nn.Module, quantization: dict) -> torch.nn.Module:
        """Sparse INT8 is applied at PTQ time and recreated in ``model_loader`` for PTQ checkpoints."""
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
