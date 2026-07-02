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

    BEVFusion-only deploy-config flags (``fuse_spconv_bn``, ``spconv_do_sort``,
    ``spconv_fuse_implicit_gemm_relu``) are read from the raw ``deploy_cfg`` passed in by the
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
        """Load the BEVFusion model onto the CUDA device for export.

        The base runner forwards the returned model to ``executor.set_pytorch_model`` after
        export, so PyTorch/ONNX/TensorRT evaluation all reuse this reference.
        """
        cuda_device = self.config.device_config.cuda
        if cuda_device is None:
            raise RuntimeError(
                "BEVFusion requires a CUDA device for sparse convolution. Set devices.cuda in deploy config."
            )

        fuse_spconv_bn = bool(self._deploy_cfg.get("fuse_spconv_bn", False))
        return build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
            fuse_spconv_bn=fuse_spconv_bn,
        )
