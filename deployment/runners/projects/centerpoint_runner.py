"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any

from deployment.core.contexts import CenterPointExportContext, ExportContext
from deployment.exporters.centerpoint.model_wrappers import CenterPointONNXWrapper
from deployment.exporters.centerpoint.onnx_workflow import CenterPointONNXExportWorkflow
from deployment.exporters.centerpoint.tensorrt_workflow import CenterPointTensorRTExportWorkflow
from deployment.exporters.common.factory import ExporterFactory
from deployment.runners.common.deployment_runner import BaseDeploymentRunner
from projects.CenterPoint.deploy.component_extractor import CenterPointComponentExtractor
from projects.CenterPoint.deploy.utils import build_centerpoint_onnx_model


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """
    CenterPoint-specific deployment runner.

    Handles CenterPoint-specific requirements:
    - Multi-file ONNX export (voxel encoder + backbone/neck/head) via workflow
    - Multi-file TensorRT export via workflow
    - Uses generic ONNX/TensorRT exporters composed into CenterPoint workflows
    - ONNX-compatible model loading

    Key improvements:
    - Uses ExporterFactory instead of passing runner methods
    - Injects CenterPointComponentExtractor for model-specific logic
    - No circular dependencies or exporter caching
    """

    def __init__(
        self,
        data_loader,
        evaluator,
        config,
        model_cfg,
        logger: logging.Logger,
        onnx_wrapper_cls=None,
        onnx_workflow=None,
        tensorrt_workflow=None,
    ):
        """
        Initialize CenterPoint deployment runner.

        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            onnx_wrapper_cls: Optional ONNX wrapper (defaults to CenterPointONNXWrapper)
            onnx_workflow: Optional custom ONNX workflow
            tensorrt_workflow: Optional custom TensorRT workflow
        """
        # Create component extractor for model-specific logic
        component_extractor = CenterPointComponentExtractor(logger=logger)

        # Initialize base runner
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=onnx_wrapper_cls or CenterPointONNXWrapper,
            onnx_workflow=onnx_workflow,
            tensorrt_workflow=tensorrt_workflow,
        )

        # Create workflows with ExporterFactory and component extractor
        if self._onnx_workflow is None:
            self._onnx_workflow = CenterPointONNXExportWorkflow(
                exporter_factory=ExporterFactory,
                component_extractor=component_extractor,
                config=self.config,
                logger=self.logger,
            )

        if self._tensorrt_workflow is None:
            self._tensorrt_workflow = CenterPointTensorRTExportWorkflow(
                exporter_factory=ExporterFactory,
                config=self.config,
                logger=self.logger,
            )

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        context: ExportContext,
    ) -> Any:
        """
        Build ONNX-compatible CenterPoint model from checkpoint.

        This method:
        1. Builds ONNX-compatible model
        2. Updates runner's config to ONNX version
        3. Explicitly injects model and config to evaluator

        Args:
            checkpoint_path: Path to checkpoint file
            context: Export context. Use CenterPointExportContext for type-safe access
                     to rot_y_axis_reference. Falls back to context.extra for compatibility.

        Returns:
            Loaded PyTorch model (ONNX-compatible)
        """
        # Extract rot_y_axis_reference from typed context or extra dict
        rot_y_axis_reference: bool = False
        if isinstance(context, CenterPointExportContext):
            rot_y_axis_reference = context.rot_y_axis_reference
        else:
            rot_y_axis_reference = context.get("rot_y_axis_reference", False)

        model, onnx_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            rot_y_axis_reference=rot_y_axis_reference,
        )

        # Update runner's internal model_cfg to ONNX-friendly version
        self.model_cfg = onnx_cfg

        # Explicitly inject model and config to evaluator
        self._inject_model_to_evaluator(model, onnx_cfg)

        return model

    def _inject_model_to_evaluator(self, model: Any, onnx_cfg: Any) -> None:
        """
        Inject PyTorch model and ONNX config to evaluator.

        Args:
            model: PyTorch model to inject
            onnx_cfg: ONNX-compatible config to inject
        """
        # Check if evaluator has the setter methods
        has_set_onnx_config = hasattr(self.evaluator, "set_onnx_config")
        has_set_pytorch_model = hasattr(self.evaluator, "set_pytorch_model")

        if not (has_set_onnx_config and has_set_pytorch_model):
            self.logger.warning(
                "Evaluator does not have set_onnx_config() and/or set_pytorch_model() methods. "
                "CenterPoint evaluator should implement these methods for proper model injection. "
                f"Has set_onnx_config: {has_set_onnx_config}, "
                f"Has set_pytorch_model: {has_set_pytorch_model}"
            )
            return

        # Inject ONNX-compatible config
        try:
            self.evaluator.set_onnx_config(onnx_cfg)
            self.logger.info("Injected ONNX-compatible config to evaluator")
        except Exception as e:
            self.logger.error(f"Failed to inject ONNX config: {e}")
            raise

        # Inject PyTorch model
        try:
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Injected PyTorch model to evaluator")
        except Exception as e:
            self.logger.error(f"Failed to inject PyTorch model: {e}")
            raise
