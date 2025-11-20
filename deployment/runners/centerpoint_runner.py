"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from deployment.exporters.centerpoint.model_wrappers import CenterPointONNXWrapper
from deployment.exporters.centerpoint.onnx_workflow import CenterPointONNXExportWorkflow
from deployment.exporters.centerpoint.tensorrt_workflow import CenterPointTensorRTExportWorkflow
from deployment.runners.deployment_runner import BaseDeploymentRunner
from projects.CenterPoint.deploy.utils import build_centerpoint_onnx_model


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """
    CenterPoint-specific deployment runner.

    Handles CenterPoint-specific requirements:
    - Multi-file ONNX export (voxel encoder + backbone/neck/head) via workflow
    - Multi-file TensorRT export via workflow
    - Uses generic ONNX/TensorRT exporters composed into CenterPoint workflows
    - ONNX-compatible model loading
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

        if self._onnx_workflow is None:
            self._onnx_workflow = CenterPointONNXExportWorkflow(exporter=self._get_onnx_exporter, logger=self.logger)
        if self._tensorrt_workflow is None:
            self._tensorrt_workflow = CenterPointTensorRTExportWorkflow(
                exporter=self._get_tensorrt_exporter,
                logger=self.logger,
            )

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        rot_y_axis_reference: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Build ONNX-compatible CenterPoint model from checkpoint.

        NOTE:
        - self.model_cfg is the "original mmdet3d config"
        - This method converts it to ONNX-friendly cfg and updates self.model_cfg

        Args:
            checkpoint_path: Path to checkpoint file
            rot_y_axis_reference: Whether to use y-axis rotation reference
            device: Device string (e.g., "cpu", "cuda:0"). Defaults to "cpu" for export.
            **kwargs: Additional arguments

        Returns:
            Loaded PyTorch model (ONNX-compatible)
        """
        # Use shared utility to build ONNX-compatible model
        model, onnx_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            rot_y_axis_reference=rot_y_axis_reference,
        )

        # Update runner's internal model_cfg to ONNX-friendly version
        self.model_cfg = onnx_cfg

        # Inject ONNX-compatible config to evaluator via setter (single source of truth)
        if hasattr(self.evaluator, "set_onnx_config"):
            self.evaluator.set_onnx_config(onnx_cfg)
            self.logger.info("Updated evaluator with ONNX-compatible config via set_onnx_config()")

        # Inject PyTorch model to evaluator via setter
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with PyTorch model via set_pytorch_model()")

        return model
