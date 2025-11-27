"""
Pipeline factory for centralized pipeline instantiation.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating deployment pipelines.
    """

    @staticmethod
    def create_centerpoint_pipeline(
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
    ) -> BaseDeploymentPipeline:
        """
        Create a CenterPoint pipeline.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance
            device: Override device (uses model_spec.device if None)

        Returns:
            CenterPoint pipeline instance
        """
        from deployment.pipelines.centerpoint import (
            CenterPointONNXPipeline,
            CenterPointPyTorchPipeline,
            CenterPointTensorRTPipeline,
        )

        device = device or model_spec.device
        backend = model_spec.backend

        if backend is Backend.PYTORCH:
            return CenterPointPyTorchPipeline(pytorch_model, device=device)
        elif backend is Backend.ONNX:
            return CenterPointONNXPipeline(pytorch_model, onnx_dir=model_spec.path, device=device)
        elif backend is Backend.TENSORRT:
            return CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir=model_spec.path, device=device)
        else:
            raise ValueError(f"Unsupported backend: {backend.value}")

    @staticmethod
    def create_yolox_pipeline(
        model_spec: ModelSpec,
        pytorch_model: Any,
        num_classes: int,
        class_names: List[str],
        device: Optional[str] = None,
    ) -> BaseDeploymentPipeline:
        """
        Create a YOLOX pipeline.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance
            num_classes: Number of classes
            class_names: List of class names
            device: Override device (uses model_spec.device if None)

        Returns:
            YOLOX pipeline instance
        """
        from deployment.pipelines.yolox import (
            YOLOXONNXPipeline,
            YOLOXPyTorchPipeline,
            YOLOXTensorRTPipeline,
        )

        device = device or model_spec.device
        backend = model_spec.backend

        if backend is Backend.PYTORCH:
            return YOLOXPyTorchPipeline(
                pytorch_model=pytorch_model,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.ONNX:
            return YOLOXONNXPipeline(
                onnx_path=model_spec.path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.TENSORRT:
            return YOLOXTensorRTPipeline(
                engine_path=model_spec.path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.value}")

    @staticmethod
    def create_calibration_pipeline(
        model_spec: ModelSpec,
        pytorch_model: Any,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> BaseDeploymentPipeline:
        """
        Create a CalibrationStatusClassification pipeline.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance
            num_classes: Number of classes (default: 2)
            class_names: List of class names (default: ["miscalibrated", "calibrated"])
            device: Override device (uses model_spec.device if None)

        Returns:
            Calibration pipeline instance
        """
        from deployment.pipelines.calibration import (
            CalibrationONNXPipeline,
            CalibrationPyTorchPipeline,
            CalibrationTensorRTPipeline,
        )

        device = device or model_spec.device
        backend = model_spec.backend
        class_names = class_names or ["miscalibrated", "calibrated"]

        if backend is Backend.PYTORCH:
            return CalibrationPyTorchPipeline(
                pytorch_model=pytorch_model,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.ONNX:
            return CalibrationONNXPipeline(
                onnx_path=model_spec.path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        elif backend is Backend.TENSORRT:
            return CalibrationTensorRTPipeline(
                engine_path=model_spec.path,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.value}")
