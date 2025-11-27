"""
Pipeline factory for centralized pipeline instantiation.

This module provides a factory for creating task-specific pipelines,
eliminating duplicated backend switching logic across evaluators.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """
    Registry for pipeline classes.

    Each task type registers its pipeline classes for different backends.
    """

    _registry: Dict[str, Dict[Backend, Type[BaseDeploymentPipeline]]] = {}

    @classmethod
    def register(
        cls,
        task_type: str,
        backend: Backend,
        pipeline_cls: Type[BaseDeploymentPipeline],
    ) -> None:
        """Register a pipeline class for a task type and backend."""
        if task_type not in cls._registry:
            cls._registry[task_type] = {}
        cls._registry[task_type][backend] = pipeline_cls

    @classmethod
    def get(cls, task_type: str, backend: Backend) -> Optional[Type[BaseDeploymentPipeline]]:
        """Get a pipeline class for a task type and backend."""
        return cls._registry.get(task_type, {}).get(backend)

    @classmethod
    def register_task(
        cls,
        task_type: str,
        pytorch_cls: Type[BaseDeploymentPipeline],
        onnx_cls: Type[BaseDeploymentPipeline],
        tensorrt_cls: Type[BaseDeploymentPipeline],
    ) -> None:
        """Register all backend pipelines for a task type."""
        cls.register(task_type, Backend.PYTORCH, pytorch_cls)
        cls.register(task_type, Backend.ONNX, onnx_cls)
        cls.register(task_type, Backend.TENSORRT, tensorrt_cls)


class PipelineFactory:
    """
    Factory for creating deployment pipelines.

    This factory centralizes pipeline creation logic, eliminating the
    duplicated backend switching code in evaluators.
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
