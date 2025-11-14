"""
CalibrationStatusClassification Deployment Pipelines.

This module provides unified deployment pipelines for CalibrationStatusClassification
across different backends (PyTorch, ONNX, TensorRT).

Example usage:

PyTorch:
    >>> from autoware_ml.deployment.pipelines.calibration import CalibrationPyTorchPipeline
    >>> pipeline = CalibrationPyTorchPipeline(model, device='cuda')
    >>> predictions, latency = pipeline.infer(input_tensor)

ONNX:
    >>> from autoware_ml.deployment.pipelines.calibration import CalibrationONNXPipeline
    >>> pipeline = CalibrationONNXPipeline(onnx_path='model.onnx', device='cuda')
    >>> predictions, latency = pipeline.infer(input_tensor)

TensorRT:
    >>> from autoware_ml.deployment.pipelines.calibration import CalibrationTensorRTPipeline
    >>> pipeline = CalibrationTensorRTPipeline(engine_path='model.engine', device='cuda')
    >>> predictions, latency = pipeline.infer(input_tensor)
"""

from autoware_ml.deployment.pipelines.calibration.calibration_pipeline import CalibrationDeploymentPipeline
from autoware_ml.deployment.pipelines.calibration.calibration_pytorch import CalibrationPyTorchPipeline
from autoware_ml.deployment.pipelines.calibration.calibration_onnx import CalibrationONNXPipeline
from autoware_ml.deployment.pipelines.calibration.calibration_tensorrt import CalibrationTensorRTPipeline


__all__ = [
    'CalibrationDeploymentPipeline',
    'CalibrationPyTorchPipeline',
    'CalibrationONNXPipeline',
    'CalibrationTensorRTPipeline',
]

