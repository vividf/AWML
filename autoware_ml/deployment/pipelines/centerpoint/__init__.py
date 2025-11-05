"""
CenterPoint Deployment Pipelines.

This module provides unified deployment pipelines for CenterPoint 3D object detection
across different backends (PyTorch, ONNX, TensorRT).

Example usage:

PyTorch:
    >>> from autoware_ml.deployment.pipelines.centerpoint import CenterPointPyTorchPipeline
    >>> pipeline = CenterPointPyTorchPipeline(model, device='cuda')
    >>> predictions, latency = pipeline.infer(points)

ONNX:
    >>> from autoware_ml.deployment.pipelines.centerpoint import CenterPointONNXPipeline
    >>> pipeline = CenterPointONNXPipeline(pytorch_model, onnx_dir='models', device='cuda')
    >>> predictions, latency = pipeline.infer(points)

TensorRT:
    >>> from autoware_ml.deployment.pipelines.centerpoint import CenterPointTensorRTPipeline
    >>> pipeline = CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir='engines', device='cuda')
    >>> predictions, latency = pipeline.infer(points)
"""

from .centerpoint_pipeline import CenterPointDeploymentPipeline
from .centerpoint_pytorch import CenterPointPyTorchPipeline
from .centerpoint_onnx import CenterPointONNXPipeline
from .centerpoint_tensorrt import CenterPointTensorRTPipeline


__all__ = [
    'CenterPointDeploymentPipeline',
    'CenterPointPyTorchPipeline',
    'CenterPointONNXPipeline',
    'CenterPointTensorRTPipeline',
]

