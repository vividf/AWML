"""
YOLOX Deployment Pipelines.

This module provides unified deployment pipelines for YOLOX object detection
across different backends (PyTorch, ONNX, TensorRT).

Example usage:

PyTorch:
    >>> from autoware_ml.deployment.pipelines.yolox import YOLOXPyTorchPipeline
    >>> pipeline = YOLOXPyTorchPipeline(model, device='cuda')
    >>> predictions, latency = pipeline.infer(image)

ONNX:
    >>> from autoware_ml.deployment.pipelines.yolox import YOLOXONNXPipeline
    >>> pipeline = YOLOXONNXPipeline(onnx_path='model.onnx', device='cuda')
    >>> predictions, latency = pipeline.infer(image)

TensorRT:
    >>> from autoware_ml.deployment.pipelines.yolox import YOLOXTensorRTPipeline
    >>> pipeline = YOLOXTensorRTPipeline(engine_path='model.engine', device='cuda')
    >>> predictions, latency = pipeline.infer(image)
"""

from autoware_ml.deployment.pipelines.yolox.yolox_pipeline import YOLOXDeploymentPipeline
from autoware_ml.deployment.pipelines.yolox.yolox_pytorch import YOLOXPyTorchPipeline
from autoware_ml.deployment.pipelines.yolox.yolox_onnx import YOLOXONNXPipeline
from autoware_ml.deployment.pipelines.yolox.yolox_tensorrt import YOLOXTensorRTPipeline


__all__ = [
    'YOLOXDeploymentPipeline',
    'YOLOXPyTorchPipeline',
    'YOLOXONNXPipeline',
    'YOLOXTensorRTPipeline',
]

