"""
CenterPoint Deployment Pipelines.

This module provides unified deployment pipelines for CenterPoint 3D object detection
across different backends (PyTorch, ONNX, TensorRT).

Example usage:

PyTorch:
    >>> from deployment.pipelines.centerpoint import CenterPointPyTorchPipeline
    >>> pipeline = CenterPointPyTorchPipeline(model, device='cuda')
    >>> predictions, latency, breakdown = pipeline.infer(points)

ONNX:
    >>> from deployment.pipelines.centerpoint import CenterPointONNXPipeline
    >>> pipeline = CenterPointONNXPipeline(pytorch_model, onnx_dir='models', device='cuda')
    >>> predictions, latency, breakdown = pipeline.infer(points)

TensorRT:
    >>> from deployment.pipelines.centerpoint import CenterPointTensorRTPipeline
    >>> pipeline = CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir='engines', device='cuda')
    >>> predictions, latency, breakdown = pipeline.infer(points)

Note:
    All pipelines now use the unified `infer()` interface from the base class.
    The `breakdown` dict contains stage-wise latencies:
    - preprocessing_ms
    - voxel_encoder_ms
    - middle_encoder_ms
    - backbone_head_ms
    - postprocessing_ms
"""

from deployment.pipelines.centerpoint.centerpoint_onnx import CenterPointONNXPipeline
from deployment.pipelines.centerpoint.centerpoint_pipeline import CenterPointDeploymentPipeline
from deployment.pipelines.centerpoint.centerpoint_pytorch import CenterPointPyTorchPipeline
from deployment.pipelines.centerpoint.centerpoint_tensorrt import CenterPointTensorRTPipeline

__all__ = [
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
]
