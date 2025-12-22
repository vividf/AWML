"""
CenterPoint Deployment Pipelines.

This module provides unified deployment pipelines for CenterPoint 3D object detection
across different backends (PyTorch, ONNX, TensorRT).

Example usage:


Using Registry:
    >>> from deployment.pipelines.common import pipeline_registry
    >>> pipeline = pipeline_registry.create_pipeline("centerpoint", model_spec, model)

Direct Instantiation:
    >>> from deployment.pipelines.centerpoint import CenterPointPyTorchPipeline
    >>> pipeline = CenterPointPyTorchPipeline(model, device='cuda')
    >>> result = pipeline.infer(points)
    >>> predictions, latency, breakdown = result.output, result.latency_ms, result.breakdown

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
from deployment.pipelines.centerpoint.factory import CenterPointPipelineFactory

__all__ = [
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
    "CenterPointPipelineFactory",
]
