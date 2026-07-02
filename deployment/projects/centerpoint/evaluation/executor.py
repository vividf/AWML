"""CenterPoint backend executor.

Thin subclass of ``PointDetectionExecutor``: declares the CenterPoint pipeline classes and
the head output-name lookup. Pipeline creation and ``(points, metainfo)`` input prep are shared
with the base (see ``deployment.evaluation.point_detection_executor``).
"""

from typing import List, Optional

from typing_extensions import override

from deployment.evaluation.point_detection_executor import PointDetectionExecutor
from deployment.projects.centerpoint.inference.onnx_inference_pipeline import CenterPointONNXInferencePipeline
from deployment.projects.centerpoint.inference.pytorch_inference_pipeline import CenterPointPyTorchInferencePipeline
from deployment.projects.centerpoint.inference.tensorrt_inference_pipeline import CenterPointTensorRTInferencePipeline


class CenterPointExecutor(PointDetectionExecutor):
    """Backend execution primitives for CenterPoint (pipeline creation, input prep)."""

    task_name = "CenterPoint"
    pytorch_pipeline_cls = CenterPointPyTorchInferencePipeline
    onnx_pipeline_cls = CenterPointONNXInferencePipeline
    tensorrt_pipeline_cls = CenterPointTensorRTInferencePipeline

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the head output names from the components config for verification logging."""
        return [out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs]
