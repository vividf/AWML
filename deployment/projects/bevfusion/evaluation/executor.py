"""BEVFusion backend executor.

Thin subclass of ``PointDetectionExecutor``: declares the BEVFusion pipeline classes, the
split/merged output-name lookup, and forwards custom spconv INT8 ``plugin_libraries`` to the
TensorRT pipeline. Pipeline creation and ``(points, metainfo)`` input prep are shared with the
base (see ``deployment.evaluation.point_detection_executor``).

This replaces the OLD ``BEVFusionPipelineFactory`` (the global pipeline registry was removed in
the refactor): pipeline construction uses the reference model on ``self.pytorch_model`` (set by
the runner after export).
"""

from typing import Any, Iterable, List, Mapping, Optional

from typing_extensions import override

from deployment.config.schema import ComponentsConfig
from deployment.evaluation.point_detection_executor import PointDetectionExecutor
from deployment.projects.bevfusion.inference.onnx_inference_pipeline import BEVFusionONNXPipeline
from deployment.projects.bevfusion.inference.pytorch_inference_pipeline import BEVFusionPyTorchPipeline
from deployment.projects.bevfusion.inference.tensorrt_inference_pipeline import BEVFusionTensorRTPipeline
from deployment.projects.bevfusion.io.component_utils import has_component, is_split_bevfusion_components


class BEVFusionExecutor(PointDetectionExecutor):
    """Backend execution primitives for BEVFusion (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the ONNX/TensorRT
            pipelines so they can resolve split (sparse+dense) vs merged main-body artifacts.
        tensorrt_plugin_libraries: Custom TensorRT plugin ``.so`` paths forwarded to the
            TensorRT pipeline (e.g. the spconv INT8 plugin); empty for the FP16 path.
    """

    task_name = "BEVFusion"
    pytorch_pipeline_cls = BEVFusionPyTorchPipeline
    onnx_pipeline_cls = BEVFusionONNXPipeline
    tensorrt_pipeline_cls = BEVFusionTensorRTPipeline

    def __init__(self, components_cfg: ComponentsConfig, tensorrt_plugin_libraries: Iterable[str] = ()) -> None:
        super().__init__(components_cfg)
        self._tensorrt_plugin_libraries = tuple(tensorrt_plugin_libraries)

    @override
    def _tensorrt_pipeline_kwargs(self) -> Mapping[str, Any]:
        """Forward the custom spconv INT8 plugin ``.so`` paths to the TensorRT pipeline."""
        return {"plugin_libraries": self._tensorrt_plugin_libraries}

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the model output names (split→dense outputs; otherwise main-body outputs)."""
        if is_split_bevfusion_components(self._components_cfg) and not has_component(
            self._components_cfg, "bevfusion_main_body"
        ):
            comp = self._components_cfg.get_component("bevfusion_dense")
        else:
            comp = self._components_cfg.get_component("bevfusion_main_body")
        return [out.name for out in comp.io.outputs]
