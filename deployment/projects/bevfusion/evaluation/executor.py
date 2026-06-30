"""BEVFusion backend executor.

Implements the task-specific backend execution primitives (pipeline creation and
input preparation) for BEVFusion, shared by the evaluator and the verification runner
via `~deployment.evaluation.backend_executor.BackendExecutor`.

This replaces the OLD ``BEVFusionPipelineFactory`` (the global pipeline registry was
removed in the refactor): pipeline construction now lives here and uses the reference
model stored on ``self.pytorch_model`` (set by the runner after export).
"""

import logging
from typing import Iterable, List, Mapping, Optional

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.evaluator_types import InferenceInput, ModelSpec
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion.inference.onnx_inference_pipeline import BEVFusionONNXPipeline
from deployment.projects.bevfusion.inference.pytorch_inference_pipeline import BEVFusionPyTorchPipeline
from deployment.projects.bevfusion.inference.tensorrt_inference_pipeline import BEVFusionTensorRTPipeline
from deployment.projects.bevfusion.io.component_utils import has_component, is_split_bevfusion_components

logger = logging.getLogger(__name__)


class BEVFusionExecutor(BackendExecutor):
    """Backend execution primitives for BEVFusion (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the ONNX/TensorRT
            pipelines so they can resolve split (sparse+dense) vs merged main-body artifacts.
        tensorrt_plugin_libraries: Custom TensorRT plugin ``.so`` paths forwarded to the
            TensorRT pipeline (e.g. the spconv INT8 plugin); empty for the FP16 path.
    """

    def __init__(self, components_cfg: ComponentsConfig, tensorrt_plugin_libraries: Iterable[str] = ()) -> None:
        super().__init__()
        self._components_cfg = components_cfg
        self._tensorrt_plugin_libraries = tuple(tensorrt_plugin_libraries)

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

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a BEVFusion inference pipeline for the given backend and device."""
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating BEVFusion PyTorch pipeline on %s", device)
            return BEVFusionPyTorchPipeline(self.pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info("Creating BEVFusion ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return BEVFusionONNXPipeline(
                self.pytorch_model,
                onnx_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating BEVFusion TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return BEVFusionTensorRTPipeline(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
                plugin_libraries=self._tensorrt_plugin_libraries,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")

    @override
    def prepare_input(
        self,
        sample: Mapping[str, object],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build InferenceInput from sample (points + metainfo)."""
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError("Sample must contain 'metainfo' for BEVFusion postprocess.")
        return InferenceInput(data=sample["points"], metadata=sample["metainfo"])
