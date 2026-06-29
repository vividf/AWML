"""
CenterPoint backend executor.

Implements the task-specific backend execution primitives (pipeline creation and
input preparation) for CenterPoint, shared by the evaluator and the verification
runner via `~deployment.core.evaluation.backend_executor.BackendExecutor`.
"""

import logging
from typing import List, Mapping, Optional

from typing_extensions import override

from deployment.configs.schema import ComponentsConfig
from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.backend_executor import BackendExecutor
from deployment.core.evaluation.evaluator_types import InferenceInput, ModelSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.base_pipeline import BaseInferencePipeline
from deployment.projects.centerpoint.pipelines.onnx import CenterPointONNXPipeline
from deployment.projects.centerpoint.pipelines.pytorch import CenterPointPyTorchPipeline
from deployment.projects.centerpoint.pipelines.tensorrt import CenterPointTensorRTPipeline

logger = logging.getLogger(__name__)


class CenterPointExecutor(BackendExecutor):
    """Backend execution primitives for CenterPoint (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the pipeline
            registry when constructing backend pipelines.
    """

    def __init__(self, components_cfg: ComponentsConfig) -> None:
        super().__init__()
        self._components_cfg = components_cfg

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the head output names from the components config for verification logging."""
        return [out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a CenterPoint inference pipeline for the given backend and device.

        Args:
            model_spec: Model specification (backend, device, path).
            device: Target device for the pipeline.

        Returns:
            CenterPoint pipeline instance (PyTorch, ONNX, or TensorRT).

        Raises:
            ValueError: If ``model_spec.backend`` is not a supported backend.
        """
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating CenterPoint PyTorch pipeline on %s", device)
            return CenterPointPyTorchPipeline(self.pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info("Creating CenterPoint ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return CenterPointONNXPipeline(
                self.pytorch_model,
                onnx_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating CenterPoint TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return CenterPointTensorRTPipeline(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")

    @override
    def prepare_input(
        self,
        sample: Mapping[str, object],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build InferenceInput from sample (points + metainfo).

        Args:
            sample: Dict with 'points' and 'metainfo'.
            data_loader: Unused; kept for interface compatibility.
            device: Unused; kept for interface compatibility.

        Returns:
            InferenceInput with data=points and metadata=metainfo.

        Raises:
            ValueError: If 'points' is missing from sample.
            KeyError: If 'metainfo' is missing from sample.
        """
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError("Sample must contain 'metainfo' for CenterPoint postprocess.")
        points = sample["points"]
        metadata = sample["metainfo"]
        return InferenceInput(data=points, metadata=metadata)
