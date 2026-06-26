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
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.backend_executor import BackendExecutor
from deployment.core.evaluation.evaluator_types import InferenceInput, ModelSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.base_pipeline import BaseInferencePipeline
from deployment.pipelines.registry import pipeline_registry

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
        """
        return pipeline_registry.create_pipeline(
            project_name="centerpoint",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

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
