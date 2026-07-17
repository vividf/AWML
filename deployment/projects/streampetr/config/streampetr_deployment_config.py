"""StreamPETR-specific deployment config.

Extends :class:`~deployment.config.base.BaseDeploymentConfig` to validate the StreamPETR
component layout and the temporal-model constraints that the generic sections intentionally
do not model.
"""

from __future__ import annotations

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig


class StreamPETRDeploymentConfig(BaseDeploymentConfig):
    """Deployment config for StreamPETR.

    StreamPETR is exported as three chained ONNX components (the split and the tensor names
    are a frozen contract consumed by Autoware / the DL4AGX TensorRT runtime):

    - ``extract_img_feat``: image backbone + neck
    - ``position_embedding``: 3D position encoding from camera geometry
    - ``pts_head_memory``: transformer decoder head with the temporal memory queue as
      explicit graph I/O (``pre_memory_*`` in, ``post_memory_*`` out)

    Temporal constraint: StreamPETR carries a memory queue across frames, so evaluation
    warmup must be disabled — warmup replays samples and would corrupt the queue. This is
    validated here, at config construction time.
    """

    #: Components StreamPETR always splits into for multi-file ONNX/TensorRT export.
    _REQUIRED_COMPONENTS = ("extract_img_feat", "position_embedding", "pts_head_memory")

    def __init__(self, deploy_cfg: Config) -> None:
        super().__init__(deploy_cfg)
        self._validate_components()
        self._validate_temporal_constraints()

    def _validate_components(self) -> None:
        """Fail early if the deploy config is missing a required StreamPETR component."""
        for component_name in self._REQUIRED_COMPONENTS:
            self.components_cfg.get_component(component_name)

    def _validate_temporal_constraints(self) -> None:
        """Reject eval settings that would corrupt the temporal memory queue.

        The shared evaluator's warmup replays samples through the pipeline before the timed
        loop; for a stateful temporal model that leaves a polluted memory queue behind, so
        ``evaluation.num_warmup`` must be 0.
        """
        if self.evaluation_config.enabled and self.evaluation_config.num_warmup != 0:
            raise ValueError(
                "StreamPETR is temporally stateful: evaluation.num_warmup must be 0 "
                f"(got {self.evaluation_config.num_warmup}). Warmup replays samples and "
                "corrupts the memory queue."
            )
