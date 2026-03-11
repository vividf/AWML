"""BEVFusion deployment bundle.

Registers a ProjectAdapter into the global project_registry so the unified CLI can invoke it.
Supports LiDAR-only BEVFusion export to ONNX/TensorRT with evaluation and verification.
"""

from __future__ import annotations

from deployment.projects.bevfusion.cli import add_args
from deployment.projects.bevfusion.entrypoint import run
from deployment.projects.bevfusion.pipelines.factory import BEVFusionPipelineFactory  # noqa: F401
from deployment.projects.registry import ProjectAdapter, project_registry

project_registry.register(
    ProjectAdapter(
        name="bevfusion",
        add_args=add_args,
        run=run,
    )
)
