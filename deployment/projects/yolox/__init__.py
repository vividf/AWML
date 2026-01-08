"""YOLOX deployment bundle.

This package owns all YOLOX_opt_elan deployment-specific code
(runner/evaluator/loader/pipelines).
It registers a ProjectAdapter into the global `project_registry` so the unified CLI can invoke it.
"""

from __future__ import annotations

from deployment.projects.registry import ProjectAdapter, project_registry
from deployment.projects.yolox.cli import add_args
from deployment.projects.yolox.entrypoint import run

# Trigger pipeline factory registration for this project.
from deployment.projects.yolox.pipelines.factory import YOLOXPipelineFactory  # noqa: F401

project_registry.register(
    ProjectAdapter(
        name="yolox",
        add_args=add_args,
        run=run,
    )
)
