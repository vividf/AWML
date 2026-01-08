"""Calibration deployment bundle.

This package owns all CalibrationStatusClassification deployment-specific code
(runner/evaluator/loader/pipelines).
It registers a ProjectAdapter into the global `project_registry` so the unified CLI can invoke it.
"""

from __future__ import annotations

from deployment.projects.calibration.cli import add_args
from deployment.projects.calibration.entrypoint import run

# Trigger pipeline factory registration for this project.
from deployment.projects.calibration.pipelines.factory import CalibrationPipelineFactory  # noqa: F401
from deployment.projects.registry import ProjectAdapter, project_registry

project_registry.register(
    ProjectAdapter(
        name="calibration",
        add_args=add_args,
        run=run,
    )
)
