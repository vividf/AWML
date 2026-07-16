"""Calibration Status Classification deployment bundle (binary image classification).

Import concrete modules (``deployment.projects.calibration.runner``, …). This ``__init__`` only
registers the project with ``deployment.projects.registry`` when the package is imported.
"""

from __future__ import annotations

from deployment.projects.calibration.entrypoint import run
from deployment.projects.registry import ProjectAdapter, project_registry

# Class names live in the deploy config (the model config records only num_classes), so the CLI
# carries no calibration-specific flags.
project_registry.register(
    ProjectAdapter(
        name="calibration",
        run=run,
    )
)
