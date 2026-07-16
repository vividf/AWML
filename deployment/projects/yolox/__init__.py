"""YOLOX deployment bundle (2D object detection).

Import concrete modules (``deployment.projects.yolox.runner``, …). This ``__init__`` only registers
the project with ``deployment.projects.registry`` when the package is imported.
"""

from __future__ import annotations

from deployment.projects.registry import ProjectAdapter, project_registry
from deployment.projects.yolox.entrypoint import run

# Everything that shapes the exported artifact (classes, thresholds, input size) is read from the
# model config, so the CLI carries no YOLOX-specific flags.
project_registry.register(
    ProjectAdapter(
        name="yolox",
        run=run,
    )
)
