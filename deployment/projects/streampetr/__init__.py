"""StreamPETR deployment bundle.

Import concrete modules (``deployment.projects.streampetr.runner``, …). This ``__init__`` only
registers the project with ``deployment.projects.registry`` when the package is imported.
"""

from __future__ import annotations

from deployment.projects.registry import ProjectAdapter, project_registry
from deployment.projects.streampetr.entrypoint import run

project_registry.register(
    ProjectAdapter(
        name="streampetr",
        run=run,
    )
)
