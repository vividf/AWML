"""
Typed context objects for deployment workflows.

Usage:
    # Create context for export
    ctx = ExportContext()

    # Project-specific context
    ctx = CenterPointExportContext(rot_y_axis_reference=True)

    # Pass to orchestrator
    result = export_orchestrator.run(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportContext:
    """
    Base context for export operations.

    Marker base class for export contexts; project-specific subclasses (e.g.
    ``CenterPointExportContext``) add typed fields for their export parameters.
    """


@dataclass(frozen=True)
class CenterPointExportContext(ExportContext):
    """
    CenterPoint-specific export context.

    Attributes:
        rot_y_axis_reference: Whether to use y-axis rotation reference for
                              ONNX-compatible output format. This affects
                              how rotation and dimensions are encoded.
    """

    rot_y_axis_reference: bool = False
