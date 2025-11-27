"""
Typed context objects for deployment workflows.

This module defines typed dataclasses that replace **kwargs with explicit,
type-checked parameters. This improves:
- Type safety: Catches mismatches at type-check time
- Discoverability: IDE autocomplete shows available parameters
- Refactoring safety: Renamed fields are caught by type checkers

Design Principles:
    1. Base contexts define common parameters across all projects
    2. Project-specific contexts extend base with additional fields
    3. Optional fields have sensible defaults
    4. Contexts are immutable (frozen=True) for safety

Usage:
    # Create context for export
    ctx = ExportContext(sample_idx=0)

    # Project-specific context
    ctx = CenterPointExportContext(rot_y_axis_reference=True)

    # Pass to orchestrator
    result = export_orchestrator.run(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExportContext:
    """
    Base context for export operations.

    This context carries parameters needed during the export workflow,
    including model loading and ONNX/TensorRT export settings.

    Attributes:
        sample_idx: Index of sample to use for tracing/shape inference (default: 0)
        extra: Dictionary for project-specific or debug-only options that don't
               warrant a dedicated field. Use sparingly.
    """

    sample_idx: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from extra dict with a default."""
        return self.extra.get(key, default)


@dataclass(frozen=True)
class YOLOXExportContext(ExportContext):
    """
    YOLOX-specific export context.

    Attributes:
        model_cfg_path: Path to model configuration file. If None, attempts
                        to extract from model_cfg.filename.
    """

    model_cfg_path: Optional[str] = None


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


@dataclass(frozen=True)
class CalibrationExportContext(ExportContext):
    """
    Calibration model export context.

    Currently uses only base ExportContext fields.
    Extend with calibration-specific parameters as needed.
    """

    pass


# Type alias for context types
ExportContextType = ExportContext | YOLOXExportContext | CenterPointExportContext | CalibrationExportContext


@dataclass(frozen=True)
class PreprocessContext:
    """
    Context for preprocessing operations.

    This context carries metadata and parameters needed during preprocessing
    in deployment pipelines.

    Attributes:
        img_info: Image metadata dictionary (height, width, scale_factor, etc.)
                  Required for 2D detection pipelines.
        extra: Dictionary for additional preprocessing parameters.
    """

    img_info: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from extra dict with a default."""
        return self.extra.get(key, default)


# Factory functions for convenience
def create_export_context(
    project_type: str = "base",
    sample_idx: int = 0,
    **kwargs: Any,
) -> ExportContext:
    """
    Factory function to create the appropriate export context.

    Args:
        project_type: One of "base", "yolox", "centerpoint", "calibration"
        sample_idx: Sample index for tracing
        **kwargs: Project-specific parameters

    Returns:
        Appropriate ExportContext subclass instance

    Example:
        ctx = create_export_context("yolox", model_cfg_path="/path/to/config.py")
        ctx = create_export_context("centerpoint", rot_y_axis_reference=True)
    """
    project_type = project_type.lower()

    if project_type == "yolox":
        return YOLOXExportContext(
            sample_idx=sample_idx,
            model_cfg_path=kwargs.pop("model_cfg_path", None),
            extra=kwargs,
        )
    elif project_type == "centerpoint":
        return CenterPointExportContext(
            sample_idx=sample_idx,
            rot_y_axis_reference=kwargs.pop("rot_y_axis_reference", False),
            extra=kwargs,
        )
    elif project_type == "calibration":
        return CalibrationExportContext(
            sample_idx=sample_idx,
            extra=kwargs,
        )
    else:
        return ExportContext(
            sample_idx=sample_idx,
            extra=kwargs,
        )
