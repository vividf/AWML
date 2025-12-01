"""
Shared constants for CenterPoint deployment.

This module centralizes default values used across the CenterPoint deployment
codebase. Note that many values (like class_names) should come from the
model config and should raise errors if missing - those are NOT defined here.

Note:
    This module only contains truly optional defaults.
    Export-related constants (file names, component names) are defined in
    deployment/exporters/centerpoint/ to maintain proper dependency direction.
"""

from typing import Tuple

# Default frame ID for evaluation metrics
# This is a reasonable default since most configs use "base_link"
DEFAULT_FRAME_ID: str = "base_link"

# CenterPoint head output names (tied to model architecture)
# These are architectural constants, not dataset-dependent
OUTPUT_NAMES: Tuple[str, ...] = ("heatmap", "reg", "height", "dim", "rot", "vel")

__all__ = [
    "DEFAULT_FRAME_ID",
    "OUTPUT_NAMES",
]
