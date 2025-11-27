"""
Typed runtime configurations for different task types.

This module provides task-specific typed configurations for runtime_io,
replacing the weakly-typed Dict[str, Any] access pattern.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BaseRuntimeConfig:
    """Base configuration for all runtime settings."""

    sample_idx: int = 0


@dataclass(frozen=True)
class Detection3DRuntimeConfig(BaseRuntimeConfig):
    """Runtime configuration for 3D detection tasks (e.g., CenterPoint)."""

    info_file: str = ""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Detection3DRuntimeConfig":
        """Create config from dictionary."""
        return cls(
            sample_idx=config_dict.get("sample_idx", 0),
            info_file=config_dict.get("info_file", ""),
        )


@dataclass(frozen=True)
class Detection2DRuntimeConfig(BaseRuntimeConfig):
    """Runtime configuration for 2D detection tasks (e.g., YOLOX)."""

    ann_file: str = ""
    img_prefix: str = ""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Detection2DRuntimeConfig":
        """Create config from dictionary."""
        return cls(
            sample_idx=config_dict.get("sample_idx", 0),
            ann_file=config_dict.get("ann_file", ""),
            img_prefix=config_dict.get("img_prefix", ""),
        )


@dataclass(frozen=True)
class ClassificationRuntimeConfig(BaseRuntimeConfig):
    """Runtime configuration for classification tasks."""

    info_pkl: str = ""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ClassificationRuntimeConfig":
        """Create config from dictionary."""
        return cls(
            sample_idx=config_dict.get("sample_idx", 0),
            info_pkl=config_dict.get("info_pkl", ""),
        )
