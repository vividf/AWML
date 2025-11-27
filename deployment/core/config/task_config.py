"""
Task configuration for unified pipeline configuration.

This module provides task-specific configuration that can be passed to pipelines,
enabling a more unified approach while still supporting task-specific parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TaskType(str, Enum):
    """Supported task types."""

    CLASSIFICATION = "classification"
    DETECTION_2D = "detection2d"
    DETECTION_3D = "detection3d"

    @classmethod
    def from_value(cls, value: str) -> "TaskType":
        """Parse string to TaskType."""
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Invalid task type '{value}'. Must be one of {[m.value for m in cls]}.")


@dataclass(frozen=True)
class TaskConfig:
    """
    Task-specific configuration for deployment pipelines (immutable).

    This configuration encapsulates all task-specific parameters needed
    by pipelines, enabling a more unified approach while still supporting
    task-specific requirements.
    """

    task_type: TaskType
    num_classes: int
    class_names: Tuple[str, ...]

    # 2D Detection specific
    input_size: Optional[Tuple[int, int]] = None

    # 3D Detection specific
    point_cloud_range: Optional[Tuple[float, ...]] = None
    voxel_size: Optional[Tuple[float, ...]] = None

    # Optional additional parameters
    score_threshold: float = 0.01
    nms_threshold: float = 0.65
    max_detections: int = 300

    @classmethod
    def for_classification(
        cls,
        num_classes: int,
        class_names: List[str],
    ) -> "TaskConfig":
        """Create configuration for classification tasks."""
        return cls(
            task_type=TaskType.CLASSIFICATION,
            num_classes=num_classes,
            class_names=tuple(class_names),
        )

    @classmethod
    def for_detection_2d(
        cls,
        num_classes: int,
        class_names: List[str],
        input_size: Tuple[int, int] = (960, 960),
        score_threshold: float = 0.01,
        nms_threshold: float = 0.65,
        max_detections: int = 300,
    ) -> "TaskConfig":
        """Create configuration for 2D detection tasks."""
        return cls(
            task_type=TaskType.DETECTION_2D,
            num_classes=num_classes,
            class_names=tuple(class_names),
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )

    @classmethod
    def for_detection_3d(
        cls,
        num_classes: int,
        class_names: List[str],
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
    ) -> "TaskConfig":
        """Create configuration for 3D detection tasks."""
        return cls(
            task_type=TaskType.DETECTION_3D,
            num_classes=num_classes,
            class_names=tuple(class_names),
            point_cloud_range=tuple(point_cloud_range) if point_cloud_range else None,
            voxel_size=tuple(voxel_size) if voxel_size else None,
        )
