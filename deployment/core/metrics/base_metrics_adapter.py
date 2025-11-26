"""
Base Metrics Adapter for unified metric computation.

This module provides the abstract base class that all task-specific metrics adapters
must implement. It ensures a consistent interface across 3D detection, 2D detection,
and classification tasks.

All metric adapters use autoware_perception_evaluation as the underlying computation
engine to ensure consistency between training (T4MetricV2) and deployment evaluation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BaseMetricsConfig:
    """Base configuration for all metrics adapters.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation (e.g., "base_link" for 3D, "camera" for 2D).
    """

    class_names: List[str]
    frame_id: str = "base_link"


class BaseMetricsAdapter(ABC):
    """
    Abstract base class for all task-specific metrics adapters.

    This class defines the common interface that all metric adapters must implement.
    Each adapter wraps autoware_perception_evaluation to compute metrics consistent
    with training evaluation (T4MetricV2).

    The workflow is:
        1. Create adapter with task-specific config
        2. Call reset() to start a new evaluation session
        3. Call add_frame() for each sample
        4. Call compute_metrics() to get final metrics
        5. Optionally call get_summary() for a human-readable summary

    Example:
        adapter = SomeMetricsAdapter(config)
        adapter.reset()
        for pred, gt in data:
            adapter.add_frame(pred, gt)
        metrics = adapter.compute_metrics()
    """

    def __init__(self, config: BaseMetricsConfig):
        """
        Initialize the metrics adapter.

        Args:
            config: Configuration for the metrics adapter.
        """
        self.config = config
        self.class_names = config.class_names
        self.frame_id = config.frame_id
        self._frame_count = 0

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the adapter for a new evaluation session.

        This method should clear all accumulated frame data and reinitialize
        the underlying evaluator.
        """
        pass

    @abstractmethod
    def add_frame(self, *args, **kwargs) -> None:
        """
        Add a frame of predictions and ground truths for evaluation.

        The specific arguments depend on the task type:
        - 3D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - 2D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - Classification: prediction: int, ground_truth: int, probabilities: List[float]
        """
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute metrics from all added frames.

        Returns:
            Dictionary of metric names to values.
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation including primary metrics.

        Returns:
            Dictionary with summary metrics and additional information.
        """
        pass

    @property
    def frame_count(self) -> int:
        """Return the number of frames added so far."""
        return self._frame_count
