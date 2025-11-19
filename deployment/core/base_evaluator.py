"""
Abstract base class for model evaluation in deployment.

Each task (classification, detection, segmentation, etc.) must implement
a concrete Evaluator that extends this base class to compute task-specific metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, TypedDict

import numpy as np

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.base_data_loader import BaseDataLoader


class EvalResultDict(TypedDict, total=False):
    """
    Structured evaluation result used across deployments.

    Attributes:
        primary_metric: Main scalar metric for quick ranking (e.g., accuracy, mAP).
        metrics: Flat dictionary of additional scalar metrics.
        per_class: Optional nested metrics keyed by class/label name.
        latency: Latency statistics as returned by compute_latency_stats().
        metadata: Arbitrary metadata that downstream components might need.
    """

    primary_metric: float
    metrics: Dict[str, float]
    per_class: Dict[str, Any]
    latency: Dict[str, float]
    metadata: Dict[str, Any]


class VerifyResultDict(TypedDict, total=False):
    """
    Structured verification outcome shared between runners and evaluators.

    Attributes:
        summary: Aggregate pass/fail counts.
        samples: Mapping of sample identifiers to boolean pass/fail states.
    """

    summary: Dict[str, int]
    samples: Dict[str, bool]
    error: str


@dataclass(frozen=True)
class ModelSpec:
    """
    Minimal description of a concrete model artifact to evaluate or verify.

    Attributes:
        backend: Backend identifier such as 'pytorch', 'onnx', or 'tensorrt'.
        device: Target device string (e.g., 'cpu', 'cuda:0').
        artifact: Filesystem representation of the produced model.
    """

    backend: Backend
    device: str
    artifact: Artifact

    @property
    def path(self) -> str:
        """Backward-compatible access to artifact path."""
        return self.artifact.path

    @property
    def multi_file(self) -> bool:
        """True if the artifact represents a multi-file bundle."""
        return self.artifact.multi_file


class BaseEvaluator(ABC):
    """
    Abstract base class for task-specific evaluators.

    This class defines the interface that all task-specific evaluators
    must implement. It handles running inference on a dataset and computing
    evaluation metrics appropriate for the task.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        model: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Run full evaluation on a model.

        Args:
            model: Specification of the artifact/backend/device triplet to evaluate
            data_loader: DataLoader for loading samples
            num_samples: Number of samples to evaluate
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing evaluation metrics. The exact metrics
            depend on the task, but should include:
            - Primary metric(s) for the task
            - Per-class metrics (if applicable)
            - Inference latency statistics
            - Any other relevant metrics

        Example:
            For classification:
            {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "per_class_accuracy": {...},
                "confusion_matrix": [...],
                "avg_latency_ms": 5.2,
            }

            For detection:
            {
                "mAP": 0.72,
                "mAP_50": 0.85,
                "mAP_75": 0.68,
                "per_class_ap": {...},
                "avg_latency_ms": 15.3,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def print_results(self, results: EvalResultDict) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary returned by evaluate()
        """
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> VerifyResultDict:
        """
        Verify exported models using scenario-based verification.

        This method compares outputs from a reference backend against a test backend
        as specified by the verification scenarios. This is a more flexible approach
        than the legacy verify() method which compares all available backends.

        Args:
            reference: Specification of backend/device/path for the reference model
            test: Specification for the backend/device/path under test
            data_loader: Data loader for test samples
            num_samples: Number of samples to verify
            tolerance: Maximum allowed difference for verification to pass
            verbose: Whether to print detailed output

        Returns:
            Verification results with pass/fail summary and per-sample outcomes.
        """
        raise NotImplementedError

    def compute_latency_stats(self, latencies: list) -> Dict[str, float]:
        """
        Compute latency statistics from a list of latency measurements.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Dictionary with latency statistics
        """
        if not latencies:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "median_ms": 0.0,
            }

        latencies_array = np.array(latencies)

        return {
            "mean_ms": float(np.mean(latencies_array)),
            "std_ms": float(np.std(latencies_array)),
            "min_ms": float(np.min(latencies_array)),
            "max_ms": float(np.max(latencies_array)),
            "median_ms": float(np.median(latencies_array)),
        }

    def format_latency_stats(self, stats: Dict[str, float]) -> str:
        """
        Format latency statistics as a readable string.

        Args:
            stats: Latency statistics dictionary

        Returns:
            Formatted string
        """
        return (
            f"Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms "
            f"(min: {stats['min_ms']:.2f}, max: {stats['max_ms']:.2f}, "
            f"median: {stats['median_ms']:.2f})"
        )
