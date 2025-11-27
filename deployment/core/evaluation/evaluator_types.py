"""
Type definitions for model evaluation in deployment.

This module contains the shared type definitions used by evaluators,
runners, and orchestrators.
"""

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend


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
