"""
Base workflow interfaces for specialized export flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from deployment.core.artifacts import Artifact
from deployment.core.base_config import BaseDeploymentConfig
from deployment.core.base_data_loader import BaseDataLoader


class OnnxExportWorkflow(ABC):
    """
    Base interface for ONNX export workflows.
    """

    @abstractmethod
    def export(
        self,
        *,
        model: Any,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
        **kwargs: Any,
    ) -> Artifact:
        """
        Execute the ONNX export workflow and return the produced artifact.
        """


class TensorRTExportWorkflow(ABC):
    """
    Base interface for TensorRT export workflows.
    """

    @abstractmethod
    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
        data_loader: BaseDataLoader,
        **kwargs: Any,
    ) -> Artifact:
        """
        Execute the TensorRT export workflow and return the produced artifact.
        """
