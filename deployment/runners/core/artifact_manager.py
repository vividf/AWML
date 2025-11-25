"""
Artifact management for deployment workflows.

This module handles registration and resolution of model artifacts (PyTorch checkpoints,
ONNX models, TensorRT engines) across different backends.
"""

import logging
import os
from typing import Dict, Optional, Tuple

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.base_config import BaseDeploymentConfig


class ArtifactManager:
    """
    Manages model artifacts and path resolution for deployment workflows.

    This class centralizes all logic for:
    - Registering artifacts after export
    - Resolving artifact paths from configuration
    - Validating artifact existence
    - Looking up artifacts by backend
    """

    def __init__(self, config: BaseDeploymentConfig, logger: logging.Logger):
        """
        Initialize artifact manager.

        Args:
            config: Deployment configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.artifacts: Dict[str, Artifact] = {}

    def register_artifact(self, backend: Backend, artifact: Artifact) -> None:
        """
        Register an artifact for a backend.

        Args:
            backend: Backend identifier
            artifact: Artifact to register
        """
        self.artifacts[backend.value] = artifact
        self.logger.debug(f"Registered {backend.value} artifact: {artifact.path}")

    def get_artifact(self, backend: Backend) -> Optional[Artifact]:
        """
        Get registered artifact for a backend.

        Args:
            backend: Backend identifier

        Returns:
            Artifact if found, None otherwise
        """
        return self.artifacts.get(backend.value)

    def resolve_pytorch_artifact(self, backend_cfg: Dict) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve PyTorch model path from backend config or registered artifacts.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an Artifact instance if a path could be resolved, otherwise None.
            is_valid indicates whether the artifact exists on disk.
        """
        # Check registered artifacts first
        artifact = self.artifacts.get(Backend.PYTORCH.value)
        if artifact:
            return artifact, artifact.exists()

        # Fallback to checkpoint path from config
        model_path = backend_cfg.get("checkpoint") or self.config.export_config.checkpoint_path
        if not model_path:
            return None, False

        artifact = Artifact(path=model_path, multi_file=False)
        return artifact, artifact.exists()

    def resolve_onnx_artifact(self, backend_cfg: Dict) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve ONNX model path from backend config or registered artifacts.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an Artifact instance if a path could be resolved, otherwise None.
            is_valid indicates whether the artifact exists on disk.
        """
        # Check registered artifacts first
        artifact = self.artifacts.get(Backend.ONNX.value)
        if artifact:
            return artifact, artifact.exists()

        # Fallback to explicit path from config
        explicit_path = backend_cfg.get("model_dir") or self.config.export_config.onnx_path
        if explicit_path:
            is_dir = os.path.isdir(explicit_path) if os.path.exists(explicit_path) else False
            fallback_artifact = Artifact(path=explicit_path, multi_file=is_dir)
            return fallback_artifact, fallback_artifact.exists()

        return None, False

    def resolve_tensorrt_artifact(self, backend_cfg: Dict) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve TensorRT model path from backend config or registered artifacts.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an Artifact instance if a path could be resolved, otherwise None.
            is_valid indicates whether the artifact exists on disk.
        """
        # Check registered artifacts first
        artifact = self.artifacts.get(Backend.TENSORRT.value)
        if artifact:
            return artifact, artifact.exists()

        # Fallback to explicit path from config
        explicit_path = backend_cfg.get("engine_dir") or self.config.export_config.tensorrt_path
        if explicit_path:
            is_dir = os.path.isdir(explicit_path) if os.path.exists(explicit_path) else False
            fallback_artifact = Artifact(path=explicit_path, multi_file=is_dir)
            return fallback_artifact, fallback_artifact.exists()

        return None, False

    def resolve_artifact(self, backend: Backend, backend_cfg: Dict) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve artifact for any backend.

        This is a convenience method that delegates to backend-specific resolvers.

        Args:
            backend: Backend identifier
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid)
        """
        if backend == Backend.PYTORCH:
            return self.resolve_pytorch_artifact(backend_cfg)
        elif backend == Backend.ONNX:
            return self.resolve_onnx_artifact(backend_cfg)
        elif backend == Backend.TENSORRT:
            return self.resolve_tensorrt_artifact(backend_cfg)
        else:
            self.logger.warning(f"Unknown backend: {backend}")
            return None, False
