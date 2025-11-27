"""
Verification orchestration for deployment workflows.

This module handles scenario-based verification across different backends.
"""

import logging
from typing import Any, Dict

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import BaseDeploymentConfig, ExportMode
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.runners.common.artifact_manager import ArtifactManager


class VerificationOrchestrator:
    """
    Orchestrates verification across backends using scenario-based verification.

    This class handles:
    - Running verification scenarios from config
    - Resolving model paths for each scenario
    - Collecting and aggregating verification results
    - Logging verification progress and results
    """

    def __init__(
        self,
        config: BaseDeploymentConfig,
        evaluator: BaseEvaluator,
        data_loader: BaseDataLoader,
        logger: logging.Logger,
    ):
        """
        Initialize verification orchestrator.

        Args:
            config: Deployment configuration
            evaluator: Evaluator instance for running verification
            data_loader: Data loader for loading samples
            logger: Logger instance
        """
        self.config = config
        self.evaluator = evaluator
        self.data_loader = data_loader
        self.logger = logger

    def run(
        self,
        artifact_manager: ArtifactManager,
        pytorch_checkpoint: str = None,
        onnx_path: str = None,
        tensorrt_path: str = None,
    ) -> Dict[str, Any]:
        """
        Run verification on exported models using policy-based verification.

        Args:
            artifact_manager: Artifact manager for resolving model paths
            pytorch_checkpoint: Path to PyTorch checkpoint (optional)
            onnx_path: Path to ONNX model file/directory (optional)
            tensorrt_path: Path to TensorRT engine file/directory (optional)

        Returns:
            Verification results dictionary
        """
        verification_cfg = self.config.verification_config

        # Check master switch
        if not verification_cfg.enabled:
            self.logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            self.logger.info(f"No verification scenarios for export mode '{export_mode.value}', skipping...")
            return {}

        # Check if PyTorch checkpoint is needed
        needs_pytorch = any(
            policy.ref_backend is Backend.PYTORCH or policy.test_backend is Backend.PYTORCH for policy in scenarios
        )

        if needs_pytorch and not pytorch_checkpoint:
            self.logger.warning(
                "PyTorch checkpoint path not available, but required by verification scenarios. "
                "Skipping verification."
            )
            return {}

        num_verify_samples = verification_cfg.num_verify_samples
        tolerance = verification_cfg.tolerance
        devices_map = verification_cfg.devices or {}

        self.logger.info("=" * 80)
        self.logger.info(f"Running Verification (mode: {export_mode.value})")
        self.logger.info("=" * 80)

        all_results = {}
        total_passed = 0
        total_failed = 0

        for i, policy in enumerate(scenarios):
            # Resolve devices using alias system
            ref_device = self._resolve_device(policy.ref_device, devices_map)
            test_device = self._resolve_device(policy.test_device, devices_map)

            self.logger.info(
                f"\nScenario {i+1}/{len(scenarios)}: "
                f"{policy.ref_backend.value}({ref_device}) vs {policy.test_backend.value}({test_device})"
            )

            # Resolve model paths based on backend
            ref_path = self._resolve_backend_path(policy.ref_backend, pytorch_checkpoint, onnx_path, tensorrt_path)
            test_path = self._resolve_backend_path(policy.test_backend, pytorch_checkpoint, onnx_path, tensorrt_path)

            if not ref_path or not test_path:
                self.logger.warning(f"  Skipping: missing paths (ref={ref_path}, test={test_path})")
                continue

            # Create artifacts and model specs
            ref_artifact = self._create_artifact(policy.ref_backend, ref_path)
            test_artifact = self._create_artifact(policy.test_backend, test_path)

            reference_spec = ModelSpec(backend=policy.ref_backend, device=ref_device, artifact=ref_artifact)

            test_spec = ModelSpec(backend=policy.test_backend, device=test_device, artifact=test_artifact)

            # Run verification
            verification_results = self.evaluator.verify(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
                verbose=False,
            )

            # Store results
            policy_key = f"{policy.ref_backend.value}_{ref_device}_vs_{policy.test_backend.value}_{test_device}"
            all_results[policy_key] = verification_results

            # Update counters
            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed

                if failed == 0:
                    self.logger.info(f"Scenario {i+1} passed ({passed} comparisons)")
                else:
                    self.logger.warning(f"Scenario {i+1} failed ({failed}/{passed+failed} comparisons)")

        # Overall summary
        self.logger.info("\n" + "=" * 80)
        if total_failed == 0:
            self.logger.info(f"All verifications passed! ({total_passed} total)")
        else:
            self.logger.warning(f"{total_failed}/{total_passed + total_failed} verifications failed")
        self.logger.info("=" * 80)

        all_results["summary"] = {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_passed + total_failed,
        }

        return all_results

    def _resolve_device(self, device_key: str, devices_map: Dict[str, str]) -> str:
        """
        Resolve device using alias system.

        Args:
            device_key: Device key from scenario
            devices_map: Device alias mapping

        Returns:
            Actual device string
        """
        if device_key in devices_map:
            return devices_map[device_key]
        else:
            # Fallback: use the key directly
            self.logger.warning(f"Device alias '{device_key}' not found in devices map, using as-is")
            return device_key

    def _resolve_backend_path(
        self, backend: Backend, pytorch_checkpoint: str, onnx_path: str, tensorrt_path: str
    ) -> str:
        """
        Resolve model path for a backend.

        Args:
            backend: Backend identifier
            pytorch_checkpoint: PyTorch checkpoint path
            onnx_path: ONNX model path
            tensorrt_path: TensorRT engine path

        Returns:
            Model path for the backend, or None if not available
        """
        if backend == Backend.PYTORCH:
            return pytorch_checkpoint
        elif backend == Backend.ONNX:
            return onnx_path
        elif backend == Backend.TENSORRT:
            return tensorrt_path
        else:
            self.logger.warning(f"Unknown backend: {backend}")
            return None

    def _create_artifact(self, backend: Backend, path: str) -> Artifact:
        """
        Create artifact from path.

        Args:
            backend: Backend identifier
            path: Model path

        Returns:
            Artifact instance
        """
        import os

        multi_file = os.path.isdir(path) if path and os.path.exists(path) else False
        return Artifact(path=path, multi_file=multi_file)
