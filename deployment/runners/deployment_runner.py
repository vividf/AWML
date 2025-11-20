"""
Unified deployment runner for common deployment workflows.

This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import torch
from mmengine.config import Config

from deployment.core import Artifact, Backend, BaseDataLoader, BaseDeploymentConfig, BaseEvaluator, ModelSpec


class DeploymentResultDict(TypedDict, total=False):
    """
    Standardized structure returned by `BaseDeploymentRunner.run()`.

    Keys:
        pytorch_model: In-memory model instance loaded from the checkpoint (if requested).
        onnx_path: Filesystem path to the exported ONNX artifact (single file or directory).
        tensorrt_path: Filesystem path to the exported TensorRT engine.
        verification_results: Arbitrary dictionary produced by `BaseEvaluator.verify()`.
        evaluation_results: Arbitrary dictionary produced by `BaseEvaluator.evaluate()`.
    """

    pytorch_model: Optional[Any]
    onnx_path: Optional[str]
    tensorrt_path: Optional[str]
    verification_results: Dict[str, Any]
    evaluation_results: Dict[str, Any]


class BaseDeploymentRunner:
    """
    Base deployment runner for common deployment workflows.

    This runner handles the standard deployment workflow:
    1. Load PyTorch model (if needed)
    2. Export to ONNX (if requested)
    3. Export to TensorRT (if requested)
    4. Verify outputs (if enabled)
    5. Evaluate models (if enabled)

    Projects should extend this class and override methods as needed:
    - Override export_onnx() for project-specific ONNX export logic
    - Override export_tensorrt() for project-specific TensorRT export logic
    - Override load_pytorch_model() for project-specific model loading
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_exporter: Any = None,
        tensorrt_exporter: Any = None,
    ):
        """
        Initialize base deployment runner.

        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            onnx_exporter: Required ONNX exporter instance (e.g., ONNXExporter with project wrapper)
            tensorrt_exporter: Required TensorRT exporter instance (e.g., TensorRTExporter with project wrapper)

        Raises:
            ValueError: If onnx_exporter or tensorrt_exporter is None
        """
        # Validate required exporters
        if onnx_exporter is None:
            raise ValueError("onnx_exporter is required and cannot be None")
        if tensorrt_exporter is None:
            raise ValueError("tensorrt_exporter is required and cannot be None")

        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger
        self._onnx_exporter = onnx_exporter
        self._tensorrt_exporter = tensorrt_exporter
        self.artifacts: Dict[str, Artifact] = {}

    @staticmethod
    def _get_backend_entry(mapping: Optional[Dict[Any, Any]], backend: Backend) -> Any:
        """
        Fetch a config value that may be keyed by either string literals or Backend enums.
        """
        if not mapping:
            return None

        if backend.value in mapping:
            return mapping[backend.value]

        return mapping.get(backend)

    def load_pytorch_model(self, checkpoint_path: str, **kwargs) -> Any:
        """
        Load PyTorch model from checkpoint.

        Subclasses must implement this method to provide project-specific model loading logic.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional project-specific arguments

        Returns:
            Loaded PyTorch model

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__}.load_pytorch_model() must be implemented by subclasses.")

    def export_onnx(self, pytorch_model: Any, **kwargs) -> Optional[Artifact]:
        """
        Export model to ONNX format.

        Uses the provided ONNX exporter instance.

        Args:
            pytorch_model: PyTorch model to export
            **kwargs: Additional project-specific arguments

        Returns:
            Artifact describing the exported ONNX output, or None if skipped
        """
        if not self.config.export_config.should_export_onnx():
            return None

        onnx_settings = self.config.get_onnx_settings()

        exporter = self._onnx_exporter
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to ONNX (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        # Save to work_dir/onnx/ directory
        onnx_dir = os.path.join(self.config.export_config.work_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        output_path = os.path.join(onnx_dir, onnx_settings.save_file)

        # Get sample input
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        single_input = self.data_loader.preprocess(sample)

        # Get batch size from configuration
        batch_size = onnx_settings.batch_size
        if batch_size is None:
            input_tensor = single_input
            self.logger.info("Using dynamic batch size")
        else:
            # Handle different input shapes
            if isinstance(single_input, (list, tuple)):
                # Multiple inputs
                input_tensor = tuple(
                    inp.repeat(batch_size, *([1] * (len(inp.shape) - 1))) if len(inp.shape) > 0 else inp
                    for inp in single_input
                )
            else:
                # Single input
                input_tensor = single_input.repeat(batch_size, *([1] * (len(single_input.shape) - 1)))
            self.logger.info(f"Using fixed batch size: {batch_size}")

        try:
            exporter.export(pytorch_model, input_tensor, output_path)
        except Exception:
            self.logger.error("ONNX export failed")
            raise

        multi_file = bool(self.config.onnx_config.get("multi_file", False))
        artifact_path = onnx_dir if multi_file else output_path
        artifact = Artifact(path=artifact_path, multi_file=multi_file)
        self.artifacts[Backend.ONNX.value] = artifact
        self.logger.info(f"ONNX export successful: {artifact.path}")
        return artifact

    def export_tensorrt(self, onnx_path: str, **kwargs) -> Optional[Artifact]:
        """
        Export ONNX model to TensorRT engine.

        Uses the provided TensorRT exporter instance.

        Args:
            onnx_path: Path to ONNX model file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Artifact describing the exported TensorRT output, or None if skipped
        """
        if not self.config.export_config.should_export_tensorrt():
            return None

        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None

        exporter = self._tensorrt_exporter
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to TensorRT (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        # Save to work_dir/tensorrt/ directory
        tensorrt_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
        os.makedirs(tensorrt_dir, exist_ok=True)

        # Determine output path based on ONNX file name
        if os.path.isdir(onnx_path):
            output_path = os.path.join(tensorrt_dir, "model.engine")
        else:
            onnx_filename = os.path.basename(onnx_path)
            engine_filename = onnx_filename.replace(".onnx", ".engine")
            output_path = os.path.join(tensorrt_dir, engine_filename)

        # Set CUDA device for TensorRT exportd
        cuda_device = self.config.export_config.cuda_device
        device_id = self.config.export_config.get_cuda_device_index()
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device for TensorRT export: {cuda_device}")

        # Get sample input for shape configuration
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        sample_input = self.data_loader.preprocess(sample)

        if isinstance(sample_input, (list, tuple)):
            sample_input = sample_input[0]  # Use first input for shape

        exporter = self._tensorrt_exporter

        try:
            exporter.export(
                model=None,
                sample_input=sample_input,
                output_path=output_path,
                onnx_path=onnx_path,
            )
        except Exception:
            self.logger.error("TensorRT export failed")
            raise

        artifact = Artifact(path=output_path, multi_file=False)
        self.artifacts[Backend.TENSORRT.value] = artifact
        self.logger.info(f"TensorRT export successful: {artifact.path}")
        return artifact

    def _resolve_pytorch_artifact(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve PyTorch model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an `Artifact` instance if a path could be resolved, otherwise None.
        """
        artifact = self.artifacts.get(Backend.PYTORCH.value)
        if artifact:
            return artifact, artifact.exists()

        model_path = backend_cfg.get("checkpoint") or self.config.export_config.checkpoint_path
        if not model_path:
            return None, False

        artifact = Artifact(path=model_path, multi_file=False)
        return artifact, artifact.exists()

    def _artifact_from_path(self, backend: Union[str, Backend], path: str) -> Artifact:
        backend_enum = Backend.from_value(backend)
        existing = self.artifacts.get(backend_enum.value)
        if existing and existing.path == path:
            return existing

        multi_file = os.path.isdir(path) if path and os.path.exists(path) else False
        return Artifact(path=path, multi_file=multi_file)

    def _build_model_spec(self, backend: Union[str, Backend], artifact: Artifact, device: str) -> ModelSpec:
        backend_enum = Backend.from_value(backend)
        return ModelSpec(
            backend=backend_enum,
            device=device,
            artifact=artifact,
        )

    def _normalize_device_for_backend(self, backend: Union[str, Backend], device: Optional[str]) -> str:
        backend_enum = Backend.from_value(backend)
        normalized_device = str(device or "cpu")

        if backend_enum in (Backend.PYTORCH, Backend.ONNX):
            if normalized_device not in ("cpu",) and not normalized_device.startswith("cuda"):
                self.logger.warning(
                    f"Unsupported device '{normalized_device}' for backend '{backend_enum.value}'. Falling back to CPU."
                )
                normalized_device = "cpu"
        elif backend_enum is Backend.TENSORRT:
            if not normalized_device or normalized_device == "cpu":
                normalized_device = self.config.export_config.cuda_device or "cuda:0"
            if not normalized_device.startswith("cuda"):
                self.logger.warning(
                    "TensorRT evaluation requires CUDA device. Overriding device "
                    f"from '{normalized_device}' to 'cuda:0'."
                )
                normalized_device = "cuda:0"

        return normalized_device

    def _resolve_onnx_artifact(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve ONNX model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an `Artifact` instance if a path could be resolved, otherwise None.
        """
        artifact = self.artifacts.get(Backend.ONNX.value)
        if artifact:
            return artifact, artifact.exists()

        explicit_path = backend_cfg.get("model_dir") or self.config.export_config.onnx_path
        if explicit_path:
            fallback_artifact = Artifact(path=explicit_path, multi_file=os.path.isdir(explicit_path))
            return fallback_artifact, fallback_artifact.exists()

        return None, False

    def _resolve_tensorrt_artifact(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve TensorRT model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an `Artifact` instance if a path could be resolved, otherwise None.
        """
        artifact = self.artifacts.get(Backend.TENSORRT.value)
        if artifact:
            return artifact, artifact.exists()

        explicit_path = backend_cfg.get("engine_dir") or self.config.export_config.tensorrt_path
        if explicit_path:
            fallback_artifact = Artifact(path=explicit_path, multi_file=os.path.isdir(explicit_path))
            return fallback_artifact, fallback_artifact.exists()

        return None, False

    def get_models_to_evaluate(self) -> List[ModelSpec]:
        """
        Get list of models to evaluate from config.

        Returns:
            List of `ModelSpec` instances describing models to evaluate.
        """
        backends = self.config.get_evaluation_backends()
        models_to_evaluate: List[ModelSpec] = []

        for backend_key, backend_cfg in backends.items():
            backend_enum = Backend.from_value(backend_key)
            if not backend_cfg.get("enabled", False):
                continue

            device = str(backend_cfg.get("device", "cpu") or "cpu")
            artifact: Optional[Artifact] = None
            is_valid = False

            if backend_enum is Backend.PYTORCH:
                artifact, is_valid = self._resolve_pytorch_artifact(backend_cfg)
            elif backend_enum is Backend.ONNX:
                artifact, is_valid = self._resolve_onnx_artifact(backend_cfg)
            elif backend_enum is Backend.TENSORRT:
                artifact, is_valid = self._resolve_tensorrt_artifact(backend_cfg)

            if is_valid and artifact:
                spec = self._build_model_spec(backend_enum, artifact, device)
                models_to_evaluate.append(spec)
                self.logger.info(f"  - {backend_enum.value}: {artifact.path} (device: {device})")
            elif artifact is not None:
                self.logger.warning(f"  - {backend_enum.value}: {artifact.path} (not found or invalid, skipping)")

        return models_to_evaluate

    def run_verification(
        self, pytorch_checkpoint: Optional[str], onnx_path: Optional[str], tensorrt_path: Optional[str], **kwargs
    ) -> Dict[str, Any]:
        """
        Run verification on exported models using policy-based verification.

        Args:
            pytorch_checkpoint: Path to PyTorch checkpoint (reference)
            onnx_path: Path to ONNX model file/directory
            tensorrt_path: Path to TensorRT engine file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Verification results dictionary
        """
        verification_cfg = self.config.verification_config

        # Check master switches
        if not verification_cfg.enabled:
            self.logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            self.logger.info(f"No verification scenarios for export mode '{export_mode.value}', skipping...")
            return {}

        # Check if any scenario actually needs PyTorch checkpoint
        needs_pytorch = any(
            policy.ref_backend is Backend.PYTORCH or policy.test_backend is Backend.PYTORCH for policy in scenarios
        )

        if needs_pytorch and not pytorch_checkpoint:
            self.logger.warning(
                "PyTorch checkpoint path not available, but required by verification scenarios. Skipping verification."
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
            ref_backend = policy.ref_backend
            # Resolve device using alias system:
            # - Scenarios use aliases (e.g., "cpu", "cuda") for flexibility
            # - Actual device strings are defined in verification["devices"]
            # - This allows easy device switching: change devices["cpu"] to affect all CPU verifications
            ref_device_key = policy.ref_device
            if ref_device_key in devices_map:
                ref_device = devices_map[ref_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                ref_device = ref_device_key
                self.logger.warning(f"Device alias '{ref_device_key}' not found in devices map, using as-is")

            test_backend = policy.test_backend
            test_device_key = policy.test_device
            if test_device_key in devices_map:
                test_device = devices_map[test_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                test_device = test_device_key
                self.logger.warning(f"Device alias '{test_device_key}' not found in devices map, using as-is")

            self.logger.info(
                f"\nScenarios {i+1}/{len(scenarios)}: "
                f"{ref_backend.value}({ref_device}) vs {test_backend.value}({test_device})"
            )

            # Resolve model paths based on backend
            ref_path = None
            test_path = None

            if ref_backend is Backend.PYTORCH:
                ref_path = pytorch_checkpoint
            elif ref_backend is Backend.ONNX:
                ref_path = onnx_path
            elif ref_backend is Backend.TENSORRT:
                ref_path = tensorrt_path

            if test_backend is Backend.ONNX:
                test_path = onnx_path
            elif test_backend is Backend.TENSORRT:
                test_path = tensorrt_path
            elif test_backend is Backend.PYTORCH:
                test_path = pytorch_checkpoint

            if not ref_path or not test_path:
                self.logger.warning(f"  Skipping: missing paths (ref={ref_path}, test={test_path})")
                continue

            ref_artifact = self._artifact_from_path(ref_backend, ref_path)
            test_artifact = self._artifact_from_path(test_backend, test_path)

            # Use policy-based verification interface
            reference_spec = self._build_model_spec(ref_backend, ref_artifact, ref_device)
            test_spec = self._build_model_spec(test_backend, test_artifact, test_device)

            verification_results = self.evaluator.verify(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
                verbose=False,
            )

            # Extract results for this specific comparison
            policy_key = f"{ref_backend.value}_{ref_device}_vs_{test_backend.value}_{test_device}"
            all_results[policy_key] = verification_results

            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed

                if failed == 0:
                    self.logger.info(f"Policy {i+1} passed ({passed} comparisons)")
                else:
                    self.logger.warning(f"Policy {i+1} failed ({failed}/{passed+failed} comparisons)")

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

    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on specified models.

        Args:
            **kwargs: Additional project-specific arguments

        Returns:
            Dictionary containing evaluation results for all backends
        """
        eval_config = self.config.evaluation_config

        if not eval_config.enabled:
            self.logger.info("Evaluation disabled, skipping...")
            return {}

        self.logger.info("=" * 80)
        self.logger.info("Running Evaluation")
        self.logger.info("=" * 80)

        models_to_evaluate = self.get_models_to_evaluate()

        if not models_to_evaluate:
            self.logger.warning("No models found for evaluation")
            return {}

        num_samples = eval_config.num_samples
        if num_samples == -1:
            num_samples = self.data_loader.get_num_samples()

        verbose_mode = eval_config.verbose

        all_results: Dict[str, Any] = {}

        for spec in models_to_evaluate:
            backend = spec.backend
            backend_device = self._normalize_device_for_backend(backend, spec.device)

            normalized_spec = self._build_model_spec(backend, spec.artifact, backend_device)

            results = self.evaluator.evaluate(
                model=normalized_spec,
                data_loader=self.data_loader,
                num_samples=num_samples,
                verbose=verbose_mode,
            )

            all_results[backend.value] = results

            self.logger.info(f"\n{backend.value.upper()} Results:")
            self.evaluator.print_results(results)

        if len(all_results) > 1:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Cross-Backend Comparison")
            self.logger.info("=" * 80)

            for backend_label, results in all_results.items():
                self.logger.info(f"\n{backend_label.upper()}:")
                if results and "error" not in results:
                    if "accuracy" in results:
                        self.logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                    if "mAP" in results:
                        self.logger.info(f"  mAP: {results.get('mAP', 0):.4f}")
                    if "latency_stats" in results:
                        stats = results["latency_stats"]
                        self.logger.info(f"  Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                    elif "latency" in results:
                        latency = results["latency"]
                        self.logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
                else:
                    self.logger.info("  No results available")

        return all_results

    def run(self, checkpoint_path: Optional[str] = None, **kwargs) -> DeploymentResultDict:
        """
        Execute the complete deployment workflow.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (optional)
            **kwargs: Additional project-specific arguments

        Returns:
            DeploymentResultDict: Structured summary of all deployment artifacts and reports.
        """
        results = {
            "pytorch_model": None,
            "onnx_path": None,
            "tensorrt_path": None,
            "verification_results": {},
            "evaluation_results": {},
        }

        export_mode = self.config.export_config.mode
        should_export_onnx = self.config.export_config.should_export_onnx()
        should_export_trt = self.config.export_config.should_export_tensorrt()

        # Resolve checkpoint / ONNX sources from config if not provided via CLI
        if checkpoint_path is None:
            checkpoint_path = self.config.export_config.checkpoint_path

        external_onnx_path = self.config.export_config.onnx_path

        # Check if we need model loading and export
        eval_config = self.config.evaluation_config
        verification_cfg = self.config.verification_config

        # Determine what we need PyTorch model for
        needs_export_onnx = should_export_onnx

        # Check if PyTorch evaluation is needed
        needs_pytorch_eval = False
        if eval_config.enabled:
            models_to_eval = eval_config.models
            if self._get_backend_entry(models_to_eval, Backend.PYTORCH):
                needs_pytorch_eval = True

        # Check if PyTorch is needed for verification
        needs_pytorch_for_verification = False
        if verification_cfg.enabled:
            export_mode = self.config.export_config.mode
            scenarios = self.config.get_verification_scenarios(export_mode)
            if scenarios:
                needs_pytorch_for_verification = any(
                    policy.ref_backend is Backend.PYTORCH or policy.test_backend is Backend.PYTORCH
                    for policy in scenarios
                )

        requires_pytorch_model = needs_export_onnx or needs_pytorch_eval or needs_pytorch_for_verification

        # Load model if needed for export or ONNX/TensorRT evaluation
        # Runner is always responsible for loading model, never reads from evaluator
        pytorch_model = None

        if requires_pytorch_model:
            if not checkpoint_path:
                self.logger.error(
                    "Checkpoint required but not provided. Please set export.checkpoint_path in config or pass via CLI."
                )
                return results

            self.logger.info("\nLoading PyTorch model...")
            try:
                pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
                results["pytorch_model"] = pytorch_model
                self.artifacts[Backend.PYTORCH.value] = Artifact(path=checkpoint_path)

                # Single-direction injection: write model to evaluator via setter (never read from it)
                if hasattr(self.evaluator, "set_pytorch_model"):
                    self.evaluator.set_pytorch_model(pytorch_model)
                    self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")
            except Exception as e:
                self.logger.error(f"Failed to load PyTorch model: {e}")
                return results

        # Export ONNX if requested
        if should_export_onnx:
            if pytorch_model is None:
                if not checkpoint_path:
                    self.logger.error("ONNX export requires checkpoint_path but none was provided.")
                    return results
                self.logger.info("\nLoading PyTorch model for ONNX export...")
                try:
                    pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
                    results["pytorch_model"] = pytorch_model
                    self.artifacts[Backend.PYTORCH.value] = Artifact(path=checkpoint_path)

                    # Single-direction injection: write model to evaluator via setter (never read from it)
                    if hasattr(self.evaluator, "set_pytorch_model"):
                        self.evaluator.set_pytorch_model(pytorch_model)
                        self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")
                except Exception as e:
                    self.logger.error(f"Failed to load PyTorch model: {e}")
                    return results

            try:
                onnx_artifact = self.export_onnx(pytorch_model, **kwargs)
                if onnx_artifact:
                    results["onnx_path"] = onnx_artifact.path
            except Exception as e:
                self.logger.error(f"Failed to export ONNX: {e}")

        # Export TensorRT if requested
        if should_export_trt:
            onnx_path = results["onnx_path"] or external_onnx_path
            if not onnx_path:
                self.logger.error(
                    "TensorRT export requires an ONNX path. Please set export.onnx_path in config or enable ONNX export."
                )
                return results
            else:
                results["onnx_path"] = onnx_path  # Ensure verification/evaluation can use this path
                if onnx_path and os.path.exists(onnx_path):
                    self.artifacts[Backend.ONNX.value] = Artifact(path=onnx_path, multi_file=os.path.isdir(onnx_path))
                try:
                    tensorrt_artifact = self.export_tensorrt(onnx_path, **kwargs)
                    if tensorrt_artifact:
                        results["tensorrt_path"] = tensorrt_artifact.path
                except Exception as e:
                    self.logger.error(f"Failed to export TensorRT: {e}")

        # Get model paths from evaluation config if not exported
        if not results["onnx_path"] or not results["tensorrt_path"]:
            eval_models = self.config.evaluation_config.models
            if not results["onnx_path"]:
                onnx_path = self._get_backend_entry(eval_models, Backend.ONNX)
                if onnx_path and os.path.exists(onnx_path):
                    results["onnx_path"] = onnx_path
                    self.artifacts[Backend.ONNX.value] = Artifact(path=onnx_path, multi_file=os.path.isdir(onnx_path))
                elif onnx_path:
                    self.logger.warning(f"ONNX file from config does not exist: {onnx_path}")
            if not results["tensorrt_path"]:
                tensorrt_path = self._get_backend_entry(eval_models, Backend.TENSORRT)
                if tensorrt_path and os.path.exists(tensorrt_path):
                    results["tensorrt_path"] = tensorrt_path
                    self.artifacts[Backend.TENSORRT.value] = Artifact(
                        path=tensorrt_path, multi_file=os.path.isdir(tensorrt_path)
                    )
                elif tensorrt_path:
                    self.logger.warning(f"TensorRT engine from config does not exist: {tensorrt_path}")

        # Verification
        verification_results = self.run_verification(
            pytorch_checkpoint=checkpoint_path,
            onnx_path=results["onnx_path"],
            tensorrt_path=results["tensorrt_path"],
            **kwargs,
        )
        results["verification_results"] = verification_results

        # Evaluation
        evaluation_results = self.run_evaluation(**kwargs)
        results["evaluation_results"] = evaluation_results

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)

        return results
