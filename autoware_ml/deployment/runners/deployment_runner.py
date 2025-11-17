"""
Unified deployment runner for common deployment workflows.

This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.config import Config

from autoware_ml.deployment.core import (
    BaseDataLoader,
    BaseDeploymentConfig,
    BaseEvaluator,
    ModelSpec,
)


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
            onnx_exporter: Required ONNX exporter instance (e.g., CenterPointONNXExporter, YOLOXONNXExporter)
            tensorrt_exporter: Required TensorRT exporter instance (e.g., CenterPointTensorRTExporter, YOLOXTensorRTExporter)

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

    def export_onnx(self, pytorch_model: Any, **kwargs) -> Optional[str]:
        """
        Export model to ONNX format.

        Uses the provided ONNX exporter instance.

        Args:
            pytorch_model: PyTorch model to export
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported ONNX file/directory, or None if export failed
        """
        # Standard ONNX export using ONNXExporter
        if not self.config.export_config.should_export_onnx():
            return None

        self.logger.info("=" * 80)
        self.logger.info("Exporting to ONNX (Using Unified ONNXExporter)")
        self.logger.info("=" * 80)

        # Get ONNX settings
        onnx_settings = self.config.get_onnx_settings()

        # Use provided exporter (required, cannot be None)
        exporter = self._onnx_exporter
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to ONNX (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        # Standard ONNX export
        # Save to work_dir/onnx/ directory
        onnx_dir = os.path.join(self.config.export_config.work_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        output_path = os.path.join(onnx_dir, onnx_settings["save_file"])

        # Get sample input
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        single_input = self.data_loader.preprocess(sample)

        # Get batch size from configuration
        batch_size = onnx_settings.get("batch_size", 1)
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

        # Use provided exporter (required, cannot be None)
        exporter = self._onnx_exporter

        success = exporter.export(pytorch_model, input_tensor, output_path)

        if success:
            self.logger.info(f"✅ ONNX export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ ONNX export failed")
            return None

    def export_tensorrt(self, onnx_path: str, **kwargs) -> Optional[str]:
        """
        Export ONNX model to TensorRT engine.

        Uses the provided TensorRT exporter instance.

        Args:
            onnx_path: Path to ONNX model file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported TensorRT engine file/directory, or None if export failed
        """
        # Standard TensorRT export using TensorRTExporter
        if not self.config.export_config.should_export_tensorrt():
            return None

        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None

        # Use provided exporter (required, cannot be None)
        exporter = self._tensorrt_exporter
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to TensorRT (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        # Standard TensorRT export
        # Save to work_dir/tensorrt/ directory
        tensorrt_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
        os.makedirs(tensorrt_dir, exist_ok=True)

        # Determine output path based on ONNX file name
        if os.path.isdir(onnx_path):
            # For multi-file ONNX (shouldn't happen in standard export, but handle it)
            # Use the directory name or a default name
            output_path = os.path.join(tensorrt_dir, "model.engine")
        else:
            # Single file: extract filename and change extension
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

        # Note: trt_settings are read from exporter.config in TensorRTExporter._configure_input_shapes
        # The exporter's config should already include model_inputs if backend_config is properly set up
        # No need to pass trt_settings here as the exporter reads from self.config

        # Use provided exporter (required, cannot be None)
        exporter = self._tensorrt_exporter

        success = exporter.export(
            model=None,  # Not used for TensorRT
            sample_input=sample_input,
            output_path=output_path,
            onnx_path=onnx_path,
        )

        if success:
            self.logger.info(f"✅ TensorRT export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ TensorRT export failed")
            return None

    def _resolve_pytorch_model(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """
        Resolve PyTorch model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (model_path, is_valid)
        """
        model_path = backend_cfg.get("checkpoint")
        if model_path:
            is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
        else:
            is_valid = False
        return model_path, is_valid

    def _resolve_onnx_model(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """
        Resolve ONNX model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (model_path, is_valid)
        """
        model_path = backend_cfg.get("model_dir")
        multi_file = self.config.onnx_config.get("multi_file", False)
        save_file = self.config.onnx_config.get("save_file", "model.onnx")

        # If model_dir is explicitly set in config
        if model_path is not None:
            if os.path.exists(model_path):
                if os.path.isfile(model_path):
                    # Single file ONNX
                    is_valid = model_path.endswith(".onnx") and not multi_file
                elif os.path.isdir(model_path):
                    # Directory: valid if multi_file is True, or if it contains ONNX files
                    if multi_file:
                        is_valid = True
                    else:
                        # Single file mode: find the ONNX file in directory
                        onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
                        if onnx_files:
                            expected_file = os.path.join(model_path, save_file)
                            if os.path.exists(expected_file):
                                model_path = expected_file
                            else:
                                model_path = os.path.join(model_path, onnx_files[0])
                            is_valid = True
                        else:
                            is_valid = False
                else:
                    is_valid = False
            else:
                is_valid = False
            return model_path, is_valid

        # Infer from export config
        work_dir = self.config.export_config.work_dir
        onnx_dir = os.path.join(work_dir, "onnx")

        if os.path.exists(onnx_dir) and os.path.isdir(onnx_dir):
            onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
            if onnx_files:
                if multi_file:
                    model_path = onnx_dir
                    is_valid = True
                else:
                    # Single file ONNX: use the save_file if it exists, otherwise use the first ONNX file found
                    expected_file = os.path.join(onnx_dir, save_file)
                    if os.path.exists(expected_file):
                        model_path = expected_file
                    else:
                        model_path = os.path.join(onnx_dir, onnx_files[0])
                    is_valid = True
            else:
                if multi_file:
                    model_path = onnx_dir
                    is_valid = True
                else:
                    # Try single file path
                    model_path = os.path.join(onnx_dir, save_file)
                    is_valid = os.path.exists(model_path) and model_path.endswith(".onnx")
        else:
            if multi_file:
                # Multi-file ONNX: return directory even if it doesn't exist yet
                model_path = onnx_dir
                is_valid = True
            else:
                # Fallback: try in work_dir directly (for backward compatibility)
                model_path = os.path.join(work_dir, save_file)
                is_valid = os.path.exists(model_path) and model_path.endswith(".onnx")

        return model_path, is_valid

    def _resolve_tensorrt_model(self, backend_cfg: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """
        Resolve TensorRT model path from backend config.

        Args:
            backend_cfg: Backend configuration dictionary

        Returns:
            Tuple of (model_path, is_valid)
        """
        model_path = backend_cfg.get("engine_dir")
        multi_file = self.config.onnx_config.get("multi_file", False)
        onnx_save_file = self.config.onnx_config.get("save_file", "model.onnx")
        expected_engine = onnx_save_file.replace(".onnx", ".engine")

        # If engine_dir is explicitly set in config
        if model_path is not None:
            if os.path.exists(model_path):
                if os.path.isfile(model_path):
                    # Single file TensorRT
                    is_valid = (model_path.endswith(".engine") or model_path.endswith(".trt")) and not multi_file
                elif os.path.isdir(model_path):
                    # Directory: valid if multi_file is True, or if it contains engine files
                    if multi_file:
                        is_valid = True
                    else:
                        # Single file mode: find the engine file in directory
                        engine_files = [f for f in os.listdir(model_path) if f.endswith(".engine")]
                        if engine_files:
                            expected_path = os.path.join(model_path, expected_engine)
                            if os.path.exists(expected_path):
                                model_path = expected_path
                            else:
                                model_path = os.path.join(model_path, engine_files[0])
                            is_valid = True
                        else:
                            is_valid = False
                else:
                    is_valid = False
            else:
                is_valid = False
            return model_path, is_valid

        # Infer from export config
        work_dir = self.config.export_config.work_dir
        engine_dir = os.path.join(work_dir, "tensorrt")

        if os.path.exists(engine_dir) and os.path.isdir(engine_dir):
            engine_files = [f for f in os.listdir(engine_dir) if f.endswith(".engine")]
            if engine_files:
                if multi_file:
                    model_path = engine_dir
                    is_valid = True
                else:
                    # Single file TensorRT: use the engine file matching ONNX filename
                    expected_path = os.path.join(engine_dir, expected_engine)
                    if os.path.exists(expected_path):
                        model_path = expected_path
                    else:
                        # Fallback: use the first engine file found
                        model_path = os.path.join(engine_dir, engine_files[0])
                    is_valid = True
            else:
                if multi_file:
                    model_path = engine_dir
                    is_valid = True
                else:
                    is_valid = False
        else:
            if multi_file:
                # Multi-file TensorRT: return directory even if it doesn't exist yet
                model_path = engine_dir
                is_valid = True
            else:
                # Fallback: try in work_dir directly (for backward compatibility)
                if os.path.exists(work_dir) and os.path.isdir(work_dir):
                    engine_files = [f for f in os.listdir(work_dir) if f.endswith(".engine")]
                    if engine_files:
                        expected_path = os.path.join(work_dir, expected_engine)
                        if os.path.exists(expected_path):
                            model_path = expected_path
                        else:
                            model_path = os.path.join(work_dir, engine_files[0])
                        is_valid = True
                    else:
                        is_valid = False
                else:
                    is_valid = False

        return model_path, is_valid

    def get_models_to_evaluate(self) -> List[ModelSpec]:
        """
        Get list of models to evaluate from config.

        Returns:
            List of tuples (backend_name, model_path, device)
        """
        backends = self.config.get_evaluation_backends()
        models_to_evaluate: List[ModelSpec] = []

        for backend_name, backend_cfg in backends.items():
            if not backend_cfg.get("enabled", False):
                continue

            device = backend_cfg.get("device", "cpu")
            model_path = None
            is_valid = False

            if backend_name == "pytorch":
                model_path, is_valid = self._resolve_pytorch_model(backend_cfg)
            elif backend_name == "onnx":
                model_path, is_valid = self._resolve_onnx_model(backend_cfg)
            elif backend_name == "tensorrt":
                model_path, is_valid = self._resolve_tensorrt_model(backend_cfg)

            if is_valid and model_path:
                normalized_device = str(device or "cpu")
                models_to_evaluate.append(
                    ModelSpec(
                        backend=backend_name,
                        device=normalized_device,
                        path=model_path,
                    )
                )
                self.logger.info(f"  - {backend_name}: {model_path} (device: {normalized_device})")
            elif model_path:
                self.logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")

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
        if not verification_cfg.get("enabled", True):
            self.logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            self.logger.info(f"No verification scenarios for export mode '{export_mode}', skipping...")
            return {}

        # Check if any scenario actually needs PyTorch checkpoint
        needs_pytorch = any(
            policy.get("ref_backend") == "pytorch" or policy.get("test_backend") == "pytorch" for policy in scenarios
        )

        if needs_pytorch and not pytorch_checkpoint:
            self.logger.warning(
                "PyTorch checkpoint path not available, but required by verification scenarios. Skipping verification."
            )
            return {}

        num_verify_samples = verification_cfg.get("num_verify_samples", 3)
        tolerance = verification_cfg.get("tolerance", 0.1)
        devices_map = verification_cfg.get("devices", {}) or {}

        self.logger.info("=" * 80)
        self.logger.info(f"Running Verification (mode: {export_mode})")
        self.logger.info("=" * 80)

        all_results = {}
        total_passed = 0
        total_failed = 0

        for i, policy in enumerate(scenarios):
            ref_backend = policy["ref_backend"]
            # Resolve device using alias system:
            # - Scenarios use aliases (e.g., "cpu", "cuda") for flexibility
            # - Actual device strings are defined in verification["devices"]
            # - This allows easy device switching: change devices["cpu"] to affect all CPU verifications
            ref_device_key = policy["ref_device"]
            if ref_device_key in devices_map:
                ref_device = devices_map[ref_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                ref_device = ref_device_key
                self.logger.warning(f"Device alias '{ref_device_key}' not found in devices map, using as-is")

            test_backend = policy["test_backend"]
            test_device_key = policy["test_device"]
            if test_device_key in devices_map:
                test_device = devices_map[test_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                test_device = test_device_key
                self.logger.warning(f"Device alias '{test_device_key}' not found in devices map, using as-is")

            self.logger.info(
                f"\nScenarios {i+1}/{len(scenarios)}: {ref_backend}({ref_device}) vs {test_backend}({test_device})"
            )

            # Resolve model paths based on backend
            ref_path = None
            test_path = None

            if ref_backend == "pytorch":
                ref_path = pytorch_checkpoint
            elif ref_backend == "onnx":
                ref_path = onnx_path

            if test_backend == "onnx":
                test_path = onnx_path
            elif test_backend == "tensorrt":
                test_path = tensorrt_path

            if not ref_path or not test_path:
                self.logger.warning(f"  Skipping: missing paths (ref={ref_path}, test={test_path})")
                continue

            # Use policy-based verification interface
            reference_spec = ModelSpec(backend=ref_backend, device=ref_device, path=ref_path)
            test_spec = ModelSpec(backend=test_backend, device=test_device, path=test_path)

            verification_results = self.evaluator.verify(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
                verbose=False,
            )

            # Extract results for this specific comparison
            policy_key = f"{ref_backend}_{ref_device}_vs_{test_backend}_{test_device}"
            all_results[policy_key] = verification_results

            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed

                if failed == 0:
                    self.logger.info(f"  ✅ Policy {i+1} passed ({passed} comparisons)")
                else:
                    self.logger.warning(f"  ⚠️  Policy {i+1} failed ({failed}/{passed+failed} comparisons)")

        # Overall summary
        self.logger.info("\n" + "=" * 80)
        if total_failed == 0:
            self.logger.info(f"✅ All verifications passed! ({total_passed} total)")
        else:
            self.logger.warning(f"⚠️  {total_failed}/{total_passed + total_failed} verifications failed")
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

        if not eval_config.get("enabled", False):
            self.logger.info("Evaluation disabled, skipping...")
            return {}

        self.logger.info("=" * 80)
        self.logger.info("Running Evaluation")
        self.logger.info("=" * 80)

        models_to_evaluate = self.get_models_to_evaluate()

        if not models_to_evaluate:
            self.logger.warning("No models found for evaluation")
            return {}

        num_samples = eval_config.get("num_samples", 10)
        if num_samples == -1:
            num_samples = self.data_loader.get_num_samples()

        verbose_mode = eval_config.get("verbose", False)

        all_results: Dict[str, Any] = {}

        # TODO(vividf): a bit ungly here, need to refactor
        for spec in models_to_evaluate:
            backend = spec.backend
            backend_device = spec.device
            model_path = spec.path

            if backend in ("pytorch", "onnx"):
                if backend_device not in ("cpu",) and not str(backend_device).startswith("cuda"):
                    self.logger.warning(
                        f"Unsupported device '{backend_device}' for backend '{backend}'. Falling back to CPU."
                    )
                    backend_device = "cpu"
            elif backend == "tensorrt":
                if not backend_device or backend_device == "cpu":
                    backend_device = self.config.export_config.cuda_device or "cuda:0"
                if not str(backend_device).startswith("cuda"):
                    self.logger.warning(
                        f"TensorRT evaluation requires CUDA device. Overriding device from '{backend_device}' to 'cuda:0'."
                    )
                    backend_device = "cuda:0"

            normalized_spec = ModelSpec(
                backend=backend,
                device=backend_device or "cpu",
                path=model_path,
            )

            results = self.evaluator.evaluate(
                model=normalized_spec,
                data_loader=self.data_loader,
                num_samples=num_samples,
                verbose=verbose_mode,
            )

            all_results[backend] = results

            self.logger.info(f"\n{backend.upper()} Results:")
            self.evaluator.print_results(results)

        if len(all_results) > 1:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Cross-Backend Comparison")
            self.logger.info("=" * 80)

            for backend, results in all_results.items():
                self.logger.info(f"\n{backend.upper()}:")
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

    def run(self, checkpoint_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete deployment workflow.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (optional)
            **kwargs: Additional project-specific arguments

        Returns:
            Dictionary containing deployment results
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
        if eval_config.get("enabled", False):
            models_to_eval = eval_config.get("models", {})
            if models_to_eval.get("pytorch"):
                needs_pytorch_eval = True

        # Check if PyTorch is needed for verification
        needs_pytorch_for_verification = False
        if verification_cfg.get("enabled", False):
            export_mode = self.config.export_config.mode
            scenarios = self.config.get_verification_scenarios(export_mode)
            if scenarios:
                needs_pytorch_for_verification = any(
                    policy.get("ref_backend") == "pytorch" or policy.get("test_backend") == "pytorch"
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

                    # Single-direction injection: write model to evaluator via setter (never read from it)
                    if hasattr(self.evaluator, "set_pytorch_model"):
                        self.evaluator.set_pytorch_model(pytorch_model)
                        self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")
                except Exception as e:
                    self.logger.error(f"Failed to load PyTorch model: {e}")
                    return results

            try:
                onnx_path = self.export_onnx(pytorch_model, **kwargs)
                results["onnx_path"] = onnx_path
            except Exception as e:
                self.logger.error(f"Failed to export ONNX: {e}")

        # Export TensorRT if requested
        if should_export_trt:
            onnx_source = results["onnx_path"] or external_onnx_path
            if not onnx_source:
                self.logger.error(
                    "TensorRT export requires an ONNX path. Please set export.onnx_path in config or enable ONNX export."
                )
                return results
            else:
                results["onnx_path"] = onnx_source  # Ensure verification/evaluation can use this path
                try:
                    tensorrt_path = self.export_tensorrt(onnx_source, **kwargs)
                    results["tensorrt_path"] = tensorrt_path
                except Exception as e:
                    self.logger.error(f"Failed to export TensorRT: {e}")

        # Get model paths from evaluation config if not exported
        if not results["onnx_path"] or not results["tensorrt_path"]:
            eval_models = self.config.evaluation_config.get("models", {})
            if not results["onnx_path"]:
                onnx_path = eval_models.get("onnx")
                if onnx_path and os.path.exists(onnx_path):
                    results["onnx_path"] = onnx_path
                elif onnx_path:
                    self.logger.warning(f"ONNX file from config does not exist: {onnx_path}")
            if not results["tensorrt_path"]:
                tensorrt_path = eval_models.get("tensorrt")
                if tensorrt_path and os.path.exists(tensorrt_path):
                    results["tensorrt_path"] = tensorrt_path
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
