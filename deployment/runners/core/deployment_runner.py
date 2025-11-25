"""
Unified deployment runner for common deployment workflows.

This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.

Refactored to use:
- ArtifactManager: Handles artifact registration and resolution
- VerificationOrchestrator: Handles verification workflows
- Reduces runner complexity from 856 lines to ~400 lines
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import torch
from mmengine.config import Config

from deployment.core import Artifact, Backend, BaseDataLoader, BaseDeploymentConfig, BaseEvaluator, ModelSpec
from deployment.exporters.base.factory import ExporterFactory
from deployment.exporters.base.model_wrappers import BaseModelWrapper
from deployment.exporters.base.onnx_exporter import ONNXExporter
from deployment.exporters.base.tensorrt_exporter import TensorRTExporter
from deployment.exporters.workflows.base import OnnxExportWorkflow, TensorRTExportWorkflow
from deployment.runners.core.artifact_manager import ArtifactManager
from deployment.runners.core.evaluation_orchestrator import EvaluationOrchestrator
from deployment.runners.core.verification_orchestrator import VerificationOrchestrator


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

    # Directory name constants
    ONNX_DIR_NAME = "onnx"
    TENSORRT_DIR_NAME = "tensorrt"
    DEFAULT_ENGINE_FILENAME = "model.engine"

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_workflow: Optional[OnnxExportWorkflow] = None,
        tensorrt_workflow: Optional[TensorRTExportWorkflow] = None,
    ):
        """
        Initialize base deployment runner.

        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            onnx_wrapper_cls: Optional ONNX model wrapper class for exporter creation
            onnx_workflow: Optional specialized ONNX workflow
            tensorrt_workflow: Optional specialized TensorRT workflow
        """
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger
        self._onnx_wrapper_cls = onnx_wrapper_cls
        self._onnx_exporter: Optional[ONNXExporter] = None
        self._tensorrt_exporter: Optional[TensorRTExporter] = None
        self._onnx_workflow = onnx_workflow
        self._tensorrt_workflow = tensorrt_workflow

        # Initialize specialized components (reduces runner complexity)
        self.artifact_manager = ArtifactManager(config, logger)
        self.verification_orchestrator = VerificationOrchestrator(config, evaluator, data_loader, logger)
        self.evaluation_orchestrator = EvaluationOrchestrator(config, evaluator, data_loader, logger)

    def _get_onnx_exporter(self) -> ONNXExporter:
        """
        Lazily instantiate and return the ONNX exporter.
        """

        if self._onnx_exporter is None:
            if self._onnx_wrapper_cls is None:
                raise RuntimeError("ONNX wrapper class not provided. Cannot create ONNX exporter.")
            self._onnx_exporter = ExporterFactory.create_onnx_exporter(
                config=self.config,
                wrapper_cls=self._onnx_wrapper_cls,
                logger=self.logger,
            )
        return self._onnx_exporter

    def _get_tensorrt_exporter(self) -> TensorRTExporter:
        """
        Lazily instantiate and return the TensorRT exporter.
        """

        if self._tensorrt_exporter is None:
            self._tensorrt_exporter = ExporterFactory.create_tensorrt_exporter(
                config=self.config,
                logger=self.logger,
            )
        return self._tensorrt_exporter

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

        Uses either a specialized workflow or the standard ONNX exporter.

        Args:
            pytorch_model: PyTorch model to export
            **kwargs: Additional project-specific arguments

        Returns:
            Artifact describing the exported ONNX output, or None if skipped
        """
        if not self.config.export_config.should_export_onnx():
            return None

        if self._onnx_workflow is None and self._onnx_wrapper_cls is None:
            raise RuntimeError("ONNX export requested but no wrapper class or workflow provided.")

        onnx_settings = self.config.get_onnx_settings()
        sample_idx = self.config.runtime_config.get("sample_idx", 0)

        # Save to work_dir/onnx/ directory
        onnx_dir = os.path.join(self.config.export_config.work_dir, self.ONNX_DIR_NAME)
        os.makedirs(onnx_dir, exist_ok=True)
        output_path = os.path.join(onnx_dir, onnx_settings.save_file)

        if self._onnx_workflow is not None:
            self.logger.info("=" * 80)
            self.logger.info(f"Exporting to ONNX via workflow ({type(self._onnx_workflow).__name__})")
            self.logger.info("=" * 80)
            try:
                artifact = self._onnx_workflow.export(
                    model=pytorch_model,
                    data_loader=self.data_loader,
                    output_dir=onnx_dir,
                    config=self.config,
                    sample_idx=sample_idx,
                    **kwargs,
                )
            except Exception:
                self.logger.error("ONNX export workflow failed")
                raise

            self.artifact_manager.register_artifact(Backend.ONNX, artifact)
            self.logger.info(f"ONNX export successful: {artifact.path}")
            return artifact

        exporter = self._get_onnx_exporter()
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to ONNX (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        # Get sample input
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
        self.artifact_manager.register_artifact(Backend.ONNX, artifact)
        self.logger.info(f"ONNX export successful: {artifact.path}")
        return artifact

    def export_tensorrt(self, onnx_path: str, **kwargs) -> Optional[Artifact]:
        """
        Export ONNX model to TensorRT engine.

        Uses either a specialized workflow or the standard TensorRT exporter.

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

        exporter_label = None if self._tensorrt_workflow else type(self._get_tensorrt_exporter()).__name__
        self.logger.info("=" * 80)
        if self._tensorrt_workflow:
            self.logger.info(f"Exporting to TensorRT via workflow ({type(self._tensorrt_workflow).__name__})")
        else:
            self.logger.info(f"Exporting to TensorRT (Using {exporter_label})")
        self.logger.info("=" * 80)

        # Save to work_dir/tensorrt/ directory
        tensorrt_dir = os.path.join(self.config.export_config.work_dir, self.TENSORRT_DIR_NAME)
        os.makedirs(tensorrt_dir, exist_ok=True)

        # Determine output path based on ONNX file name
        output_path = self._get_tensorrt_output_path(onnx_path, tensorrt_dir)

        # Set CUDA device for TensorRT export
        cuda_device = self.config.export_config.cuda_device
        device_id = self.config.export_config.get_cuda_device_index()
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device for TensorRT export: {cuda_device}")

        # Get sample input for shape configuration
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample_input = self.data_loader.get_shape_sample(sample_idx)

        if self._tensorrt_workflow is not None:
            try:
                artifact = self._tensorrt_workflow.export(
                    onnx_path=onnx_path,
                    output_dir=tensorrt_dir,
                    config=self.config,
                    device=cuda_device,
                    data_loader=self.data_loader,
                    **kwargs,
                )
            except Exception:
                self.logger.error("TensorRT export workflow failed")
                raise

            self.artifact_manager.register_artifact(Backend.TENSORRT, artifact)
            self.logger.info(f"TensorRT export successful: {artifact.path}")
            return artifact

        exporter = self._get_tensorrt_exporter()

        try:
            artifact = exporter.export(
                model=None,
                sample_input=sample_input,
                output_path=output_path,
                onnx_path=onnx_path,
            )
        except Exception:
            self.logger.error("TensorRT export failed")
            raise

        self.artifact_manager.register_artifact(Backend.TENSORRT, artifact)
        self.logger.info(f"TensorRT export successful: {artifact.path}")
        return artifact

    def _get_tensorrt_output_path(self, onnx_path: str, tensorrt_dir: str) -> str:
        """
        Determine TensorRT output path based on ONNX file name.

        Args:
            onnx_path: Path to ONNX model file or directory
            tensorrt_dir: Directory for TensorRT engines

        Returns:
            Path for TensorRT engine output
        """
        if os.path.isdir(onnx_path):
            return os.path.join(tensorrt_dir, self.DEFAULT_ENGINE_FILENAME)
        else:
            onnx_filename = os.path.basename(onnx_path)
            engine_filename = onnx_filename.replace(".onnx", ".engine")
            return os.path.join(tensorrt_dir, engine_filename)

    def _load_and_register_pytorch_model(
        self, checkpoint_path: str, results: DeploymentResultDict, **kwargs
    ) -> Optional[Any]:
        """
        Load PyTorch model and register it with artifact manager and evaluator.

        Args:
            checkpoint_path: Path to checkpoint file
            results: Results dictionary to update
            **kwargs: Additional project-specific arguments

        Returns:
            Loaded PyTorch model, or None if loading failed
        """
        if not checkpoint_path:
            self.logger.error(
                "Checkpoint required but not provided. Please set export.checkpoint_path in config or pass via CLI."
            )
            return None

        self.logger.info("\nLoading PyTorch model...")
        try:
            pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
            results["pytorch_model"] = pytorch_model
            self.artifact_manager.register_artifact(Backend.PYTORCH, Artifact(path=checkpoint_path))

            # Single-direction injection: write model to evaluator via setter (never read from it)
            if hasattr(self.evaluator, "set_pytorch_model"):
                self.evaluator.set_pytorch_model(pytorch_model)
                self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")
            return pytorch_model
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            return None

    def _determine_pytorch_requirements(self) -> bool:
        """
        Determine if PyTorch model is required based on configuration.

        Returns:
            True if PyTorch model is needed, False otherwise
        """
        should_export_onnx = self.config.export_config.should_export_onnx()
        eval_config = self.config.evaluation_config
        verification_cfg = self.config.verification_config

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

        return should_export_onnx or needs_pytorch_eval or needs_pytorch_for_verification

    def _resolve_and_register_artifact(
        self, backend: Backend, results_key: str, results: DeploymentResultDict
    ) -> None:
        """
        Resolve artifact path from evaluation config and register it.

        Args:
            backend: Backend type (ONNX or TENSORRT)
            results_key: Key in results dict to update ("onnx_path" or "tensorrt_path")
            results: Results dictionary to update
        """
        if results[results_key]:
            return  # Already resolved

        eval_models = self.config.evaluation_config.models
        artifact_path = self._get_backend_entry(eval_models, backend)

        if artifact_path and os.path.exists(artifact_path):
            results[results_key] = artifact_path
            multi_file = os.path.isdir(artifact_path)
            self.artifact_manager.register_artifact(backend, Artifact(path=artifact_path, multi_file=multi_file))
        elif artifact_path:
            self.logger.warning(f"{backend.value} file from config does not exist: {artifact_path}")

    def run_verification(
        self, pytorch_checkpoint: Optional[str], onnx_path: Optional[str], tensorrt_path: Optional[str], **kwargs
    ) -> Dict[str, Any]:
        """
        Run verification on exported models using policy-based verification.

        This method now delegates to VerificationOrchestrator, which handles all
        verification logic. This reduces runner complexity.

        Args:
            pytorch_checkpoint: Path to PyTorch checkpoint (reference)
            onnx_path: Path to ONNX model file/directory
            tensorrt_path: Path to TensorRT engine file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Verification results dictionary
        """
        # Delegate to verification orchestrator
        return self.verification_orchestrator.run(
            artifact_manager=self.artifact_manager,
            pytorch_checkpoint=pytorch_checkpoint,
            onnx_path=onnx_path,
            tensorrt_path=tensorrt_path,
        )

    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on specified models.

        This method now delegates to EvaluationOrchestrator, which handles all
        evaluation logic. This further reduces runner complexity.

        Args:
            **kwargs: Additional project-specific arguments (not currently used)

        Returns:
            Dictionary containing evaluation results for all backends
        """
        # Delegate to evaluation orchestrator
        return self.evaluation_orchestrator.run(self.artifact_manager)

    def _execute_exports(
        self,
        should_export_onnx: bool,
        should_export_trt: bool,
        pytorch_model: Optional[Any],
        checkpoint_path: Optional[str],
        external_onnx_path: Optional[str],
        results: DeploymentResultDict,
        **kwargs,
    ) -> None:
        """
        Execute ONNX and TensorRT exports.

        Args:
            should_export_onnx: Whether to export ONNX
            should_export_trt: Whether to export TensorRT
            pytorch_model: Loaded PyTorch model (may be None)
            checkpoint_path: Path to checkpoint (for loading model if needed)
            external_onnx_path: External ONNX path from config
            results: Results dictionary to update
            **kwargs: Additional project-specific arguments
        """
        # Export ONNX if requested
        if should_export_onnx:
            # Load model if not already loaded
            if pytorch_model is None:
                if not checkpoint_path:
                    self.logger.error("ONNX export requires checkpoint_path but none was provided.")
                    return
                pytorch_model = self._load_and_register_pytorch_model(checkpoint_path, results, **kwargs)
                if pytorch_model is None:
                    return  # Loading failed, error already logged

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
                return

            # Ensure verification/evaluation can use this path
            results["onnx_path"] = onnx_path
            if onnx_path and os.path.exists(onnx_path):
                multi_file = os.path.isdir(onnx_path)
                self.artifact_manager.register_artifact(Backend.ONNX, Artifact(path=onnx_path, multi_file=multi_file))

            try:
                tensorrt_artifact = self.export_tensorrt(onnx_path, **kwargs)
                if tensorrt_artifact:
                    results["tensorrt_path"] = tensorrt_artifact.path
            except Exception as e:
                self.logger.error(f"Failed to export TensorRT: {e}")

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
        checkpoint_path = checkpoint_path or self.config.export_config.checkpoint_path
        external_onnx_path = self.config.export_config.onnx_path

        # Determine if PyTorch model is needed
        requires_pytorch_model = self._determine_pytorch_requirements()

        # Load model if needed
        pytorch_model = None
        if requires_pytorch_model:
            pytorch_model = self._load_and_register_pytorch_model(checkpoint_path, results, **kwargs)
            if pytorch_model is None:
                return results  # Loading failed, error already logged

        # Execute exports
        self._execute_exports(
            should_export_onnx,
            should_export_trt,
            pytorch_model,
            checkpoint_path,
            external_onnx_path,
            results,
            **kwargs,
        )

        # Resolve paths from evaluation config if not exported
        self._resolve_and_register_artifact(Backend.ONNX, "onnx_path", results)
        self._resolve_and_register_artifact(Backend.TENSORRT, "tensorrt_path", results)

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
