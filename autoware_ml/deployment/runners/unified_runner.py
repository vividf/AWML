"""
Unified deployment runner for common deployment workflows.

This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.
"""

import os
import logging
from typing import Any, Dict, Optional, Callable, List, Tuple

import torch
from mmengine.config import Config

from autoware_ml.deployment.core import BaseDeploymentConfig, BaseDataLoader, BaseEvaluator
from autoware_ml.deployment.exporters.onnx_exporter import ONNXExporter
from autoware_ml.deployment.exporters.tensorrt_exporter import TensorRTExporter


class DeploymentRunner:
    """
    Unified deployment runner for common deployment workflows.
    
    This runner handles the standard deployment workflow:
    1. Load PyTorch model (if needed)
    2. Export to ONNX (if requested)
    3. Export to TensorRT (if requested)
    4. Verify outputs (if enabled)
    5. Evaluate models (if enabled)
    
    Projects can customize behavior by:
    - Overriding methods (load_pytorch_model, export_onnx, export_tensorrt)
    - Providing custom callbacks
    - Extending this class
    """
    
    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        load_model_fn: Optional[Callable] = None,
        export_onnx_fn: Optional[Callable] = None,
        export_tensorrt_fn: Optional[Callable] = None,
    ):
        """
        Initialize unified deployment runner.
        
        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            load_model_fn: Optional custom function to load PyTorch model
            export_onnx_fn: Optional custom function to export ONNX
            export_tensorrt_fn: Optional custom function to export TensorRT
        """
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger
        self._load_model_fn = load_model_fn
        self._export_onnx_fn = export_onnx_fn
        self._export_tensorrt_fn = export_tensorrt_fn
    
    def load_pytorch_model(
        self,
        checkpoint_path: str,
        **kwargs
    ) -> Any:
        """
        Load PyTorch model from checkpoint.
        
        Uses custom function if provided, otherwise uses default implementation.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional project-specific arguments
            
        Returns:
            Loaded PyTorch model
        """
        if self._load_model_fn:
            return self._load_model_fn(checkpoint_path, **kwargs)
        
        # Default implementation - should be overridden by projects
        self.logger.warning("Using default load_pytorch_model - projects should override this")
        raise NotImplementedError("load_pytorch_model must be implemented or provided via load_model_fn")
    
    def export_onnx(
        self,
        pytorch_model: Any,
        **kwargs
    ) -> Optional[str]:
        """
        Export model to ONNX format.
        
        Uses custom function if provided, otherwise uses standard ONNXExporter.
        
        Args:
            pytorch_model: PyTorch model to export
            **kwargs: Additional project-specific arguments
            
        Returns:
            Path to exported ONNX file/directory, or None if export failed
        """
        if self._export_onnx_fn:
            return self._export_onnx_fn(pytorch_model, self.data_loader, self.config, self.logger, **kwargs)
        
        # Standard ONNX export using ONNXExporter
        if not self.config.export_config.should_export_onnx():
            return None
        
        self.logger.info("=" * 80)
        self.logger.info("Exporting to ONNX (Using Unified ONNXExporter)")
        self.logger.info("=" * 80)
        
        # Get ONNX settings
        onnx_settings = self.config.get_onnx_settings()
        output_path = os.path.join(self.config.export_config.work_dir, onnx_settings["save_file"])
        os.makedirs(self.config.export_config.work_dir, exist_ok=True)
        
        # Get sample input
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        single_input = self.data_loader.preprocess(sample)
        
        # Ensure tensor is float32
        if single_input.dtype != torch.float32:
            single_input = single_input.float()
        
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
        
        # Use unified ONNXExporter
        exporter = ONNXExporter(onnx_settings, self.logger)
        success = exporter.export(pytorch_model, input_tensor, output_path)
        
        if success:
            self.logger.info(f"✅ ONNX export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ ONNX export failed")
            return None
    
    def export_tensorrt(
        self,
        onnx_path: str,
        **kwargs
    ) -> Optional[str]:
        """
        Export ONNX model to TensorRT engine.
        
        Uses custom function if provided, otherwise uses standard TensorRTExporter.
        
        Args:
            onnx_path: Path to ONNX model file/directory
            **kwargs: Additional project-specific arguments
            
        Returns:
            Path to exported TensorRT engine file/directory, or None if export failed
        """
        if self._export_tensorrt_fn:
            return self._export_tensorrt_fn(onnx_path, self.config, self.data_loader, self.logger, **kwargs)
        
        # Standard TensorRT export using TensorRTExporter
        if not self.config.export_config.should_export_tensorrt():
            return None
        
        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None
        
        self.logger.info("=" * 80)
        self.logger.info("Exporting to TensorRT (Using Unified TensorRTExporter)")
        self.logger.info("=" * 80)
        
        trt_settings = self.config.get_tensorrt_settings()
        
        # Determine output path
        if os.path.isdir(onnx_path):
            # Directory: create tensorrt subdirectory
            output_path = os.path.join(onnx_path, "tensorrt")
            os.makedirs(output_path, exist_ok=True)
        else:
            # Single file: replace extension
            output_path = onnx_path.replace(".onnx", ".engine")
        
        # Get sample input for shape configuration
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        sample_input = self.data_loader.preprocess(sample)
        
        # Ensure tensor is float32
        if isinstance(sample_input, (list, tuple)):
            sample_input = sample_input[0]  # Use first input for shape
        if sample_input.dtype != torch.float32:
            sample_input = sample_input.float()
        
        # Merge backend_config.model_inputs into trt_settings for TensorRTExporter
        if hasattr(self.config, 'backend_config') and hasattr(self.config.backend_config, 'model_inputs'):
            trt_settings = trt_settings.copy()
            trt_settings['model_inputs'] = self.config.backend_config.model_inputs
        
        # Use unified TensorRTExporter
        exporter = TensorRTExporter(trt_settings, self.logger)
        success = exporter.export(
            model=None,  # Not used for TensorRT
            sample_input=sample_input,
            output_path=output_path,
            onnx_path=onnx_path
        )
        
        if success:
            self.logger.info(f"✅ TensorRT export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ TensorRT export failed")
            return None

    def get_models_to_evaluate(self) -> List[Tuple[str, str]]:
        """
        Get list of models to evaluate from config.

        Returns:
            List of tuples (backend_name, model_path)
        """
        eval_config = self.config.evaluation_config
        models_config = eval_config.get("models", {})
        models_to_evaluate: List[Tuple[str, str]] = []

        backend_mapping = {
            "pytorch": "pytorch",
            "onnx": "onnx",
            "tensorrt": "tensorrt",
        }

        for backend_key, model_path in models_config.items():
            backend_name = backend_mapping.get(backend_key.lower())
            if backend_name and model_path:
                is_valid = False

                if backend_name == "pytorch":
                    is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
                elif backend_name == "onnx":
                    if os.path.exists(model_path):
                        if os.path.isfile(model_path):
                            is_valid = model_path.endswith('.onnx')
                        elif os.path.isdir(model_path):
                            onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
                            is_valid = len(onnx_files) > 0
                elif backend_name == "tensorrt":
                    if os.path.exists(model_path):
                        if os.path.isfile(model_path):
                            is_valid = model_path.endswith('.engine') or model_path.endswith('.trt')
                        elif os.path.isdir(model_path):
                            engine_files = [f for f in os.listdir(model_path) if f.endswith('.engine')]
                            is_valid = len(engine_files) > 0

                if is_valid:
                    models_to_evaluate.append((backend_name, model_path))
                    self.logger.info(f"  - {backend_name}: {model_path}")
                else:
                    self.logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")

        return models_to_evaluate

    def run_verification(
        self,
        pytorch_checkpoint: Optional[str],
        onnx_path: Optional[str],
        tensorrt_path: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run verification on exported models.

        Args:
            pytorch_checkpoint: Path to PyTorch checkpoint (reference)
            onnx_path: Path to ONNX model file/directory
            tensorrt_path: Path to TensorRT engine file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Verification results dictionary
        """
        if not self.config.export_config.verify:
            self.logger.info("Verification disabled, skipping...")
            return {}

        if not pytorch_checkpoint:
            self.logger.warning("PyTorch checkpoint path not available, skipping verification")
            return {}

        if not onnx_path and not tensorrt_path:
            self.logger.info("No exported models to verify, skipping verification")
            return {}

        num_verify_samples = self.config.verification_config.get("num_verify_samples", 3)
        tolerance = self.config.verification_config.get("tolerance", 0.1)

        verification_results = self.evaluator.verify(
            pytorch_model_path=pytorch_checkpoint,
            onnx_model_path=onnx_path,
            tensorrt_model_path=tensorrt_path,
            data_loader=self.data_loader,
            num_samples=num_verify_samples,
            device=self.config.export_config.device,
            tolerance=tolerance,
            verbose=False,
        )

        if 'summary' in verification_results:
            summary = verification_results['summary']
            if summary['failed'] == 0:
                if summary.get('skipped', 0) > 0:
                    self.logger.info(f"\n✅ All verifications passed! ({summary['skipped']} skipped)")
                else:
                    self.logger.info("\n✅ All verifications passed!")
            else:
                self.logger.warning(f"\n⚠️  {summary['failed']}/{summary['total']} verifications failed")
        else:
            self.logger.error("\n❌ Verification encountered errors")

        return verification_results

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

        for backend, model_path in models_to_evaluate:
            results = self.evaluator.evaluate(
                model_path=model_path,
                data_loader=self.data_loader,
                num_samples=num_samples,
                backend=backend,
                device=self.config.export_config.device,
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
                    if 'latency_stats' in results:
                        stats = results['latency_stats']
                        self.logger.info(f"  Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                    elif 'latency' in results:
                        latency = results['latency']
                        self.logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
                else:
                    self.logger.info("  No results available")

        return all_results
    
    def run(
        self,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
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
        
        # Check if we need model loading and export
        eval_config = self.config.evaluation_config
        needs_export = self.config.export_config.mode != "none"
        needs_onnx_eval = False
        if eval_config.get("enabled", False):
            models_to_eval = eval_config.get("models", {})
            if models_to_eval.get("onnx") or models_to_eval.get("tensorrt"):
                needs_onnx_eval = True
        
        # Load model if needed for export or ONNX/TensorRT evaluation
        if needs_export or needs_onnx_eval:
            if not checkpoint_path:
                if needs_export:
                    self.logger.error("Checkpoint required for export")
                else:
                    self.logger.error("Checkpoint required for ONNX/TensorRT evaluation")
                self.logger.error("Please provide checkpoint_path argument")
                return results
            
            # Load PyTorch model
            self.logger.info("\nLoading PyTorch model...")
            try:
                pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
                results["pytorch_model"] = pytorch_model
            except Exception as e:
                self.logger.error(f"Failed to load PyTorch model: {e}")
                return results
            
            # Export ONNX
            if self.config.export_config.should_export_onnx():
                try:
                    onnx_path = self.export_onnx(pytorch_model, **kwargs)
                    results["onnx_path"] = onnx_path
                except Exception as e:
                    self.logger.error(f"Failed to export ONNX: {e}")
            
            # Export TensorRT
            if self.config.export_config.should_export_tensorrt() and results["onnx_path"]:
                try:
                    tensorrt_path = self.export_tensorrt(results["onnx_path"], **kwargs)
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
            **kwargs
        )
        results["verification_results"] = verification_results
        
        # Evaluation
        evaluation_results = self.run_evaluation(**kwargs)
        results["evaluation_results"] = evaluation_results
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)
        
        return results


