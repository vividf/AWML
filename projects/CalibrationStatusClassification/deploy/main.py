"""
CalibrationStatusClassification Deployment Main Script (New Pipeline Architecture).

This script demonstrates the new unified pipeline architecture for CalibrationStatusClassification deployment:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import mmengine
import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters.onnx_exporter import ONNXExporter
from autoware_ml.deployment.exporters.tensorrt_exporter import TensorRTExporter
from projects.CalibrationStatusClassification.deploy.data_loader import CalibrationDataLoader
from projects.CalibrationStatusClassification.deploy.evaluator import ClassificationEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(model_cfg: Config, checkpoint_path: str, device: str):
    """Load PyTorch model from checkpoint."""
    torch_device = torch.device(device)
    model = get_model(model_cfg, checkpoint_path, device=torch_device)
    model.eval()
    return model


def export_onnx(
    pytorch_model,
    data_loader: CalibrationDataLoader,
    config: BaseDeploymentConfig,
    logger: logging.Logger
) -> str:
    """
    Export model to ONNX format using the unified ONNXExporter.
    
    Returns:
        Path to exported ONNX file, or None if export failed
    """
    logger.info("=" * 80)
    logger.info("Exporting to ONNX (Using Unified ONNXExporter)")
    logger.info("=" * 80)

    # Get ONNX settings
    onnx_settings = config.get_onnx_settings()
    output_path = os.path.join(config.export_config.work_dir, onnx_settings["save_file"])
    os.makedirs(config.export_config.work_dir, exist_ok=True)

    # Get sample input
    # Use the same preprocessing as inference (via data_loader.preprocess)
    # This ensures ONNX export and inference use identical input format
    sample_idx = config.runtime_config.get("sample_idx", 0)
    sample = data_loader.load_sample(sample_idx)
    single_input = data_loader.preprocess(sample)
    
    # Ensure tensor is float32 (same as data_loader preprocessing)
    if single_input.dtype != torch.float32:
        single_input = single_input.float()

    # Get batch size from configuration
    batch_size = onnx_settings.get("batch_size", 1)
    if batch_size is None:
        input_tensor = single_input
        logger.info("Using dynamic batch size")
    else:
        input_tensor = single_input.repeat(batch_size, 1, 1, 1)
        logger.info(f"Using fixed batch size: {batch_size}")

    # Use unified ONNXExporter
    exporter = ONNXExporter(onnx_settings, logger)
    success = exporter.export(pytorch_model, input_tensor, output_path)

    if success:
        logger.info(f"✅ ONNX export successful: {output_path}")
        return output_path
    else:
        logger.error(f"❌ ONNX export failed")
        return None


def export_tensorrt(
    onnx_path: str,
    config: BaseDeploymentConfig,
    data_loader: CalibrationDataLoader,
    logger: logging.Logger
) -> str:
    """
    Export ONNX model to TensorRT engine using the unified TensorRTExporter.
    
    Returns:
        Path to exported TensorRT engine, or None if export failed
    """
    logger.info("=" * 80)
    logger.info("Exporting to TensorRT (Using Unified TensorRTExporter)")
    logger.info("=" * 80)

    trt_settings = config.get_tensorrt_settings()
    output_path = onnx_path.replace(".onnx", ".engine")

    # Get sample input for shape configuration
    sample_idx = config.runtime_config.get("sample_idx", 0)
    sample = data_loader.load_sample(sample_idx)
    sample_input = data_loader.preprocess(sample)
    
    # Ensure tensor is float32
    if sample_input.dtype != torch.float32:
        sample_input = sample_input.float()

    # Merge backend_config.model_inputs into trt_settings for TensorRTExporter
    # TensorRTExporter expects model_inputs in the config
    if hasattr(config, 'backend_config') and hasattr(config.backend_config, 'model_inputs'):
        trt_settings = trt_settings.copy()
        trt_settings['model_inputs'] = config.backend_config.model_inputs

    # Use unified TensorRTExporter
    exporter = TensorRTExporter(trt_settings, logger)
    success = exporter.export(
        model=None,  # Not used for TensorRT
        sample_input=sample_input,
        output_path=output_path,
        onnx_path=onnx_path
    )

    if success:
        logger.info(f"✅ TensorRT export successful: {output_path}")
        return output_path
    else:
        logger.error(f"❌ TensorRT export failed")
        return None


def get_models_to_evaluate(eval_config: dict, logger: logging.Logger) -> list:
    """
    Get list of models to evaluate from config.
    
    Args:
        eval_config: Evaluation configuration
        logger: Logger instance
        
    Returns:
        List of tuples (backend_name, model_path)
    """
    models_config = eval_config.get("models", {})
    models_to_evaluate = []
    
    backend_mapping = {
        "pytorch": "pytorch",
        "onnx": "onnx",
        "tensorrt": "tensorrt",
    }
    
    for backend_key, model_path in models_config.items():
        backend_name = backend_mapping.get(backend_key.lower())
        if backend_name and model_path:
            # Check if path exists and is valid for the backend
            is_valid = False
            
            if backend_name == "pytorch":
                # PyTorch: check if checkpoint file exists
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
            elif backend_name == "onnx":
                # ONNX: check if file exists
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path) and model_path.endswith('.onnx')
            elif backend_name == "tensorrt":
                # TensorRT: check if engine file exists
                is_valid = os.path.exists(model_path) and os.path.isfile(model_path) and (model_path.endswith('.engine') or model_path.endswith('.trt'))
            
            if is_valid:
                models_to_evaluate.append((backend_name, model_path))
                logger.info(f"  - {backend_name}: {model_path}")
            else:
                logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")
    
    return models_to_evaluate


def run_evaluation(
    data_loader: CalibrationDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """
    Run evaluation on specified models.
    
    Args:
        data_loader: Data loader for samples
        config: Deployment configuration
        model_cfg: Model configuration
        logger: Logger instance
    """
    eval_config = config.evaluation_config
    
    if not eval_config.get("enabled", False):
        logger.info("Evaluation disabled, skipping...")
        return
    
    logger.info("=" * 80)
    logger.info("Running Evaluation")
    logger.info("=" * 80)
    
    # Get models to evaluate from config
    models_to_evaluate = get_models_to_evaluate(eval_config, logger)
    
    if not models_to_evaluate:
        logger.warning("No models found for evaluation")
        return
    
    evaluator = ClassificationEvaluator(model_cfg)
    
    num_samples = eval_config.get("num_samples", 10)
    if num_samples == -1:
        num_samples = data_loader.get_num_samples()
    
    verbose_mode = eval_config.get("verbose", False)
    
    all_results = {}
    
    for backend, model_path in models_to_evaluate:
        results = evaluator.evaluate(
            model_path=model_path,
            data_loader=data_loader,
            num_samples=num_samples,
            backend=backend,
            device=config.export_config.device,
            verbose=verbose_mode,
        )
        
        all_results[backend] = results
        
        logger.info(f"\n{backend.upper()} Results:")
        evaluator.print_results(results)
    
    # Compare results across backends
    if len(all_results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Comparison")
        logger.info("=" * 80)
        
        for backend, results in all_results.items():
            logger.info(f"\n{backend.upper()}:")
            if results and "error" not in results:
                logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                if 'latency_stats' in results:
                    logger.info(f"  Latency: {results['latency_stats']['mean_ms']:.2f} ± {results['latency_stats']['std_ms']:.2f} ms")
            else:
                logger.info("  No results available")


def run_verification(
    pytorch_checkpoint: str,
    onnx_path: str,
    tensorrt_path: str,
    data_loader: CalibrationDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """
    Run verification on exported models.
    
    Args:
        pytorch_checkpoint: Path to PyTorch checkpoint (reference)
        onnx_path: Path to ONNX model file
        tensorrt_path: Path to TensorRT engine file
        data_loader: Data loader for test samples
        config: Deployment configuration
        model_cfg: Model configuration
        logger: Logger instance
    """
    # Verification
    if not config.export_config.verify:
        logger.info("Verification disabled, skipping...")
        return

    if not pytorch_checkpoint:
        logger.warning("PyTorch checkpoint path not available, skipping verification")
        return
    
    if not onnx_path and not tensorrt_path:
        logger.info("No exported models to verify, skipping verification")
        return
    
    # Get verification parameters
    num_verify_samples = config.verification_config.get("num_verify_samples", 3)
    tolerance = config.verification_config.get("tolerance", 0.1)
    
    # Create evaluator
    evaluator = ClassificationEvaluator(model_cfg)
    
    # Run verification
    verification_results = evaluator.verify(
        pytorch_model_path=pytorch_checkpoint,
        onnx_model_path=onnx_path,
        tensorrt_model_path=tensorrt_path,
        data_loader=data_loader,
        num_samples=num_verify_samples,
        device=config.export_config.device,
        tolerance=tolerance,
        verbose=False,
    )
    
    # Check if verification succeeded
    if 'summary' in verification_results:
        summary = verification_results['summary']
        if summary['failed'] == 0:
            if summary.get('skipped', 0) > 0:
                logger.info(f"\n✅ All verifications passed! ({summary['skipped']} skipped)")
            else:
                logger.info("\n✅ All verifications passed!")
        else:
            logger.warning(f"\n⚠️  {summary['failed']}/{summary['total']} verifications failed")
    else:
        logger.error("\n❌ Verification encountered errors")


def main():
    """Main deployment pipeline with new architecture."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("CalibrationStatusClassification Deployment - New Pipeline Architecture")
    logger.info("=" * 80)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    # Override from command line
    if args.work_dir:
        config.export_config.work_dir = args.work_dir
    if args.device:
        config.export_config.device = args.device

    logger.info("=" * 80)
    logger.info("CalibrationStatusClassification Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")

    # Get info_pkl path
    info_pkl = config.runtime_config.get("info_pkl")
    if not info_pkl:
        logger.error("info_pkl path must be provided in config")
        return

    # Create data loader (calibrated version for export)
    logger.info("\nCreating data loader...")
    data_loader = CalibrationDataLoader(
        info_pkl_path=info_pkl,
        model_cfg=model_cfg,
        miscalibration_probability=0.0,
        device=config.export_config.device,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Check if we need model loading and export
    eval_config = config.evaluation_config
    needs_export = config.export_config.mode != "none"
    needs_onnx_eval = False
    if eval_config.get("enabled", False):
        models_to_eval = eval_config.get("models", {})
        if models_to_eval.get("onnx") or models_to_eval.get("tensorrt"):
            needs_onnx_eval = True
    
    # Load model if needed for export or ONNX/TensorRT evaluation
    pytorch_model = None
    onnx_path = None
    tensorrt_path = None
    
    if needs_export or needs_onnx_eval:
        if not args.checkpoint:
            if needs_export:
                logger.error("Checkpoint required for export")
            else:
                logger.error("Checkpoint required for ONNX/TensorRT evaluation")
            logger.error("Please provide --checkpoint argument")
            return
        
        # Load PyTorch model
        logger.info("\nLoading PyTorch model...")
        pytorch_model = load_pytorch_model(model_cfg, args.checkpoint, config.export_config.device)
        
        # Export ONNX
        if config.export_config.should_export_onnx():
            onnx_path = export_onnx(pytorch_model, data_loader, config, logger)
        
        # Export TensorRT
        if config.export_config.should_export_tensorrt() and onnx_path:
            tensorrt_path = export_tensorrt(onnx_path, config, data_loader, logger)
    
    # Get model paths from evaluation config if not exported
    if not onnx_path or not tensorrt_path:
        eval_models = config.evaluation_config.get("models", {})
        if not onnx_path:
            onnx_path = eval_models.get("onnx")
            if onnx_path and not os.path.exists(onnx_path):
                logger.warning(f"ONNX file from config does not exist: {onnx_path}")
                onnx_path = None
        if not tensorrt_path:
            tensorrt_path = eval_models.get("tensorrt")
            if tensorrt_path and not os.path.exists(tensorrt_path):
                logger.warning(f"TensorRT engine from config does not exist: {tensorrt_path}")
                tensorrt_path = None

    # Verification
    run_verification(
        pytorch_checkpoint=args.checkpoint,
        onnx_path=onnx_path,
        tensorrt_path=tensorrt_path,
        data_loader=data_loader,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    # Run evaluation
    run_evaluation(data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

