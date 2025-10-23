"""
CalibrationStatusClassification Model Deployment Script (Refactored)

This script exports CalibrationStatusClassification models using the unified
deployment framework, with comprehensive verification and performance benchmarking.

Features:
- ONNX export with optimization
- TensorRT conversion with precision policy support
- Unified verification across backends
- Full model evaluation with metrics
- Performance benchmarking with latency statistics
- Confusion matrix and per-class accuracy analysis
"""

import logging
import os.path as osp

import mmengine
import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

from autoware_ml.deployment.core import BaseDeploymentConfig, parse_base_args, setup_logging, verify_model_outputs
from autoware_ml.deployment.exporters import ONNXExporter, TensorRTExporter
from projects.CalibrationStatusClassification.deploy.data_loader import CalibrationDataLoader
from projects.CalibrationStatusClassification.deploy.evaluator import (
    ClassificationEvaluator,
    get_models_to_evaluate,
    run_full_evaluation,
)


class CalibrationDeploymentConfig(BaseDeploymentConfig):
    """Extended configuration for CalibrationStatusClassification deployment."""

    def __init__(self, deploy_cfg: Config):
        super().__init__(deploy_cfg)
        # CalibrationStatus-specific config can be added here if needed


def main():
    """Main deployment function."""
    # Parse arguments
    parser = parse_base_args()
    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Load configurations
    logger.info(f"Loading deploy config from: {args.deploy_cfg}")
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    config = CalibrationDeploymentConfig(deploy_cfg)

    logger.info(f"Loading model config from: {args.model_cfg}")
    model_cfg = Config.fromfile(args.model_cfg)

    # Get configuration
    work_dir = args.work_dir or config.export_config.work_dir
    device = args.device or config.export_config.device
    info_pkl = config.runtime_config.get("info_pkl")
    sample_idx = config.runtime_config.get("sample_idx", 0)
    existing_onnx = config.runtime_config.get("onnx_file")
    export_mode = config.export_config.mode

    # Validate required parameters
    if not info_pkl:
        logger.error("info_pkl path must be provided in config")
        return

    # Setup working directory
    mmengine.mkdir_or_exist(osp.abspath(work_dir))
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Export mode: {export_mode}")

    # Check if eval-only mode
    is_eval_only = export_mode == "none"

    # Validate eval-only mode configuration
    if is_eval_only:
        eval_enabled = config.evaluation_config.get("enabled", False)
        if not eval_enabled:
            logger.error(
                "Configuration error: export mode is 'none' but evaluation.enabled is False. "
                "Please set evaluation.enabled = True in your config."
            )
            return

    # Validate checkpoint requirement
    if not is_eval_only and not args.checkpoint:
        logger.error("Checkpoint is required when export mode is not 'none'")
        return

    # Export phase
    if not is_eval_only:
        logger.info(f"\n{'='*70}")
        logger.info("Starting model export...")
        logger.info(f"{'='*70}\n")

        # Determine export paths
        onnx_path = None
        trt_path = None

        onnx_settings = config.get_onnx_settings()

        if config.export_config.should_export_onnx():
            onnx_path = osp.join(work_dir, onnx_settings["save_file"])

        if config.export_config.should_export_tensorrt():
            if existing_onnx and not config.export_config.should_export_onnx():
                onnx_path = existing_onnx
                if not osp.exists(onnx_path):
                    logger.error(f"Provided ONNX file does not exist: {onnx_path}")
                    return
            elif not onnx_path:
                logger.error("TensorRT export requires ONNX file. Set mode='both' or provide onnx_file in config.")
                return

            trt_file = onnx_settings["save_file"].replace(".onnx", ".engine")
            trt_path = osp.join(work_dir, trt_file)

        # Load model
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        torch_device = torch.device(device)
        model = get_model(model_cfg, args.checkpoint, device=torch_device)

        # Create data loaders for calibrated and miscalibrated samples
        logger.info(f"Loading sample data from info.pkl: {info_pkl}")
        data_loader_calibrated = CalibrationDataLoader(
            info_pkl_path=info_pkl,
            model_cfg=model_cfg,
            miscalibration_probability=0.0,
            device=device,
        )
        data_loader_miscalibrated = CalibrationDataLoader(
            info_pkl_path=info_pkl,
            model_cfg=model_cfg,
            miscalibration_probability=1.0,
            device=device,
        )

        # Load sample inputs
        input_tensor_calibrated = data_loader_calibrated.load_and_preprocess(sample_idx)
        input_tensor_miscalibrated = data_loader_miscalibrated.load_and_preprocess(sample_idx)

        # Export ONNX
        if config.export_config.should_export_onnx() and onnx_path:
            logger.info(f"\nExporting to ONNX...")
            onnx_exporter = ONNXExporter(onnx_settings, logger)
            success = onnx_exporter.export(model, input_tensor_calibrated, onnx_path)
            if success:
                logger.info(f"✓ ONNX export successful: {onnx_path}")
                onnx_exporter.validate_export(onnx_path)
            else:
                logger.error("✗ ONNX export failed")
                return

        # Export TensorRT
        if config.export_config.should_export_tensorrt() and trt_path and onnx_path:
            logger.info(f"\nExporting to TensorRT...")
            trt_settings = config.get_tensorrt_settings()
            trt_exporter = TensorRTExporter(trt_settings, logger)
            success = trt_exporter.export(model, input_tensor_calibrated, trt_path, onnx_path=onnx_path)
            if success:
                logger.info(f"✓ TensorRT export successful: {trt_path}")
            else:
                logger.error("✗ TensorRT export failed")
                return

        # Run verification if requested
        if config.export_config.verify:
            logger.info(f"\n{'='*70}")
            logger.info("Running model verification...")
            logger.info(f"{'='*70}\n")

            test_inputs = {
                "miscalibrated_sample": input_tensor_miscalibrated,
                "calibrated_sample": input_tensor_calibrated,
            }

            verify_onnx = (
                onnx_path if config.export_config.should_export_onnx() else (existing_onnx if existing_onnx else None)
            )
            verify_trt = trt_path if config.export_config.should_export_tensorrt() else None

            verification_results = verify_model_outputs(
                pytorch_model=model,
                test_inputs=test_inputs,
                onnx_path=verify_onnx,
                tensorrt_path=verify_trt,
                device=device,
                logger=logger,
                model_type="CalibrationStatusClassification",  # Explicitly specify model type
            )

            # Check if all verifications passed
            all_passed = all(verification_results.values())
            if all_passed:
                logger.info("✓ All verifications PASSED")
            else:
                logger.warning("⚠ Some verifications FAILED")

        # Log exported formats
        exported_formats = []
        if config.export_config.should_export_onnx():
            exported_formats.append("ONNX")
        if config.export_config.should_export_tensorrt():
            exported_formats.append("TensorRT")
        if exported_formats:
            logger.info(f"\nExported formats: {', '.join(exported_formats)}")

        logger.info(f"\n{'='*70}")
        logger.info("Deployment completed successfully!")
        logger.info(f"{'='*70}\n")
    else:
        logger.info("Evaluation-only mode: Skipping model loading and export\n")

    # Evaluation phase
    eval_cfg = config.evaluation_config
    should_evaluate = eval_cfg.get("enabled", False)
    num_samples = eval_cfg.get("num_samples", 10)
    verbose_mode = eval_cfg.get("verbose", False)

    if should_evaluate:
        logger.info(f"\n{'='*70}")
        logger.info("Starting full model evaluation...")
        logger.info(f"{'='*70}\n")

        models_to_evaluate = get_models_to_evaluate(eval_cfg, logger)

        if models_to_evaluate:
            run_full_evaluation(
                models_to_evaluate,
                model_cfg,
                info_pkl,
                device,
                num_samples,
                verbose_mode,
                logger,
            )
        else:
            logger.warning("No models found for evaluation")


if __name__ == "__main__":
    main()
