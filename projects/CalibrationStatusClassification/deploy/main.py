"""
CalibrationStatusClassification Model Deployment Script

This script exports CalibrationStatusClassification models to ONNX and TensorRT formats,
with comprehensive verification and performance benchmarking.

Features:
- ONNX export with optimization
- TensorRT conversion with precision policy support
- Dual verification (ONNX + TensorRT) on single samples
- Full model evaluation on multiple samples with metrics
- Performance benchmarking with latency statistics
- Confusion matrix and per-class accuracy analysis
"""

import os.path as osp

import mmengine
import torch
from config import DeploymentConfig, parse_args, setup_logging
from evaluator import get_models_to_evaluate, run_full_evaluation
from exporters import (
    export_models,
    load_sample_data_from_info_pkl,
    run_model_verification,
    validate_and_prepare_paths,
)
from mmengine.config import Config
from mmpretrain.apis import get_model


def main():
    """Main deployment function."""
    args = parse_args()
    logger = setup_logging(args.log_level)

    # Load configurations
    logger.info(f"Loading deploy config from: {args.deploy_cfg}")
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    config = DeploymentConfig(deploy_cfg)

    logger.info(f"Loading model config from: {args.model_cfg}")
    model_cfg = Config.fromfile(args.model_cfg)

    # Extract runtime configuration
    work_dir = config.work_dir
    device = config.device
    info_pkl = config.runtime_io_config.get("info_pkl")
    sample_idx = config.runtime_io_config.get("sample_idx", 0)
    existing_onnx = config.runtime_io_config.get("onnx_file")
    export_mode = config.export_config.get("mode", "both")

    # Validate required parameters
    if not info_pkl:
        logger.error("info_pkl path must be provided either in config or via --info-pkl")
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
                "Configuration error: export mode is 'none' (evaluation-only mode) "
                "but evaluation.enabled is False. "
                "Please set evaluation.enabled = True in your config."
            )
            return

    # Validate checkpoint requirement
    if not is_eval_only and not args.checkpoint:
        logger.error("Checkpoint is required when export mode is not 'none'")
        return

    # Export phase
    if not is_eval_only:
        # Determine export paths
        onnx_path, trt_path = validate_and_prepare_paths(config, work_dir, existing_onnx, logger)
        if onnx_path is None and trt_path is None:
            return

        # Load model
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        device = torch.device(device)
        model = get_model(model_cfg, args.checkpoint, device=device)

        # Load sample data
        logger.info(f"Loading sample data from info.pkl: {info_pkl}")
        input_tensor_calibrated = load_sample_data_from_info_pkl(info_pkl, model_cfg, 0.0, sample_idx, device=device)
        input_tensor_miscalibrated = load_sample_data_from_info_pkl(
            info_pkl, model_cfg, 1.0, sample_idx, device=device
        )

        # Export models
        success, device = export_models(model, config, onnx_path, trt_path, input_tensor_calibrated, device, logger)
        if not success:
            logger.error("Export failed")
            return

        # Update tensors if device changed
        if device.type == "cuda":
            input_tensor_calibrated = input_tensor_calibrated.to(device)
            input_tensor_miscalibrated = input_tensor_miscalibrated.to(device)

        # Run verification
        run_model_verification(
            model,
            config,
            onnx_path,
            trt_path,
            input_tensor_calibrated,
            input_tensor_miscalibrated,
            existing_onnx,
            logger,
        )

        # Log exported formats
        exported_formats = []
        if config.should_export_onnx:
            exported_formats.append("ONNX")
        if config.should_export_tensorrt:
            exported_formats.append("TensorRT")
        if exported_formats:
            logger.info(f"Exported formats: {', '.join(exported_formats)}")

        logger.info("Deployment completed successfully!")
    else:
        logger.info("Evaluation-only mode: Skipping model loading and export")

    # Evaluation phase
    eval_cfg = config.evaluation_config
    should_evaluate = eval_cfg.get("enabled", False)
    num_samples = eval_cfg.get("num_samples", 10)
    verbose_mode = eval_cfg.get("verbose", False)

    if should_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info("Starting full model evaluation...")
        logger.info(f"{'='*60}")

        models_to_evaluate = get_models_to_evaluate(eval_cfg, logger)
        run_full_evaluation(
            models_to_evaluate,
            model_cfg,
            info_pkl,
            device,
            num_samples,
            verbose_mode,
            logger,
        )


if __name__ == "__main__":
    main()
