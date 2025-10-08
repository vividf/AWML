"""
YOLOX Deployment Main Script.

This script handles the complete deployment pipeline for YOLOX:
- Export to ONNX and/or TensorRT
- Verify outputs across backends
- Evaluate model performance
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmpretrain.apis import get_model as get_classification_model

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.core.verification import verify_models
from autoware_ml.deployment.exporters import ONNXExporter, TensorRTExporter

from .data_loader import YOLOXDataLoader
from .evaluator import YOLOXEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(model_cfg: Config, checkpoint_path: str, device: str):
    """
    Load PyTorch model from checkpoint.

    Args:
        model_cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    from mmdet.apis import init_detector

    model = init_detector(model_cfg, checkpoint_path, device=device)
    model.eval()

    return model


def export_onnx(model, data_loader: YOLOXDataLoader, config: BaseDeploymentConfig, logger: logging.Logger) -> str:
    """
    Export model to ONNX format.

    Returns:
        Path to exported ONNX file
    """
    logger.info("=" * 80)
    logger.info("Exporting to ONNX")
    logger.info("=" * 80)

    # Get sample input
    sample_idx = config.runtime_config.get("sample_idx", 0)
    input_tensor = data_loader.load_and_preprocess(sample_idx)

    # Get ONNX settings
    onnx_settings = config.get_onnx_settings()
    output_path = os.path.join(config.export_config.work_dir, onnx_settings["save_file"])

    os.makedirs(config.export_config.work_dir, exist_ok=True)

    # Export
    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output path: {output_path}")

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        opset_version=onnx_settings["opset_version"],
        do_constant_folding=onnx_settings["do_constant_folding"],
        input_names=onnx_settings["input_names"],
        output_names=onnx_settings["output_names"],
        dynamic_axes=onnx_settings.get("dynamic_axes"),
        export_params=onnx_settings.get("export_params", True),
        keep_initializers_as_inputs=onnx_settings.get("keep_initializers_as_inputs", False),
    )

    logger.info(f"✅ ONNX export successful: {output_path}")

    return output_path


def export_tensorrt(onnx_path: str, config: BaseDeploymentConfig, logger: logging.Logger) -> str:
    """
    Export ONNX model to TensorRT.

    Returns:
        Path to exported TensorRT engine
    """
    logger.info("=" * 80)
    logger.info("Exporting to TensorRT")
    logger.info("=" * 80)

    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not installed. Please install TensorRT.")
        return None

    trt_settings = config.get_tensorrt_settings()
    output_path = onnx_path.replace(".onnx", ".engine")

    # Build TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # Build config
    config_trt = builder.create_builder_config()
    config_trt.max_workspace_size = trt_settings["max_workspace_size"]

    # Set precision
    precision_flags = trt_settings["policy_flags"]
    if "FP16" in precision_flags and precision_flags["FP16"]:
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision")
    if "TF32" in precision_flags and precision_flags["TF32"]:
        config_trt.set_flag(trt.BuilderFlag.TF32)
        logger.info("Enabled TF32 precision")

    # Build engine
    logger.info("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config_trt)

    if engine is None:
        logger.error("Failed to build TensorRT engine")
        return None

    # Serialize and save
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    logger.info(f"✅ TensorRT export successful: {output_path}")

    return output_path


def run_evaluation(
    model_paths: dict,
    data_loader: YOLOXDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger: logging.Logger,
):
    """Run evaluation on specified models."""
    eval_config = config.evaluation_config

    if not eval_config.get("enabled", False):
        logger.info("Evaluation disabled, skipping...")
        return

    logger.info("=" * 80)
    logger.info("Running Evaluation")
    logger.info("=" * 80)

    evaluator = YOLOXEvaluator(model_cfg)

    num_samples = eval_config.get("num_samples", 100)
    if num_samples == -1:
        num_samples = data_loader.get_num_samples()

    models_to_eval = eval_config.get("models_to_evaluate", ["pytorch"])

    all_results = {}

    for backend in models_to_eval:
        if backend not in model_paths or model_paths[backend] is None:
            logger.warning(f"Model for backend '{backend}' not available, skipping...")
            continue

        results = evaluator.evaluate(
            model_path=model_paths[backend],
            data_loader=data_loader,
            num_samples=num_samples,
            backend=backend,
            device=config.export_config.device,
            verbose=eval_config.get("verbose", False),
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
            logger.info(f"  mAP: {results['mAP']:.4f}")
            logger.info(f"  mAP@50: {results['mAP_50']:.4f}")
            logger.info(f"  Latency: {results['latency']['mean_ms']:.2f} ± {results['latency']['std_ms']:.2f} ms")


def main():
    """Main deployment pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)

    config = BaseDeploymentConfig(deploy_cfg)

    # Override from command line
    if args.work_dir:
        config.export_config.work_dir = args.work_dir
    if args.device:
        config.export_config.device = args.device

    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = YOLOXDataLoader(
        ann_file=config.runtime_config["ann_file"],
        img_prefix=config.runtime_config["img_prefix"],
        model_cfg=model_cfg,
        device=config.export_config.device,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Track model paths
    model_paths = {}

    # Load PyTorch model if needed
    pytorch_model = None
    if config.export_config.mode != "none" or "pytorch" in config.evaluation_config.get("models_to_evaluate", []):
        if args.checkpoint:
            logger.info("\nLoading PyTorch model...")
            pytorch_model = load_pytorch_model(model_cfg, args.checkpoint, config.export_config.device)
            model_paths["pytorch"] = args.checkpoint
        else:
            logger.error("Checkpoint required for PyTorch model")
            return

    # Export ONNX
    onnx_path = None
    if config.export_config.should_export_onnx():
        onnx_path = export_onnx(pytorch_model, data_loader, config, logger)
        if onnx_path:
            model_paths["onnx"] = onnx_path
    elif config.runtime_config.get("onnx_file"):
        onnx_path = config.runtime_config["onnx_file"]
        model_paths["onnx"] = onnx_path

    # Export TensorRT
    trt_path = None
    if config.export_config.should_export_tensorrt() and onnx_path:
        trt_path = export_tensorrt(onnx_path, config, logger)
        if trt_path:
            model_paths["tensorrt"] = trt_path

    # Verification
    if config.export_config.verify and len(model_paths) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Verification")
        logger.info("=" * 80)
        logger.info("TODO: Implement verification")
        # verify_models(...)

    # Evaluation
    run_evaluation(model_paths, data_loader, config, model_cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Deployment Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
