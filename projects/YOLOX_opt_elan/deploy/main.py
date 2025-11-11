"""
YOLOX_opt_elan Deployment Main Script (Unified Runner Architecture).

This script uses the unified deployment runner to handle the complete deployment workflow:
- Unified interface across PyTorch, ONNX, and TensorRT backends
- Simplified export and evaluation workflow
- Easy cross-backend verification
"""

import os
import sys
from pathlib import Path

import torch
from mmdet.apis import init_detector
from mmengine.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import BaseDeploymentConfig, setup_logging
from autoware_ml.deployment.core.base_config import parse_base_args
from autoware_ml.deployment.exporters.onnx_exporter import ONNXExporter
from autoware_ml.deployment.runners import DeploymentRunner
from projects.YOLOX_opt_elan.deploy.data_loader import YOLOXOptElanDataLoader
from projects.YOLOX_opt_elan.deploy.evaluator import YOLOXOptElanEvaluator
from projects.YOLOX_opt_elan.deploy.onnx_wrapper import YOLOXONNXWrapper


def parse_args():
    """Parse command line arguments."""
    parser = parse_base_args()
    args = parser.parse_args()
    return args


def load_pytorch_model(checkpoint_path: str, model_cfg_path: str, device: str, **kwargs):
    """Load PyTorch model from checkpoint."""
    model = init_detector(model_cfg_path, checkpoint_path, device=device)
    model.eval()
    return model


def export_onnx(
    pytorch_model,
    data_loader: YOLOXOptElanDataLoader,
    config: BaseDeploymentConfig,
    model_cfg: Config,
    logger,
    **kwargs
) -> str:
    """
    Export model to ONNX format using the unified ONNXExporter.
    
    Note: YOLOX requires special wrapper (YOLOXONNXWrapper) for ONNX export.
    
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
    sample_idx = config.runtime_config.get("sample_idx", 0)
    sample = data_loader.load_sample(sample_idx)
    single_input = data_loader.preprocess(sample)
    
    # Ensure tensor is float32
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

    # Replace ReLU6 with ReLU for better ONNX compatibility
    def replace_relu6_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU6):
                setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
            else:
                replace_relu6_with_relu(child)

    replace_relu6_with_relu(pytorch_model)

    # Wrap model for ONNX export (YOLOX-specific requirement)
    num_classes = model_cfg.model.bbox_head.num_classes
    wrapped_model = YOLOXONNXWrapper(model=pytorch_model, num_classes=num_classes)
    wrapped_model.eval()

    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output format: [batch_size, num_predictions, {4 + 1 + num_classes}]")
    logger.info(f"Output path: {output_path}")

    # Update output_names in onnx_settings to match YOLOX requirement
    onnx_settings_updated = onnx_settings.copy()
    onnx_settings_updated["output_names"] = ["output"]

    # Use unified ONNXExporter
    exporter = ONNXExporter(onnx_settings_updated, logger)
    success = exporter.export(wrapped_model, input_tensor, output_path)

    if success:
        logger.info(f"✅ ONNX export successful: {output_path}")
        return output_path
    else:
        logger.error(f"❌ ONNX export failed")
        return None


def main():
    """Main deployment pipeline using unified runner."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment - Unified Runner Architecture")
    logger.info("=" * 80)

    # Load configs
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    model_cfg_path = args.model_cfg
    config = BaseDeploymentConfig(deploy_cfg)

    # Override from command line
    if args.work_dir:
        config.export_config.work_dir = args.work_dir
    if args.device:
        config.export_config.device = args.device

    logger.info("=" * 80)
    logger.info("YOLOX_opt_elan Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("Deployment Configuration:")
    logger.info(f"  Export mode: {config.export_config.mode}")
    logger.info(f"  Device: {config.export_config.device}")
    logger.info(f"  Work dir: {config.export_config.work_dir}")
    logger.info(f"  Verify: {config.export_config.verify}")

    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = YOLOXOptElanDataLoader(
        ann_file=config.runtime_config["ann_file"],
        img_prefix=config.runtime_config.get("img_prefix", ""),
        model_cfg=model_cfg,
        device=config.export_config.device,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    # Create evaluator
    evaluator = YOLOXOptElanEvaluator(model_cfg, model_cfg_path=model_cfg_path)

    # Create unified runner with custom functions
    runner = DeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
        load_model_fn=lambda checkpoint_path, **kwargs: load_pytorch_model(
            checkpoint_path, model_cfg_path=model_cfg_path, device=config.export_config.device, **kwargs
        ),
        export_onnx_fn=lambda pytorch_model, data_loader, config, logger, **kwargs: export_onnx(
            pytorch_model, data_loader, config, model_cfg, logger, **kwargs
        ),
    )

    # Execute deployment workflow
    runner.run(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
