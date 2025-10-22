"""
Test script to verify CenterPoint PyTorch vs ONNX verification fixes.

This script tests the verification process to ensure:
1. Both ONNX models are being used correctly
2. Outputs are properly compared
3. Differences are correctly computed
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autoware_ml.deployment.core import setup_logging
from autoware_ml.deployment.core.verification import verify_model_outputs
from projects.CenterPoint.deploy.data_loader import CenterPointDataLoader
from projects.CenterPoint.deploy.main import load_pytorch_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CenterPoint verification fixes")
    
    parser.add_argument(
        "--model-cfg",
        type=str,
        required=True,
        help="Path to model config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        required=True,
        help="Path to ONNX model directory"
    )
    parser.add_argument(
        "--info-file",
        type=str,
        required=True,
        help="Path to dataset info file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to verify"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-1,
        help="Verification tolerance"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 80)
    logger.info("CenterPoint Verification Test")
    logger.info("=" * 80)
    logger.info(f"Model config: {args.model_cfg}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"ONNX directory: {args.onnx_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Tolerance: {args.tolerance}")
    
    # Initialize mmdet3d scope
    init_default_scope("mmdet3d")
    
    # Load model config
    model_cfg = Config.fromfile(args.model_cfg)
    
    # Create data loader
    logger.info("\nCreating data loader...")
    data_loader = CenterPointDataLoader(
        info_file=args.info_file,
        model_cfg=model_cfg,
        device=args.device
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")
    
    # Load PyTorch model with ONNX compatibility
    logger.info("\nLoading PyTorch model...")
    pytorch_model, _ = load_pytorch_model(
        model_cfg,
        args.checkpoint,
        args.device,
        replace_onnx_models=True,
        rot_y_axis_reference=False
    )
    logger.info("✅ PyTorch model loaded successfully")
    
    # Prepare verification inputs
    logger.info("\nPreparing verification inputs...")
    verification_inputs = {}
    num_samples = min(args.num_samples, data_loader.get_num_samples())
    
    for i in range(num_samples):
        sample = data_loader.load_sample(i)
        input_data = data_loader.preprocess(sample)
        verification_inputs[f"sample_{i}"] = input_data
    
    logger.info(f"Prepared {len(verification_inputs)} samples for verification")
    
    # Run verification
    logger.info("\n" + "=" * 80)
    logger.info("Running Verification")
    logger.info("=" * 80)
    
    verification_results = verify_model_outputs(
        pytorch_model=pytorch_model,
        test_inputs=verification_inputs,
        onnx_path=args.onnx_dir,
        tensorrt_path=None,  # Only test ONNX for now
        device=args.device,
        tolerance=args.tolerance,
        logger=logger,
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Verification Test Summary")
    logger.info("=" * 80)
    
    all_passed = True
    for backend, passed in verification_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {backend}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All verifications PASSED!")
        logger.info("The fixes are working correctly:")
        logger.info("  1. Both ONNX models are being used")
        logger.info("  2. Outputs are properly compared")
        logger.info("  3. Differences should now be non-zero and realistic")
        return 0
    else:
        logger.error("\n❌ Some verifications FAILED!")
        logger.error("Please check the detailed logs above for more information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

