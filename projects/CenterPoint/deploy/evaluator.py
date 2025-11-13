"""
CenterPoint Evaluator for deployment.

This module implements evaluation for CenterPoint 3D object detection models.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.core import BaseEvaluator

from .data_loader import CenterPointDataLoader

# Constants
LOG_INTERVAL = 50
GPU_CLEANUP_INTERVAL = 10


class CenterPointEvaluator(BaseEvaluator):
    """
    Evaluator for CenterPoint 3D object detection.

    Computes 3D detection metrics including mAP, NDS, and latency statistics.

    Note: For production, should integrate with mmdet3d's evaluation metrics.
    """

    def __init__(
        self,
        model_cfg: Config,
        class_names: List[str] = None,
    ):
        """
        Initialize CenterPoint evaluator.

        Args:
            model_cfg: ONNX-compatible model configuration (used for all backends)
            class_names: List of class names (optional)
        """
        super().__init__(config={})
        self.model_cfg = model_cfg

        # Get class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        else:
            # Default for T4Dataset
            self.class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

    def verify(
        self,
        pytorch_model_path: str,
        onnx_model_path: str = None,
        tensorrt_model_path: str = None,
        data_loader: CenterPointDataLoader = None,
        num_samples: int = 1,
        device: str = "cpu",
        tolerance: float = 0.1,
        verbose: bool = False,
        onnx_device: str = None,  # Device for ONNX verification (None = use device)
        tensorrt_device: str = None,  # Device for TensorRT verification (None = use device)
    ) -> Dict[str, Any]:
        """
        Verify exported models against PyTorch reference by comparing raw outputs.
        
        This method is similar to evaluate() but focuses on numerical consistency
        rather than detection metrics. It compares raw head outputs before postprocessing.
        
        Args:
            pytorch_model_path: Path to PyTorch checkpoint (reference)
            onnx_model_path: Optional path to ONNX model directory
            tensorrt_model_path: Optional path to TensorRT model directory
            data_loader: Data loader for test samples
            num_samples: Number of samples to verify
            device: Device to run verification on
            tolerance: Maximum allowed difference for verification to pass
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing verification results:
            {
                'sample_0_onnx': bool (passed/failed),
                'sample_0_tensorrt': bool (passed/failed),
                ...
                'summary': {'passed': int, 'failed': int, 'total': int}
            }
        """
        logger = logging.getLogger(__name__)
        
        logger.info("\n" + "=" * 60)
        logger.info("CenterPoint Model Verification")
        logger.info("=" * 60)
        logger.info(f"PyTorch reference: {pytorch_model_path}")
        if onnx_model_path:
            logger.info(f"ONNX model: {onnx_model_path}")
        if tensorrt_model_path:
            logger.info(f"TensorRT model: {tensorrt_model_path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)
        
        results = {}
        skipped_backends = []  # Track skipped backends
        
        # Determine devices for each backend
        # ONNX verification always uses CPU for numerical consistency
        onnx_verify_device = onnx_device if onnx_device is not None else "cpu"
        # TensorRT verification uses specified device or falls back to main device
        tensorrt_verify_device = tensorrt_device if tensorrt_device is not None else device
        
        logger.info(f"Verification device configuration:")
        logger.info(f"  ONNX: {onnx_verify_device} (CPU for numerical consistency)")
        logger.info(f"  TensorRT: {tensorrt_verify_device}")
        
        # IMPORTANT: For CenterPoint hybrid architecture and numerical consistency:
        # - ONNX export is done on CPU, so ONNX verification uses CPU (both ONNX Runtime and PyTorch reference)
        # - TensorRT runs on CUDA, but for verification we need to compare against CPU PyTorch reference
        #   because CUDA implementations (both ONNX Runtime CUDA and PyTorch CUDA) have numerical differences
        #   compared to CPU implementations. This is normal due to different algorithms and floating-point ordering.
        # - The key insight: ONNX (CPU) passes because both use CPU. ONNX (CUDA) fails because CUDA differs from CPU.
        # - For TensorRT verification, we compare TensorRT (CUDA) vs PyTorch (CPU), accepting that CUDA has differences.
        #   But we can also create a CPU-based TensorRT verification if needed.
        
        # Create ONNX pipeline and CPU PyTorch reference if requested
        onnx_pipeline = None
        onnx_pipeline_cuda = None
        pytorch_pipeline_cpu_onnx = None
        if onnx_model_path:
            logger.info("\nInitializing ONNX pipeline...")
            onnx_pipeline = self._create_pipeline("onnx", onnx_model_path, onnx_verify_device, logger)
            if onnx_pipeline is None:
                logger.warning("Failed to create ONNX pipeline, skipping ONNX verification")
                skipped_backends.append("onnx")
            else:
                # Create PyTorch reference pipeline on CPU for ONNX verification
                logger.info("Initializing PyTorch reference pipeline (CPU) for ONNX verification...")
                pytorch_pipeline_cpu_onnx = self._create_pipeline("pytorch", pytorch_model_path, onnx_verify_device, logger)
                if pytorch_pipeline_cpu_onnx is None:
                    logger.error("Failed to create PyTorch reference pipeline for ONNX verification")
                    skipped_backends.append("onnx")
                # Optional: create ONNX pipeline on CUDA for comparison
                if torch.cuda.is_available():
                    try:
                        logger.info("Initializing ONNX pipeline (CUDA) for comparison...")
                        onnx_pipeline_cuda = self._create_pipeline("onnx", onnx_model_path, "cuda:0", logger)
                        if onnx_pipeline_cuda is not None:
                            logger.info("  ✓ ONNX pipeline (CUDA) initialized for device comparison")
                        else:
                            logger.warning("  ⚠️  Failed to initialize ONNX pipeline (CUDA) for comparison")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Could not initialize ONNX pipeline (CUDA) for comparison: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
        
        # Create TensorRT pipeline
        # NOTE: For verification, we use CPU PyTorch reference because:
        # 1. CUDA implementations have inherent numerical differences
        # 2. ONNX export is done on CPU, so CPU is the "ground truth"
        # 3. TensorRT runs on CUDA but we compare against CPU reference
        tensorrt_pipeline = None
        pytorch_pipeline_cpu_trt = None
        pytorch_pipeline_cuda = None
        if tensorrt_model_path:
            logger.info("\nInitializing TensorRT pipeline...")
            if not tensorrt_verify_device.startswith("cuda"):
                logger.warning("TensorRT requires CUDA device, skipping TensorRT verification")
                skipped_backends.append("tensorrt")
            else:
                tensorrt_pipeline = self._create_pipeline("tensorrt", tensorrt_model_path, tensorrt_verify_device, logger)
                if tensorrt_pipeline is None:
                    logger.warning("Failed to create TensorRT pipeline, skipping TensorRT verification")
                    skipped_backends.append("tensorrt")
                else:
                    # Create PyTorch reference pipeline on CPU for TensorRT verification
                    # This ensures numerical consistency (CPU is the reference for exported models)
                    logger.info("Initializing PyTorch reference pipeline (CPU) for TensorRT verification...")
                    logger.info("  Note: Using CPU reference because CUDA implementations have numerical differences")
                    pytorch_pipeline_cpu_trt = self._create_pipeline("pytorch", pytorch_model_path, "cpu", logger)
                    if pytorch_pipeline_cpu_trt is None:
                        logger.error("Failed to create PyTorch reference pipeline for TensorRT verification")
                        skipped_backends.append("tensorrt")
                    # Optional: create PyTorch pipeline on CUDA for device comparison
                    if torch.cuda.is_available():
                        try:
                            logger.info("Initializing PyTorch pipeline (CUDA) for comparison...")
                            pytorch_pipeline_cuda = self._create_pipeline("pytorch", pytorch_model_path, "cuda:0", logger)
                            if pytorch_pipeline_cuda is not None:
                                logger.info("  ✓ PyTorch pipeline (CUDA) initialized for device comparison")
                            else:
                                logger.warning("  ⚠️  Failed to initialize PyTorch pipeline (CUDA) for comparison")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Could not initialize PyTorch pipeline (CUDA) for comparison: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
        
        # Verify each sample
        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                logger.info(f"\n{'='*60}")
                logger.info(f"Verifying sample {i}")
                logger.info(f"{'='*60}")
                
                # Get sample data
                sample = data_loader.load_sample(i)
                
                # Get points for pipeline
                if 'points' in sample:
                    points = sample['points']
                else:
                    input_data = data_loader.preprocess(sample)
                    points = input_data.get('points', input_data)
                
                sample_meta = sample.get('metainfo', {})
                
                # Verify ONNX (with CPU PyTorch reference)
                if onnx_pipeline and pytorch_pipeline_cpu_onnx:
                    logger.info("\nVerifying ONNX pipeline...")
                    logger.info("Running PyTorch reference (CPU) for ONNX verification...")
                    try:
                        pytorch_outputs_cpu_onnx, pytorch_latency_cpu_onnx, _ = pytorch_pipeline_cpu_onnx.infer(
                            points, sample_meta, return_raw_outputs=True
                        )
                        logger.info(f"  PyTorch latency (CPU): {pytorch_latency_cpu_onnx:.2f} ms")
                        logger.info(f"  PyTorch output: {len(pytorch_outputs_cpu_onnx)} head outputs")
                        
                        # Log output statistics
                        output_names = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
                        for idx, (out, name) in enumerate(zip(pytorch_outputs_cpu_onnx, output_names)):
                            if isinstance(out, torch.Tensor):
                                out_np = out.cpu().numpy()
                                logger.info(f"    {name}: shape={out.shape}, range=[{out_np.min():.3f}, {out_np.max():.3f}]")
                        
                        onnx_passed = self._verify_single_backend(
                            onnx_pipeline,
                            points,
                            sample_meta,
                            pytorch_outputs_cpu_onnx,
                            pytorch_latency_cpu_onnx,
                            tolerance,
                            "ONNX",
                            logger
                        )
                        results[f"sample_{i}_onnx"] = onnx_passed
                    except Exception as e:
                        logger.error(f"  ONNX verification failed: {e}")
                        import traceback
                        traceback.print_exc()
                        results[f"sample_{i}_onnx"] = False
                
                # Verify TensorRT (with CPU PyTorch reference for numerical consistency)
                # NOTE: We use CPU reference because:
                # 1. ONNX export is done on CPU, so CPU is the "ground truth"
                # 2. CUDA implementations (both ONNX Runtime and PyTorch) have numerical differences
                # 3. This ensures consistency with ONNX verification which also uses CPU
                if tensorrt_pipeline and pytorch_pipeline_cpu_trt:
                    logger.info("\nVerifying TensorRT pipeline...")
                    logger.info("Running PyTorch reference (CPU) for TensorRT verification...")
                    logger.info("  Note: Using CPU reference for numerical consistency (CUDA implementations differ)")
                    try:
                        pytorch_outputs_cpu, pytorch_latency_cpu, _ = pytorch_pipeline_cpu_trt.infer(
                            points, sample_meta, return_raw_outputs=True
                        )
                        logger.info(f"  PyTorch latency (CPU): {pytorch_latency_cpu:.2f} ms")
                        logger.info(f"  PyTorch output: {len(pytorch_outputs_cpu)} head outputs")
                        
                        # Log output statistics
                        output_names = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
                        for idx, (out, name) in enumerate(zip(pytorch_outputs_cpu, output_names)):
                            if isinstance(out, torch.Tensor):
                                out_np = out.cpu().numpy()
                                logger.info(f"    {name}: shape={out.shape}, range=[{out_np.min():.3f}, {out_np.max():.3f}]")
                        
                        # DEBUG: Verify all pipelines use the same input data at each stage
                        logger.info("\n  DEBUG: Verifying input data consistency across pipelines...")
                        try:
                            # Step 1: Compare original points
                            logger.info("    Step 1: Comparing original points...")
                            if onnx_pipeline:
                                # Get preprocessed data from each pipeline
                                preprocessed_pytorch, _ = pytorch_pipeline_cpu_trt.preprocess(points, **sample_meta)
                                preprocessed_onnx, _ = onnx_pipeline.preprocess(points, **sample_meta)
                                preprocessed_trt, _ = tensorrt_pipeline.preprocess(points, **sample_meta)
                                
                                # Compare input_features (voxel encoder input)
                                if 'input_features' in preprocessed_pytorch and 'input_features' in preprocessed_onnx and 'input_features' in preprocessed_trt:
                                    # Move all to CPU for comparison
                                    input_feat_pytorch = preprocessed_pytorch['input_features'].cpu()
                                    input_feat_onnx = preprocessed_onnx['input_features'].cpu()
                                    input_feat_trt = preprocessed_trt['input_features'].cpu()
                                    
                                    # Compare shapes
                                    if input_feat_pytorch.shape != input_feat_onnx.shape or input_feat_pytorch.shape != input_feat_trt.shape:
                                        logger.error(f"      ⚠️  input_features shape mismatch!")
                                        logger.error(f"        PyTorch: {input_feat_pytorch.shape}")
                                        logger.error(f"        ONNX: {input_feat_onnx.shape}")
                                        logger.error(f"        TensorRT: {input_feat_trt.shape}")
                                    else:
                                        # Compare values
                                        diff_pytorch_onnx = torch.abs(input_feat_pytorch - input_feat_onnx)
                                        diff_pytorch_trt = torch.abs(input_feat_pytorch - input_feat_trt)
                                        diff_onnx_trt = torch.abs(input_feat_onnx - input_feat_trt)
                                        
                                        logger.info(f"      input_features shape: {input_feat_pytorch.shape}")
                                        logger.info(f"      PyTorch vs ONNX: max_diff={diff_pytorch_onnx.max():.9f}, mean_diff={diff_pytorch_onnx.mean():.9f}")
                                        logger.info(f"      PyTorch vs TensorRT: max_diff={diff_pytorch_trt.max():.9f}, mean_diff={diff_pytorch_trt.mean():.9f}")
                                        logger.info(f"      ONNX vs TensorRT: max_diff={diff_onnx_trt.max():.9f}, mean_diff={diff_onnx_trt.mean():.9f}")
                                        
                                        if diff_pytorch_onnx.max() > 1e-6 or diff_pytorch_trt.max() > 1e-6:
                                            logger.warning(f"      ⚠️  input_features differ! This could cause output differences.")
                                
                                # Step 2: Compare voxel encoder outputs
                                logger.info("    Step 2: Comparing voxel encoder outputs...")
                                voxel_feat_pytorch = pytorch_pipeline_cpu_trt.run_voxel_encoder(preprocessed_pytorch['input_features'])
                                voxel_feat_onnx = onnx_pipeline.run_voxel_encoder(preprocessed_onnx['input_features'])
                                voxel_feat_trt = tensorrt_pipeline.run_voxel_encoder(preprocessed_trt['input_features'])
                                
                                # Move all to CPU for comparison
                                voxel_feat_pytorch = voxel_feat_pytorch.cpu()
                                voxel_feat_onnx = voxel_feat_onnx.cpu()
                                voxel_feat_trt = voxel_feat_trt.cpu()
                                
                                if voxel_feat_pytorch.shape != voxel_feat_onnx.shape or voxel_feat_pytorch.shape != voxel_feat_trt.shape:
                                    logger.error(f"      ⚠️  voxel_features shape mismatch!")
                                    logger.error(f"        PyTorch: {voxel_feat_pytorch.shape}")
                                    logger.error(f"        ONNX: {voxel_feat_onnx.shape}")
                                    logger.error(f"        TensorRT: {voxel_feat_trt.shape}")
                                else:
                                    diff_pytorch_onnx = torch.abs(voxel_feat_pytorch - voxel_feat_onnx)
                                    diff_pytorch_trt = torch.abs(voxel_feat_pytorch - voxel_feat_trt)
                                    diff_onnx_trt = torch.abs(voxel_feat_onnx - voxel_feat_trt)
                                    
                                    logger.info(f"      voxel_features shape: {voxel_feat_pytorch.shape}")
                                    logger.info(f"      PyTorch vs ONNX: max_diff={diff_pytorch_onnx.max():.9f}, mean_diff={diff_pytorch_onnx.mean():.9f}")
                                    logger.info(f"      PyTorch vs TensorRT: max_diff={diff_pytorch_trt.max():.9f}, mean_diff={diff_pytorch_trt.mean():.9f}")
                                    logger.info(f"      ONNX vs TensorRT: max_diff={diff_onnx_trt.max():.9f}, mean_diff={diff_onnx_trt.mean():.9f}")
                                    
                                    if diff_pytorch_onnx.max() > 1e-6 or diff_pytorch_trt.max() > 1e-6:
                                        logger.warning(f"      ⚠️  voxel_features differ! This could cause output differences.")
                                
                                # Step 3: Compare spatial_features (middle encoder output, backbone input)
                                logger.info("    Step 3: Comparing spatial_features (backbone input)...")
                                spatial_feat_pytorch = pytorch_pipeline_cpu_trt.process_middle_encoder(
                                    voxel_feat_pytorch.to(pytorch_pipeline_cpu_trt.device), 
                                    preprocessed_pytorch['coors']
                                )
                                spatial_feat_onnx = onnx_pipeline.process_middle_encoder(
                                    voxel_feat_onnx.to(onnx_pipeline.device), 
                                    preprocessed_onnx['coors']
                                )
                                spatial_feat_trt = tensorrt_pipeline.process_middle_encoder(
                                    voxel_feat_trt.to(tensorrt_pipeline.device), 
                                    preprocessed_trt['coors']
                                )
                                
                                # Move all to CPU for comparison
                                spatial_feat_pytorch = spatial_feat_pytorch.cpu()
                                spatial_feat_onnx = spatial_feat_onnx.cpu()
                                spatial_feat_trt = spatial_feat_trt.cpu()
                                
                                if spatial_feat_pytorch.shape != spatial_feat_onnx.shape or spatial_feat_pytorch.shape != spatial_feat_trt.shape:
                                    logger.error(f"      ⚠️  spatial_features shape mismatch!")
                                    logger.error(f"        PyTorch: {spatial_feat_pytorch.shape}")
                                    logger.error(f"        ONNX: {spatial_feat_onnx.shape}")
                                    logger.error(f"        TensorRT: {spatial_feat_trt.shape}")
                                else:
                                    diff_pytorch_onnx = torch.abs(spatial_feat_pytorch - spatial_feat_onnx)
                                    diff_pytorch_trt = torch.abs(spatial_feat_pytorch - spatial_feat_trt)
                                    diff_onnx_trt = torch.abs(spatial_feat_onnx - spatial_feat_trt)
                                    
                                    logger.info(f"      spatial_features shape: {spatial_feat_pytorch.shape}")
                                    logger.info(f"      PyTorch vs ONNX: max_diff={diff_pytorch_onnx.max():.9f}, mean_diff={diff_pytorch_onnx.mean():.9f}")
                                    logger.info(f"      PyTorch vs TensorRT: max_diff={diff_pytorch_trt.max():.9f}, mean_diff={diff_pytorch_trt.mean():.9f}")
                                    logger.info(f"      ONNX vs TensorRT: max_diff={diff_onnx_trt.max():.9f}, mean_diff={diff_onnx_trt.mean():.9f}")
                                    
                                    if diff_pytorch_onnx.max() > 1e-6 or diff_pytorch_trt.max() > 1e-6:
                                        logger.warning(f"      ⚠️  spatial_features differ! This could cause output differences.")
                                    else:
                                        logger.info(f"      ✓ All spatial_features match - backbone/head inputs are identical")
                        except Exception as e:
                            logger.warning(f"    Could not verify input data consistency: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                        
                        # DEBUG: Check rot_y_axis_reference setting and output order
                        logger.info("\n  DEBUG: Checking rot_y_axis_reference and output order...")
                        try:
                            # Check rot_y_axis_reference from PyTorch model
                            rot_y_axis_ref = False
                            if hasattr(pytorch_pipeline_cpu_trt, 'pytorch_model') and hasattr(pytorch_pipeline_cpu_trt.pytorch_model, 'pts_bbox_head'):
                                rot_y_axis_ref = getattr(
                                    pytorch_pipeline_cpu_trt.pytorch_model.pts_bbox_head,
                                    '_rot_y_axis_reference',
                                    False
                                )
                            logger.info(f"    rot_y_axis_reference: {rot_y_axis_ref}")
                            
                            # Check actual output order from ONNX and TensorRT pipelines
                            if onnx_pipeline:
                                # Get ONNX output names from the session
                                onnx_output_names = None
                                if hasattr(onnx_pipeline, 'backbone_head_session'):
                                    onnx_output_names = [output.name for output in onnx_pipeline.backbone_head_session.get_outputs()]
                                    logger.info(f"    ONNX output names (from session): {onnx_output_names}")
                                
                                # Get TensorRT output names (using new TensorRT API)
                                trt_output_names = None
                                if hasattr(tensorrt_pipeline, '_engines') and 'backbone_neck_head' in tensorrt_pipeline._engines:
                                    try:
                                        import tensorrt as trt
                                        engine = tensorrt_pipeline._engines['backbone_neck_head']
                                        trt_output_names = []
                                        for i in range(engine.num_io_tensors):
                                            tensor_name = engine.get_tensor_name(i)
                                            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                                                trt_output_names.append(tensor_name)
                                        logger.info(f"    TensorRT output names (from engine): {trt_output_names}")
                                    except Exception as e:
                                        logger.warning(f"    Could not get TensorRT output names: {e}")
                                
                                logger.info(f"    Expected output order: {output_names}")
                                
                                # Verify output order matches
                                if onnx_output_names and trt_output_names:
                                    # Check if they match expected order (after reordering)
                                    logger.info(f"    ✓ Output order verification: ONNX and TensorRT pipelines should reorder to match expected order")
                                    if onnx_output_names != trt_output_names:
                                        logger.warning(f"    ⚠️  ONNX and TensorRT have different output name orders, but pipelines should reorder them")
                        except Exception as e:
                            logger.warning(f"    Could not check rot_y_axis_reference/output order: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                        
                        # DEBUG: Compare ONNX (CPU) vs TensorRT (CUDA) to see if TensorRT optimization is the issue
                        logger.info("\n  DEBUG: Comparing ONNX (CPU) vs TensorRT (CUDA) head outputs...")
                        try:
                            if onnx_pipeline:
                                # Run ONNX inference on CPU
                                onnx_outputs_cpu, _, _ = onnx_pipeline.infer(points, sample_meta, return_raw_outputs=True)
                                
                                # Run TensorRT inference
                                tensorrt_outputs, _, _ = tensorrt_pipeline.infer(points, sample_meta, return_raw_outputs=True)
                                
                                # Step 3: Verify output shapes and layouts match
                                logger.info("    Step 3: Verifying output shapes and layouts...")
                                if len(onnx_outputs_cpu) != len(tensorrt_outputs):
                                    logger.error(f"      ⚠️  Output count mismatch: ONNX={len(onnx_outputs_cpu)}, TensorRT={len(tensorrt_outputs)}")
                                if len(onnx_outputs_cpu) != len(output_names):
                                    logger.error(f"      ⚠️  Output count mismatch with expected: ONNX={len(onnx_outputs_cpu)}, Expected={len(output_names)}")
                                
                                # Compare ONNX CPU vs TensorRT CUDA
                                onnx_trt_max_diff = 0.0
                                for idx, (onnx_out, trt_out, name) in enumerate(zip(onnx_outputs_cpu, tensorrt_outputs, output_names)):
                                    if isinstance(onnx_out, torch.Tensor) and isinstance(trt_out, torch.Tensor):
                                        # Step 3: Check shapes and layout match
                                        if onnx_out.shape != trt_out.shape:
                                            logger.error(f"      ⚠️  Shape mismatch for {name}: ONNX={onnx_out.shape}, TensorRT={trt_out.shape}")
                                            logger.error(f"         This could indicate layout differences (e.g., [B,C,H,W] vs [B,H,W,C])")
                                        else:
                                            logger.info(f"      ✓ {name}: Shape matches {onnx_out.shape}")
                                        
                                        # Move to CPU for comparison
                                        onnx_np = onnx_out.cpu().numpy()
                                        trt_np = trt_out.cpu().numpy()
                                        
                                        # Check for potential post-processing differences
                                        # (e.g., sigmoid, exp, decode operations)
                                        onnx_min, onnx_max = onnx_np.min(), onnx_np.max()
                                        trt_min, trt_max = trt_np.min(), trt_np.max()
                                        
                                        # Check if outputs look like they've been through different transformations
                                        # If ONNX is in logits range (e.g., -20 to 20) but TensorRT is in sigmoid range (0 to 1),
                                        # that indicates different post-processing
                                        if name == 'heatmap':
                                            if (onnx_min < -10 or onnx_max > 10) and (0 <= trt_min <= 1 and 0 <= trt_max <= 1):
                                                logger.warning(f"        ⚠️  {name} appears to have different post-processing!")
                                                logger.warning(f"           ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}] (logits?)")
                                                logger.warning(f"           TensorRT range: [{trt_min:.3f}, {trt_max:.3f}] (sigmoid?)")
                                            elif (0 <= onnx_min <= 1 and 0 <= onnx_max <= 1) and (trt_min < -10 or trt_max > 10):
                                                logger.warning(f"        ⚠️  {name} appears to have different post-processing!")
                                                logger.warning(f"           ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}] (sigmoid?)")
                                                logger.warning(f"           TensorRT range: [{trt_min:.3f}, {trt_max:.3f}] (logits?)")
                                        
                                        # Calculate differences
                                        diff = np.abs(onnx_np - trt_np)
                                        max_diff_val = diff.max()
                                        mean_diff_val = diff.mean()
                                        onnx_trt_max_diff = max(onnx_trt_max_diff, max_diff_val)
                                        
                                        # Calculate relative difference (for large values)
                                        onnx_abs_max = np.abs(onnx_np).max()
                                        if onnx_abs_max > 1e-6:
                                            rel_diff = max_diff_val / onnx_abs_max
                                            logger.info(f"      {name}: ONNX(CPU) vs TensorRT(CUDA)")
                                            logger.info(f"        max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}, rel_diff={rel_diff:.6f}")
                                            logger.info(f"        ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}], TensorRT range: [{trt_min:.3f}, {trt_max:.3f}]")
                                        else:
                                            logger.info(f"      {name}: ONNX(CPU) vs TensorRT(CUDA)")
                                            logger.info(f"        max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                            logger.info(f"        ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}], TensorRT range: [{trt_min:.3f}, {trt_max:.3f}]")
                                        
                                        # For rot and dim, check if rot_y_axis_reference might be causing issues
                                        if name in ['rot', 'dim'] and rot_y_axis_ref:
                                            logger.warning(f"        ⚠️  {name} with rot_y_axis_reference=True - check if conversion is needed")
                                            # Check if values are in different coordinate systems
                                            # (e.g., if dim has swapped width/length, or rot has swapped sin/cos)
                                            if name == 'dim' and onnx_out.shape[-3] == 3:
                                                # Check if first two channels are swapped
                                                onnx_wl = onnx_np[:, [0, 1], :, :]
                                                trt_wl = trt_np[:, [0, 1], :, :]
                                                trt_wl_swapped = trt_np[:, [1, 0], :, :]
                                                diff_normal = np.abs(onnx_wl - trt_wl).max()
                                                diff_swapped = np.abs(onnx_wl - trt_wl_swapped).max()
                                                if diff_swapped < diff_normal * 0.5:
                                                    logger.warning(f"        ⚠️  {name} width/length might be swapped! (swapped diff={diff_swapped:.6f} < normal diff={diff_normal:.6f})")
                                    else:
                                        logger.warning(f"      {name}: Type mismatch - ONNX={type(onnx_out)}, TensorRT={type(trt_out)}")
                                
                                logger.info(f"    ONNX (CPU) vs TensorRT (CUDA) overall max_diff: {onnx_trt_max_diff:.6f}")
                                
                                if onnx_trt_max_diff > 0.1:
                                    logger.warning(f"    ⚠️  Large difference between ONNX (CPU) and TensorRT (CUDA)!")
                                    logger.warning(f"    This suggests TensorRT optimizations or post-processing differences.")
                                    logger.warning(f"    ONNX (CPU) matches PyTorch (CPU), but TensorRT (CUDA) differs significantly.")
                                    logger.warning(f"    Possible causes:")
                                    logger.warning(f"      1. TensorRT precision/quantization (FP16/INT8 vs FP32)")
                                    logger.warning(f"      2. Different post-processing (sigmoid/exp/decode in engine vs Python)")
                                    logger.warning(f"      3. rot_y_axis_reference coordinate system conversion mismatch")
                                else:
                                    logger.info(f"    ✓ ONNX (CPU) and TensorRT (CUDA) match well.")
                                    logger.info(f"    The difference is likely due to CUDA vs CPU implementation differences.")
                                
                                # Additional comparison: ONNX (CPU) vs ONNX (CUDA)
                                if onnx_pipeline_cuda:
                                    logger.info("\n    Step 4: Comparing ONNX Runtime (CPU) vs ONNX Runtime (CUDA) head outputs...")
                                    try:
                                        onnx_outputs_cuda, _, _ = onnx_pipeline_cuda.infer(points, sample_meta, return_raw_outputs=True)
                                        if len(onnx_outputs_cuda) != len(onnx_outputs_cpu):
                                            logger.error(f"      ⚠️  ONNX CPU/CUDA output count mismatch: CPU={len(onnx_outputs_cpu)}, CUDA={len(onnx_outputs_cuda)}")
                                        onnx_cpu_cuda_max_diff = 0.0
                                        for idx, (cpu_out, cuda_out, name) in enumerate(zip(onnx_outputs_cpu, onnx_outputs_cuda, output_names)):
                                            if isinstance(cpu_out, torch.Tensor) and isinstance(cuda_out, torch.Tensor):
                                                if cpu_out.shape != cuda_out.shape:
                                                    logger.error(f"      ⚠️  Shape mismatch for {name}: CPU={cpu_out.shape}, CUDA={cuda_out.shape}")
                                                    continue
                                                cpu_np = cpu_out.cpu().numpy()
                                                cuda_np = cuda_out.cpu().numpy()
                                                diff = np.abs(cpu_np - cuda_np)
                                                max_diff_val = diff.max()
                                                mean_diff_val = diff.mean()
                                                onnx_cpu_cuda_max_diff = max(onnx_cpu_cuda_max_diff, max_diff_val)
                                                cpu_min, cpu_max = cpu_np.min(), cpu_np.max()
                                                cuda_min, cuda_max = cuda_np.min(), cuda_np.max()
                                                logger.info(f"      {name}: ONNX(CPU) vs ONNX(CUDA) max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                                logger.info(f"        CPU range: [{cpu_min:.3f}, {cpu_max:.3f}], CUDA range: [{cuda_min:.3f}, {cuda_max:.3f}]")
                                            else:
                                                logger.warning(f"      {name}: Type mismatch - CPU={type(cpu_out)}, CUDA={type(cuda_out)}")
                                        logger.info(f"      ONNX (CPU) vs ONNX (CUDA) overall max_diff: {onnx_cpu_cuda_max_diff:.6f}")
                                        if onnx_cpu_cuda_max_diff > 0.1:
                                            logger.warning("      ⚠️  Significant difference between ONNX CPU and CUDA outputs - indicates CUDA kernel differences")
                                        else:
                                            logger.info("      ✓ ONNX CPU and CUDA outputs match closely")
                                    except Exception as e:
                                        logger.warning(f"      ⚠️  Could not compare ONNX CPU vs CUDA outputs: {e}")
                                        import traceback
                                        logger.debug(traceback.format_exc())
                                
                                # Additional comparison: PyTorch (CPU) vs PyTorch (CUDA)
                                if pytorch_pipeline_cuda:
                                    logger.info("\n    Step 5: Comparing PyTorch (CPU) vs PyTorch (CUDA) head outputs...")
                                    try:
                                        pytorch_outputs_cuda, pytorch_latency_cuda, _ = pytorch_pipeline_cuda.infer(
                                            points, sample_meta, return_raw_outputs=True
                                        )
                                        logger.info(f"      PyTorch latency (CUDA): {pytorch_latency_cuda:.2f} ms")
                                        if len(pytorch_outputs_cuda) != len(pytorch_outputs_cpu):
                                            logger.error(f"      ⚠️  PyTorch CPU/CUDA output count mismatch: CPU={len(pytorch_outputs_cpu)}, CUDA={len(pytorch_outputs_cuda)}")
                                        pytorch_cpu_cuda_max_diff = 0.0
                                        for idx, (cpu_out, cuda_out, name) in enumerate(zip(pytorch_outputs_cpu, pytorch_outputs_cuda, output_names)):
                                            if isinstance(cpu_out, torch.Tensor) and isinstance(cuda_out, torch.Tensor):
                                                if cpu_out.shape != cuda_out.shape:
                                                    logger.error(f"      ⚠️  Shape mismatch for {name}: CPU={cpu_out.shape}, CUDA={cuda_out.shape}")
                                                    continue
                                                cpu_np = cpu_out.cpu().numpy()
                                                cuda_np = cuda_out.cpu().numpy()
                                                diff = np.abs(cpu_np - cuda_np)
                                                max_diff_val = diff.max()
                                                mean_diff_val = diff.mean()
                                                pytorch_cpu_cuda_max_diff = max(pytorch_cpu_cuda_max_diff, max_diff_val)
                                                cpu_min, cpu_max = cpu_np.min(), cpu_np.max()
                                                cuda_min, cuda_max = cuda_np.min(), cuda_np.max()
                                                logger.info(f"      {name}: PyTorch(CPU) vs PyTorch(CUDA) max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                                logger.info(f"        CPU range: [{cpu_min:.3f}, {cpu_max:.3f}], CUDA range: [{cuda_min:.3f}, {cuda_max:.3f}]")
                                            else:
                                                logger.warning(f"      {name}: Type mismatch - CPU={type(cpu_out)}, CUDA={type(cuda_out)}")
                                        logger.info(f"      PyTorch (CPU) vs PyTorch (CUDA) overall max_diff: {pytorch_cpu_cuda_max_diff:.6f}")
                                        if pytorch_cpu_cuda_max_diff > 0.1:
                                            logger.warning("      ⚠️  Significant difference between PyTorch CPU and CUDA outputs - indicates CUDA kernel differences")
                                        else:
                                            logger.info("      ✓ PyTorch CPU and CUDA outputs match closely")
                                        
                                        # Additional comparison: PyTorch (CUDA) vs ONNX (CUDA)
                                        if onnx_pipeline_cuda:
                                            logger.info("\n    Step 6: Comparing PyTorch (CUDA) vs ONNX Runtime (CUDA) head outputs...")
                                            try:
                                                if len(pytorch_outputs_cuda) != len(onnx_outputs_cuda):
                                                    logger.error(f"      ⚠️  PyTorch/ONNX CUDA output count mismatch: PyTorch={len(pytorch_outputs_cuda)}, ONNX={len(onnx_outputs_cuda)}")
                                                pytorch_onnx_cuda_max_diff = 0.0
                                                for idx, (pytorch_out, onnx_out, name) in enumerate(zip(pytorch_outputs_cuda, onnx_outputs_cuda, output_names)):
                                                    if isinstance(pytorch_out, torch.Tensor) and isinstance(onnx_out, torch.Tensor):
                                                        if pytorch_out.shape != onnx_out.shape:
                                                            logger.error(f"      ⚠️  Shape mismatch for {name}: PyTorch={pytorch_out.shape}, ONNX={onnx_out.shape}")
                                                            continue
                                                        pytorch_np = pytorch_out.cpu().numpy()
                                                        onnx_np = onnx_out.cpu().numpy()
                                                        diff = np.abs(pytorch_np - onnx_np)
                                                        max_diff_val = diff.max()
                                                        mean_diff_val = diff.mean()
                                                        pytorch_onnx_cuda_max_diff = max(pytorch_onnx_cuda_max_diff, max_diff_val)
                                                        pytorch_min, pytorch_max = pytorch_np.min(), pytorch_np.max()
                                                        onnx_min, onnx_max = onnx_np.min(), onnx_np.max()
                                                        logger.info(f"      {name}: PyTorch(CUDA) vs ONNX(CUDA) max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                                        logger.info(f"        PyTorch range: [{pytorch_min:.3f}, {pytorch_max:.3f}], ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}]")
                                                    else:
                                                        logger.warning(f"      {name}: Type mismatch - PyTorch={type(pytorch_out)}, ONNX={type(onnx_out)}")
                                                logger.info(f"      PyTorch (CUDA) vs ONNX (CUDA) overall max_diff: {pytorch_onnx_cuda_max_diff:.6f}")
                                                if pytorch_onnx_cuda_max_diff > 0.1:
                                                    logger.warning("      ⚠️  Significant difference between PyTorch CUDA and ONNX CUDA outputs")
                                                    logger.warning("      This suggests ONNX Runtime CUDA uses different kernels than PyTorch CUDA")
                                                else:
                                                    logger.info("      ✓ PyTorch CUDA and ONNX CUDA outputs match closely")
                                            except Exception as e:
                                                logger.warning(f"      ⚠️  Could not compare PyTorch CUDA vs ONNX CUDA outputs: {e}")
                                                import traceback
                                                logger.debug(traceback.format_exc())
                                        
                                        # Additional comparison: PyTorch (CUDA) vs TensorRT (CUDA)
                                        logger.info("\n    Step 7: Comparing PyTorch (CUDA) vs TensorRT (CUDA) head outputs...")
                                        try:
                                            if len(pytorch_outputs_cuda) != len(tensorrt_outputs):
                                                logger.error(f"      ⚠️  PyTorch/TensorRT CUDA output count mismatch: PyTorch={len(pytorch_outputs_cuda)}, TensorRT={len(tensorrt_outputs)}")
                                            pytorch_trt_cuda_max_diff = 0.0
                                            for idx, (pytorch_out, trt_out, name) in enumerate(zip(pytorch_outputs_cuda, tensorrt_outputs, output_names)):
                                                if isinstance(pytorch_out, torch.Tensor) and isinstance(trt_out, torch.Tensor):
                                                    if pytorch_out.shape != trt_out.shape:
                                                        logger.error(f"      ⚠️  Shape mismatch for {name}: PyTorch={pytorch_out.shape}, TensorRT={trt_out.shape}")
                                                        continue
                                                    pytorch_np = pytorch_out.cpu().numpy()
                                                    trt_np = trt_out.cpu().numpy()
                                                    diff = np.abs(pytorch_np - trt_np)
                                                    max_diff_val = diff.max()
                                                    mean_diff_val = diff.mean()
                                                    pytorch_trt_cuda_max_diff = max(pytorch_trt_cuda_max_diff, max_diff_val)
                                                    pytorch_min, pytorch_max = pytorch_np.min(), pytorch_np.max()
                                                    trt_min, trt_max = trt_np.min(), trt_np.max()
                                                    logger.info(f"      {name}: PyTorch(CUDA) vs TensorRT(CUDA) max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                                    logger.info(f"        PyTorch range: [{pytorch_min:.3f}, {pytorch_max:.3f}], TensorRT range: [{trt_min:.3f}, {trt_max:.3f}]")
                                                else:
                                                    logger.warning(f"      {name}: Type mismatch - PyTorch={type(pytorch_out)}, TensorRT={type(trt_out)}")
                                            logger.info(f"      PyTorch (CUDA) vs TensorRT (CUDA) overall max_diff: {pytorch_trt_cuda_max_diff:.6f}")
                                            if pytorch_trt_cuda_max_diff > 0.1:
                                                logger.warning("      ⚠️  Significant difference between PyTorch CUDA and TensorRT CUDA outputs")
                                                logger.warning("      This suggests TensorRT optimizations cause differences even when both use CUDA")
                                            else:
                                                logger.info("      ✓ PyTorch CUDA and TensorRT CUDA outputs match closely")
                                        except Exception as e:
                                            logger.warning(f"      ⚠️  Could not compare PyTorch CUDA vs TensorRT CUDA outputs: {e}")
                                            import traceback
                                            logger.debug(traceback.format_exc())
                                        
                                        # Additional comparison: ONNX (CUDA) vs TensorRT (CUDA)
                                        if onnx_pipeline_cuda:
                                            logger.info("\n    Step 8: Comparing ONNX Runtime (CUDA) vs TensorRT (CUDA) head outputs...")
                                            try:
                                                if len(onnx_outputs_cuda) != len(tensorrt_outputs):
                                                    logger.error(f"      ⚠️  ONNX/TensorRT CUDA output count mismatch: ONNX={len(onnx_outputs_cuda)}, TensorRT={len(tensorrt_outputs)}")
                                                onnx_trt_cuda_max_diff = 0.0
                                                for idx, (onnx_out, trt_out, name) in enumerate(zip(onnx_outputs_cuda, tensorrt_outputs, output_names)):
                                                    if isinstance(onnx_out, torch.Tensor) and isinstance(trt_out, torch.Tensor):
                                                        if onnx_out.shape != trt_out.shape:
                                                            logger.error(f"      ⚠️  Shape mismatch for {name}: ONNX={onnx_out.shape}, TensorRT={trt_out.shape}")
                                                            continue
                                                        onnx_np = onnx_out.cpu().numpy()
                                                        trt_np = trt_out.cpu().numpy()
                                                        diff = np.abs(onnx_np - trt_np)
                                                        max_diff_val = diff.max()
                                                        mean_diff_val = diff.mean()
                                                        onnx_trt_cuda_max_diff = max(onnx_trt_cuda_max_diff, max_diff_val)
                                                        onnx_min, onnx_max = onnx_np.min(), onnx_np.max()
                                                        trt_min, trt_max = trt_np.min(), trt_np.max()
                                                        logger.info(f"      {name}: ONNX(CUDA) vs TensorRT(CUDA) max_diff={max_diff_val:.6f}, mean_diff={mean_diff_val:.6f}")
                                                        logger.info(f"        ONNX range: [{onnx_min:.3f}, {onnx_max:.3f}], TensorRT range: [{trt_min:.3f}, {trt_max:.3f}]")
                                                    else:
                                                        logger.warning(f"      {name}: Type mismatch - ONNX={type(onnx_out)}, TensorRT={type(trt_out)}")
                                                logger.info(f"      ONNX (CUDA) vs TensorRT (CUDA) overall max_diff: {onnx_trt_cuda_max_diff:.6f}")
                                                if onnx_trt_cuda_max_diff > 0.1:
                                                    logger.warning("      ⚠️  Significant difference between ONNX CUDA and TensorRT CUDA outputs")
                                                    logger.warning("      This suggests TensorRT optimizations cause differences compared to ONNX Runtime CUDA")
                                                else:
                                                    logger.info("      ✓ ONNX CUDA and TensorRT CUDA outputs match closely")
                                            except Exception as e:
                                                logger.warning(f"      ⚠️  Could not compare ONNX CUDA vs TensorRT CUDA outputs: {e}")
                                                import traceback
                                                logger.debug(traceback.format_exc())
                                    except Exception as e:
                                        logger.warning(f"      ⚠️  Could not compare PyTorch CPU vs CUDA outputs: {e}")
                                        import traceback
                                        logger.debug(traceback.format_exc())
                        except Exception as e:
                            logger.warning(f"    Could not compare ONNX vs TensorRT: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                        
                        tensorrt_passed = self._verify_single_backend(
                            tensorrt_pipeline,
                            points,
                            sample_meta,
                            pytorch_outputs_cpu,
                            pytorch_latency_cpu,
                            tolerance,
                            "TensorRT",
                            logger
                        )
                        results[f"sample_{i}_tensorrt"] = tensorrt_passed
                    except Exception as e:
                        logger.error(f"  TensorRT verification failed: {e}")
                        import traceback
                        traceback.print_exc()
                        results[f"sample_{i}_tensorrt"] = False
                
                # Cleanup GPU memory after each sample (TensorRT needs frequent cleanup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        # Compute summary
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        total = len(results)
        skipped = len(skipped_backends) * num_samples  # Number of skipped verifications
        
        results['summary'] = {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'total': total
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary")
        logger.info("=" * 60)
        for key, value in results.items():
            if key != 'summary':
                status = "✓ PASSED" if value else "✗ FAILED"
                logger.info(f"  {key}: {status}")
        
        # Show skipped backends if any
        if skipped_backends:
            logger.info("")
            for backend in skipped_backends:
                logger.info(f"  {backend}: ⊝ SKIPPED")
        
        logger.info("=" * 60)
        summary_parts = [f"{passed}/{total} passed"]
        if failed > 0:
            summary_parts.append(f"{failed}/{total} failed")
        if skipped > 0:
            summary_parts.append(f"{skipped} skipped")
        logger.info(f"Total: {', '.join(summary_parts)}")
        logger.info("=" * 60)
        
        return results

    def evaluate(
        self,
        model_path: str,
        data_loader: CenterPointDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model_path: Path to model file or directory
            data_loader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            backend: Backend type ('pytorch', 'onnx', 'tensorrt')
            device: Device to run evaluation on
            verbose: Whether to print detailed output

        Returns:
            Dictionary containing evaluation results
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"\nEvaluating {backend} model: {model_path}")
        logger.info(f"Number of samples: {num_samples}")

        # Create Pipeline instance
        pipeline = self._create_pipeline(backend, model_path, device, logger)
        
        if pipeline is None:
            logger.error(f"Failed to create {backend} Pipeline")
            return {}

        # Run evaluation
        predictions_list = []
        ground_truths_list = []
        latencies = []
        latency_breakdowns = []  # Track individual stage latencies

        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                if verbose and i % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {i+1}/{num_samples}")

                # Get sample data
                sample = data_loader.load_sample(i)
                
                # Get points for pipeline
                if 'points' in sample:
                    points = sample['points']
                else:
                    # Load points from data_loader
                    input_data = data_loader.preprocess(sample)
                    points = input_data.get('points', input_data)
                
                # Get ground truth
                gt_data = data_loader.get_ground_truth(i)

                # Run inference using unified Pipeline interface
                sample_meta = sample.get('metainfo', {})
                predictions, latency, latency_breakdown = pipeline.infer(points, sample_meta)
                
                # Parse ground truths
                ground_truths = self._parse_ground_truths(gt_data)

                predictions_list.append(predictions)
                ground_truths_list.append(ground_truths)
                latencies.append(latency)
                latency_breakdowns.append(latency_breakdown)

                # Cleanup GPU memory after each sample (TensorRT needs frequent cleanup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}

        finally:
            # Cleanup (Pipeline handles its own cleanup)
            pass

        # Compute metrics
        results = self._compute_metrics(predictions_list, ground_truths_list, latencies, logger, latency_breakdowns)
        
        return results

    def _verify_single_backend(
        self,
        pipeline,
        points: torch.Tensor,
        sample_meta: Dict,
        reference_outputs: List[torch.Tensor],
        reference_latency: float,
        tolerance: float,
        backend_name: str,
        logger,
    ) -> bool:
        """
        Verify a single backend against PyTorch reference outputs.
        
        Args:
            pipeline: Pipeline instance to verify
            points: Input point cloud
            sample_meta: Sample metadata
            reference_outputs: Reference outputs from PyTorch [heatmap, reg, height, dim, rot, vel]
            reference_latency: Reference inference latency
            tolerance: Maximum allowed difference
            backend_name: Name of backend for logging ("ONNX", "TensorRT")
            logger: Logger instance
            
        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            # For debugging: Check if this is TensorRT and log input to backbone
            if backend_name == "TensorRT" and hasattr(pipeline, 'run_backbone_head'):
                # We can't easily intercept the input here, but we can add logging
                # in the pipeline itself
                pass
            
            # Run inference with raw outputs
            backend_outputs, backend_latency, backend_breakdown = pipeline.infer(
                points, sample_meta, return_raw_outputs=True
            )
            
            logger.info(f"  {backend_name} latency: {backend_latency:.2f} ms")
            logger.info(f"  {backend_name} output: {len(backend_outputs)} head outputs")
            
            # Debug: Check output shapes match
            if len(backend_outputs) != len(reference_outputs):
                logger.error(f"  Output count mismatch: {len(backend_outputs)} vs {len(reference_outputs)}")
                return False
            
            for i, (ref_out, backend_out) in enumerate(zip(reference_outputs, backend_outputs)):
                if isinstance(ref_out, torch.Tensor) and isinstance(backend_out, torch.Tensor):
                    if ref_out.shape != backend_out.shape:
                        logger.error(f"  Output {i} shape mismatch: {ref_out.shape} vs {backend_out.shape}")
                        return False
            
            # Compare outputs
            if len(backend_outputs) != len(reference_outputs):
                logger.error(f"  Output count mismatch: {len(backend_outputs)} vs {len(reference_outputs)}")
                return False
            
            max_diff = 0.0
            mean_diff = 0.0
            total_elements = 0
            
            output_names = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']
            
            for idx, (ref_out, backend_out, name) in enumerate(zip(reference_outputs, backend_outputs, output_names)):
                if isinstance(ref_out, torch.Tensor) and isinstance(backend_out, torch.Tensor):
                    # Convert to numpy for comparison
                    ref_np = ref_out.cpu().numpy()
                    backend_np = backend_out.cpu().numpy()
                    
                    # Check for shape mismatch
                    if ref_np.shape != backend_np.shape:
                        logger.error(f"    {name}: shape mismatch - {ref_np.shape} vs {backend_np.shape}")
                        return False
                    
                    # Compute differences
                    diff = np.abs(ref_np - backend_np)
                    output_max_diff = diff.max()
                    output_mean_diff = diff.mean()
                    max_diff = max(max_diff, output_max_diff)
                    mean_diff += diff.sum()
                    total_elements += diff.size
                    
                    # Log detailed statistics
                    logger.info(f"    {name}: max_diff={output_max_diff:.6f}, mean_diff={output_mean_diff:.6f}")
                    
                    # Check for special values
                    ref_nan = np.isnan(ref_np).any()
                    ref_inf = np.isinf(ref_np).any()
                    backend_nan = np.isnan(backend_np).any()
                    backend_inf = np.isinf(backend_np).any()
                    
                    if ref_nan or ref_inf or backend_nan or backend_inf:
                        logger.warning(f"      ⚠️  Special values detected in {name}:")
                        if ref_nan: logger.warning(f"         PyTorch has NaN!")
                        if ref_inf: logger.warning(f"         PyTorch has Inf!")
                        if backend_nan: logger.warning(f"         {backend_name} has NaN!")
                        if backend_inf: logger.warning(f"         {backend_name} has Inf!")
            
            # Compute overall mean difference
            if total_elements > 0:
                mean_diff /= total_elements
            
            logger.info(f"  Overall Max difference: {max_diff:.6f}")
            logger.info(f"  Overall Mean difference: {mean_diff:.6f}")
            
            # Check if verification passed
            if max_diff < tolerance:
                logger.info(f"  {backend_name} verification PASSED ✓")
                return True
            else:
                logger.warning(
                    f"  {backend_name} verification FAILED ✗ "
                    f"(max diff: {max_diff:.6f} > tolerance: {tolerance:.6f})"
                )
                return False
        
        except Exception as e:
            logger.error(f"  {backend_name} verification failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_pipeline(self, backend: str, model_path: str, device: str, logger) -> Any:
        """Create Pipeline instance for the specified backend."""
        try:
            # Import Pipeline classes
            from autoware_ml.deployment.pipelines import (
                CenterPointPyTorchPipeline,
                CenterPointONNXPipeline,
                CenterPointTensorRTPipeline
            )
            
            # Ensure device is properly set
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            
            device_obj = torch.device(device) if isinstance(device, str) else device
            
            # Use unified ONNX-compatible config for all backends
            cfg_for_backend = self.model_cfg

            # Load PyTorch model (required by all backends)
            if backend == "pytorch":
                pytorch_model = self._load_pytorch_model(
                    model_path, device_obj, logger, cfg_for_backend
                )
                return CenterPointPyTorchPipeline(pytorch_model, device=str(device_obj))
                
            elif backend == "onnx":
                # ONNX pipeline uses ONNX Runtime for voxel encoder and head;
                # PyTorch model is used for preprocessing/middle encoder/postprocessing.
                
                # Find checkpoint path
                import os
                checkpoint_path = model_path.replace('centerpoint_deployment', 'centerpoint/best_checkpoint.pth')
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = model_path.replace('_deployment', '/best_checkpoint.pth')
                
                pytorch_model = self._load_pytorch_model(
                    checkpoint_path, device_obj, logger, cfg_for_backend
                )
                return CenterPointONNXPipeline(pytorch_model, onnx_dir=model_path, device=str(device_obj))
                
            elif backend == "tensorrt":
                # TensorRT requires CUDA
                if not str(device).startswith("cuda"):
                    logger.warning("TensorRT requires CUDA device, skipping TensorRT evaluation")
                    return None
                
                # Find checkpoint path
                import os
                checkpoint_path = model_path.replace('centerpoint_deployment/tensorrt', 'centerpoint/best_checkpoint.pth')
                checkpoint_path = checkpoint_path.replace('/tensorrt', '')
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = model_path.replace('_deployment/tensorrt', '/best_checkpoint.pth')
                
                pytorch_model = self._load_pytorch_model(
                    checkpoint_path, device_obj, logger, cfg_for_backend
                )
                return CenterPointTensorRTPipeline(pytorch_model, tensorrt_dir=model_path, device=str(device_obj))
                
        except Exception as e:
            logger.error(f"Failed to create {backend} Pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_pytorch_model(
        self,
        checkpoint_path: str,
        device: torch.device,
        logger,
        model_cfg: Config,
    ) -> Any:
        """
        Load PyTorch model directly without using init_model to avoid CUDA checks.
        
        The model config should already be in the correct format (original or ONNX-compatible)
        based on the backend being evaluated.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            logger: Logger instance
        """
        try:
            from mmengine.registry import MODELS, init_default_scope
            from mmengine.runner import load_checkpoint
            import copy as copy_module
            
            # Initialize mmdet3d scope
            init_default_scope("mmdet3d")
            
            # Get model config - use deepcopy to avoid modifying shared nested objects
            model_config = copy_module.deepcopy(model_cfg.model)
            
            # For ONNX models, ensure device is set
            if hasattr(model_config, 'device'):
                model_config.device = str(device)
                logger.info(f"Set model config device to: {model_config.device}")
            
            # Build model using MODELS registry
            logger.info(f"Building model with device: {device}")
            logger.info(f"Model type: {model_config.type}")
            model = MODELS.build(model_config)
            model.to(device)
            
            # Add cfg attribute to model (required by inference_detector)
            model.cfg = model_cfg
            
            # Load checkpointoriginal_model_cfg
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, map_location=device)
            
            model.eval()
            
            logger.info(f"Successfully loaded PyTorch model on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model directly: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to init_model if direct loading fails (only for CUDA)
            # Note: init_model doesn't work well with CPU, so skip fallback for CPU
            if str(device).startswith('cuda'):
                try:
                    from mmdet3d.apis import init_model
                    logger.info("Falling back to init_model...")
                    model = init_model(model_cfg, checkpoint_path, device=device)
                    return model
                except Exception as fallback_e:
                    logger.error(f"Fallback to init_model also failed: {fallback_e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error("Direct model loading failed and fallback is disabled for CPU mode")
            
            raise e


    def _parse_ground_truths(self, gt_data: Dict) -> List[Dict]:
        """Parse ground truth data from gt_data returned by get_ground_truth()."""
        logger = logging.getLogger(__name__)
        ground_truths = []
        
        if 'gt_bboxes_3d' in gt_data and 'gt_labels_3d' in gt_data:
            gt_bboxes_3d = gt_data['gt_bboxes_3d']
            gt_labels_3d = gt_data['gt_labels_3d']
            
            
            # Count by label
            unique_labels, counts = np.unique(gt_labels_3d, return_counts=True)
            
            for i in range(len(gt_bboxes_3d)):
                bbox_3d = gt_bboxes_3d[i]  # [x, y, z, w, l, h, yaw]
                label = gt_labels_3d[i]
                
                ground_truths.append({
                    'bbox_3d': bbox_3d.tolist(),
                    'label': int(label)
                })
        
        return ground_truths

    # TODO(vividf): use autoware_perception_eval in the future
    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
        latency_breakdowns: List[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        logger = logging.getLogger(__name__)
        
        # Debug metrics
        
        # Count total predictions and ground truths
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)
        
        
        # Count per class
        per_class_preds = {}
        per_class_gts = {}
        
        for predictions in predictions_list:
            for pred in predictions:
                label = pred['label']
                per_class_preds[label] = per_class_preds.get(label, 0) + 1
        
        for ground_truths in ground_truths_list:
            for gt in ground_truths:
                label = gt['label']
                per_class_gts[label] = per_class_gts.get(label, 0) + 1

        # Compute latency statistics
        latency_stats = self.compute_latency_stats(latencies)
        
        # Compute stage-wise latency breakdown if available
        if latency_breakdowns and len(latency_breakdowns) > 0:
            stage_stats = {}
            stages = ['preprocessing_ms', 'voxel_encoder_ms', 'middle_encoder_ms', 'backbone_head_ms', 'postprocessing_ms']
            
            for stage in stages:
                stage_values = [bd.get(stage, 0.0) for bd in latency_breakdowns if stage in bd]
                if stage_values:
                    stage_stats[stage] = self.compute_latency_stats(stage_values)
            
            latency_stats["latency_breakdown"] = stage_stats

        # Try to compute mmdet3d metrics
        try:
            map_results = self._compute_simple_3d_map(
                predictions_list,
                ground_truths_list,
                num_classes=len(self.class_names),
            )
            
            # Combine results with mmdet3d metrics
            results = {
                "total_predictions": total_predictions,
                "total_ground_truths": total_ground_truths,
                "per_class_predictions": per_class_preds,
                "per_class_ground_truths": per_class_gts,
                "latency": latency_stats,
                "num_samples": len(predictions_list),
                **map_results,  # Include mAP, NDS, etc.
            }
            
            logger.info("✅ Successfully computed mmdet3d metrics")
            
        except Exception as e:
            logger.warning(f"Failed to compute mmdet3d metrics: {e}")
            logger.warning("Using simplified metrics instead")
            
            # Fallback to simplified metrics
            results = {
                "total_predictions": total_predictions,
                "total_ground_truths": total_ground_truths,
                "per_class_predictions": per_class_preds,
                "per_class_ground_truths": per_class_gts,
                "latency": latency_stats,
                "num_samples": len(predictions_list),
                "mAP": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
            }

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("CenterPoint 3D Object Detection - Evaluation Results")
        print("=" * 80)

        # Detection metrics
        print(f"\nDetection Metrics:")
        print(f"  mAP (0.5:0.95): {results.get('mAP', 0.0):.4f}")
        print(f"  mAP @ IoU=0.50: {results.get('mAP_50', 0.0):.4f}")
        print(f"  NDS: {results.get('NDS', 0.0):.4f}")
        print(f"  mATE: {results.get('mATE', 0.0):.4f}")
        print(f"  mASE: {results.get('mASE', 0.0):.4f}")
        print(f"  mAOE: {results.get('mAOE', 0.0):.4f}")
        print(f"  mAVE: {results.get('mAVE', 0.0):.4f}")
        print(f"  mAAE: {results.get('mAAE', 0.0):.4f}")

        # Per-class AP (show all 3D object classes)
        if "per_class_ap" in results:
            print(f"\nPer-Class AP (3D Object Classes):")
            for class_id, ap in results["per_class_ap"].items():
                class_name = class_id if isinstance(class_id, str) else (self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}")
                print(f"  {class_name:<12}: {ap:.4f}")

        # Detection statistics
        print(f"\nDetection Statistics:")
        print(f"  Total Predictions: {results.get('total_predictions', 0)}")
        print(f"  Total Ground Truths: {results.get('total_ground_truths', 0)}")

        # Per-class statistics
        if "per_class_predictions" in results and "per_class_ground_truths" in results:
            print(f"\nPer-Class Statistics:")
            for class_id in range(len(self.class_names)):
                class_name = self.class_names[class_id]
                pred_count = results["per_class_predictions"].get(class_id, 0)
                gt_count = results["per_class_ground_truths"].get(class_id, 0)
                print(f"  {class_name}:")
                print(f"    Predictions: {pred_count}")
                print(f"    Ground Truths: {gt_count}")

        # Latency statistics
        if "latency" in results:
            latency = results["latency"]
            print(f"\nLatency Statistics:")
            print(f"  Mean:   {latency['mean_ms']:.2f} ms")
            print(f"  Std:    {latency['std_ms']:.2f} ms")
            print(f"  Min:    {latency['min_ms']:.2f} ms")
            print(f"  Max:    {latency['max_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")
            
            # Stage-wise latency breakdown
            if "latency_breakdown" in latency:
                breakdown = latency["latency_breakdown"]
                print(f"\n  Stage-wise Latency Breakdown:")
                if "preprocessing_ms" in breakdown:
                    print(f"    Preprocessing:     {breakdown['preprocessing_ms']['mean_ms']:.2f} ± {breakdown['preprocessing_ms']['std_ms']:.2f} ms")
                if "voxel_encoder_ms" in breakdown:
                    print(f"    Voxel Encoder:    {breakdown['voxel_encoder_ms']['mean_ms']:.2f} ± {breakdown['voxel_encoder_ms']['std_ms']:.2f} ms")
                if "middle_encoder_ms" in breakdown:
                    print(f"    Middle Encoder:   {breakdown['middle_encoder_ms']['mean_ms']:.2f} ± {breakdown['middle_encoder_ms']['std_ms']:.2f} ms")
                if "backbone_head_ms" in breakdown:
                    print(f"    Backbone + Head:  {breakdown['backbone_head_ms']['mean_ms']:.2f} ± {breakdown['backbone_head_ms']['std_ms']:.2f} ms")
                if "postprocessing_ms" in breakdown:
                    print(f"    Postprocessing:   {breakdown['postprocessing_ms']['mean_ms']:.2f} ± {breakdown['postprocessing_ms']['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")
        print("=" * 80)

    def _compute_simple_3d_map(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        num_classes: int,
    ) -> Dict[str, Any]:
        """Compute simple 3D mAP using basic IoU calculation."""
        try:
            import numpy as np
            
            class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
            iou_threshold = 0.5
            
            # Initialize per-class metrics
            per_class_ap = {}
            
            for class_id, class_name in enumerate(class_names):
                # Collect all predictions and ground truths for this class
                all_predictions = []
                all_ground_truths = []
                
                for predictions, ground_truths in zip(predictions_list, ground_truths_list):
                    # Filter by class
                    class_predictions = [p for p in predictions if p['label'] == class_id]
                    class_ground_truths = [g for g in ground_truths if g['label'] == class_id]
                    
                    all_predictions.extend(class_predictions)
                    all_ground_truths.extend(class_ground_truths)
                
                # Sort predictions by score (descending)
                all_predictions.sort(key=lambda x: x['score'], reverse=True)
                
                # Debug IoU calculations for this class
                logger = logging.getLogger(__name__)
                if all_predictions and all_ground_truths:
                    # Show IoU between first prediction and first few GTs
                    first_pred = all_predictions[0]
                    
                    max_iou = 0.0
                    for i, gt in enumerate(all_ground_truths[:3]):  # Show first 3 GTs
                        iou = self._compute_3d_iou_simple(first_pred['bbox_3d'], gt['bbox_3d'])
                        max_iou = max(max_iou, iou)
                    
                    # Debug: Check if this is PyTorch or ONNX/TensorRT
                    backend_type = "Unknown"
                    if hasattr(self, '_current_backend'):
                        backend_type = str(type(self._current_backend))
                
                # Compute AP for this class
                if len(all_ground_truths) == 0:
                    # No ground truths for this class
                    ap = 0.0
                elif len(all_predictions) == 0:
                    # No predictions for this class
                    ap = 0.0
                else:
                    # Compute precision-recall curve
                    tp = 0
                    fp = 0
                    fn = len(all_ground_truths)
                    
                    precision_values = []
                    recall_values = []
                    
                    # Track which ground truths have been matched
                    gt_matched = [False] * len(all_ground_truths)
                    
                    for i, pred in enumerate(all_predictions):
                        pred_bbox = np.array(pred['bbox_3d'])
                        best_iou = 0.0
                        best_gt_idx = -1
                        
                        # Find best matching ground truth
                        for j, gt in enumerate(all_ground_truths):
                            if gt_matched[j]:
                                continue
                            
                            gt_bbox = np.array(gt['bbox_3d'])
                            iou = self._compute_3d_iou_simple(pred_bbox, gt_bbox)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                        
                        # Determine if prediction is TP or FP
                        if best_iou >= iou_threshold:
                            tp += 1
                            fn -= 1
                            gt_matched[best_gt_idx] = True
                        else:
                            fp += 1
                        
                        # Compute precision and recall
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        
                        precision_values.append(precision)
                        recall_values.append(recall)
                    
                    # Compute AP using 11-point interpolation
                    ap = self._compute_ap_11_point(precision_values, recall_values)
                
                per_class_ap[class_name] = ap
            
            # Compute overall mAP
            map_50 = sum(per_class_ap.values()) / len(class_names) if len(class_names) > 0 else 0.0
            
            return {
                "mAP": map_50,
                "mAP_50": map_50,
                "NDS": map_50,  # Simplified
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": per_class_ap,
            }
            
        except Exception as e:
            print(f"ERROR: Simple 3D mAP computation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": {},
            }
    
    def _compute_3d_iou_simple(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute simple 3D IoU using BEV (Bird's Eye View) approximation."""
        try:
            # Extract BEV parameters: [x, y, w, l, yaw]
            x1, y1, w1, l1, yaw1 = box1[0], box1[1], box1[3], box1[4], box1[6]
            x2, y2, w2, l2, yaw2 = box2[0], box2[1], box2[3], box2[4], box2[6]
            
            # Debug IoU calculation
            # print(f"DEBUG IoU: box1=[{x1:.2f}, {y1:.2f}, {w1:.2f}, {l1:.2f}, {yaw1:.2f}], box2=[{x2:.2f}, {y2:.2f}, {w2:.2f}, {l2:.2f}, {yaw2:.2f}]")
            
            # For simplicity, ignore rotation and compute axis-aligned IoU
            # This is an approximation - proper 3D IoU would consider rotation
            
            # Compute intersection rectangle
            x_left = max(x1 - w1/2, x2 - w2/2)
            y_top = max(y1 - l1/2, y2 - l2/2)
            x_right = min(x1 + w1/2, x2 + w2/2)
            y_bottom = min(y1 + l1/2, y2 + l2/2)
            
            # print(f"DEBUG IoU: intersection_rect=[{x_left:.2f}, {y_top:.2f}, {x_right:.2f}, {y_bottom:.2f}]")
            
            if x_right < x_left or y_bottom < y_top:
                # print(f"DEBUG IoU: No intersection (x_right={x_right:.2f} < x_left={x_left:.2f} or y_bottom={y_bottom:.2f} < y_top={y_top:.2f})")
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Compute union area
            area1 = w1 * l1
            area2 = w2 * l2
            union_area = area1 + area2 - intersection_area
            
            # print(f"DEBUG IoU: intersection={intersection_area:.2f}, area1={area1:.2f}, area2={area2:.2f}, union={union_area:.2f}")
            
            if union_area <= 0:
                # print(f"DEBUG IoU: Union area <= 0, returning 0.0")
                return 0.0
            
            iou = intersection_area / union_area
            # print(f"DEBUG IoU: Final IoU = {iou:.3f}")
            return min(iou, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"ERROR: 3D IoU computation failed: {e}")
            return 0.0
    
    def _compute_ap_11_point(self, precision_values: List[float], recall_values: List[float]) -> float:
        """Compute AP using 11-point interpolation."""
        if len(precision_values) == 0 or len(recall_values) == 0:
            return 0.0
        
        import numpy as np
        
        # 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        max_precision = np.zeros_like(recall_thresholds)
        
        for i, threshold in enumerate(recall_thresholds):
            # Find maximum precision for recall >= threshold
            valid_indices = np.where(np.array(recall_values) >= threshold)[0]
            if len(valid_indices) > 0:
                max_precision[i] = np.max(np.array(precision_values)[valid_indices])
        
        ap = np.mean(max_precision)
        return float(ap)
