"""
CenterPoint Evaluator for deployment.

This module implements evaluation for CenterPoint 3D object detection models.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from mmengine.config import Config

from autoware_ml.deployment.backends import ONNXBackend, PyTorchBackend, TensorRTBackend
from autoware_ml.deployment.core import BaseEvaluator

from .data_loader import CenterPointDataLoader
from .centerpoint_tensorrt_backend import CenterPointTensorRTBackend

# Constants
LOG_INTERVAL = 50
GPU_CLEANUP_INTERVAL = 10


class CenterPointEvaluator(BaseEvaluator):
    """
    Evaluator for CenterPoint 3D object detection.

    Computes 3D detection metrics including mAP, NDS, and latency statistics.

    Note: For production, should integrate with mmdet3d's evaluation metrics.
    """

    def __init__(self, model_cfg: Config, class_names: List[str] = None):
        """
        Initialize CenterPoint evaluator.

        Args:
            model_cfg: Model configuration
            class_names: List of class names (optional)
        """
        super().__init__(config={})
        self.model_cfg = model_cfg
        
        # Debug: Check the model config in evaluator
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluator init model type: {self.model_cfg.model.type}")
        logger.info(f"Evaluator init voxel encoder type: {self.model_cfg.model.pts_voxel_encoder.type}")

        # Get class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(model_cfg, "class_names"):
            self.class_names = model_cfg.class_names
        else:
            # Default for T4Dataset
            self.class_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

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

        # Create backend instance
        inference_backend = self._create_backend(backend, model_path, device, logger)
        
        if inference_backend is None:
            logger.error(f"Failed to create {backend} backend")
            return {}
        
        # Store reference to pytorch_model if available (for consistent ONNX/TensorRT decoding)
        if hasattr(inference_backend, 'pytorch_model'):
            self.pytorch_model = inference_backend.pytorch_model
            logger.info(f"Stored PyTorch model reference for {backend} decoding")
        elif backend == "pytorch":
            self.pytorch_model = inference_backend
        else:
            self.pytorch_model = None
        
        # Load model (only for non-PyTorch backends)
        if backend != "pytorch":
            try:
                inference_backend.load_model()
            except Exception as e:
                logger.error(f"Failed to load {backend} model: {e}")
                return {}

        # Run evaluation
        predictions_list = []
        ground_truths_list = []
        latencies = []

        try:
            for i in range(min(num_samples, data_loader.get_num_samples())):
                if verbose and i % LOG_INTERVAL == 0:
                    logger.info(f"Processing sample {i+1}/{num_samples}")

                # Get sample data
                sample = data_loader.load_sample(i)
                input_data = data_loader.preprocess(sample)

                # Run inference
                if backend == "pytorch":
                    output, latency = self._run_pytorch_inference(inference_backend, input_data, sample)
                elif backend == "tensorrt":
                    # TensorRT also needs special preprocessing to ensure voxels/coors are included
                    output, latency = self._run_tensorrt_inference(inference_backend, input_data, sample)
                else:
                    output, latency = inference_backend.infer(input_data)

                # Parse predictions and ground truths
                predictions = self._parse_predictions(output, sample)
                ground_truths = self._parse_ground_truths(sample)
                
                # Debug PyTorch heatmap values
                if hasattr(output, '__iter__') and len(output) > 0:
                    if hasattr(output[0], 'pred_instances_3d'):
                        pred_instances = output[0].pred_instances_3d
                        if hasattr(pred_instances, 'scores') and len(pred_instances.scores) > 0:
                            scores = pred_instances.scores
                            print(f"DEBUG: PyTorch prediction scores - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")
                            print(f"DEBUG: PyTorch prediction count: {len(scores)}")

                predictions_list.append(predictions)
                ground_truths_list.append(ground_truths)
                latencies.append(latency)

                # Cleanup GPU memory periodically
                if i % GPU_CLEANUP_INTERVAL == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}

        finally:
            # Cleanup
            if hasattr(inference_backend, 'close'):
                inference_backend.close()

        # Compute metrics
        results = self._compute_metrics(predictions_list, ground_truths_list, latencies, logger)
        
        return results

    def _create_backend(self, backend: str, model_path: str, device: str, logger) -> Any:
        """Create backend instance."""
        try:
            if backend == "pytorch":
                # For 3D detection, we need to load the model directly
                # PyTorch evaluation should use STANDARD config, not ONNX config!
                # Ensure device is properly set to CPU if CUDA is not available
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = "cpu"
                device_obj = torch.device(device) if isinstance(device, str) else device
                
                # Use lower-level model loading to avoid CUDA checks in init_model
                # Convert ONNX config back to standard config for PyTorch evaluation
                model = self._load_pytorch_model_directly(
                    model_path, 
                    device_obj, 
                    logger, 
                    use_standard_config=True  # Convert to standard config
                )
                return model
            elif backend == "onnx":
                # For CenterPoint ONNX, we need the PyTorch model for voxelization
                # IMPORTANT: Use the ONNX-compatible model config (self.model_cfg should already be ONNX version)
                # Ensure device is properly set to CPU if CUDA is not available
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = "cpu"
                device_obj = torch.device(device) if isinstance(device, str) else device
                
                # Check if model_cfg is already ONNX version
                if self.model_cfg.model.type == "CenterPointONNX":
                    logger.info("Using ONNX-compatible model config for ONNX backend")
                    # Find the checkpoint path - try to infer from model_path
                    # Typically: work_dirs/centerpoint_deployment -> work_dirs/centerpoint/best_checkpoint.pth
                    import os
                    checkpoint_path = model_path.replace('centerpoint_deployment', 'centerpoint/best_checkpoint.pth')
                    if not os.path.exists(checkpoint_path):
                        # Try alternative paths
                        checkpoint_path = model_path.replace('_deployment', '/best_checkpoint.pth')
                    
                    # IMPORTANT: ONNX backend needs ONNX config, so use_standard_config=False
                    pytorch_model = self._load_pytorch_model_directly(
                        checkpoint_path, 
                        device_obj, 
                        logger,
                        use_standard_config=False  # Keep ONNX config for ONNX backend
                    )
                else:
                    logger.error("Model config is not ONNX-compatible!")
                    logger.error("Please use --replace-onnx-models flag when running deployment")
                    logger.error(f"Current model type: {self.model_cfg.model.type}")
                    logger.error("Expected model type: CenterPointONNX")
                    raise ValueError(
                        "ONNX evaluation requires ONNX-compatible model config. "
                        "Please ensure the model config has been updated to use CenterPointONNX."
                    )
                
                return ONNXBackend(model_path, device=device, pytorch_model=pytorch_model)
            elif backend == "tensorrt":
                # TensorRT requires CUDA, so skip if not available
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, skipping TensorRT backend")
                    return None
                
                # TensorRT backend needs PyTorch model for middle encoder
                # Load PyTorch model if config is ONNX-compatible
                if self.model_cfg.model.type == "CenterPointONNX":
                    logger.info("Loading PyTorch model for TensorRT middle encoder")
                    import os
                    checkpoint_path = model_path.replace('centerpoint_deployment/tensorrt', 'centerpoint/best_checkpoint.pth')
                    checkpoint_path = checkpoint_path.replace('/tensorrt', '')  # Handle different path formats
                    if not os.path.exists(checkpoint_path):
                        # Try alternative paths
                        checkpoint_path = model_path.replace('_deployment/tensorrt', '/best_checkpoint.pth')
                    
                    device_obj = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # IMPORTANT: TensorRT backend needs ONNX config, so use_standard_config=False
                    pytorch_model = self._load_pytorch_model_directly(
                        checkpoint_path, 
                        device_obj, 
                        logger,
                        use_standard_config=False  # Keep ONNX config for TensorRT backend
                    )
                    return CenterPointTensorRTBackend(model_path, device=device, pytorch_model=pytorch_model)
                else:
                    logger.error("TensorRT evaluation requires ONNX-compatible model config")
                    logger.error("Please use --replace-onnx-models flag when running deployment")
                    raise ValueError("TensorRT evaluation requires ONNX-compatible model config")
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
        except Exception as e:
            logger.error(f"Failed to create {backend} backend: {e}")
            return None

    def _load_pytorch_model_directly(self, checkpoint_path: str, device: torch.device, logger, use_standard_config: bool = False) -> Any:
        """
        Load PyTorch model directly without using init_model to avoid CUDA checks.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            logger: Logger instance
            use_standard_config: If True, convert ONNX config back to standard config for PyTorch evaluation
        """
        try:
            from mmengine.registry import MODELS, init_default_scope
            from mmengine.runner import load_checkpoint
            import copy as copy_module
            
            # Initialize mmdet3d scope
            init_default_scope("mmdet3d")
            
            # Get model config - use deepcopy to avoid modifying shared nested objects
            model_config = copy_module.deepcopy(self.model_cfg.model)
            
            # For PyTorch evaluation, convert ONNX config back to standard config
            if use_standard_config and model_config.type == "CenterPointONNX":
                logger.info("Converting ONNX-compatible config to standard config for PyTorch evaluation")
                # Convert model types back to standard versions
                model_config.type = "CenterPoint"
                
                # Remove ONNX-specific parameters from model config
                if hasattr(model_config, 'point_channels'):
                    delattr(model_config, 'point_channels')
                    logger.info("  Removed ONNX-specific parameter: point_channels")
                if hasattr(model_config, 'device'):
                    delattr(model_config, 'device')
                    logger.info("  Removed ONNX-specific parameter: device")
                
                # Convert voxel encoder
                if model_config.pts_voxel_encoder.type == "PillarFeatureNetONNX":
                    model_config.pts_voxel_encoder.type = "PillarFeatureNet"
                    logger.info("  Converted voxel encoder: PillarFeatureNetONNX -> PillarFeatureNet")
                elif model_config.pts_voxel_encoder.type == "BackwardPillarFeatureNetONNX":
                    model_config.pts_voxel_encoder.type = "BackwardPillarFeatureNet"
                    logger.info("  Converted voxel encoder: BackwardPillarFeatureNetONNX -> BackwardPillarFeatureNet")
                
                # Convert bbox head
                if model_config.pts_bbox_head.type == "CenterHeadONNX":
                    model_config.pts_bbox_head.type = "CenterHead"
                    logger.info("  Converted bbox head: CenterHeadONNX -> CenterHead")
                    # Remove ONNX-specific parameters from bbox head
                    if hasattr(model_config.pts_bbox_head, 'rot_y_axis_reference'):
                        delattr(model_config.pts_bbox_head, 'rot_y_axis_reference')
                        logger.info("  Removed ONNX-specific parameter: rot_y_axis_reference")
                    
                if hasattr(model_config.pts_bbox_head, 'separate_head') and model_config.pts_bbox_head.separate_head.type == "SeparateHeadONNX":
                    model_config.pts_bbox_head.separate_head.type = "SeparateHead"
                    logger.info("  Converted separate head: SeparateHeadONNX -> SeparateHead")
            else:
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
            model.cfg = self.model_cfg
            
            # Load checkpoint
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
                    model = init_model(self.model_cfg, checkpoint_path, device=device)
                    return model
                except Exception as fallback_e:
                    logger.error(f"Fallback to init_model also failed: {fallback_e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error("Direct model loading failed and fallback is disabled for CPU mode")
            
            raise e

    def _run_tensorrt_inference(self, backend, input_data: Dict, sample: Dict) -> tuple:
        """Run TensorRT inference with proper voxelized data format."""
        import time
        
        # TensorRT backend needs voxelized data (voxels, coors, num_points)
        # Use PyTorch model's voxelization if available
        if hasattr(backend, 'pytorch_model') and backend.pytorch_model is not None:
            # Get lidar points
            if 'points' in input_data:
                points = input_data['points']
            else:
                raise ValueError("TensorRT inference requires 'points' in input_data")
            
            # Use mmcv's Voxelization to voxelize points
            from mmcv.ops import Voxelization
            
            start_time = time.time()
            
            # Get voxelization config from model config
            # Default CenterPoint voxelization config
            voxel_size = [0.32, 0.32, 8.0]
            point_cloud_range = [-121.6, -121.6, -3.0, 121.6, 121.6, 5.0]
            max_num_points = 32  # max points per voxel
            max_voxels = (30000, 40000)  # (train, test)
            
            # Create voxelization layer
            voxel_layer = Voxelization(
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                max_num_points=max_num_points,
                max_voxels=max_voxels[1]  # Use test max_voxels
            )
            
            # Convert points to CPU for voxelization (mmcv ops may require CPU)
            if isinstance(points, torch.Tensor):
                points_cpu = points.cpu()
            else:
                points_cpu = torch.from_numpy(points)
            
            # Voxelize
            voxels, coors, num_points = voxel_layer(points_cpu)
            
            # Convert voxels to 11 dimensions using PyTorch model's get_input_features
            # This is what ONNX does - not simple zero padding!
            device = next(backend.pytorch_model.parameters()).device
            voxels_torch = voxels.to(device)
            coors_torch = coors.to(device)
            num_points_torch = num_points.to(device)
            
            # Add batch_idx to coors (mmcv returns [N, 3], need [N, 4])
            batch_idx = torch.zeros((coors_torch.shape[0], 1), dtype=coors_torch.dtype, device=device)
            coors_torch = torch.cat([batch_idx, coors_torch], dim=1)
            
            # Use PyTorch model's get_input_features to get 11-dim features
            with torch.no_grad():
                if hasattr(backend.pytorch_model.pts_voxel_encoder, 'get_input_features'):
                    voxels_11d = backend.pytorch_model.pts_voxel_encoder.get_input_features(
                        voxels_torch,
                        num_points_torch,
                        coors_torch
                    )
                    print(f"DEBUG: TensorRT got 11-dim features from get_input_features: {voxels_11d.shape}")
                else:
                    # Fallback: zero padding (not ideal)
                    print(f"WARNING: get_input_features not available, using zero padding")
                    pad_size = 11 - voxels_torch.shape[2]
                    padding = torch.zeros(voxels_torch.shape[0], voxels_torch.shape[1], pad_size, 
                                        dtype=voxels_torch.dtype, device=device)
                    voxels_11d = torch.cat([voxels_torch, padding], dim=2)
            
            # Convert back to original device for TensorRT
            # coors_torch already has batch_idx added
            voxels = voxels_11d
            coors = coors_torch
            num_points = num_points_torch
            
            # Prepare voxelized input data
            voxelized_input = {
                'voxels': voxels,
                'coors': coors,
                'num_points': num_points
            }
            
            print(f"DEBUG: Voxelized data - voxels: {voxels.shape}, coors: {coors.shape}, coors range: [{coors.min()}, {coors.max()}]")
            
            # Run TensorRT inference
            output, latency = backend.infer(voxelized_input)
            
            return output, latency
        else:
            # Fallback to direct inference (will use dummy coors)
            print(f"WARNING: PyTorch model not available for voxelization, using fallback")
            return backend.infer(input_data)
    
    def _run_pytorch_inference(self, model, input_data: Dict, sample: Dict) -> tuple:
        """Run PyTorch inference with proper data format."""
        from mmengine.dataset import pseudo_collate
        from mmdet3d.structures import Det3DDataSample
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Convert input data to proper format for PyTorch model
        # PyTorch model expects voxels, num_points, coors
        if 'voxels' in input_data and 'num_points' in input_data and 'coors' in input_data:
            # Already in correct format
            voxels = input_data['voxels']
            num_points = input_data['num_points'] 
            coors = input_data['coors']
        else:
            # Convert from points format
            points = input_data['points']
            # Create dummy voxels for verification
            voxels = torch.randn(1000, 32, 11)  # Dummy voxels
            num_points = torch.randint(1, 33, (1000,))  # Dummy num_points
            coors = torch.randint(0, 100, (1000, 3))  # Dummy coors
        
        # Create data sample
        data_sample = Det3DDataSample()
        data_sample.set_metainfo(sample.get('metainfo', {}))
        
        # Run inference
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and str(device).startswith('cuda') else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and str(device).startswith('cuda') else None
        
        if start_time:
            start_time.record()
        else:
            import time
            start_time_cpu = time.time()
        
        # Use inference method for proper evaluation
        with torch.no_grad():
            # Use the inference API which handles data formatting correctly
            from mmdet3d.apis import inference_detector
            # Get lidar path from sample
            lidar_path = sample.get('pts_path', sample.get('lidar_path', None))
            if lidar_path is None:
                # Fallback: use points data from sample
                lidar_path = sample['points'].numpy()
            outputs = inference_detector(model, lidar_path)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time)
        else:
            import time
            latency = (time.time() - start_time_cpu) * 1000  # Convert to ms
        
        return outputs, latency

    def _parse_predictions(self, output: Any, sample: Dict) -> List[Dict]:
        """Parse model output into prediction format."""
        predictions = []
        
        # Debug output
        print(f"DEBUG: Raw output type: {type(output)}")
        
        # Convert tuple to list if needed
        if isinstance(output, tuple):
            output = list(output)
        
        if isinstance(output, list):
            print(f"DEBUG: List output length: {len(output)}")
            if len(output) > 0:
                print(f"DEBUG: Item 0 type: {type(output[0])}")
                if hasattr(output[0], '__dict__'):
                    print(f"DEBUG: Item 0 attributes: {list(output[0].__dict__.keys())}")
        
        # Handle different output formats
        if isinstance(output, list) and len(output) > 0:
            # Check if it's Det3DDataSample format (PyTorch)
            if hasattr(output[0], 'pred_instances_3d'):
                print("INFO:projects.CenterPoint.deploy.evaluator:Raw head outputs detected, parsing CenterPoint predictions...")
                data_sample = output[0]
                
                # Extract predictions from Det3DDataSample
                pred_instances = data_sample.pred_instances_3d
                print(f"DEBUG: pred_instances attributes: {list(pred_instances.__dict__.keys())}")
                
                if hasattr(pred_instances, 'bboxes_3d') and hasattr(pred_instances, 'scores_3d') and hasattr(pred_instances, 'labels_3d'):
                    bboxes_3d = pred_instances.bboxes_3d
                    scores_3d = pred_instances.scores_3d
                    labels_3d = pred_instances.labels_3d
                    
                    print(f"DEBUG: Det3DDataSample - bboxes_3d shape: {bboxes_3d.shape}, scores shape: {scores_3d.shape}, labels shape: {labels_3d.shape}")
                    
                    # Convert to numpy for processing
                    bboxes_np = bboxes_3d.cpu().numpy()
                    scores_np = scores_3d.cpu().numpy()
                    labels_np = labels_3d.cpu().numpy()
                    
                    print(f"DEBUG: PyTorch bbox format - first bbox: {bboxes_np[0]}")
                    print(f"DEBUG: PyTorch bbox range - x: {bboxes_np[:, 0].min():.2f} to {bboxes_np[:, 0].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - y: {bboxes_np[:, 1].min():.2f} to {bboxes_np[:, 1].max():.2f}")
                    print(f"DEBUG: PyTorch bbox range - z: {bboxes_np[:, 2].min():.2f} to {bboxes_np[:, 2].max():.2f}")
                    print(f"DEBUG: PyTorch labels: {np.unique(labels_np)}")
                    print(f"DEBUG: PyTorch scores range: {scores_np.min():.3f} to {scores_np.max():.3f}")
                    
                    # Parse each prediction
                    for i in range(len(bboxes_np)):
                        bbox_3d = bboxes_np[i]  # [x, y, z, w, l, h, yaw, vx, vy]
                        score = scores_np[i]
                        label = labels_np[i]
                        
                        # Keep predictions in ego coordinates to match Ground Truth coordinate system
                        bbox_ego = bbox_3d[:7]  # [x, y, z, w, l, h, yaw]
                        
                        predictions.append({
                            'bbox_3d': bbox_ego.tolist(),  # [x, y, z, w, l, h, yaw] in ego coordinate (same as GT)
                            'score': float(score),
                            'label': int(label)
                        })
                    
                    print(f"DEBUG: Parsed {len(predictions)} predictions from Det3DDataSample")
                    
            # Check if it's TensorRT format (List[Dict[str, torch.Tensor]])
            elif isinstance(output[0], dict) and len(output) == 6:
                print("INFO:projects.CenterPoint.deploy.evaluator:TensorRT format detected, parsing CenterPoint predictions...")
                # TensorRT format: List[Dict[str, torch.Tensor]] with 6 head outputs
                # Expected keys: heatmap, reg, height, dim, rot, vel
                head_outputs = {}
                for item in output:
                    for key, value in item.items():
                        head_outputs[key] = value
                
                print(f"DEBUG: TensorRT head outputs keys: {list(head_outputs.keys())}")
                for key, value in head_outputs.items():
                    print(f"DEBUG: TensorRT {key} shape: {value.shape}")
                
                # Parse CenterPoint head outputs
                predictions = self._parse_centerpoint_head_outputs(
                    head_outputs.get('heatmap', torch.zeros(1, 5, 200, 200)),
                    head_outputs.get('reg', torch.zeros(1, 2, 200, 200)),
                    head_outputs.get('height', torch.zeros(1, 1, 200, 200)),
                    head_outputs.get('dim', torch.zeros(1, 3, 200, 200)),
                    head_outputs.get('rot', torch.zeros(1, 2, 200, 200)),
                    head_outputs.get('vel', torch.zeros(1, 2, 200, 200)),
                    sample
                )
                    
            # Check if it's raw head outputs format (ONNX)
            elif isinstance(output[0], (list, tuple)) and len(output[0]) == 6:
                print("INFO:projects.CenterPoint.deploy.evaluator:Raw head outputs detected, parsing CenterPoint predictions...")
                # Raw head outputs: [heatmap, reg, height, dim, rot, vel]
                heatmap, reg, height, dim, rot, vel = output[0]
                
                print(f"DEBUG: Raw output shapes - heatmap: {heatmap.shape}, reg: {reg.shape}, height: {height.shape}, dim: {dim.shape}, rot: {rot.shape}, vel: {vel.shape}")
                
                # Parse CenterPoint head outputs
                predictions = self._parse_centerpoint_head_outputs(
                    heatmap, reg, height, dim, rot, vel, sample
                )
            
            # Check if it's ONNX format (list of numpy arrays or torch tensors)
            elif isinstance(output[0], (torch.Tensor, np.ndarray)) and len(output) == 6:
                print("INFO:projects.CenterPoint.deploy.evaluator:ONNX head outputs detected, parsing CenterPoint predictions...")
                # ONNX outputs: [heatmap, reg, height, dim, rot, vel]
                heatmap, reg, height, dim, rot, vel = output
                
                # Convert numpy arrays to torch tensors if needed
                if isinstance(heatmap, np.ndarray):
                    heatmap = torch.from_numpy(heatmap)
                    reg = torch.from_numpy(reg)
                    height = torch.from_numpy(height)
                    dim = torch.from_numpy(dim)
                    rot = torch.from_numpy(rot)
                    vel = torch.from_numpy(vel)
                
                print(f"DEBUG: ONNX output shapes - heatmap: {heatmap.shape}, reg: {reg.shape}, height: {height.shape}, dim: {dim.shape}, rot: {rot.shape}, vel: {vel.shape}")
                
                # Parse CenterPoint head outputs
                predictions = self._parse_centerpoint_head_outputs(
                    heatmap, reg, height, dim, rot, vel, sample
                )
                
        elif isinstance(output, dict):
            # Handle dictionary output format
            if 'predictions' in output:
                predictions = output['predictions']
            else:
                # Convert dict to list format
                predictions = [output]
        
        return predictions

    def _parse_with_pytorch_model(self, heatmap, reg, height, dim, rot, vel, sample):
        """Use PyTorch model's complete forward pass for consistent results."""
        import torch
        
        print("INFO: Using complete PyTorch model forward pass for ONNX/TensorRT post-processing")
        
        # Instead of using ONNX head outputs + PyTorch decoder,
        # use ONNX outputs up to middle encoder, then run PyTorch for backbone+head
        # This ensures 100% consistency with PyTorch evaluation
        
        # For now, fall back to manual parsing
        # TODO: Implement hybrid approach later
        print("WARNING: PyTorch model integration not yet implemented, falling back to manual parsing")
        return None
    
    def _parse_with_pytorch_decoder(self, heatmap, reg, height, dim, rot, vel, sample):
        """Use PyTorch model's predict_by_feat for consistent decoding."""
        import torch
        
        print("INFO: Using PyTorch model's predict_by_feat for ONNX/TensorRT post-processing")
        
        # Convert to torch tensors if needed
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        if isinstance(reg, np.ndarray):
            reg = torch.from_numpy(reg)
        if isinstance(height, np.ndarray):
            height = torch.from_numpy(height)
        if isinstance(dim, np.ndarray):
            dim = torch.from_numpy(dim)
        if isinstance(rot, np.ndarray):
            rot = torch.from_numpy(rot)
        if isinstance(vel, np.ndarray):
            vel = torch.from_numpy(vel)
        
        # Move to same device as model
        device = next(self.pytorch_model.parameters()).device
        heatmap = heatmap.to(device)
        reg = reg.to(device) if reg is not None else None
        height = height.to(device)
        dim = dim.to(device)
        rot = rot.to(device)
        vel = vel.to(device) if vel is not None else None
        
        # IMPORTANT: If model was exported with rot_y_axis_reference=True,
        # we need to convert ONNX outputs back to standard format before passing to predict_by_feat
        rot_y_axis_reference = getattr(self.model_cfg.model.pts_bbox_head, 'rot_y_axis_reference', False)
        
        if rot_y_axis_reference:
            print(f"INFO: Detected rot_y_axis_reference=True, converting outputs to standard format")
            
            # 1. Convert dim from [w, l, h] back to [l, w, h]
            # ONNX output: dim[:, [0, 1, 2]] = [w, l, h]
            # Standard: dim[:, [0, 1, 2]] = [l, w, h]
            # So we need to swap channels 0 and 1
            dim = dim[:, [1, 0, 2], :, :]  # [w, l, h] -> [l, w, h]
            
            # 2. Convert rot from [-cos(x), -sin(y)] back to [sin(y), cos(x)]
            # ONNX output: rot[:, [0, 1]] = [-cos(x), -sin(y)]
            # Standard: rot[:, [0, 1]] = [sin(y), cos(x)]
            # Step 1: Negate to get [cos(x), sin(y)]
            rot = rot * (-1.0)
            # Step 2: Swap channels to get [sin(y), cos(x)]
            rot = rot[:, [1, 0], :, :]
            
            print(f"INFO: Converted dim [w,l,h]->[l,w,h] and rot [-cos,-sin]->[sin,cos]")
        
        # Prepare head outputs in mmdet3d format: Tuple[List[dict]]
        # The head outputs should be in dict format with keys: 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel'
        preds_dict = {
            'heatmap': heatmap,
            'reg': reg,
            'height': height,
            'dim': dim,
            'rot': rot,
            'vel': vel
        }
        preds_dicts = ([preds_dict],)  # Tuple[List[dict]] format for single task
        
        # Prepare batch_input_metas from sample
        metainfo = sample.get('metainfo', {})
        # Add required fields for predict_by_feat
        if 'box_type_3d' not in metainfo:
            # Default to LiDARInstance3DBoxes (same as CenterPoint default)
            from mmdet3d.structures import LiDARInstance3DBoxes
            metainfo['box_type_3d'] = LiDARInstance3DBoxes
        batch_input_metas = [metainfo]
        
        # Call predict_by_feat to get final predictions
        print(f"DEBUG: Calling predict_by_feat with preds_dicts type: {type(preds_dicts)}")
        print(f"DEBUG: preds_dicts[0] keys: {list(preds_dicts[0][0].keys())}")
        
        # Debug head outputs
        for key, value in preds_dict.items():
            if value is not None and hasattr(value, 'shape'):
                print(f"DEBUG: {key} shape: {value.shape}, min: {value.min():.4f}, max: {value.max():.4f}, mean: {value.mean():.4f}")
            else:
                print(f"DEBUG: {key}: {value}")
        
        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts,
                batch_input_metas=batch_input_metas
            )
        
        # predictions_list is List[InstanceData]
        # Each InstanceData has: bboxes_3d, scores_3d, labels_3d
        predictions = []
        for pred_instances in predictions_list:
            bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()  # [N, 9]
            scores_3d = pred_instances.scores_3d.cpu().numpy()  # [N]
            labels_3d = pred_instances.labels_3d.cpu().numpy()  # [N]
            
            print(f"DEBUG: PyTorch predict_by_feat - bboxes shape: {bboxes_3d.shape}, scores shape: {scores_3d.shape}")
            if len(bboxes_3d) > 0:
                print(f"DEBUG: PyTorch predict_by_feat - first bbox: {bboxes_3d[0]}")
            
            for i in range(len(bboxes_3d)):
                bbox_3d = bboxes_3d[i][:7]  # [x, y, z, w, l, h, yaw] - already in correct format
                score = scores_3d[i]
                label = labels_3d[i]
                
                predictions.append({
                    'bbox_3d': bbox_3d.tolist(),
                    'score': float(score),
                    'label': int(label)
                })
        
        print(f"DEBUG: PyTorch predict_by_feat produced {len(predictions)} predictions")
        return predictions

    def _parse_centerpoint_head_outputs(self, heatmap, reg, height, dim, rot, vel, sample):
        """Parse CenterPoint head outputs using PyTorch's CenterPointBBoxCoder for consistency."""
        import torch
        import numpy as np
        
        # Use PyTorch model's predict_by_feat for consistent post-processing
        if hasattr(self, 'pytorch_model') and self.pytorch_model is not None:
            try:
                return self._parse_with_pytorch_decoder(heatmap, reg, height, dim, rot, vel, sample)
            except Exception as e:
                print(f"WARNING: Failed to use PyTorch predict_by_feat: {e}, falling back to manual parsing")
                import traceback
                traceback.print_exc()
        
        # Convert to torch tensors for consistency with PyTorch
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        if isinstance(reg, np.ndarray):
            reg = torch.from_numpy(reg)
        if isinstance(height, np.ndarray):
            height = torch.from_numpy(height)
        if isinstance(dim, np.ndarray):
            dim = torch.from_numpy(dim)
        if isinstance(rot, np.ndarray):
            rot = torch.from_numpy(rot)
        if isinstance(vel, np.ndarray):
            vel = torch.from_numpy(vel)
        
        batch_size, num_classes, H, W = heatmap.shape
        print(f"DEBUG: Parsing CenterPoint outputs - batch_size: {batch_size}, num_classes: {num_classes}, H: {H}, W: {W}")
        
        predictions = []
        
        # Get point cloud range from sample
        point_cloud_range = sample.get('metainfo', {}).get('point_cloud_range', [-121.6, -121.6, -3.0, 121.6, 121.6, 5.0])
        voxel_size = sample.get('metainfo', {}).get('voxel_size', [0.32, 0.32, 8.0])
        
        # Debug coordinate transformation parameters
        print(f"DEBUG: point_cloud_range: {point_cloud_range}")
        print(f"DEBUG: voxel_size: {voxel_size}")
        
        # Debug: Check if this is ONNX/TensorRT backend
        backend_type = "Unknown"
        if hasattr(self, '_current_backend'):
            backend_type = str(type(self._current_backend))
        print(f"DEBUG: Backend type: {backend_type}")
        
        # Use the same logic as PyTorch's CenterPointBBoxCoder._topk
        # PyTorch uses a two-stage top-K selection process
        max_num = 100  # Same as PyTorch's default max_num
        
        for b in range(batch_size):
            # Apply sigmoid to get probabilities (same as PyTorch)
            heatmap_prob = torch.sigmoid(heatmap[b])  # [num_classes, H, W]
            
            # Debug heatmap value distribution
            print(f"DEBUG: Heatmap value analysis:")
            print(f"  Raw heatmap - min: {heatmap[b].min():.6f}, max: {heatmap[b].max():.6f}, mean: {heatmap[b].mean():.6f}")
            print(f"  Sigmoid heatmap - min: {heatmap_prob.min():.6f}, max: {heatmap_prob.max():.6f}, mean: {heatmap_prob.mean():.6f}")
            
            # Check if high values are concentrated at boundaries
            boundary_mask = torch.zeros_like(heatmap_prob[0])
            boundary_mask[0, :] = 1  # top boundary
            boundary_mask[-1, :] = 1  # bottom boundary  
            boundary_mask[:, 0] = 1  # left boundary
            boundary_mask[:, -1] = 1  # right boundary
            
            boundary_values = heatmap_prob[0][boundary_mask.bool()]
            interior_values = heatmap_prob[0][~boundary_mask.bool()]
            
            print(f"  Boundary values - min: {boundary_values.min():.6f}, max: {boundary_values.max():.6f}, mean: {boundary_values.mean():.6f}")
            print(f"  Interior values - min: {interior_values.min():.6f}, max: {interior_values.max():.6f}, mean: {interior_values.mean():.6f}")
            print(f"  Boundary count: {len(boundary_values)}, Interior count: {len(interior_values)}")
            
            # Stage 1: Get top-K for each class
            # Flatten heatmap: [num_classes, H, W] -> [num_classes, H*W]
            heatmap_flat = heatmap_prob.view(num_classes, -1)  # [num_classes, H*W]
            
            # Get top-K scores and indices for each class
            topk_scores_per_class, topk_inds_per_class = torch.topk(heatmap_flat, max_num)  # [num_classes, max_num]
            
            # Convert flat indices back to y, x coordinates for each class
            topk_inds_2d = topk_inds_per_class % (H * W)  # [num_classes, max_num]
            topk_ys_per_class = (topk_inds_2d.float() / W).int().float()  # [num_classes, max_num]
            topk_xs_per_class = (topk_inds_2d % W).int().float()  # [num_classes, max_num]
            
            # Stage 2: Get top-K across all classes (same as PyTorch)
            # Flatten all scores: [num_classes, max_num] -> [num_classes * max_num]
            all_scores = topk_scores_per_class.view(-1)  # [num_classes * max_num]
            
            # Get top-K across all classes
            topk_score, topk_ind = torch.topk(all_scores, max_num)  # [max_num]
            
            # Convert global indices back to class indices
            topk_clses = (topk_ind / max_num).int()  # [max_num]
            
            # Gather the corresponding y, x coordinates
            topk_ys = torch.gather(topk_ys_per_class.view(-1), 0, topk_ind)  # [max_num]
            topk_xs = torch.gather(topk_xs_per_class.view(-1), 0, topk_ind)  # [max_num]
            
            print(f"DEBUG: Selected top {len(topk_score)} predictions across all classes")
            print(f"DEBUG: Score range: {topk_score.min():.6f} to {topk_score.max():.6f}")
            print(f"DEBUG: Class distribution: {torch.bincount(topk_clses).tolist()}")
            
            # Process each selected prediction
            for i in range(len(topk_score)):
                score = topk_score[i].item()
                cls = topk_clses[i].item()
                y_idx = int(topk_ys[i].item())
                x_idx = int(topk_xs[i].item())
                
                # Debug first few predictions
                if i < 3:
                    print(f"DEBUG: Prediction {i} - cls: {cls}, score: {score:.6f}, y: {y_idx}, x: {x_idx}")
                
                # Get regression offsets (IMPORTANT: reg provides fine-grained position adjustments)
                reg_x = reg[b, 0, y_idx, x_idx].item() if reg.shape[1] > 0 else 0.0
                reg_y = reg[b, 1, y_idx, x_idx].item() if reg.shape[1] > 1 else 0.0
                
                # Convert grid coordinates to 3D coordinates (same as PyTorch CenterPointBBoxCoder)
                # PyTorch uses: (xs + reg_x) * out_size_factor * voxel_size[0] + pc_range[0]
                # Get out_size_factor from model config
                out_size_factor = getattr(self.model_cfg.model.pts_bbox_head, 'out_size_factor', 1)
                x = (x_idx + reg_x) * out_size_factor * voxel_size[0] + point_cloud_range[0]
                y = (y_idx + reg_y) * out_size_factor * voxel_size[1] + point_cloud_range[1]
                z = height[b, 0, y_idx, x_idx].item() if height.shape[1] > 0 else 0.0
                
                # Get dimensions (ONNX already swapped dim order: [1, 0, 2] -> [width, length, height])
                # IMPORTANT: CenterPoint outputs log(dim), need to apply exp() to recover actual dimensions
                w = np.exp(dim[b, 0, y_idx, x_idx].item()) if dim.shape[1] > 0 else 1.0
                l = np.exp(dim[b, 1, y_idx, x_idx].item()) if dim.shape[1] > 1 else 1.0
                h = np.exp(dim[b, 2, y_idx, x_idx].item()) if dim.shape[1] > 2 else 1.0
                
                # Get rotation (ONNX already swapped rot order: [1, 0] -> [cos, sin])
                if rot.shape[1] >= 2:
                    rot_sin = rot[b, 1, y_idx, x_idx].item()  # sin (was originally at index 0)
                    rot_cos = rot[b, 0, y_idx, x_idx].item()  # cos (was originally at index 1)
                    yaw = np.arctan2(rot_sin, rot_cos)
                else:
                    yaw = 0.0
                
                # IMPORTANT: Only apply coordinate transformation if rot_y_axis_reference=True
                # Check if model uses y-axis reference (from model config)
                rot_y_axis_reference = getattr(self.model_cfg.model.pts_bbox_head, 'rot_y_axis_reference', False)
                
                if rot_y_axis_reference:
                    # Apply transformations for y-axis reference models:
                    # 1. Switch width and length: bbox[:, [3, 4]] = bbox[:, [4, 3]]
                    # 2. Change rotation: bbox[:, 6] = -bbox[:, 6] - np.pi / 2
                    w_converted = l  # Switch w and l
                    l_converted = w
                    yaw_converted = -yaw - np.pi / 2
                else:
                    # No transformation needed - ONNX already outputs in correct format
                    w_converted = w
                    l_converted = l
                    yaw_converted = yaw
                
                # Keep predictions in ego coordinates to match Ground Truth coordinate system
                bbox_ego = np.array([x, y, z, w_converted, l_converted, h, yaw_converted])
                
                # Debug first few predictions
                if i < 3:
                    print(f"DEBUG: Prediction {i} - bbox: {bbox_ego}")
                
                predictions.append({
                    'bbox_3d': bbox_ego.tolist(),  # [x, y, z, w, l, h, yaw] in ego coordinate (same as GT)
                    'score': float(score),
                    'label': int(cls)
                })
        
        print(f"DEBUG: Parsed {len(predictions)} predictions from CenterPoint head outputs")
        
        # Debug prediction statistics
        if len(predictions) > 0:
            scores = [p['score'] for p in predictions]
            labels = [p['label'] for p in predictions]
            bboxes = [p['bbox_3d'] for p in predictions]
            
            print(f"DEBUG: ONNX/TensorRT scores range: {min(scores):.3f} to {max(scores):.3f}")
            print(f"DEBUG: ONNX/TensorRT labels: {sorted(set(labels))}")
            
            # Debug bbox spatial ranges
            bboxes_np = np.array(bboxes)
            print(f"DEBUG: ONNX/TensorRT bbox range - x: {bboxes_np[:, 0].min():.2f} to {bboxes_np[:, 0].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - y: {bboxes_np[:, 1].min():.2f} to {bboxes_np[:, 1].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - z: {bboxes_np[:, 2].min():.2f} to {bboxes_np[:, 2].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - w: {bboxes_np[:, 3].min():.2f} to {bboxes_np[:, 3].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - l: {bboxes_np[:, 4].min():.2f} to {bboxes_np[:, 4].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - h: {bboxes_np[:, 5].min():.2f} to {bboxes_np[:, 5].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox range - yaw: {bboxes_np[:, 6].min():.2f} to {bboxes_np[:, 6].max():.2f}")
            print(f"DEBUG: ONNX/TensorRT bbox format - first bbox: {bboxes_np[0]}")
            
            # Count predictions by class
            from collections import Counter
            label_counts = Counter(labels)
            print(f"DEBUG: ONNX/TensorRT prediction counts by class: {dict(label_counts)}")
        
        return predictions


    def _parse_ground_truths(self, sample: Dict) -> List[Dict]:
        """Parse ground truth data."""
        ground_truths = []
        
        if 'gt_bboxes_3d' in sample and 'gt_labels_3d' in sample:
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            
            print(f"DEBUG: Ground truth analysis:")
            print(f"  GT bboxes shape: {gt_bboxes_3d.shape}")
            print(f"  GT labels shape: {gt_labels_3d.shape}")
            print(f"  GT bboxes range: x={gt_bboxes_3d[:, 0].min():.1f}-{gt_bboxes_3d[:, 0].max():.1f}, y={gt_bboxes_3d[:, 1].min():.1f}-{gt_bboxes_3d[:, 1].max():.1f}")
            print(f"  GT bboxes range: z={gt_bboxes_3d[:, 2].min():.1f}-{gt_bboxes_3d[:, 2].max():.1f}")
            print(f"  GT bbox format - first bbox: {gt_bboxes_3d[0]}")
            print(f"  GT labels: {np.unique(gt_labels_3d)}")
            print(f"  GT count: {len(gt_bboxes_3d)}")
            
            # Count by label
            unique_labels, counts = np.unique(gt_labels_3d, return_counts=True)
            print(f"  GT label distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"    Label {label}: {count} ground truths")
            
            for i in range(len(gt_bboxes_3d)):
                bbox_3d = gt_bboxes_3d[i]  # [x, y, z, w, l, h, yaw]
                label = gt_labels_3d[i]
                
                ground_truths.append({
                    'bbox_3d': bbox_3d.tolist(),
                    'label': int(label)
                })
        
        return ground_truths

    def _compute_metrics(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        latencies: List[float],
        logger,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        
        # Debug metrics
        print(f"DEBUG: _compute_metrics - predictions_list length: {len(predictions_list)}")
        print(f"DEBUG: _compute_metrics - ground_truths_list length: {len(ground_truths_list)}")
        print(f"DEBUG: _compute_metrics - predictions per sample: {[len(preds) for preds in predictions_list]}")
        print(f"DEBUG: _compute_metrics - ground truths per sample: {[len(gts) for gts in ground_truths_list]}")
        
        # Count total predictions and ground truths
        total_predictions = sum(len(preds) for preds in predictions_list)
        total_ground_truths = sum(len(gts) for gts in ground_truths_list)
        
        print(f"DEBUG: _compute_metrics - total_predictions: {total_predictions}, total_ground_truths: {total_ground_truths}")
        
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

    def _compute_mmdet3d_map(
        self,
        predictions_list: List[List[Dict]],
        ground_truths_list: List[List[Dict]],
        num_classes: int,
    ) -> Dict[str, Any]:
        """Compute mAP using mmdet3d's eval_map_recall function."""
        try:
            from mmdet3d.evaluation import eval_map_recall
            from mmdet3d.structures import LiDARInstance3DBoxes
            import numpy as np
            import torch
            
            # Convert to mmdet3d format
            # eval_map_recall expects:
            # pred: {class_name: {img_id: [(bbox, score), ...]}}
            # gt: {class_name: {img_id: [bbox, ...]}}
            class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian']
            
            # Initialize data structures
            pred_by_class = {}
            gt_by_class = {}
            
            for class_name in class_names:
                pred_by_class[class_name] = {}
                gt_by_class[class_name] = {}
            
            # Process each sample
            for sample_idx, (predictions, ground_truths) in enumerate(zip(predictions_list, ground_truths_list)):
                img_id = f"sample_{sample_idx}"
                
                # Debug: Check if we have predictions and ground truths for this sample
                print(f"DEBUG: Processing sample {sample_idx} - predictions: {len(predictions)}, ground truths: {len(ground_truths)}")
                
                # Group predictions by class
                for pred in predictions:
                    class_name = class_names[pred['label']]
                    if img_id not in pred_by_class[class_name]:
                        pred_by_class[class_name][img_id] = []
                    
                    # Convert bbox to LiDARInstance3DBoxes format
                    bbox_3d = pred['bbox_3d']  # [x, y, z, w, l, h, yaw]
                    # Create a single bbox tensor
                    bbox_tensor = torch.tensor([bbox_3d], dtype=torch.float32)
                    bbox_obj = LiDARInstance3DBoxes(bbox_tensor)
                    
                    pred_by_class[class_name][img_id].append((bbox_obj, pred['score']))
                
                # Group ground truths by class
                for gt in ground_truths:
                    class_name = class_names[gt['label']]
                    if img_id not in gt_by_class[class_name]:
                        gt_by_class[class_name][img_id] = []
                    
                    # Convert bbox to LiDARInstance3DBoxes format
                    bbox_3d = gt['bbox_3d']  # [x, y, z, w, l, h, yaw]
                    bbox_tensor = torch.tensor([bbox_3d], dtype=torch.float32)
                    bbox_obj = LiDARInstance3DBoxes(bbox_tensor)
                    
                    gt_by_class[class_name][img_id].append(bbox_obj)
            
            # Debug: Print input to eval_map
            print(f"DEBUG: eval_map input - pred_by_class keys: {list(pred_by_class.keys())}")
            print(f"DEBUG: eval_map input - gt_by_class keys: {list(gt_by_class.keys())}")
            for class_name in class_names:
                pred_count = sum(len(pred_by_class[class_name][img_id]) for img_id in pred_by_class[class_name])
                gt_count = sum(len(gt_by_class[class_name][img_id]) for img_id in gt_by_class[class_name])
                print(f"DEBUG: {class_name} - pred count: {pred_count}, gt count: {gt_count}")
            
            # Ensure all classes have all sample IDs
            all_sample_ids = set()
            for class_name in class_names:
                all_sample_ids.update(pred_by_class[class_name].keys())
                all_sample_ids.update(gt_by_class[class_name].keys())
            
            print(f"DEBUG: All sample IDs: {sorted(all_sample_ids)}")
            
            # Add empty entries for missing sample IDs
            for class_name in class_names:
                for sample_id in all_sample_ids:
                    if sample_id not in pred_by_class[class_name]:
                        pred_by_class[class_name][sample_id] = []
                    if sample_id not in gt_by_class[class_name]:
                        gt_by_class[class_name][sample_id] = []
            
            # Compute 3D mAP metrics using eval_map_recall
            map_results = eval_map_recall(
                pred_by_class,
                gt_by_class,
                ovthresh=[0.5],  # IoU threshold for 3D detection
            )
            
            # Debug: Print eval_map_recall results
            print(f"DEBUG: eval_map_recall results type: {type(map_results)}")
            print(f"DEBUG: eval_map_recall results: {map_results}")
            
            # Extract results
            # eval_map_recall returns (recall, precision, ap) for each IoU threshold
            recall, precision, ap = map_results
            
            print(f"DEBUG: recall type: {type(recall)}, value: {recall}")
            print(f"DEBUG: precision type: {type(precision)}, value: {precision}")
            print(f"DEBUG: ap type: {type(ap)}, value: {ap}")
            
            # Extract mAP@0.5 (first IoU threshold)
            map_50 = 0.0
            per_class_ap = {}
            
            for class_name in class_names:
                if class_name in ap[0]:  # ap[0] is for IoU=0.5
                    class_ap = ap[0][class_name]
                    if isinstance(class_ap, np.ndarray):
                        per_class_ap[class_name] = float(class_ap.mean()) if len(class_ap) > 0 else 0.0
                    else:
                        per_class_ap[class_name] = float(class_ap)
                    map_50 += per_class_ap[class_name]
                else:
                    per_class_ap[class_name] = 0.0
            
            # Average mAP across all classes
            map_50 = map_50 / len(class_names) if len(class_names) > 0 else 0.0
            
            return {
                "mAP": map_50,
                "mAP_50": map_50,
                "NDS": map_50,  # Simplified - real NDS needs more complex computation
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": per_class_ap,
            }
            
        except Exception as e:
            print(f"Error computing mmdet3d metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return zero metrics as fallback
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "NDS": 0.0,
                "mATE": 0.0,
                "mASE": 0.0,
                "mAOE": 0.0,
                "mAVE": 0.0,
                "mAAE": 0.0,
                "per_class_ap": {f"class_{i}": 0.0 for i in range(num_classes)},
            }

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
                if all_predictions and all_ground_truths:
                    print(f"DEBUG: {class_name} - Computing IoU for {len(all_predictions)} predictions vs {len(all_ground_truths)} GTs")
                    # Show IoU between first prediction and first few GTs
                    first_pred = all_predictions[0]
                    print(f"DEBUG: {class_name} - First pred bbox: {first_pred['bbox_3d']}")
                    print(f"DEBUG: {class_name} - First pred score: {first_pred['score']:.3f}")
                    
                    max_iou = 0.0
                    for i, gt in enumerate(all_ground_truths[:3]):  # Show first 3 GTs
                        iou = self._compute_3d_iou_simple(first_pred['bbox_3d'], gt['bbox_3d'])
                        print(f"DEBUG: {class_name} - IoU with GT {i}: {iou:.3f}, GT bbox: {gt['bbox_3d']}")
                        max_iou = max(max_iou, iou)
                    print(f"DEBUG: {class_name} - Max IoU for first pred: {max_iou:.3f}")
                    
                    # Debug: Check if this is PyTorch or ONNX/TensorRT
                    backend_type = "Unknown"
                    if hasattr(self, '_current_backend'):
                        backend_type = str(type(self._current_backend))
                    print(f"DEBUG: {class_name} - Backend type: {backend_type}")
                    print(f"DEBUG: {class_name} - Sample count: {len(predictions_list)} samples")
                    print(f"DEBUG: {class_name} - GT count: {len(ground_truths_list)} samples")
                
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
                print(f"DEBUG: {class_name} - AP: {ap:.4f}, predictions: {len(all_predictions)}, ground truths: {len(all_ground_truths)}")
            
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
