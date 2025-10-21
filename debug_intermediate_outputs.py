#!/usr/bin/env python3
"""
Debug script to compare intermediate outputs between PyTorch and ONNX.
This helps identify the first diverging layer in CenterPoint.
"""

import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from projects.CenterPoint.deploy.main import load_pytorch_model
from autoware_ml.deployment.backends.pytorch_backend import PyTorchBackend
from mmengine import Config


def compare_intermediate_outputs():
    """Compare intermediate outputs between PyTorch and ONNX."""
    
    # Load configuration
    config_path = "projects/CenterPoint/deploy/configs/deploy_config.py"
    model_config_path = "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py"
    checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"
    
    # Load deploy config
    deploy_config = Config.fromfile(config_path)
    device = deploy_config.export.device
    
    print(f"Using device: {device}")
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model(
        Config.fromfile(model_config_path), 
        checkpoint_path, 
        device,
        replace_onnx_models=True,
        rot_y_axis_reference=True
    )
    
    # Export with intermediate outputs
    print("Exporting ONNX model with intermediate outputs...")
    pytorch_model.save_onnx_with_intermediate_outputs(
        "work_dirs/centerpoint_deployment_debug",
        onnx_opset_version=deploy_config.onnx_config.opset_version
    )
    
    # Generate test input
    print("Generating test input...")
    random_inputs = pytorch_model._get_random_inputs()
    input_features, voxel_dict = pytorch_model._extract_random_features()
    
    # Get PyTorch intermediate outputs
    print("Getting PyTorch intermediate outputs...")
    pytorch_model.eval()
    
    # Voxel encoder output
    voxel_features = pytorch_model.pts_voxel_encoder(input_features)
    voxel_features = voxel_features.squeeze(1)
    
    # Middle encoder output
    coors = voxel_dict["coors"]
    batch_size = coors[-1, 0] + 1
    spatial_features = pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)
    
    # Backbone intermediate outputs
    backbone_outs = []
    x = spatial_features
    for i in range(len(pytorch_model.pts_backbone.blocks)):
        x = pytorch_model.pts_backbone.blocks[i](x)
        backbone_outs.append(x)
    
    print(f"PyTorch backbone outputs shapes:")
    for i, out in enumerate(backbone_outs):
        print(f"  Stage {i}: {out.shape}")
    
    # Load ONNX model and get intermediate outputs
    print("Getting ONNX intermediate outputs...")
    onnx_path = "work_dirs/centerpoint_deployment_debug/pts_backbone_with_intermediate.onnx"
    
    # Create ONNX Runtime session
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    if device.startswith("cuda"):
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]
    
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    
    # Convert input to numpy
    spatial_features_np = spatial_features.detach().cpu().numpy()
    
    # Run ONNX inference
    onnx_outputs = session.run(None, {"spatial_features": spatial_features_np})
    
    print(f"ONNX backbone outputs shapes:")
    for i, out in enumerate(onnx_outputs):
        print(f"  Stage {i}: {out.shape}")
    
    # Compare outputs
    print("\nComparing intermediate outputs:")
    max_diffs = []
    mean_diffs = []
    
    for i, (pytorch_out, onnx_out) in enumerate(zip(backbone_outs, onnx_outputs)):
        # Convert PyTorch to numpy
        pytorch_np = pytorch_out.detach().cpu().numpy()
        
        # Compute differences
        diff = np.abs(pytorch_np - onnx_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        
        print(f"Stage {i}:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  PyTorch stats: min={pytorch_np.min():.6f}, max={pytorch_np.max():.6f}, mean={pytorch_np.mean():.6f}")
        print(f"  ONNX stats: min={onnx_out.min():.6f}, max={onnx_out.max():.6f}, mean={onnx_out.mean():.6f}")
        print()
    
    # Find first diverging stage
    tolerance = 0.01
    first_diverging_stage = None
    for i, max_diff in enumerate(max_diffs):
        if max_diff > tolerance:
            first_diverging_stage = i
            break
    
    if first_diverging_stage is not None:
        print(f"🚨 First diverging stage: Stage {first_diverging_stage} (max diff: {max_diffs[first_diverging_stage]:.6f})")
    else:
        print("✅ All stages within tolerance")
    
    return max_diffs, mean_diffs, first_diverging_stage


if __name__ == "__main__":
    try:
        max_diffs, mean_diffs, first_diverging_stage = compare_intermediate_outputs()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
