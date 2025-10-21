#!/usr/bin/env python3
"""
Detailed debug script to analyze Stage 1 layer-by-layer differences.
This helps identify the specific operation causing numerical divergence.
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


def analyze_stage1_layer_by_layer():
    """Analyze Stage 1 layer-by-layer to find the exact diverging operation."""
    
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
    
    # Generate test input
    print("Generating test input...")
    input_features, voxel_dict = pytorch_model._extract_random_features()
    
    # Get to Stage 1 input
    pytorch_model.eval()
    voxel_features = pytorch_model.pts_voxel_encoder(input_features)
    voxel_features = voxel_features.squeeze(1)
    coors = voxel_dict["coors"]
    batch_size = coors[-1, 0] + 1
    spatial_features = pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)
    
    # Get Stage 0 output (input to Stage 1)
    stage0_output = pytorch_model.pts_backbone.blocks[0](spatial_features)
    print(f"Stage 0 output shape: {stage0_output.shape}")
    
    # Analyze Stage 1 layer by layer
    stage1_block = pytorch_model.pts_backbone.blocks[1]
    print(f"\nStage 1 block structure:")
    print(f"Number of layers: {len(stage1_block)}")
    
    # Hook to capture intermediate outputs
    intermediate_outputs = []
    
    def hook_fn(module, input, output):
        intermediate_outputs.append(output.detach().cpu().numpy())
    
    # Register hooks for each layer in Stage 1
    hooks = []
    for i, layer in enumerate(stage1_block):
        hooks.append(layer.register_forward_hook(hook_fn))
        print(f"Layer {i}: {type(layer).__name__}")
    
    # Run Stage 1 forward pass
    with torch.no_grad():
        stage1_input = stage0_output
        stage1_output = stage1_block(stage1_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"\nStage 1 intermediate outputs:")
    for i, output in enumerate(intermediate_outputs):
        print(f"Layer {i} output shape: {output.shape}")
        print(f"Layer {i} stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")
    
    # Now create ONNX model for Stage 1 only
    print("\nCreating ONNX model for Stage 1 only...")
    
    class Stage1Only(torch.nn.Module):
        def __init__(self, stage1_block):
            super().__init__()
            self.stage1_block = stage1_block
            
        def forward(self, x):
            return self.stage1_block(x)
    
    stage1_only = Stage1Only(stage1_block)
    stage1_only.eval()
    
    # Export Stage 1 to ONNX
    os.makedirs("work_dirs/stage1_debug", exist_ok=True)
    stage1_onnx_path = "work_dirs/stage1_debug/stage1_only.onnx"
    
    torch.onnx.export(
        stage1_only,
        (stage0_output,),
        stage1_onnx_path,
        export_params=True,
        opset_version=deploy_config.onnx_config.opset_version,
        do_constant_folding=True,
        input_names=['stage0_output'],
        output_names=['stage1_output'],
        dynamic_axes={
            'stage0_output': {0: 'batch_size', 2: 'H', 3: 'W'},
            'stage1_output': {0: 'batch_size', 2: 'H', 3: 'W'},
        }
    )
    
    print(f"Stage 1 ONNX model saved to: {stage1_onnx_path}")
    
    # Load ONNX model and compare
    print("\nComparing Stage 1 PyTorch vs ONNX...")
    
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
    
    session = ort.InferenceSession(stage1_onnx_path, sess_options=so, providers=providers)
    
    # Convert input to numpy
    stage0_input_np = stage0_output.detach().cpu().numpy()
    
    # Run ONNX inference
    onnx_outputs = session.run(None, {"stage0_output": stage0_input_np})
    onnx_stage1_output = onnx_outputs[0]
    
    # Compare final outputs
    pytorch_stage1_np = stage1_output.detach().cpu().numpy()
    
    diff = np.abs(pytorch_stage1_np - onnx_stage1_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nStage 1 Final Output Comparison:")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"PyTorch stats: min={pytorch_stage1_np.min():.6f}, max={pytorch_stage1_np.max():.6f}, mean={pytorch_stage1_np.mean():.6f}")
    print(f"ONNX stats: min={onnx_stage1_output.min():.6f}, max={onnx_stage1_output.max():.6f}, mean={onnx_stage1_output.mean():.6f}")
    
    # Analyze which layer starts diverging
    print(f"\nLayer-by-layer analysis:")
    print(f"Input to Stage 1 stats: min={stage0_input_np.min():.6f}, max={stage0_input_np.max():.6f}, mean={stage0_input_np.mean():.6f}")
    
    # Check if the issue is in the first layer (stride=2 convolution)
    print(f"\nFirst layer analysis (stride=2 conv):")
    first_layer = stage1_block[0]  # Conv2d
    second_layer = stage1_block[1]  # BatchNorm
    third_layer = stage1_block[2]  # ReLU
    
    # Test first layer only
    with torch.no_grad():
        conv_output = first_layer(stage0_output)
        bn_output = second_layer(conv_output)
        relu_output = third_layer(bn_output)
    
    print(f"Conv2d output stats: min={conv_output.min():.6f}, max={conv_output.max():.6f}, mean={conv_output.mean():.6f}")
    print(f"BatchNorm output stats: min={bn_output.min():.6f}, max={bn_output.max():.6f}, mean={bn_output.mean():.6f}")
    print(f"ReLU output stats: min={relu_output.min():.6f}, max={relu_output.max():.6f}, mean={relu_output.mean():.6f}")
    
    # Check BatchNorm parameters
    print(f"\nBatchNorm parameters:")
    print(f"Weight shape: {second_layer.weight.shape}")
    print(f"Bias shape: {second_layer.bias.shape}")
    print(f"Running mean shape: {second_layer.running_mean.shape}")
    print(f"Running var shape: {second_layer.running_var.shape}")
    print(f"Eps: {second_layer.eps}")
    print(f"Momentum: {second_layer.momentum}")
    
    return max_diff, mean_diff


if __name__ == "__main__":
    try:
        max_diff, mean_diff = analyze_stage1_layer_by_layer()
        print(f"\n🎯 Stage 1 Analysis Complete")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
