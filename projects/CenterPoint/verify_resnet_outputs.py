"""Script to verify ResNet backbone output shapes for BEV-friendly configuration.

This script helps verify that the ResNet backbone outputs have the correct
spatial dimensions: (H, W), (H/2, W/2), (H/4, W/4)

Usage:
    python verify_resnet_outputs.py
"""

import os
import sys

# Add workspace root to path if not already there
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config


def verify_resnet_outputs(config_path: str, input_shape: tuple = (1, 32, 760, 760)):
    """Verify ResNet backbone output shapes.

    Args:
        config_path: Path to config file
        input_shape: Input tensor shape (N, C, H, W)
    """
    # Convert to absolute path
    if not os.path.isabs(config_path):
        # Try relative to current directory first
        if not os.path.exists(config_path):
            # Try relative to workspace root
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            config_path = os.path.join(workspace_root, config_path)

    print(f"Loading config from: {config_path}")

    # Load config (this will automatically import custom modules via custom_imports)
    cfg = Config.fromfile(config_path, import_custom_modules=True)

    print(f"Config loaded. Custom imports: {getattr(cfg, 'custom_imports', None)}")

    # Explicitly import BEVResNet to ensure it's registered
    # This is a backup in case custom_imports didn't trigger the import
    try:
        from projects.CenterPoint.models.backbones.resnet import BEVResNet  # noqa: F401

        print("✓ BEVResNet module imported successfully")
    except ImportError as e:
        print(f"Warning: Could not explicitly import BEVResNet: {e}")
        # Try alternative import path
        try:
            import projects.CenterPoint.models  # This should trigger __init__.py

            print("✓ projects.CenterPoint.models imported")
        except ImportError as e2:
            print(f"Warning: Could not import models package: {e2}")

    # Build backbone
    backbone_cfg = cfg.model.pts_backbone.copy()
    backbone_cfg.pop("_delete_", None)

    # Get base_channels for expected shape calculation
    base_channels = backbone_cfg.get("base_channels", 32)

    print(f"Building backbone with config: {backbone_cfg.get('type', 'Unknown')}")
    print(f"Base channels: {base_channels}")

    # Use mmdet3d registry to build the model
    try:
        backbone = MODELS.build(backbone_cfg)
        backbone.eval()
    except KeyError as e:
        print(f"\nError: {e}")
        print("\nAvailable models in registry:")
        print(f"  ResNet: {'ResNet' in MODELS._module_dict}")
        print(f"  All registered models: {list(MODELS._module_dict.keys())[:20]}...")
        raise

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Forward pass
    with torch.no_grad():
        outputs = backbone(dummy_input)

    # Print output shapes
    print(f"\n{'='*60}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"{'='*60}")
    print(f"\nResNet Backbone Output Shapes:")
    print(f"{'='*60}")

    for i, out in enumerate(outputs):
        print(f"Stage {i} (out_indices[{i}]): {out.shape}")

    print(f"{'='*60}\n")

    # Verify expected shapes
    # Note: ResNet50 uses Bottleneck with expansion=4
    # Output channels = base_channels * 2^i * 4
    # base_channels=32 → [128, 256, 512] (32*4, 64*4, 128*4)
    # base_channels=64 → [256, 512, 1024] (64*4, 128*4, 256*4)
    # base_channels=16 → [64, 128, 256] (16*4, 32*4, 64*4)
    H, W = input_shape[2], input_shape[3]

    # Calculate expected channels based on base_channels
    # ResNet50: planes = base_channels * 2^i, output = planes * expansion (4)
    expected_shapes = [
        (input_shape[0], base_channels * 4, H, W),  # stage 0: base_channels * 2^0 * 4
        (input_shape[0], base_channels * 8, H // 2, W // 2),  # stage 1: base_channels * 2^1 * 4
        (input_shape[0], base_channels * 16, H // 4, W // 4),  # stage 2: base_channels * 2^2 * 4
    ]

    print("Expected shapes:")
    for i, expected in enumerate(expected_shapes):
        print(f"  Stage {i}: {expected}")

    print("\nVerification:")
    all_match = True
    for i, (out, expected) in enumerate(zip(outputs, expected_shapes)):
        match = out.shape == expected
        status = "✓" if match else "✗"
        print(f"  Stage {i}: {status} {out.shape} {'==' if match else '!='} {expected}")
        if not match:
            all_match = False

    # Critical: Verify spatial dimension alignment for FPN
    print(f"\n{'='*60}")
    print("Spatial Dimension Alignment Check (Critical for FPN):")
    print(f"{'='*60}")
    H, W = input_shape[2], input_shape[3]
    spatial_dims = [(out.shape[2], out.shape[3]) for out in outputs]
    expected_spatial = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]

    spatial_match = True
    for i, ((h, w), (eh, ew)) in enumerate(zip(spatial_dims, expected_spatial)):
        h_match = h == eh
        w_match = w == ew
        status = "✓" if (h_match and w_match) else "✗"
        print(f"  Stage {i}: {status} ({h}, {w}) {'==' if (h_match and w_match) else '!='} ({eh}, {ew})")
        if not (h_match and w_match):
            spatial_match = False
            print(f"    ⚠️  MISMATCH: Expected ({eh}, {ew}) but got ({h}, {w})")
            print(f"    ⚠️  This will cause FPN torch.cat() to fail!")

    if all_match and spatial_match:
        print("\n✓ All output shapes match expected dimensions!")
        print("✓ Spatial dimensions are perfectly aligned for FPN!")
    else:
        print("\n✗ Some output shapes don't match. Please check configuration.")
        if not spatial_match:
            print("✗ CRITICAL: Spatial dimension mismatch will break FPN!")

    return outputs


if __name__ == "__main__":
    # Default config path (relative to workspace root)
    default_config = "projects/CenterPoint/configs/t4dataset/Centerpoint/second_resnet50_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Try default path relative to workspace root
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        config_path = os.path.join(workspace_root, default_config)
        if not os.path.exists(config_path):
            # Try relative to current directory
            config_path = default_config

    verify_resnet_outputs(config_path)
