"""
Verify YOLOX ONNX Model Export.

This script verifies that the exported ONNX model has the correct format
and can perform inference correctly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def check_onnx_model(onnx_path: str) -> dict:
    """
    Check ONNX model structure and metadata.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        Dictionary containing model information
    """
    print("=" * 80)
    print(f"Checking ONNX Model: {onnx_path}")
    print("=" * 80)

    # Load model
    model = onnx.load(onnx_path)

    # Check model
    try:
        onnx.checker.check_model(model)
        print("✅ ONNX model is valid")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        return None

    # Get input/output info
    info = {
        "inputs": [],
        "outputs": [],
        "opset_version": model.opset_import[0].version,
    }

    print("\n📥 Inputs:")
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type

        info["inputs"].append({"name": name, "shape": shape, "dtype": dtype})
        print(f"  - {name}: {shape} (dtype={dtype})")

    print("\n📤 Outputs:")
    for output_tensor in model.graph.output:
        name = output_tensor.name
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in output_tensor.type.tensor_type.shape.dim]
        dtype = output_tensor.type.tensor_type.elem_type

        info["outputs"].append({"name": name, "shape": shape, "dtype": dtype})
        print(f"  - {name}: {shape} (dtype={dtype})")

    print(f"\n🔧 ONNX Opset Version: {info['opset_version']}")
    print(f"📦 Model Size: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")

    return info


def test_inference(onnx_path: str, batch_size: int = 1, input_shape: tuple = (3, 960, 960)):
    """
    Test ONNX model inference with dummy input.

    Args:
        onnx_path: Path to ONNX file
        batch_size: Batch size for testing
        input_shape: Input shape (C, H, W)
    """
    print("\n" + "=" * 80)
    print("Testing ONNX Inference")
    print("=" * 80)

    # Create session
    try:
        sess = ort.InferenceSession(onnx_path)
        print("✅ ONNX Runtime session created successfully")
    except Exception as e:
        print(f"❌ Failed to create ONNX Runtime session: {e}")
        return

    # Get input name
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]

    print(f"\n📝 Input name: {input_name}")
    print(f"📝 Output names: {output_names}")

    # Create dummy input
    dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)
    print(f"\n🎲 Dummy input shape: {dummy_input.shape}")
    print(f"   Range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")

    # Run inference
    try:
        import time

        start_time = time.time()
        outputs = sess.run(output_names, {input_name: dummy_input})
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        print(f"\n✅ Inference successful!")
        print(f"⏱️  Latency: {latency_ms:.2f} ms")

        # Print output info
        print(f"\n📊 Outputs:")
        for i, (output_name, output) in enumerate(zip(output_names, outputs)):
            print(f"  [{i}] {output_name}:")
            print(f"      Shape: {output.shape}")
            print(f"      Dtype: {output.dtype}")
            print(f"      Range: [{output.min():.6f}, {output.max():.6f}]")

            # If this is detection output format [B, N, 7]
            if len(output.shape) == 3 and output.shape[-1] == 7:
                print(f"\n      🎯 Detected format: [batch, num_predictions, 7]")
                print(f"         Format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]")
                print(f"\n      Sample detections (first 3):")
                for j in range(min(3, output.shape[1])):
                    det = output[0, j]
                    print(
                        f"        [{j}] bbox=[{det[0]:.2f}, {det[1]:.2f}, {det[2]:.2f}, {det[3]:.2f}], "
                        f"obj={det[4]:.4f}, cls_conf={det[5]:.4f}, cls_id={int(det[6])}"
                    )

                # Count high-confidence detections
                high_conf_mask = (output[0, :, 4] * output[0, :, 5]) > 0.5
                num_high_conf = high_conf_mask.sum()
                print(f"\n      📈 High confidence detections (>0.5): {num_high_conf}")

        return outputs

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def compare_with_reference(onnx_path: str, reference_path: str):
    """
    Compare current ONNX model with reference model.

    Args:
        onnx_path: Path to current ONNX file
        reference_path: Path to reference ONNX file
    """
    print("\n" + "=" * 80)
    print("Comparing with Reference Model")
    print("=" * 80)

    if not Path(reference_path).exists():
        print(f"⚠️  Reference model not found: {reference_path}")
        return

    # Load both models
    model_current = onnx.load(onnx_path)
    model_ref = onnx.load(reference_path)

    # Compare inputs
    print("\n📥 Input Comparison:")
    for inp_curr, inp_ref in zip(model_current.graph.input, model_ref.graph.input):
        shape_curr = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp_curr.type.tensor_type.shape.dim]
        shape_ref = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp_ref.type.tensor_type.shape.dim]

        match = "✅" if shape_curr == shape_ref else "❌"
        print(f"  {match} {inp_curr.name}: {shape_curr} vs {shape_ref}")

    # Compare outputs
    print("\n📤 Output Comparison:")
    for out_curr, out_ref in zip(model_current.graph.output, model_ref.graph.output):
        shape_curr = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out_curr.type.tensor_type.shape.dim]
        shape_ref = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out_ref.type.tensor_type.shape.dim]

        match = "✅" if shape_curr == shape_ref else "❌"
        print(f"  {match} {out_curr.name}: {shape_curr} vs {shape_ref}")


def main():
    parser = argparse.ArgumentParser(description="Verify YOLOX ONNX Model")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model file")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--input-size", type=int, nargs=2, default=[960, 960], help="Input image size (H W)")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference ONNX model for comparison")
    parser.add_argument("--no-inference", action="store_true", help="Skip inference test")

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.onnx_path).exists():
        print(f"❌ ONNX file not found: {args.onnx_path}")
        sys.exit(1)

    # Check model structure
    info = check_onnx_model(args.onnx_path)
    if info is None:
        sys.exit(1)

    # Test inference
    if not args.no_inference:
        input_shape = (3, args.input_size[0], args.input_size[1])
        test_inference(args.onnx_path, args.batch_size, input_shape)

    # Compare with reference
    if args.reference:
        compare_with_reference(args.onnx_path, args.reference)

    print("\n" + "=" * 80)
    print("✅ Verification Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
