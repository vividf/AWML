"""
Detailed ONNX Model Analysis Script.

This script performs deep analysis of ONNX models to understand their structure.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import helper, numpy_helper


def analyze_onnx_detailed(onnx_path: str) -> Dict:
    """
    Perform detailed analysis of ONNX model.

    Returns dictionary with complete model information.
    """
    print("\n" + "=" * 100)
    print(f"📊 DETAILED ANALYSIS: {onnx_path}")
    print("=" * 100)

    if not Path(onnx_path).exists():
        print(f"❌ File not found: {onnx_path}")
        return None

    # Load model
    model = onnx.load(onnx_path)
    graph = model.graph

    info = {
        "path": onnx_path,
        "inputs": [],
        "outputs": [],
        "nodes": [],
        "initializers": [],
        "value_info": [],
    }

    # === Basic Information ===
    print("\n📋 BASIC INFORMATION")
    print("-" * 100)
    print(f"Model name: {graph.name}")
    print(f"Opset version: {model.opset_import[0].version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"IR version: {model.ir_version}")
    file_size_mb = Path(onnx_path).stat().st_size / 1024 / 1024
    print(f"File size: {file_size_mb:.2f} MB")

    # === Inputs ===
    print("\n📥 INPUTS")
    print("-" * 100)
    for idx, input_tensor in enumerate(graph.input):
        name = input_tensor.name
        dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)

        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")

        info["inputs"].append({"index": idx, "name": name, "shape": shape, "dtype": dtype})

        print(f"  [{idx}] {name}")
        print(f"      Shape: {shape}")
        print(f"      Dtype: {dtype}")

    # === Outputs ===
    print("\n📤 OUTPUTS")
    print("-" * 100)
    for idx, output_tensor in enumerate(graph.output):
        name = output_tensor.name
        dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)

        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")

        info["outputs"].append({"index": idx, "name": name, "shape": shape, "dtype": dtype})

        print(f"  [{idx}] {name}")
        print(f"      Shape: {shape}")
        print(f"      Dtype: {dtype}")

    # === Nodes (Operations) ===
    print("\n🔧 GRAPH NODES (Operations)")
    print("-" * 100)
    print(f"Total nodes: {len(graph.node)}")

    # Count node types
    node_type_counts = {}
    for node in graph.node:
        op_type = node.op_type
        node_type_counts[op_type] = node_type_counts.get(op_type, 0) + 1

    print("\nNode type distribution:")
    for op_type, count in sorted(node_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_type:20s}: {count:4d}")

    # Show first and last few nodes
    print("\n📍 First 10 nodes:")
    for idx, node in enumerate(graph.node[:10]):
        print(f"  [{idx:4d}] {node.op_type:15s} | {node.name[:60]}")
        print(f"         Inputs:  {[inp[:40] for inp in node.input[:3]]}")
        print(f"         Outputs: {[out[:40] for out in node.output[:2]]}")

    print("\n📍 Last 10 nodes:")
    for idx, node in enumerate(graph.node[-10:], start=len(graph.node) - 10):
        print(f"  [{idx:4d}] {node.op_type:15s} | {node.name[:60]}")
        print(f"         Inputs:  {[inp[:40] for inp in node.input[:3]]}")
        print(f"         Outputs: {[out[:40] for out in node.output[:2]]}")

    # === Initializers (Weights) ===
    print("\n⚖️  INITIALIZERS (Weights/Constants)")
    print("-" * 100)
    print(f"Total initializers: {len(graph.initializer)}")

    total_params = 0
    print("\nFirst 10 initializers:")
    for idx, init in enumerate(graph.initializer[:10]):
        dims = list(init.dims)
        num_elements = np.prod(dims) if dims else 0
        total_params += num_elements
        dtype = onnx.TensorProto.DataType.Name(init.data_type)

        print(f"  [{idx:4d}] {init.name[:50]}")
        print(f"         Shape: {dims}, Dtype: {dtype}, Elements: {num_elements:,}")

    # Calculate total params
    for init in graph.initializer:
        dims = list(init.dims)
        total_params += np.prod(dims) if dims else 0

    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # === Value Info ===
    print("\n📊 VALUE INFO (Intermediate tensors)")
    print("-" * 100)
    print(f"Total value_info entries: {len(graph.value_info)}")

    if len(graph.value_info) > 0:
        print("\nFirst 10 intermediate tensors:")
        for idx, val_info in enumerate(graph.value_info[:10]):
            name = val_info.name
            shape = []
            for dim in val_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            dtype = onnx.TensorProto.DataType.Name(val_info.type.tensor_type.elem_type)
            print(f"  [{idx}] {name[:60]}")
            print(f"      Shape: {shape}, Dtype: {dtype}")

    # === Output Connection Analysis ===
    print("\n🔗 OUTPUT CONNECTION ANALYSIS")
    print("-" * 100)

    # Find which nodes produce the outputs
    output_names = [out.name for out in graph.output]
    print(f"Looking for nodes that produce: {output_names}")

    for output_name in output_names:
        print(f"\n🎯 Tracing output: {output_name}")

        # Find the node that produces this output
        producer_nodes = [node for node in graph.node if output_name in node.output]

        if producer_nodes:
            for node in producer_nodes:
                print(f"  ✓ Produced by node:")
                print(f"    Type: {node.op_type}")
                print(f"    Name: {node.name}")
                print(f"    Inputs: {list(node.input)}")
                print(f"    Outputs: {list(node.output)}")
                print(f"    Attributes: {[(attr.name, attr.type) for attr in node.attribute]}")

                # Trace back one level
                print(f"\n    ⬅️  Input sources:")
                for inp_name in node.input[:5]:  # Show first 5 inputs
                    inp_producers = [n for n in graph.node if inp_name in n.output]
                    if inp_producers:
                        for inp_node in inp_producers:
                            print(f"      - {inp_name[:40]}")
                            print(f"        From: {inp_node.op_type} ({inp_node.name[:40]})")
                    else:
                        # Check if it's an initializer or input
                        if any(init.name == inp_name for init in graph.initializer):
                            print(f"      - {inp_name[:40]} (initializer/weight)")
                        elif any(inp.name == inp_name for inp in graph.input):
                            print(f"      - {inp_name[:40]} (graph input)")
        else:
            print(f"  ❌ No producer node found! This might be a graph input or initializer.")

    return info


def compare_onnx_models(old_path: str, new_path: str):
    """
    Compare two ONNX models side by side.
    """
    print("\n" + "=" * 100)
    print("🔄 COMPARISON: OLD vs NEW")
    print("=" * 100)

    if not Path(old_path).exists():
        print(f"❌ Old model not found: {old_path}")
        return
    if not Path(new_path).exists():
        print(f"❌ New model not found: {new_path}")
        return

    old_model = onnx.load(old_path)
    new_model = onnx.load(new_path)

    old_graph = old_model.graph
    new_graph = new_model.graph

    # Compare inputs
    print("\n📥 INPUT COMPARISON")
    print("-" * 100)
    print(f"{'OLD':^40s} | {'NEW':^40s}")
    print("-" * 100)

    max_inputs = max(len(old_graph.input), len(new_graph.input))
    for i in range(max_inputs):
        if i < len(old_graph.input):
            old_inp = old_graph.input[i]
            old_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in old_inp.type.tensor_type.shape.dim]
            old_str = f"{old_inp.name}: {old_shape}"
        else:
            old_str = "(none)"

        if i < len(new_graph.input):
            new_inp = new_graph.input[i]
            new_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in new_inp.type.tensor_type.shape.dim]
            new_str = f"{new_inp.name}: {new_shape}"
        else:
            new_str = "(none)"

        match = "✅" if old_str == new_str else "❌"
        print(f"{old_str:40s} | {new_str:40s} {match}")

    # Compare outputs
    print("\n📤 OUTPUT COMPARISON")
    print("-" * 100)
    print(f"{'OLD':^40s} | {'NEW':^40s}")
    print("-" * 100)

    max_outputs = max(len(old_graph.output), len(new_graph.output))
    for i in range(max_outputs):
        if i < len(old_graph.output):
            old_out = old_graph.output[i]
            old_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in old_out.type.tensor_type.shape.dim]
            old_str = f"{old_out.name}: {old_shape}"
        else:
            old_str = "(none)"

        if i < len(new_graph.output):
            new_out = new_graph.output[i]
            new_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in new_out.type.tensor_type.shape.dim]
            new_str = f"{new_out.name}: {new_shape}"
        else:
            new_str = "(none)"

        match = "✅" if old_str == new_str else "❌"
        print(f"{old_str:40s} | {new_str:40s} {match}")

    # Compare node counts
    print("\n🔧 NODE COUNT COMPARISON")
    print("-" * 100)
    print(f"Old model: {len(old_graph.node)} nodes")
    print(f"New model: {len(new_graph.node)} nodes")
    print(f"Difference: {len(new_graph.node) - len(old_graph.node)} nodes")

    # Compare node types
    old_types = {}
    for node in old_graph.node:
        old_types[node.op_type] = old_types.get(node.op_type, 0) + 1

    new_types = {}
    for node in new_graph.node:
        new_types[node.op_type] = new_types.get(node.op_type, 0) + 1

    all_types = set(old_types.keys()) | set(new_types.keys())

    print("\n🔧 NODE TYPE DISTRIBUTION")
    print("-" * 100)
    print(f"{'Op Type':20s} | {'OLD':>8s} | {'NEW':>8s} | {'Diff':>8s}")
    print("-" * 100)

    for op_type in sorted(all_types):
        old_count = old_types.get(op_type, 0)
        new_count = new_types.get(op_type, 0)
        diff = new_count - old_count

        if diff != 0:
            marker = "❌"
        else:
            marker = "✅"

        print(f"{op_type:20s} | {old_count:8d} | {new_count:8d} | {diff:+8d} {marker}")


def main():
    parser = argparse.ArgumentParser(description="Detailed ONNX Model Analysis")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model")
    parser.add_argument("--compare", type=str, default=None, help="Path to second ONNX model for comparison")
    parser.add_argument("--output", type=str, default=None, help="Save analysis to file")

    args = parser.parse_args()

    # Redirect output to file if specified
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, "w")

    # Analyze first model
    info = analyze_onnx_detailed(args.onnx_path)

    # Compare if second model provided
    if args.compare:
        compare_onnx_models(args.onnx_path, args.compare)

    if args.output:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
