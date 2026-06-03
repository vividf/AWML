#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_initializer_hashes(model: onnx.ModelProto) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for tensor in model.graph.initializer:
        name = tensor.name[6:] if tensor.name.startswith("model.") else tensor.name
        raw = tensor.raw_data if tensor.raw_data else numpy_helper.to_array(tensor).tobytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def load_onnx_summary(path: Path) -> Dict[str, object]:
    model = onnx.load(str(path))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "opset_import": [(op.domain, op.version) for op in model.opset_import],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_hashes": model_initializer_hashes(model),
    }


def run_command(cmd: Iterable[str]) -> None:
    printable = " ".join(cmd)
    print(f"[RUN] {printable}")
    subprocess.run(list(cmd), check=True)


def ort_outputs(
    path: Path, input_name: str, input_shape: Tuple[int, ...], output_names: Iterable[str]
) -> Dict[str, np.ndarray]:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feed = {input_name: rng.standard_normal(input_shape, dtype=np.float32)}
    outputs = session.run(list(output_names), feed)
    return {name: out for name, out in zip(output_names, outputs)}


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    diff = np.abs(a - b)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "allclose_1e5": bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)),
        "allclose_1e6": bool(np.allclose(a, b, atol=1e-6, rtol=1e-6)),
    }


def trtexec_build(
    trtexec_bin: str,
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    shape: str,
    workspace_mib: int,
) -> None:
    run_command(
        [
            trtexec_bin,
            f"--onnx={onnx_path}",
            "--fp16",
            f"--minShapes={input_name}:{shape}",
            f"--optShapes={input_name}:{shape}",
            f"--maxShapes={input_name}:{shape}",
            f"--memPoolSize=workspace:{workspace_mib}",
            f"--saveEngine={engine_path}",
            "--skipInference",
        ]
    )


def trtexec_dump(
    trtexec_bin: str,
    engine_path: Path,
    input_name: str,
    shape: str,
    input_bin: Path,
    output_json: Path,
) -> None:
    run_command(
        [
            trtexec_bin,
            f"--loadEngine={engine_path}",
            f"--shapes={input_name}:{shape}",
            f"--loadInputs={input_name}:{input_bin}",
            "--dumpOutput",
            f"--exportOutput={output_json}",
        ]
    )


def load_trtexec_output(path: Path) -> Dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    records = raw if isinstance(raw, list) else [raw]
    outputs: Dict[str, np.ndarray] = {}
    for item in records:
        if isinstance(item, dict) and "name" in item and "values" in item:
            outputs[item["name"]] = np.array(item["values"], dtype=np.float32)
    return outputs


def compare_initializer_hashes(left: Dict[str, str], right: Dict[str, str]) -> int:
    diff_keys = [key for key in set(left) & set(right) if left[key] != right[key]]
    return len(diff_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CenterPoint ONNX/TRT export equivalence.")
    parser.add_argument("left_dir", type=Path, help="First work_dir, e.g. work_dirs/centerpoint_2_6_fp16_copy")
    parser.add_argument("right_dir", type=Path, help="Second work_dir, e.g. work_dirs/centerpoint_2_6_fp16")
    parser.add_argument(
        "--trtexec",
        type=str,
        default="/usr/src/tensorrt/bin/trtexec",
        help="Path to trtexec binary.",
    )
    parser.add_argument("--workspace-mib", type=int, default=1024, help="TRT workspace size in MiB.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT build/output comparison.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Do not delete temp build artifacts.")
    parser.add_argument("--report-json", type=Path, help="Optional path to save full comparison report as JSON.")
    args = parser.parse_args()

    left_dir = args.left_dir.resolve()
    right_dir = args.right_dir.resolve()
    left_onnx = left_dir / "onnx"
    right_onnx = right_dir / "onnx"

    onnx_pairs = [
        {
            "name": "voxel_encoder",
            "file": "pts_voxel_encoder.onnx",
            "input_name": "input_features",
            "shape_tuple": (20000, 32, 11),
            "trt_shape_tuple": (96000, 32, 11),
            "shape_trtexec": "96000x32x11",
            "outputs": ["pillar_features"],
        },
        {
            "name": "backbone_head",
            "file": "pts_backbone_neck_head.onnx",
            "input_name": "spatial_features",
            "shape_tuple": (1, 32, 1020, 1020),
            "trt_shape_tuple": (1, 32, 1020, 1020),
            "shape_trtexec": "1x32x1020x1020",
            "outputs": ["heatmap", "reg", "height", "dim", "rot", "vel"],
        },
    ]

    report: Dict[str, object] = {"left_dir": str(left_dir), "right_dir": str(right_dir), "onnx": {}, "trt": {}}

    print("\n=== ONNX Static + ORT Comparison ===")
    for pair in onnx_pairs:
        name = pair["name"]
        left_file = left_onnx / pair["file"]
        right_file = right_onnx / pair["file"]

        left_summary = load_onnx_summary(left_file)
        right_summary = load_onnx_summary(right_file)
        hash_diff_count = compare_initializer_hashes(
            left_summary["initializer_hashes"],  # type: ignore[arg-type]
            right_summary["initializer_hashes"],  # type: ignore[arg-type]
        )

        print(f"\n[{name}]")
        print(f"left_file:  {left_file}")
        print(f"right_file: {right_file}")
        print(f"left_sha256:  {left_summary['sha256']}")
        print(f"right_sha256: {right_summary['sha256']}")
        print(f"opset_left/right: {left_summary['opset_import']} / {right_summary['opset_import']}")
        print(
            "node_left/right: "
            f"{left_summary['node_count']} / {right_summary['node_count']}, "
            f"init_left/right: {left_summary['initializer_count']} / {right_summary['initializer_count']}, "
            f"initializer_hash_diff_count: {hash_diff_count}"
        )

        left_ort = ort_outputs(left_file, pair["input_name"], pair["shape_tuple"], pair["outputs"])
        right_ort = ort_outputs(right_file, pair["input_name"], pair["shape_tuple"], pair["outputs"])
        ort_comp = {output: compare_arrays(left_ort[output], right_ort[output]) for output in pair["outputs"]}
        for output_name, metrics in ort_comp.items():
            print(
                f"ORT {output_name}: max_abs={metrics['max_abs']:.8f}, "
                f"mean_abs={metrics['mean_abs']:.8f}, allclose_1e6={metrics['allclose_1e6']}"
            )

        report["onnx"][name] = {  # type: ignore[index]
            "left": left_summary,
            "right": right_summary,
            "initializer_hash_diff_count": hash_diff_count,
            "ort": ort_comp,
        }

    if args.skip_trt:
        if args.report_json:
            args.report_json.write_text(json.dumps(report, indent=2))
        return

    print("\n=== TensorRT Rebuild + Output Comparison ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="centerpoint_compare_"))
    print(f"Artifacts dir: {temp_dir}")

    try:
        for pair in onnx_pairs:
            name = pair["name"]
            input_name = pair["input_name"]
            shape = pair["shape_trtexec"]
            left_onnx_file = left_onnx / pair["file"]
            right_onnx_file = right_onnx / pair["file"]

            rng_seed = 123 if name == "voxel_encoder" else 456
            input_bin = temp_dir / f"{name}_input.bin"
            rng = np.random.default_rng(rng_seed)
            rng.standard_normal(pair["trt_shape_tuple"], dtype=np.float32).tofile(input_bin)

            left_engine = temp_dir / f"{name}_left.engine"
            right_engine = temp_dir / f"{name}_right.engine"
            left_json = temp_dir / f"{name}_left.json"
            right_json = temp_dir / f"{name}_right.json"

            trtexec_build(args.trtexec, left_onnx_file, left_engine, input_name, shape, args.workspace_mib)
            trtexec_build(args.trtexec, right_onnx_file, right_engine, input_name, shape, args.workspace_mib)
            trtexec_dump(args.trtexec, left_engine, input_name, shape, input_bin, left_json)
            trtexec_dump(args.trtexec, right_engine, input_name, shape, input_bin, right_json)

            left_outputs = load_trtexec_output(left_json)
            right_outputs = load_trtexec_output(right_json)
            output_comp = {
                output_name: compare_arrays(left_outputs[output_name], right_outputs[output_name])
                for output_name in sorted(set(left_outputs) & set(right_outputs))
            }

            print(f"\n[{name}]")
            print(f"engine_left_sha256:  {file_sha256(left_engine)}")
            print(f"engine_right_sha256: {file_sha256(right_engine)}")
            for output_name, metrics in output_comp.items():
                print(
                    f"TRT {output_name}: max_abs={metrics['max_abs']:.8f}, "
                    f"mean_abs={metrics['mean_abs']:.8f}, allclose_1e6={metrics['allclose_1e6']}"
                )

            report["trt"][name] = {  # type: ignore[index]
                "engine_left": {"path": str(left_engine), "sha256": file_sha256(left_engine)},
                "engine_right": {"path": str(right_engine), "sha256": file_sha256(right_engine)},
                "outputs": output_comp,
            }
    finally:
        if args.keep_artifacts:
            print(f"Kept artifacts in {temp_dir}")
        else:
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()

    if args.report_json:
        args.report_json.write_text(json.dumps(report, indent=2))
        print(f"\nSaved report: {args.report_json}")


if __name__ == "__main__":
    main()
