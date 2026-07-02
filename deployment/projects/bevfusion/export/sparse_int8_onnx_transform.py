"""Post-process an ONNX model to replace ImplicitGemm → ImplicitGemmInt8.

sparse INT8 approach: The standard Autoware ONNX export (via torch.onnx.export +
sparse_functional.py symbolic methods) produces autoware::ImplicitGemm nodes
with 5 inputs. This script enriches them to autoware::ImplicitGemmInt8 nodes
with 7 inputs (+ channel_scale + bias_scaled) and INT8 scale attributes.

Usage::

    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform \\
        --onnx work_dirs/bevfusion/sparse_encoder.onnx \\
        --checkpoint work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth \\
        --config projects/BEVFusion/configs/.../bevfusion_..._120m.py \\
        --output work_dirs/bevfusion/sparse_encoder_int8.onnx

The output ONNX can be loaded by TensorRT with the ImplicitGemmInt8Plugin.

Debugging / scale audit::

    # Per-layer JSON (matched ONNX node ↔ PTQ stem, input/output scales, channel_scale stats)
    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform ... --audit-report int8_layers.json

    # Read-only dump of an already-transformed INT8 ONNX (no checkpoint)
    python -m deployment.projects.bevfusion.export.sparse_int8_onnx_audit --onnx sparse_int8.onnx
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import _try_get_constant_numpy
from deployment.projects.bevfusion.export.sparse_int8_transform_ops import (
    _LEGACY_W_AMAX_PTQ,
    _append_implicit_gemm_int8_plugin_attributes,
    _build_layer_scale_map,
    _build_scale_info,
    _collect_occupied_tensor_names,
    _implicit_gemm_attrs_from_node,
    _implicit_gemm_filter_c_out,
    _implicit_gemm_filter_c_out_c_in,
    _implicit_gemm_matches_fp16_pattern,
    _implicit_gemm_node_precision,
    _implicit_gemm_to_int8_path,
    _load_amax_from_checkpoint,
    _load_deploy_transform_options,
    _match_onnx_node_to_layer,
    _print_int8_census,
    _safe_trt_scale_names,
    _set_implicit_gemm_node_precision,
    _terminal_boundary_amax,
)
from deployment.quantization.sparse.naming import (
    topologically_sorted_sparse_stems as _topologically_sorted_sparse_stems,
)


def transform_onnx_int8(
    model: onnx.ModelProto,
    layer_scales: Dict[str, dict],
    encoder_sd: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = False,
    amax_dict: Optional[Dict[str, torch.Tensor]] = None,
    override_terminal_absmax: Optional[float] = None,
    audit_records: Optional[List[Dict[str, Any]]] = None,
    fp16_layer_patterns: Optional[List[str]] = None,
    fuse_implicit_gemm_trailing_relu: bool = True,
) -> onnx.ModelProto:
    """Replace ImplicitGemm nodes with ImplicitGemmInt8 nodes.

    Args:
        model: ONNX model with autoware::ImplicitGemm nodes.
        layer_scales: dict from _build_layer_scale_map.
        encoder_sd: **Required** for strict weight ``_amax`` layout (5D ``(C_out,C_in)``)
            and per-``C_out`` vectors; also used for bias.
        amax_dict: Raw ``_amax`` keys from the PTQ checkpoint (same as ``_load_amax_from_checkpoint``).
            Used for terminal ``output_scale`` via ``conv_out.*._input_quantizer._amax`` when sparse INT8
            buffers are absent. Prefer ``encoder_sd['_last_int8_conv_output_absmax']`` from sparse
            PTQ, or pass ``--terminal-absmax``.
        audit_records: If provided, append one JSON-serializable dict per converted
            ``ImplicitGemmInt8`` (for ``--audit-report`` or custom tooling).
        fp16_layer_patterns: Optional list of case-insensitive substring patterns.
            Any ``ImplicitGemm`` node whose **name** contains one of the patterns is
            **kept as FP16** ``ImplicitGemm`` (skipped INT8 replacement). Driven by
            ``spconv_int8_fp16_layers`` in the BEVFusion deploy_config. ``conv_out`` follows
            the same rule as other layers (no ONNX special-case skip).
        fuse_implicit_gemm_trailing_relu: When True, run ``fuse_autoware_implicit_gemm_trailing_relu``
            so standalone ``Relu`` chains on sparse conv outputs become ``ImplicitGemm.act_type=kReLU``.

    Returns:
        Modified ONNX model with autoware::ImplicitGemmInt8 nodes.

    Raises:
        ValueError: if terminal boundary amax is missing, stem/scale resolution fails, ONNX nodes
            cannot be matched, or checkpoint used legacy weight quantizer ``axis=(4)``.
        RuntimeError: if the count of converted ImplicitGemm nodes does not match the graph.
    """
    if encoder_sd is None:
        raise ValueError(
            "transform_onnx_int8 requires encoder state_dict (--checkpoint) to validate "
            "weight _amax vs each layer's (C_out, C_in). " + _LEGACY_W_AMAX_PTQ
        )

    model = copy.deepcopy(model)

    if fuse_implicit_gemm_trailing_relu:
        from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import (
            fuse_autoware_implicit_gemm_trailing_relu,
        )

        n_relu_in = sum(1 for n in model.graph.node if n.op_type == "Relu")
        n_fused = fuse_autoware_implicit_gemm_trailing_relu(model)
        if n_fused:
            print(
                f"  [onnx-fuse] spconv_fuse_implicit_gemm_relu=True: merged {n_fused} "
                "ImplicitGemm→Relu chain(s): act_type=kReLU (1), Relu nodes removed."
            )
        else:
            print(
                "  [onnx-fuse] spconv_fuse_implicit_gemm_relu=True but fused 0 chains "
                f"(Relu nodes in graph={n_relu_in}). No direct ImplicitGemm→Relu adjacency was "
                "found — check for BatchNorm/Add nodes sitting between ImplicitGemm and Relu."
            )
    else:
        print("  [onnx-fuse] spconv_fuse_implicit_gemm_relu=False: skipping ImplicitGemm→Relu fusion.")

    graph = model.graph
    layer_stems = list(layer_scales.keys())
    occupied_names = _collect_occupied_tensor_names(graph)

    # Derive output_amax from the **activation successor** in topo order (not raw dict / lexical
    # order). See _topologically_sorted_sparse_stems docstring.
    topo_stems = _topologically_sorted_sparse_stems(list(layer_scales.keys()))
    if verbose:
        print(f"  [debug-topo] first_stems={topo_stems[:6]}... last={topo_stems[-3:]}")

    term_np, term_src = _terminal_boundary_amax(
        encoder_sd, amax_dict, override_terminal_absmax=override_terminal_absmax
    )
    if term_np is None:
        raise ValueError(
            "Sparse INT8 ONNX transform: checkpoint has no terminal boundary amax. Sparse PTQ must save "
            "pts_middle_encoder._last_int8_conv_output_absmax (preferred) or "
            "_sparse_tail_absmax, or calibrate conv_out._input_quantizer._amax, or pass "
            "--terminal-absmax when running sparse_int8_onnx_transform."
        )
    print(
        f"  [int8-output-scale] Terminal boundary: source={term_src} "
        f"amax={float(term_np.reshape(-1)[0]):.6f} → output_scale={float(term_np.reshape(-1)[0]) / 127.0:.6f}"
    )
    if verbose and term_src == "sparse_tail_absmax_legacy":
        print(
            "  [int8-output-scale] Using pts_middle_encoder._sparse_tail_absmax for terminal "
            "scale (legacy). Prefer re-running sparse PTQ to get "
            "_last_int8_conv_output_absmax — tail-at-conv_out can over-scale the last "
            "ImplicitGemmInt8 and inflate TRT lidar_bev."
        )

    scale_info = _build_scale_info(topo_stems, layer_scales, encoder_sd, term_np, term_src, verbose)

    fp16_patterns_norm: List[str] = [p.lower() for p in (fp16_layer_patterns or []) if p]
    if fp16_patterns_norm:
        print(f"  [int8] spconv_int8_fp16_layers patterns (kept FP16): {fp16_patterns_norm}")

    n_expected_int8 = sum(1 for n in graph.node if _implicit_gemm_to_int8_path(n, fp16_patterns_norm))
    # Track which patterns matched nodes, to warn about typos / dead patterns.
    fp16_pattern_hits: Dict[str, int] = {p: 0 for p in fp16_patterns_norm}

    # Replace nodes.
    new_nodes = []
    transform_count = 0
    stem_assigned_to_node: Dict[str, str] = {}
    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "ImplicitGemm" or node.domain != "autoware":
            new_nodes.append(node)
            continue

        # Already-converted INT8 node (precision=1) from a prior run — keep as-is (idempotent).
        if _implicit_gemm_node_precision(node) == 1:
            new_nodes.append(node)
            continue

        matched_fp16 = _implicit_gemm_matches_fp16_pattern(node, fp16_patterns_norm)
        if matched_fp16 is not None:
            fp16_pattern_hits[matched_fp16] += 1
            # Stamp precision=0 explicitly so the FP-kept node is self-describing in the ONNX.
            _set_implicit_gemm_node_precision(node, 0)
            print(
                f"  [int8] Keep FP16 ImplicitGemm per spconv_int8_fp16_layers "
                f"(pattern={matched_fp16!r}): name={node.name!r} (precision=0)"
            )
            new_nodes.append(node)
            continue

        stem = _match_onnx_node_to_layer(
            node,
            model,
            layer_stems,
            layer_scales,
            encoder_sd=encoder_sd,
            verbose=verbose,
        )
        if stem is None or stem not in scale_info:
            raise ValueError(
                "Sparse INT8 ONNX transform: ImplicitGemm node could not be matched to a calibrated stem "
                f"or scales were not built for it: name={node.name!r} inputs={list(node.input)}. "
                "Use --verbose on sparse_int8_onnx_transform to debug stem matching."
            )

        prev = stem_assigned_to_node.get(stem)
        if prev is not None:
            raise ValueError(
                "Sparse INT8 ONNX transform: duplicate PTQ stem assignment — two ImplicitGemm nodes "
                f"matched the same stem {stem!r} (first={prev!r}, second={node.name!r}). "
                "This usually means substring-based matching is ambiguous; run with --verbose "
                "and fix ONNX tensor naming or matching heuristics."
            )
        stem_assigned_to_node[stem] = node.name or f"<unnamed_{transform_count}>"

        si = scale_info[stem]
        c_scale = len(si["channel_scale"])
        c_filter = _implicit_gemm_filter_c_out(model, node)
        if c_filter is not None and c_scale != c_filter:
            raise ValueError(
                "Sparse INT8 ONNX transform: channel_scale length does not match filter C_out "
                f"(wrong stem match?): stem={stem!r} channel_scale_len={c_scale} "
                f"filter_c_out={c_filter} node={node.name!r}"
            )

        c_out = c_scale

        # FP16 fusion may add a 6th tensor input (Add folded into ImplicitGemm). INT8 uses 5 sparse
        # tensors + scales; merge that extra FP32/Half bias into bias_scaled (= bias / output_scale).
        bs_arr = np.asarray(si["bias_scaled"], dtype=np.float32).reshape(-1).copy()
        if len(node.input) == 6:
            extra_name = node.input[5]
            extra = _try_get_constant_numpy(graph, extra_name, init_map)
            if extra is None:
                raise ValueError(
                    "Sparse INT8 ONNX transform: ImplicitGemm "
                    f"{node.name!r} has 6 inputs (ONNX-fused bias) but constant "
                    f"{extra_name!r} is not an initializer or Constant node value."
                )
            ex = np.asarray(extra, dtype=np.float32).reshape(-1)
            if ex.size != bs_arr.size:
                raise ValueError(
                    "Sparse INT8 ONNX transform: fused 6th-input bias length "
                    f"{ex.size} != C_out {bs_arr.size} (stem={stem!r}, node={node.name!r})."
                )
            out_sc = float(si["output_scale"])
            bs_arr = bs_arr + (ex / out_sc)
            if verbose:
                print(
                    f"  [int8] Merged ONNX 6th-input fused bias into bias_scaled "
                    f"(stem={stem!r}, node={node.name!r})"
                )

        # Create ONNX initializers for channel_scale and bias_scaled.
        cs_name, bs_name = _safe_trt_scale_names(stem, occupied_names)

        cs_init = numpy_helper.from_array(si["channel_scale"], name=cs_name)
        bs_init = numpy_helper.from_array(bs_arr, name=bs_name)
        graph.initializer.append(cs_init)
        graph.initializer.append(bs_init)
        init_map[cs_name] = np.asarray(si["channel_scale"], dtype=np.float32)
        init_map[bs_name] = bs_arr

        # TRT's ONNX parser requires graph.input entries with type info
        # for all initializers referenced by custom plugin nodes.
        cs_vi = helper.make_tensor_value_info(cs_name, TensorProto.FLOAT, [c_out])
        bs_vi = helper.make_tensor_value_info(bs_name, TensorProto.FLOAT, [c_out])
        graph.input.append(cs_vi)
        graph.input.append(bs_vi)

        # Preserve existing attributes (normalize names), override INT8 scales from PTQ.
        attrs = _implicit_gemm_attrs_from_node(node)
        attrs["output_scale"] = float(si["output_scale"])
        attrs["input_scale"] = float(si["input_scale"])

        # Fused FP16 export may have 6 inputs (optional per-channel bias); Int8 uses 5 sparse + scales.
        if len(node.input) not in (5, 6):
            raise ValueError(
                f"sparse INT8: autoware::ImplicitGemm {node.name!r} has {len(node.input)} inputs; "
                "expected 5 or 6 (6 = ONNX-fused bias). Take first 5 as sparse tensors."
            )
        sparse_in = list(node.input[:5])

        # Keep op_type "ImplicitGemm" (same Autoware plugin/creator as FP16); precision=1 attr +
        # the two extra FP32 scale inputs switch the plugin into INT8 mode. Avoids a second plugin.
        int8_node = helper.make_node(
            "ImplicitGemm",
            inputs=sparse_in + [cs_name, bs_name],
            outputs=list(node.output),
            domain="autoware",
            name=f"{node.name}_int8" if node.name else f"ImplicitGemmInt8_{transform_count}",
        )
        _append_implicit_gemm_int8_plugin_attributes(int8_node, attrs)

        new_nodes.append(int8_node)
        transform_count += 1
        print(
            f"  [int8] {stem}: input_scale={si['input_scale']:.6f} "
            f"output_scale={si['output_scale']:.6f} "
            f"channel_scale_shape={si['channel_scale'].shape}"
        )

        if audit_records is not None:
            cs = si["channel_scale"].reshape(-1).astype(np.float64)
            c_out_i, c_in_i = _implicit_gemm_filter_c_out_c_in(model, node)
            audit_records.append(
                {
                    "implicit_gemm_node_name": node.name or "",
                    "implicit_gemm_int8_node_name": int8_node.name or "",
                    "matched_stem": stem,
                    "filter_input": node.input[1] if len(node.input) > 1 else "",
                    "c_out": int(c_out_i) if c_out_i is not None else None,
                    "c_in": int(c_in_i) if c_in_i is not None else None,
                    "input_scale": float(si["input_scale"]),
                    "output_scale": float(si["output_scale"]),
                    "channel_scale_len": int(cs.size),
                    "channel_scale_min": float(cs.min()) if cs.size else None,
                    "channel_scale_max": float(cs.max()) if cs.size else None,
                    "channel_scale_mean": float(cs.mean()) if cs.size else None,
                    "channel_scale_initializer": cs_name,
                    "bias_scaled_initializer": bs_name,
                }
            )

    # Replace nodes in graph.
    del graph.node[:]
    graph.node.extend(new_nodes)

    if transform_count != n_expected_int8:
        raise RuntimeError(
            f"Sparse INT8 ONNX transform: expected {n_expected_int8} ImplicitGemm → ImplicitGemmInt8 "
            f"replacements (excluding conv_out), got {transform_count}. "
            "Graph/calibration mismatch."
        )

    unused_stems = set(scale_info.keys()) - set(stem_assigned_to_node.keys())
    if unused_stems:
        print(
            "\n  [int8-audit] WARNING: calibrated stems with no matched ImplicitGemm node "
            f"(count={len(unused_stems)}): {sorted(unused_stems)[:12]}"
            f"{'...' if len(unused_stems) > 12 else ''}"
        )

    _print_int8_census(graph, fp16_pattern_hits)

    print(f"\nTransformed {transform_count} ImplicitGemm → ImplicitGemmInt8 nodes")
    return model


def main():
    parser = argparse.ArgumentParser(description="Transform ONNX ImplicitGemm nodes to ImplicitGemmInt8")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="PTQ checkpoint with NVIDIA _amax calibration values",
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--config",
        default=None,
        help="MMEngine config (optional, for bias extraction from fresh model)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print topology, scale chain, and per-node stem matching diagnostics",
    )
    parser.add_argument(
        "--terminal-absmax",
        type=float,
        default=None,
        help="Override scalar amax for the last INT8 layer output_scale (= absmax/127). "
        "Use if the checkpoint lacks sparse INT8 buffers or you need a one-off fix.",
    )
    parser.add_argument(
        "--audit-report",
        default=None,
        help="Write JSON array of per-layer INT8 scale summaries (matched stem, scales, channel_scale stats).",
    )
    parser.add_argument(
        "--fp16-layers",
        default=None,
        help=(
            "Comma-separated list of case-insensitive substring patterns. Any ImplicitGemm "
            "node whose name/inputs/outputs contains one of these substrings is kept FP16 "
            "instead of being replaced by ImplicitGemmInt8 (for accuracy tuning). "
            "Example: --fp16-layers 'encoder_layer3.encoder_layer3.2,conv_input.0'"
        ),
    )
    parser.add_argument(
        "--deploy-cfg",
        default=None,
        help="Loads deploy_config .py for spconv_int8_fp16_layers.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load ONNX.
    print(f"Loading ONNX: {args.onnx}")
    model = onnx.load(args.onnx)

    # Load _amax values.
    print(f"Loading _amax from: {args.checkpoint}")
    amax_dict = _load_amax_from_checkpoint(args.checkpoint)
    print(f"  Found {len(amax_dict)} _amax keys")

    layer_scales = _build_layer_scale_map(amax_dict)
    print(f"  Matched {len(layer_scales)} sparse conv layers")
    ci0 = layer_scales.get("conv_input.0", {})
    wa0 = ci0.get("weight_amax")
    if wa0 is not None:
        s = np.asarray(wa0).shape
        print(
            f"  [sanity] conv_input.0 weight_amax shape={s} "
            f"(sparse INT8 expects per-C_out, e.g. (16,1,1,1,1); "
            f"(1,1,1,1,5) means an old checkpoint or wrong --checkpoint path)"
        )

    # Optional: load encoder state_dict for bias.
    encoder_sd = None
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_sd = ckpt.get("state_dict", ckpt)

    deploy_cfg_path = getattr(args, "deploy_cfg", None)
    deploy_opts = _load_deploy_transform_options(deploy_cfg_path)
    fp16_layer_patterns = list(deploy_opts.fp16_layer_patterns)

    if args.fp16_layers:
        fp16_layer_patterns.extend(p.strip() for p in args.fp16_layers.split(",") if p.strip())

    fp16_layer_patterns = list(dict.fromkeys(fp16_layer_patterns))

    print(f"\nTransforming ImplicitGemm → ImplicitGemmInt8...")
    if fp16_layer_patterns:
        print(f"  FP16 keep-list ({len(fp16_layer_patterns)}): {fp16_layer_patterns}")
    print(
        "  ImplicitGemm ReLU/Add ONNX fuse: "
        f"{'enabled' if deploy_opts.fuse_implicit_gemm_relu else 'disabled'}"
        + (f" (from deploy_cfg {deploy_cfg_path!r})" if deploy_cfg_path else " (default)")
    )
    audit_records: List[Dict[str, Any]] = []
    model = transform_onnx_int8(
        model,
        layer_scales,
        encoder_sd,
        verbose=args.verbose,
        amax_dict=amax_dict,
        override_terminal_absmax=args.terminal_absmax,
        audit_records=audit_records if args.audit_report else None,
        fp16_layer_patterns=fp16_layer_patterns,
        fuse_implicit_gemm_trailing_relu=deploy_opts.fuse_implicit_gemm_relu,
    )

    if args.audit_report:
        path = args.audit_report
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "terminal": {
                        "note": "See print log [int8-output-scale] for terminal amax source",
                    },
                    "layers": audit_records,
                },
                f,
                indent=2,
            )
        print(f"\n  [int8-audit] Wrote {len(audit_records)} layer entries to {path!r}")

    # Save (save_model avoids some TRT edge cases with large graphs).
    onnx.save_model(model, args.output)
    print(f"\nSaved INT8 ONNX: {args.output}")


if __name__ == "__main__":
    main()
