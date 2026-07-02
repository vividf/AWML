"""Custom ONNX exporter for libspconv INT8 sparse encoder.

Adapts CUDA-BEVFusion's ``exptool.py`` for spconv v2 and the AWML
BEVFusionSparseEncoder.  Produces a custom ONNX graph consumable by
``libspconv``'s C++ ONNX parser (``lidar-scn-onnx-parser.cpp``).

Custom ONNX ops: SparseConvolution, Relu, ScatterDense, Reshape, Transpose.

Usage::

    from deployment.projects.bevfusion.export.libspconv_onnx_exporter import (
        LibspconvExporter,
    )

    exporter = LibspconvExporter()
    exporter.export(model, voxels, coors, batch_size, save_path)
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnx.helper as helper
import torch
import torch.nn as nn


class LibspconvExporter:
    """Trace-based exporter that monkey-patches spconv v2 operations to build a
    custom ONNX graph compatible with libspconv's C++ ONNX parser.
    """

    def __init__(self) -> None:
        self._nodes: List[onnx.NodeProto] = []
        self._initializers: List[onnx.TensorProto] = []
        self._obj_to_id: Dict[int, str] = {}
        self._avoid_reuse: list = []
        self._enabled: bool = False
        self._saved_fns: Dict[str, Any] = {}
        self._conv_counter: int = 0
        self._relu_counter: int = 0
        self._dense_counter: int = 0
        self._reshape_counter: int = 0
        self._permute_counter: int = 0

    # ------------------------------------------------------------------
    # Tensor ID management (same approach as CUDA-BEVFusion exptool)
    # ------------------------------------------------------------------
    def _obj_id(self, obj: Any) -> int:
        try:
            from spconv.pytorch.core import SparseConvTensor

            if isinstance(obj, SparseConvTensor):
                return id(obj.features)
        except ImportError:
            pass
        return id(obj)

    def _register(self, obj: Any) -> str:
        oid = self._obj_id(obj)
        tid = str(len(self._obj_to_id))
        self._obj_to_id[oid] = tid
        return tid

    def _get_id(self, obj: Any) -> str:
        oid = self._obj_id(obj)
        if oid not in self._obj_to_id:
            raise KeyError(
                f"Tensor not registered (id={oid}). An intermediate operation "
                f"may not be traced. Object type: {type(obj).__name__}"
            )
        return self._obj_to_id[oid]

    def _append_initializer(self, value: torch.Tensor, name: str) -> str:
        data = value.cpu().detach().numpy().astype(np.float16).tobytes()
        self._initializers.append(
            helper.make_tensor(
                name=name,
                data_type=helper.TensorProto.DataType.FLOAT16,
                dims=list(value.shape),
                vals=data,
                raw=True,
            )
        )
        return name

    # ------------------------------------------------------------------
    # Operation hooks
    # ------------------------------------------------------------------
    def _make_sparse_conv_hook(self, original_fn):
        exporter = self

        def hooked_forward(module_self, *args, **kwargs):
            if not exporter._enabled:
                return original_fn(module_self, *args, **kwargs)

            exporter._enabled = False
            x_input = args[0] if args else kwargs.get("input")
            y = original_fn(module_self, *args, **kwargs)
            exporter._enabled = True

            exporter._trace_sparse_conv(module_self, x_input, y)
            exporter._avoid_reuse.extend([x_input, y])
            return y

        return hooked_forward

    def _trace_sparse_conv(self, module: nn.Module, x: Any, y: Any) -> None:
        from spconv.pytorch import ops

        self._register(y)
        idx = self._conv_counter
        self._conv_counter += 1

        subm = getattr(module, "subm", False)
        label = "subm" if subm else "conv"
        print(f"   --> SparseConvolution{idx}[{label}]  " f"Input {self._get_id(x)}, Output {self._get_id(y)}")

        # Weight: spconv v2 layout (*kernel_size, C_in, C_out) → (C_out, *kernel_size, C_in)
        w = module.weight.data
        if w.dim() == 5:
            w_export = w.permute(4, 0, 1, 2, 3).contiguous()
        else:
            w_export = w.contiguous()

        inputs = [
            self._get_id(x),
            self._append_initializer(w_export, f"spconv{idx}.weight"),
        ]
        if module.bias is not None:
            inputs.append(self._append_initializer(module.bias.data, f"spconv{idx}.bias"))

        output_bound = getattr(module, "output_bound", 200000)

        precision = getattr(module, "precision", "int8")
        output_precision = getattr(module, "output_precision", "int8")

        attrs: Dict[str, Any] = dict(
            ndim=getattr(module, "ndim", 3),
            input_spatial_shape=list(x.spatial_shape),
            output_spatial_shape=list(y.spatial_shape),
            in_channels=int(module.in_channels),
            out_channels=int(module.out_channels),
            kernel_size=list(module.kernel_size),
            output_bound=output_bound,
            stride=list(module.stride),
            dilation=list(module.dilation),
            padding=list(module.padding),
            transposed=bool(getattr(module, "transposed", False)),
            inverse=bool(getattr(module, "inverse", False)),
            output_padding=list(getattr(module, "output_padding", [0, 0, 0])),
            groups=int(getattr(module, "groups", 1)),
            subm=bool(subm),
            rulebook=getattr(module, "indice_key", "") or "",
            activation=getattr(module, "act_type_str", "None"),
            input_shape=list(x.features.shape),
            output_shape=list(y.features.shape),
            precision=precision,
            output_precision=output_precision,
        )

        idr = getattr(module, "_input_dynamic_range", 0.0)
        wdr = getattr(module, "_weight_dynamic_ranges", [])
        attrs["input_dynamic_range"] = float(idr)
        attrs["weight_dynamic_ranges"] = [float(v) for v in wdr]

        self._nodes.append(
            helper.make_node(
                "SparseConvolution",
                inputs,
                [self._get_id(y)],
                f"conv{idx}",
                **attrs,
            )
        )

    def _make_relu_hook(self, original_fn):
        exporter = self

        def hooked_forward(module_self, *args, **kwargs):
            if not exporter._enabled:
                return original_fn(module_self, *args, **kwargs)

            exporter._enabled = False
            x_input = args[0] if args else kwargs.get("input")
            y = original_fn(module_self, *args, **kwargs)
            exporter._enabled = True

            exporter._trace_relu(x_input, y)
            exporter._avoid_reuse.extend([x_input, y])
            return y

        return hooked_forward

    def _trace_relu(self, x: Any, y: Any) -> None:
        self._register(y)
        idx = self._relu_counter
        self._relu_counter += 1
        print(f"   --> ReLU{idx}  Input {self._get_id(x)}, Output {self._get_id(y)}")
        self._nodes.append(
            helper.make_node(
                "Relu",
                [self._get_id(x)],
                [self._get_id(y)],
                f"relu{idx}",
            )
        )

    def _make_dense_hook(self, original_fn):
        exporter = self

        def hooked_dense(sct_self, *args, **kwargs):
            if not exporter._enabled:
                return original_fn(sct_self, *args, **kwargs)

            exporter._enabled = False
            y = original_fn(sct_self, *args, **kwargs)
            exporter._enabled = True

            exporter._trace_dense(sct_self, y)
            exporter._avoid_reuse.extend([sct_self, y])
            return y

        return hooked_dense

    def _trace_dense(self, sct: Any, y: torch.Tensor) -> None:
        self._register(y)
        idx = self._dense_counter
        self._dense_counter += 1
        print(
            f"   --> ToDense{idx}[{sct.spatial_shape}][{list(y.size())}]  "
            f"Input {self._get_id(sct)}, Output {self._get_id(y)}"
        )
        self._nodes.append(
            helper.make_node(
                "ScatterDense",
                [self._get_id(sct)],
                [self._get_id(y)],
                f"scatter{idx}",
                input_spatial_shape=list(sct.spatial_shape),
                format="xyz",
                output_shape=list(y.size()),
            )
        )

    def _make_permute_hook(self, original_fn):
        exporter = self

        def hooked_permute(tensor_self, *dims):
            if not exporter._enabled:
                return original_fn(tensor_self, *dims)

            exporter._enabled = False
            y = original_fn(tensor_self, *dims)
            exporter._enabled = True

            exporter._trace_permute(tensor_self, y, dims)
            exporter._avoid_reuse.extend([tensor_self, y])
            return y

        return hooked_permute

    def _trace_permute(self, x: torch.Tensor, y: torch.Tensor, dims: tuple) -> None:
        self._register(y)
        idx = self._permute_counter
        self._permute_counter += 1
        print(f"   --> Permute{idx}[{dims}][{list(y.shape)}]  " f"Input {self._get_id(x)}, Output {self._get_id(y)}")
        self._nodes.append(
            helper.make_node(
                "Transpose",
                [self._get_id(x)],
                [self._get_id(y)],
                f"transpose{idx}",
                dims=list(dims),
            )
        )

    def _make_reshape_hook(self, original_fn):
        exporter = self

        def hooked_reshape(tensor_self, *dims):
            if not exporter._enabled:
                return original_fn(tensor_self, *dims)

            exporter._enabled = False
            y = original_fn(tensor_self, *dims)
            exporter._enabled = True

            exporter._trace_reshape(tensor_self, y, dims)
            exporter._avoid_reuse.extend([tensor_self, y])
            return y

        return hooked_reshape

    def _trace_reshape(self, x: torch.Tensor, y: torch.Tensor, dims: tuple) -> None:
        self._register(y)
        idx = self._reshape_counter
        self._reshape_counter += 1
        print(f"   --> Reshape{idx}[{dims}]  " f"Input {self._get_id(x)}, Output {self._get_id(y)}")
        self._nodes.append(
            helper.make_node(
                "Reshape",
                [self._get_id(x)],
                [self._get_id(y)],
                f"reshape{idx}",
                dims=list(dims),
            )
        )

    # ------------------------------------------------------------------
    # Hook installation / cleanup
    # ------------------------------------------------------------------
    def _install_hooks(self) -> None:
        from spconv.pytorch.conv import SparseConvolution as SpconvSparseConv
        from spconv.pytorch.core import SparseConvTensor

        try:
            from spconv.pytorch import SparseReLU
        except ImportError:
            SparseReLU = None

        # SparseConvolution.forward
        self._saved_fns["SpconvSparseConv.forward"] = SpconvSparseConv.forward
        SpconvSparseConv.forward = self._make_sparse_conv_hook(SpconvSparseConv.forward)

        # SparseReLU.forward
        if SparseReLU is not None:
            self._saved_fns["SparseReLU.forward"] = SparseReLU.forward
            SparseReLU.forward = self._make_relu_hook(SparseReLU.forward)

        # torch.nn.ReLU.forward (fallback)
        self._saved_fns["nn.ReLU.forward"] = nn.ReLU.forward
        nn.ReLU.forward = self._make_relu_hook(nn.ReLU.forward)

        # SparseConvTensor.dense
        self._saved_fns["SparseConvTensor.dense"] = SparseConvTensor.dense
        SparseConvTensor.dense = self._make_dense_hook(SparseConvTensor.dense)

        # Tensor.permute
        self._saved_fns["Tensor.permute"] = torch.Tensor.permute
        torch.Tensor.permute = self._make_permute_hook(torch.Tensor.permute)

        # Tensor.reshape (our forward uses reshape, not view)
        self._saved_fns["Tensor.reshape"] = torch.Tensor.reshape
        torch.Tensor.reshape = self._make_reshape_hook(torch.Tensor.reshape)

    def _restore_hooks(self) -> None:
        from spconv.pytorch.conv import SparseConvolution as SpconvSparseConv
        from spconv.pytorch.core import SparseConvTensor

        if "SpconvSparseConv.forward" in self._saved_fns:
            SpconvSparseConv.forward = self._saved_fns["SpconvSparseConv.forward"]

        try:
            from spconv.pytorch import SparseReLU

            if "SparseReLU.forward" in self._saved_fns:
                SparseReLU.forward = self._saved_fns["SparseReLU.forward"]
        except ImportError:
            pass

        if "nn.ReLU.forward" in self._saved_fns:
            nn.ReLU.forward = self._saved_fns["nn.ReLU.forward"]

        if "SparseConvTensor.dense" in self._saved_fns:
            SparseConvTensor.dense = self._saved_fns["SparseConvTensor.dense"]

        if "Tensor.permute" in self._saved_fns:
            torch.Tensor.permute = self._saved_fns["Tensor.permute"]

        if "Tensor.reshape" in self._saved_fns:
            torch.Tensor.reshape = self._saved_fns["Tensor.reshape"]

        self._saved_fns.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(
        self,
        model: nn.Module,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
        save_path: str,
        *,
        forward_fn: Optional[Any] = None,
    ) -> str:
        """Export sparse encoder to custom libspconv ONNX.

        Args:
            model: BEVFusionSparseEncoder (BN fused, FP16, with precision attrs).
            voxels: Input voxel features ``(N, C)``, FP16.
            coors: Voxel coordinates ``(N, 4)``, INT32.
            batch_size: Batch size (typically 1).
            save_path: Output ``.onnx`` file path.
            forward_fn: Optional custom forward callable. If ``None``, uses
                ``_make_encoder_forward(model)``.

        Returns:
            The ``save_path``.
        """
        self._nodes = []
        self._initializers = []
        self._obj_to_id = {}
        self._avoid_reuse = []
        self._conv_counter = 0
        self._relu_counter = 0
        self._dense_counter = 0
        self._reshape_counter = 0
        self._permute_counter = 0

        if forward_fn is None:
            forward_fn = _make_encoder_forward(model)

        coors = coors.int()

        self._install_hooks()
        try:
            self._register(voxels)

            print("Tracing model inference for libspconv ONNX export...")
            with torch.no_grad():
                self._enabled = True
                y = forward_fn(voxels, coors, batch_size)
                self._enabled = False
            print("Tracing done!")

            inputs = [
                helper.make_value_info(
                    name="0",
                    type_proto=helper.make_tensor_type_proto(
                        elem_type=helper.TensorProto.DataType.FLOAT16,
                        shape=list(voxels.size()),
                    ),
                )
            ]

            outputs = [
                helper.make_value_info(
                    name=self._get_id(y),
                    type_proto=helper.make_tensor_type_proto(
                        elem_type=helper.TensorProto.DataType.FLOAT16,
                        shape=list(y.size()),
                    ),
                )
            ]

            graph = helper.make_graph(
                name="scn",
                inputs=inputs,
                outputs=outputs,
                nodes=self._nodes,
                initializer=self._initializers,
            )
            opset = [helper.make_operatorsetid("ai.onnx", 11)]
            onnx_model = helper.make_model(
                graph,
                opset_imports=opset,
                producer_name="awml-bevfusion",
                producer_version="1.0",
            )
            onnx.save_model(onnx_model, save_path)
            print(f"Export completed. ONNX saved to {save_path}")
            print(f"  Nodes: {len(self._nodes)}, Initializers: {len(self._initializers)}")
        finally:
            self._restore_hooks()
            self._avoid_reuse = []

        return save_path


def _make_encoder_forward(model: nn.Module):
    """Build a forward callable that mirrors BEVFusionSparseEncoder.forward
    but works on a BN-fused model for ONNX export."""
    from spconv.pytorch.core import SparseConvTensor

    def forward_fn(voxel_features: torch.Tensor, coors: torch.Tensor, batch_size: int):
        coors = coors.int()
        sparse_shape = getattr(model, "sparse_shape", [1440, 1440, 41])

        num_aug = getattr(model, "num_aug_features", 0)
        if num_aug:
            import numpy as np

            num_points = voxel_features.shape[0]
            min_vals = model.aug_features_min_values.view(1, -1)
            max_vals = model.aug_features_max_values.view(1, -1)
            x_normed = (voxel_features - min_vals) / (max_vals - min_vals)
            y_enc = x_normed.reshape(-1, 1) * np.pi * model.exponents.reshape(1, -1)
            y_enc = y_enc.reshape(num_points, -1)
            voxel_features = torch.cat([torch.cos(y_enc), torch.sin(y_enc)], dim=1)

        input_sp_tensor = SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        x = model.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in model.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = model.conv_out(encode_features[-1])

        spatial_features = out.dense()
        N, C, X, Y, Z = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3)
        spatial_features = spatial_features.reshape(N, C * Z, X, Y)
        return spatial_features

    return forward_fn


def set_precision_attributes(
    model: nn.Module,
    *,
    default_precision: str = "int8",
    default_output_precision: str = "int8",
    conv_input_precision: str = "fp16",
    conv_out_output_precision: str = "fp16",
) -> None:
    """Set ``precision`` and ``output_precision`` on every module,
    following the CUDA-BEVFusion convention.

    - All layers default to INT8 precision.
    - ``conv_input`` keeps FP16 input (receives FP16 features from voxelization).
    - ``conv_out`` output returns to FP16 (feeds into TRT dense engine).
    """
    for name, module in model.named_modules():
        module.precision = default_precision
        module.output_precision = default_output_precision

    conv_input = getattr(model, "conv_input", None)
    if conv_input is not None:
        first = conv_input[0] if hasattr(conv_input, "__getitem__") else conv_input
        first.precision = conv_input_precision

    conv_out = getattr(model, "conv_out", None)
    if conv_out is not None:
        first = conv_out[0] if hasattr(conv_out, "__getitem__") else conv_out
        first.output_precision = conv_out_output_precision


def extract_dynamic_ranges_from_checkpoint(
    model: nn.Module,
    checkpoint_state_dict: Dict[str, torch.Tensor],
) -> int:
    """Extract ``_amax`` values from a PTQ checkpoint and set them as
    ``_input_dynamic_range`` / ``_weight_dynamic_ranges`` on each
    SparseConvolution module.

    Returns the number of modules that received dynamic range attributes.
    """
    from spconv.pytorch.conv import SparseConvolution as SpconvSparseConv

    amax_map: Dict[str, torch.Tensor] = {}
    for k, v in checkpoint_state_dict.items():
        if "_amax" in k:
            amax_map[k] = v

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, SpconvSparseConv):
            continue

        input_key = f"{name}._input_quantizer._amax"
        weight_key = f"{name}._weight_quantizer._amax"

        alt_prefixes = [
            "",
            "pts_middle_encoder.",
            "module.",
            "module.pts_middle_encoder.",
        ]

        idr = None
        wdr = None
        for prefix in alt_prefixes:
            ik = f"{prefix}{input_key}"
            wk = f"{prefix}{weight_key}"
            if ik in amax_map and idr is None:
                idr = float(amax_map[ik].cpu().flatten()[0])
            if wk in amax_map and wdr is None:
                wdr = amax_map[wk].cpu().flatten().tolist()

        if idr is not None:
            module._input_dynamic_range = idr
        if wdr is not None:
            module._weight_dynamic_ranges = wdr

        if idr is not None or wdr is not None:
            count += 1
            print(f"  [dynamic-range] {name}: input_dr={idr}, " f"weight_dr_len={len(wdr) if wdr else 0}")

    return count
