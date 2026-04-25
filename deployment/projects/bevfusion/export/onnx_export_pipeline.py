"""BEVFusion ONNX export pipeline.

Exports the BEVFusion main_body to a single ONNX file, including the TopK fix.
Replicates the logic from projects/BEVFusion/deploy/ within the new deployment framework.
"""

from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn

from deployment.configs import BaseDeploymentConfig
from deployment.core.artifacts import Artifact
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.export_pipelines.base import OnnxExportPipeline
from deployment.projects.bevfusion.io.component_utils import is_split_bevfusion_components
from deployment.projects.bevfusion.io.model_loader import setup_quantization_for_onnx_export

logger = logging.getLogger(__name__)


def _head_dict_to_export_outputs(outputs: dict) -> tuple:
    """Match ``BEVFusionMainBodyWrapper`` post-processing (ONNX outputs)."""
    import torch.nn.functional as F

    score = outputs["heatmap"].sigmoid()
    one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
    score = score * outputs["query_heatmap_score"] * one_hot
    score = score[0].max(dim=0)[0]

    bbox_pred = torch.cat(
        [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]],
        dim=0,
    )

    return bbox_pred, score, outputs["query_labels"][0]


class BEVFusionSparseWrapper(nn.Module):
    """LiDAR sparse tower only: voxels/coors/num_points → BEV feature map."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.mod = model

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        voxels = voxels.to(dtype=torch.float32)
        coors = coors.to(dtype=torch.int32)
        if coors.shape[1] == 3:
            num_points = coors.shape[0]
            batch_coors = torch.zeros(num_points, 1, dtype=torch.int32, device=coors.device)
            coors = torch.cat([batch_coors, coors], dim=1).contiguous()

        return self.mod.extract_pts_feat(voxels, coors, num_points_per_voxel, points=None)


class BEVFusionDenseWrapper(nn.Module):
    """SECOND + neck + head (+ ONNX postprocess). Input: ``lidar_bev`` [B,C,H,W]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.mod = model

    def forward(self, lidar_bev: torch.Tensor) -> tuple:
        x = lidar_bev
        if self.mod.pts_backbone is not None:
            x = self.mod.pts_backbone(x)
        if self.mod.pts_neck is not None:
            x = self.mod.pts_neck(x)
        x = self.mod._align_lidar_bev_to_head_grid(x)
        outputs = self.mod.bbox_head(x, [])
        head_out = outputs[0][0]
        return _head_dict_to_export_outputs(head_out)


class BEVFusionMainBodyWrapper(nn.Module):
    """Wrapper for BEVFusion that matches the ONNX export interface.

    Takes voxels/coors/num_points_per_voxel and returns bbox_pred/score/label_pred.
    Replicates TrtBevFusionMainContainer from projects/BEVFusion/deploy/containers.py.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.mod = model

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> tuple:
        import torch.nn.functional as F

        # spconv requires int32 indices; float batch column (torch.zeros default) + int coors
        # yields float tensor and can CUDA fault when SPCONV_FX_TRACE_MODE relaxes dtype checks.
        # INT8 sparse path: quantize_per_tensor only accepts float; keep voxels FP32.
        voxels = voxels.to(dtype=torch.float32)
        coors = coors.to(dtype=torch.int32)
        if coors.shape[1] == 3:
            num_points = coors.shape[0]
            batch_coors = torch.zeros(num_points, 1, dtype=torch.int32, device=coors.device)
            coors = torch.cat([batch_coors, coors], dim=1).contiguous()

        batch_inputs_dict = {
            "voxels": {"voxels": voxels, "coors": coors, "num_points_per_voxel": num_points_per_voxel},
        }

        outputs = self.mod._forward(batch_inputs_dict, using_image_features=True)

        score = outputs["heatmap"].sigmoid()
        one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
        score = score * outputs["query_heatmap_score"] * one_hot
        score = score[0].max(dim=0)[0]

        bbox_pred = torch.cat(
            [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]],
            dim=0,
        )

        return bbox_pred, score, outputs["query_labels"][0]


class BEVFusionONNXExportPipeline(OnnxExportPipeline):
    """ONNX export for BEVFusion.

    - **Single-file** (``bevfusion_main_body``): full graph, TopK fix applied.
    - **Split** (``bevfusion_sparse`` + ``bevfusion_dense``): sparse tower ONNX + dense ONNX
      (route 1: sparse can go to libspconv / plugin; dense → TensorRT without spconv ops).
    """

    def __init__(
        self,
        module: str = "main_body",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.module = module
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
    ) -> Artifact:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if is_split_bevfusion_components(config.components_cfg):
            return self._export_split(model, data_loader, output_dir_path, config, sample_idx)

        self.logger.info("=" * 80)
        self.logger.info("Exporting BEVFusion to ONNX (single-file)")
        self.logger.info("=" * 80)

        device = next(model.parameters()).device
        self.logger.info(f"Model device: {device}")

        component_cfg = config.components_cfg.get_component("bevfusion_main_body")
        onnx_filename = component_cfg.onnx_file
        output_path = output_dir_path / onnx_filename
        temp_path = output_dir_path / onnx_filename.replace(".onnx", "_temp_to_be_fixed.onnx")

        self.logger.info(f"Loading sample {sample_idx} for export tracing...")
        sample = data_loader.load_sample(sample_idx)
        points = sample["points"]
        self.logger.info(f"Sample loaded: points shape={points.shape}")

        self.logger.info("Running voxelization...")
        voxels, coors, num_points_per_voxel = self._voxelize(model, points)
        self.logger.info(f"Voxelization done: {voxels.shape[0]} voxels")

        onnx_cfg = self._get_onnx_config(config, "bevfusion_main_body")
        self.logger.info(
            f"ONNX config: opset={onnx_cfg['opset_version']}, inputs={onnx_cfg['input_names']}, outputs={onnx_cfg['output_names']}"
        )

        # Use QuantizeLinear/DequantizeLinear in ONNX (same as CenterPoint). Without this,
        # pytorch_quantization TensorQuantizer exports as primitive ops (Mul, Round, Clip, Div).
        setup_quantization_for_onnx_export()
        from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
            ensure_spconv_quantize_per_tensor_float_activations,
        )

        ensure_spconv_quantize_per_tensor_float_activations()
        self.logger.info("Running torch.onnx.export...")
        self._export_to_onnx(model, voxels, coors, num_points_per_voxel, str(temp_path), onnx_cfg)

        num_proposals = self._get_num_proposals(model)
        self._fix_topk(str(temp_path), str(output_path), num_proposals)

        self.logger.info("=" * 80)
        self.logger.info(f"BEVFusion ONNX export successful: {output_path}")
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))

    def _export_split(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir_path: Path,
        config: BaseDeploymentConfig,
        sample_idx: int,
    ) -> Artifact:
        """Export ``bevfusion_sparse.onnx`` and ``bevfusion_dense.onnx``."""
        self.logger.info("=" * 80)
        self.logger.info("Exporting BEVFusion to ONNX (split: sparse + dense)")
        self.logger.info("=" * 80)

        self._assert_split_model_ok(model)

        device = next(model.parameters()).device
        self.logger.info(f"Model device: {device}")

        self.logger.info(f"Loading sample {sample_idx} for export tracing...")
        sample = data_loader.load_sample(sample_idx)
        points = sample["points"]
        self.logger.info(f"Sample loaded: points shape={points.shape}")

        self.logger.info("Running voxelization...")
        voxels, coors, num_points_per_voxel = self._voxelize(model, points)
        self.logger.info(f"Voxelization done: {voxels.shape[0]} voxels")

        setup_quantization_for_onnx_export()
        from deployment.projects.bevfusion.quantization.spconv_quantized_add_patch import (
            ensure_spconv_quantize_per_tensor_float_activations,
        )

        ensure_spconv_quantize_per_tensor_float_activations()

        sparse_cfg = config.components_cfg.get_component("bevfusion_sparse")
        dense_cfg = config.components_cfg.get_component("bevfusion_dense")
        sparse_onnx = output_dir_path / sparse_cfg.onnx_file
        dense_onnx = output_dir_path / dense_cfg.onnx_file
        dense_temp = output_dir_path / dense_cfg.onnx_file.replace(".onnx", "_temp_to_be_fixed.onnx")

        onnx_cfg_sparse = self._get_onnx_config(config, "bevfusion_sparse")
        onnx_cfg_dense = self._get_onnx_config(config, "bevfusion_dense")

        self.logger.info(
            "Sparse ONNX: inputs=%s outputs=%s",
            onnx_cfg_sparse["input_names"],
            onnx_cfg_sparse["output_names"],
        )
        self.logger.info("Running torch.onnx.export (sparse)...")
        self._export_to_onnx(
            model, voxels, coors, num_points_per_voxel, str(sparse_onnx), onnx_cfg_sparse, wrapper="sparse"
        )

        with torch.no_grad():
            sw = BEVFusionSparseWrapper(model)
            sw.eval()
            trace_dev = device
            lidar_bev = sw(
                voxels.to(trace_dev), coors.to(trace_dev, dtype=torch.int32), num_points_per_voxel.to(trace_dev)
            )
        self.logger.info("Dense trace input lidar_bev shape: %s", tuple(lidar_bev.shape))

        self.logger.info(
            "Dense ONNX: inputs=%s outputs=%s",
            onnx_cfg_dense["input_names"],
            onnx_cfg_dense["output_names"],
        )
        self.logger.info("Running torch.onnx.export (dense)...")
        self._export_dense_to_onnx(model, lidar_bev, str(dense_temp), onnx_cfg_dense)

        num_proposals = self._get_num_proposals(model)
        self._fix_topk(str(dense_temp), str(dense_onnx), num_proposals)

        self.logger.info("=" * 80)
        self.logger.info("Split ONNX export OK: %s , %s", sparse_onnx, dense_onnx)
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))

    @staticmethod
    def _assert_split_model_ok(model: torch.nn.Module) -> None:
        if getattr(model, "fusion_layer", None) is not None:
            raise RuntimeError(
                "Split ONNX export requires LiDAR-only path (fusion_layer must be None). "
                "Use single-file export or implement a fusion ONNX branch."
            )
        if getattr(model, "img_backbone", None) is not None:
            raise RuntimeError("Split ONNX export is for LiDAR-only BEVFusion (img_backbone must be None).")
        if getattr(model, "pts_middle_encoder", None) is None:
            raise RuntimeError("pts_middle_encoder is required for split sparse export.")

    def _voxelize(self, model: torch.nn.Module, points: torch.Tensor) -> tuple:
        """Run voxelization on a point cloud sample."""
        device = next(model.parameters()).device
        points = points.to(device).float()

        with torch.no_grad():
            ret = model.pts_voxel_layer(points)
            if len(ret) == 3:
                feats, coords, sizes = ret
            else:
                feats, coords = ret
                sizes = torch.ones(feats.shape[0], device=device)
            coords = coords[:, :].to(dtype=torch.int32)  # [M, 3] (z, y, x); spconv expects int32

        return feats, coords, sizes

    def _get_onnx_config(self, config: BaseDeploymentConfig, component_name: str) -> Dict[str, Any]:
        """Build ONNX export configuration for a components_cfg entry."""
        component_cfg = config.components_cfg.get_component(component_name)
        io_cfg = component_cfg.io

        input_names = [inp.name for inp in io_cfg.inputs]
        output_names = [out.name for out in io_cfg.outputs]

        dynamic_axes = {}
        if hasattr(io_cfg, "dynamic_axes") and io_cfg.dynamic_axes:
            dynamic_axes = dict(io_cfg.dynamic_axes)

        onnx_settings = config.onnx_config
        opset_version = getattr(onnx_settings, "opset_version", 17)
        # Default "auto" = trace on the same device as the model (usually CUDA). CUDA-built spconv
        # implicit_gemm runs GPU kernels: indices/features on CPU + those kernels => cudaErrorIllegalAddress.
        # If you lack GPU memory for dense(), set trace_device=cpu only with a CPU spconv build, or export
        # on another machine; see docs/4_spconv_int8_implementation_history_zh.md (this project).
        trace_device = onnx_settings.trace_device or os.environ.get("BEVFUSION_ONNX_TRACE_DEVICE", "auto")

        return {
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "opset_version": opset_version,
            "export_params": True,
            "keep_initializers_as_inputs": False,
            "verbose": False,
            "trace_device": trace_device,
        }

    def _export_dense_to_onnx(
        self,
        model: torch.nn.Module,
        lidar_bev: torch.Tensor,
        output_path: str,
        onnx_cfg: Dict[str, Any],
    ) -> None:
        """Export pts_backbone + neck + head (+ postprocess) to ONNX."""
        model_device = next(model.parameters()).device
        raw_td = onnx_cfg.get("trace_device") or "auto"
        if raw_td in ("auto", "", None):
            trace_dev = model_device
        else:
            trace_dev = torch.device(raw_td)

        if model_device.type == "cuda" and trace_dev.type == "cpu":
            trace_dev = model_device

        moved_for_trace = trace_dev != model_device
        if moved_for_trace:
            model.to(trace_dev)

        try:
            wrapper = BEVFusionDenseWrapper(model)
            wrapper.eval()
            wrapper.to(trace_dev)
            bev = lidar_bev.to(trace_dev)

            export_kw: Dict[str, Any] = dict(
                export_params=onnx_cfg["export_params"],
                input_names=onnx_cfg["input_names"],
                output_names=onnx_cfg["output_names"],
                opset_version=onnx_cfg["opset_version"],
                dynamic_axes=onnx_cfg["dynamic_axes"],
                keep_initializers_as_inputs=onnx_cfg["keep_initializers_as_inputs"],
                verbose=onnx_cfg["verbose"],
            )
            try:
                from torch.onnx import TrainingMode

                export_kw["training"] = TrainingMode.EVAL
            except Exception:
                pass

            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*non-tuple sequence for multidimensional indexing.*",
                        category=UserWarning,
                    )
                    torch.onnx.export(wrapper, (bev,), output_path, **export_kw)
        finally:
            if moved_for_trace:
                try:
                    model.to(model_device)
                except Exception as e:
                    self.logger.warning("Could not move model back after dense ONNX export: %s", e)
                if model_device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        self.logger.info("Exported dense ONNX to %s", output_path)

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        output_path: str,
        onnx_cfg: Dict[str, Any],
        *,
        wrapper: str = "main",
    ) -> None:
        """Export voxel-based subgraph to ONNX (full main_body or sparse-only)."""
        model_device = next(model.parameters()).device
        raw_td = onnx_cfg.get("trace_device") or "auto"
        if raw_td in ("auto", "", None):
            trace_dev = model_device
        else:
            trace_dev = torch.device(raw_td)

        if model_device.type == "cuda" and trace_dev.type == "cpu":
            self.logger.warning(
                "trace_device=cpu while model is on %s: CUDA spconv implicit_gemm does not support "
                "CPU indices (merge_sort illegal address). Tracing on %s instead. "
                "For large dense() OOM, use a larger GPU or export elsewhere; do not use CPU trace with CUDA spconv.",
                model_device,
                model_device,
            )
            trace_dev = model_device

        moved_for_trace = trace_dev != model_device
        if moved_for_trace:
            self.logger.info(
                "Moving model to %s for ONNX tracing (model was on %s; avoids GPU OOM from sparse dense()).",
                trace_dev,
                model_device,
            )
            model.to(trace_dev)

        orig_sparse_encoder: Optional[nn.Module] = None
        try:
            enc = getattr(model, "pts_middle_encoder", None)
            if enc is not None:
                from deployment.projects.bevfusion.export.sparse_encoder_float_shadow import (
                    build_float_sparse_encoder_shadow,
                    encoder_has_nvidia_tensor_quantizers,
                    resolve_sparse_onnx_shadow,
                )

                gm_src, cfg_ov = resolve_sparse_onnx_shadow(enc, model)
                if gm_src is not None:
                    gm_cls = getattr(torch.fx, "GraphModule", None)
                    nvidia_shadow = gm_src is enc and encoder_has_nvidia_tensor_quantizers(enc)
                    if nvidia_shadow:
                        self.logger.info(
                            "Sparse tower (NVIDIA TensorQuantizer path, scheme A): using a fused FP32 "
                            "shadow encoder for torch.onnx.export so sparse ONNX has no Q/DQ around "
                            "ImplicitGemm. PTQ _amax stay in checkpoint for Path B transform. See "
                            "docs/11_int8_pathb_autoware_plugin.md §8-4."
                        )
                    elif gm_cls is not None and isinstance(gm_src, gm_cls):
                        self.logger.info(
                            "Sparse tower uses FX GraphModule (spconv INT8 path): using a fused FP32 "
                            "shadow encoder only for torch.onnx.export (ONNX cannot represent "
                            "aten::_empty_affine_quantized; same idea as Lidar exptool exporting a float "
                            "graph). PyTorch PTQ inference is unchanged after export. See "
                            "docs/5_bevfusion_onnx_trt_spconv_int8.md (bevfusion project) §3 / §十-A."
                        )
                    else:
                        self.logger.info(
                            "Sparse tower: using fused FP32 shadow encoder for ONNX export "
                            "(weights from nested GraphModule)."
                        )
                    if cfg_ov:
                        self.logger.info(
                            "Shadow rebuild merges %d attribute(s) from model.cfg pts_middle_encoder.",
                            len(cfg_ov),
                        )
                    if gm_src is not enc:
                        self.logger.info(
                            "Using nested GraphModule for shadow weights (type(enc)=%s, gm=%s).",
                            type(enc).__name__,
                            type(gm_src).__name__,
                        )
                    orig_sparse_encoder = enc
                    model.pts_middle_encoder = build_float_sparse_encoder_shadow(
                        gm_src,
                        trace_dev,
                        cfg_overrides=cfg_ov if cfg_ov else None,
                    )

            if wrapper == "sparse":
                wrapper_mod: nn.Module = BEVFusionSparseWrapper(model)
            elif wrapper == "main":
                wrapper_mod = BEVFusionMainBodyWrapper(model)
            else:
                raise ValueError(f"Unknown wrapper '{wrapper}' for ONNX export")

            model_inputs = (
                voxels.to(trace_dev),
                coors.to(device=trace_dev, dtype=torch.int32),
                num_points_per_voxel.to(trace_dev),
            )
            wrapper_mod.eval()
            wrapper_mod.to(trace_dev)

            export_kw: Dict[str, Any] = dict(
                export_params=onnx_cfg["export_params"],
                input_names=onnx_cfg["input_names"],
                output_names=onnx_cfg["output_names"],
                opset_version=onnx_cfg["opset_version"],
                dynamic_axes=onnx_cfg["dynamic_axes"],
                keep_initializers_as_inputs=onnx_cfg["keep_initializers_as_inputs"],
                verbose=onnx_cfg["verbose"],
            )
            try:
                from torch.onnx import TrainingMode

                export_kw["training"] = TrainingMode.EVAL
            except Exception:
                pass

            unsupported_onnx_op = getattr(getattr(torch.onnx, "errors", None), "UnsupportedOperatorError", None)
            with torch.no_grad():
                # Pip spconv dense()/scatter uses list indexing; PyTorch 2.x deprecates x[list] (use x[tuple(list)]).
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*non-tuple sequence for multidimensional indexing.*",
                        category=UserWarning,
                    )
                    try:
                        torch.onnx.export(wrapper_mod, model_inputs, output_path, **export_kw)
                    except Exception as e:
                        if (
                            unsupported_onnx_op is not None
                            and isinstance(e, unsupported_onnx_op)
                            and "_empty_affine_quantized" in str(e)
                        ):
                            raise RuntimeError(
                                "torch.onnx.export failed on quantized tensors (aten::_empty_affine_quantized). "
                                "Expected fix: FX GraphModule under pts_middle_encoder with "
                                "sparse_encoder_float_shadow.SPARSE_ENCODER_SHADOW_ATTRS so the exporter "
                                "swaps in an FP32 BEVFusionSparseEncoder for tracing only. "
                                "If your encoder is wrapped, ensure a child GraphModule exposes those attributes."
                            ) from e
                        raise
        finally:
            if orig_sparse_encoder is not None:
                model.pts_middle_encoder = orig_sparse_encoder
            if moved_for_trace:
                try:
                    model.to(model_device)
                except Exception as e:
                    self.logger.warning(
                        "Could not move model back to %s after ONNX export (GPU may be in error state): %s",
                        model_device,
                        e,
                    )
                if model_device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        self.logger.info("Exported ONNX to %s", output_path)

    def _get_num_proposals(self, model: torch.nn.Module) -> int:
        """Extract num_proposals from the BEVFusion model config."""
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            num_proposals = cfg.get("num_proposals", None)
            if num_proposals is not None:
                return int(num_proposals)

        if hasattr(model, "bbox_head") and hasattr(model.bbox_head, "num_proposals"):
            return int(model.bbox_head.num_proposals)

        raise ValueError(
            "num_proposals not found in model config or bbox_head. "
            "Ensure model_cfg or bbox_head.num_proposals is set."
        )

    def _fix_topk(self, input_path: str, output_path: str, num_proposals: int) -> None:
        """Fix the TopK node in the ONNX graph to use a constant K.

        TensorRT requires TopK's K to be a constant, but torch.onnx.export
        may produce a dynamic K. This replaces it with num_proposals.
        """
        self.logger.info(f"Fixing TopK (K={num_proposals}) in ONNX graph...")
        model = onnx.load(input_path)
        graph = gs.import_onnx(model)

        topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
        if len(topk_nodes) == 0:
            self.logger.warning("No TopK node found; skipping fix")
            onnx.save_model(model, output_path)
            return

        if len(topk_nodes) != 1:
            self.logger.warning(f"Expected 1 TopK node, found {len(topk_nodes)}; fixing the first one")

        topk = topk_nodes[0]
        topk.inputs[1] = gs.Constant("K", values=np.array([num_proposals], dtype=np.int64))
        topk.outputs[0].shape = [1, num_proposals]
        topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
        topk.outputs[1].shape = [1, num_proposals]
        topk.outputs[1].dtype = np.int64

        graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(graph), output_path)

        # Clean up temp file
        if os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)

        self.logger.info(f"TopK fixed. Final ONNX: {output_path}")
