"""BEVFusion ONNX export pipeline.

Exports the BEVFusion main_body to a single ONNX file, including the TopK fix.
Replicates the logic from projects/BEVFusion/deploy/ within the new deployment framework.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.config.base import BaseDeploymentConfig
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.artifacts import Artifact
from deployment.projects.bevfusion.io.component_utils import (
    has_component,
    is_split_bevfusion_components,
    should_merge_split_bevfusion,
)

logger = logging.getLogger(__name__)


def _normalize_sparse_coors_for_autoware(coors: torch.Tensor) -> torch.Tensor:
    """Normalize sparse coordinates to the legacy Autoware export contract.

    Graph **inputs** must be ``[z, y, x]`` (no batch). This wrapper flips to
    ``[x, y, z]`` and prepends batch — same as ``projects/BEVFusion/deploy/containers.py``.
    Voxelization outputs ``[x, y, z]``; convert with ``voxel_indices_xyz_to_graph_input_zyx``
    before tracing or feeding ONNX/TRT.
    """
    from deployment.projects.bevfusion.io.coors_contract import graph_input_zyx_to_model_indices_xyz

    coors = coors.to(dtype=torch.int32)
    if coors.shape[1] == 3:
        num_points = coors.shape[0]
        coors = graph_input_zyx_to_model_indices_xyz(coors)
        batch_coors = torch.zeros(num_points, 1, dtype=torch.int32, device=coors.device)
        coors = torch.cat([batch_coors, coors], dim=1).contiguous()
    return coors


def _head_dict_to_export_outputs(outputs: dict) -> tuple:
    """Turn the detection-head output dict into the (bbox_pred, score, label) ONNX outputs."""
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
        coors = _normalize_sparse_coors_for_autoware(coors)

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
        # spconv requires int32 indices; float batch column (torch.zeros default) + int coors
        # yields float tensor and can CUDA fault in implicit_gemm. Keep voxels FP32.
        voxels = voxels.to(dtype=torch.float32)
        coors = _normalize_sparse_coors_for_autoware(coors)

        batch_inputs_dict = {
            "voxels": {"voxels": voxels, "coors": coors, "num_points_per_voxel": num_points_per_voxel},
        }

        outputs = self.mod._forward(batch_inputs_dict, using_image_features=True)
        return _head_dict_to_export_outputs(outputs)


class BEVFusionONNXExportPipeline:
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

        self.logger.info("Running torch.onnx.export...")
        self._export_to_onnx(
            model,
            voxels,
            coors,
            num_points_per_voxel,
            str(temp_path),
            onnx_cfg,
            fuse_spconv_bn=bool(config.deploy_cfg.get("fuse_spconv_bn", False)),
        )

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
            model,
            voxels,
            coors,
            num_points_per_voxel,
            str(sparse_onnx),
            onnx_cfg_sparse,
            wrapper="sparse",
            fuse_spconv_bn=bool(config.deploy_cfg.get("fuse_spconv_bn", False)),
        )
        self._postprocess_sparse_onnx_fp(config=config, sparse_onnx_path=sparse_onnx)

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

        if should_merge_split_bevfusion(config.deploy_cfg):
            self._merge_split_onnx_artifact(
                config=config,
                sparse_onnx_path=sparse_onnx,
                dense_onnx_path=dense_onnx,
                output_dir_path=output_dir_path,
            )

        self.logger.info("=" * 80)
        self.logger.info("Split ONNX export OK: %s , %s", sparse_onnx, dense_onnx)
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))

    @staticmethod
    def _deploy_cfg_fuse_implicit_gemm_relu(deploy_cfg: Any, *, default: bool = True) -> bool:
        """Read ``spconv_fuse_implicit_gemm_relu`` (fuse trailing Relu into ImplicitGemm nodes)."""
        val = deploy_cfg.get("spconv_fuse_implicit_gemm_relu", None)
        if val is not None:
            return bool(val)
        return default

    def _postprocess_sparse_onnx_fp(self, *, config: BaseDeploymentConfig, sparse_onnx_path: Path) -> None:
        """Optional FP sparse ONNX postprocess (ImplicitGemm activation fusion)."""
        enable_fuse = self._deploy_cfg_fuse_implicit_gemm_relu(config.deploy_cfg, default=False)
        if not enable_fuse:
            self.logger.info("Sparse ONNX postprocess: ImplicitGemm ReLU fuse disabled by deploy config.")
            return
        if not sparse_onnx_path.exists():
            raise FileNotFoundError(f"Sparse ONNX not found for postprocess: {sparse_onnx_path}")

        from deployment.projects.bevfusion.export.onnx_fuse_implicit_gemm_activation import (
            fuse_autoware_implicit_gemm_trailing_relu,
        )

        model = onnx.load(str(sparse_onnx_path))
        n_relu = fuse_autoware_implicit_gemm_trailing_relu(model)
        onnx.save_model(model, str(sparse_onnx_path))

        self.logger.info(
            "Sparse ONNX postprocess: ImplicitGemm fuse done (trailing Relu=%d): %s",
            n_relu,
            sparse_onnx_path,
        )

    def _merge_split_onnx_artifact(
        self,
        *,
        config: BaseDeploymentConfig,
        sparse_onnx_path: Path,
        dense_onnx_path: Path,
        output_dir_path: Path,
    ) -> None:
        """Merge split sparse+dense ONNX into single main_body ONNX."""
        if not has_component(config.components_cfg, "bevfusion_main_body"):
            raise KeyError(
                "bevfusion_merge is enabled but components_cfg has no 'bevfusion_main_body'. "
                "Ensure merge overlay is applied before export."
            )
        merged_cfg = config.components_cfg.get_component("bevfusion_main_body")
        merged_path = output_dir_path / merged_cfg.onnx_file

        try:
            from onnx import compose as onnx_compose
        except Exception as e:
            raise RuntimeError("ONNX compose utilities unavailable; cannot merge split ONNX.") from e

        if not sparse_onnx_path.exists():
            raise FileNotFoundError(f"Sparse ONNX not found: {sparse_onnx_path}")
        if not dense_onnx_path.exists():
            raise FileNotFoundError(f"Dense ONNX not found: {dense_onnx_path}")

        sparse_model = onnx.load(str(sparse_onnx_path))
        dense_model = onnx.load(str(dense_onnx_path))

        # onnx.compose.merge_models requires identical IR/opset metadata.
        target_ir = max(int(sparse_model.ir_version), int(dense_model.ir_version))
        sparse_model.ir_version = target_ir
        dense_model.ir_version = target_ir

        sparse_opsets = {imp.domain: int(imp.version) for imp in sparse_model.opset_import}
        dense_opsets = {imp.domain: int(imp.version) for imp in dense_model.opset_import}
        merged_opsets = dict(sparse_opsets)
        for domain, version in dense_opsets.items():
            merged_opsets[domain] = max(version, merged_opsets.get(domain, version))
        merged_opset_ids = [onnx.helper.make_operatorsetid(d, v) for d, v in merged_opsets.items()]
        del sparse_model.opset_import[:]
        sparse_model.opset_import.extend(merged_opset_ids)
        del dense_model.opset_import[:]
        dense_model.opset_import.extend(merged_opset_ids)

        sparse_pref = onnx_compose.add_prefix(sparse_model, prefix="sparse/")
        dense_pref = onnx_compose.add_prefix(dense_model, prefix="dense/")

        sparse_out_name = config.components_cfg.get_component("bevfusion_sparse").io.outputs[0].name
        dense_in_name = config.components_cfg.get_component("bevfusion_dense").io.inputs[0].name
        io_map = [(f"sparse/{sparse_out_name}", f"dense/{dense_in_name}")]

        merged_model = onnx_compose.merge_models(sparse_pref, dense_pref, io_map=io_map)
        merged_graph = gs.import_onnx(merged_model)

        sparse_inputs = [inp.name for inp in config.components_cfg.get_component("bevfusion_sparse").io.inputs]
        dense_outputs = [out.name for out in config.components_cfg.get_component("bevfusion_dense").io.outputs]
        if len(merged_graph.inputs) != len(sparse_inputs):
            self.logger.warning(
                "Merged ONNX input count mismatch: graph=%d expected=%d",
                len(merged_graph.inputs),
                len(sparse_inputs),
            )
        if len(merged_graph.outputs) != len(dense_outputs):
            self.logger.warning(
                "Merged ONNX output count mismatch: graph=%d expected=%d",
                len(merged_graph.outputs),
                len(dense_outputs),
            )
        for i, name in enumerate(sparse_inputs):
            if i < len(merged_graph.inputs):
                merged_graph.inputs[i].name = name
        for i, name in enumerate(dense_outputs):
            if i < len(merged_graph.outputs):
                merged_graph.outputs[i].name = name

        merged_graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(merged_graph), str(merged_path))
        self.logger.info("Merged split ONNX -> %s", merged_path)

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
            from deployment.projects.bevfusion.io.coors_contract import voxel_indices_xyz_to_graph_input_zyx

            coords = coords[:, :].to(dtype=torch.int32)  # [M, 3] (x, y, z) from voxel layer
            coords = voxel_indices_xyz_to_graph_input_zyx(coords)  # ONNX graph input: [z, y, x]

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

        onnx_settings = config.deploy_cfg.get("onnx_config", {}) or {}
        opset_version = getattr(onnx_settings, "opset_version", 17)
        # Default "auto" = trace on the same device as the model (usually CUDA). CUDA-built spconv
        # implicit_gemm runs GPU kernels: indices/features on CPU + those kernels => cudaErrorIllegalAddress.
        # If you lack GPU memory for dense(), set trace_device=cpu only with a CPU spconv build, or export
        # on another machine.
        trace_device = getattr(onnx_settings, "trace_device", None) or os.environ.get(
            "BEVFUSION_ONNX_TRACE_DEVICE", "auto"
        )

        return {
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "opset_version": opset_version,
            "do_constant_folding": bool(getattr(onnx_settings, "do_constant_folding", True)),
            "export_params": True,
            "keep_initializers_as_inputs": False,
            "verbose": False,
            "trace_device": trace_device,
        }

    def _torch_onnx_export_module(
        self,
        module: nn.Module,
        model_inputs: tuple,
        output_path: str,
        onnx_cfg: Dict[str, Any],
    ) -> None:
        """Run ``torch.onnx.export`` with deploy ``onnx_config`` (incl. ``do_constant_folding``)."""
        export_kw: Dict[str, Any] = dict(
            export_params=onnx_cfg["export_params"],
            input_names=onnx_cfg["input_names"],
            output_names=onnx_cfg["output_names"],
            opset_version=onnx_cfg["opset_version"],
            dynamic_axes=onnx_cfg["dynamic_axes"],
            keep_initializers_as_inputs=onnx_cfg["keep_initializers_as_inputs"],
            verbose=onnx_cfg["verbose"],
            do_constant_folding=bool(onnx_cfg.get("do_constant_folding", True)),
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
                torch.onnx.export(module, model_inputs, output_path, **export_kw)

    def _resolve_trace_device(
        self,
        model_device: torch.device,
        onnx_cfg: Dict[str, Any],
        *,
        warn_on_cpu: bool = False,
    ) -> torch.device:
        """Resolve the tracing device from ``onnx_cfg``, coercing CPU back to the model's GPU.

        CUDA-built spconv implicit_gemm cannot run with CPU indices (merge_sort illegal
        address), so a requested ``cpu`` trace device is overridden to the model's CUDA device.
        """
        raw_td = onnx_cfg.get("trace_device") or "auto"
        trace_dev = model_device if raw_td in ("auto", "", None) else torch.device(raw_td)

        if model_device.type == "cuda" and trace_dev.type == "cpu":
            if warn_on_cpu:
                self.logger.warning(
                    "trace_device=cpu while model is on %s: CUDA spconv implicit_gemm does not support "
                    "CPU indices (merge_sort illegal address). Tracing on %s instead. "
                    "For large dense() OOM, use a larger GPU or export elsewhere; do not use CPU trace with CUDA spconv.",
                    model_device,
                    model_device,
                )
            trace_dev = model_device
        return trace_dev

    @contextlib.contextmanager
    def _model_on_trace_device(
        self,
        model: torch.nn.Module,
        model_device: torch.device,
        trace_dev: torch.device,
    ):
        """Temporarily move ``model`` to ``trace_dev`` for tracing, restoring it afterward."""
        moved = trace_dev != model_device
        if moved:
            self.logger.info(
                "Moving model to %s for ONNX tracing (model was on %s; avoids GPU OOM from sparse dense()).",
                trace_dev,
                model_device,
            )
            model.to(trace_dev)
        try:
            yield
        finally:
            if moved:
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

    def _export_dense_to_onnx(
        self,
        model: torch.nn.Module,
        lidar_bev: torch.Tensor,
        output_path: str,
        onnx_cfg: Dict[str, Any],
    ) -> None:
        """Export pts_backbone + neck + head (+ postprocess) to ONNX."""
        model_device = next(model.parameters()).device
        trace_dev = self._resolve_trace_device(model_device, onnx_cfg)

        with self._model_on_trace_device(model, model_device, trace_dev):
            wrapper = BEVFusionDenseWrapper(model)
            wrapper.eval()
            wrapper.to(trace_dev)
            bev = lidar_bev.to(trace_dev)
            self._torch_onnx_export_module(wrapper, (bev,), output_path, onnx_cfg)

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
        fuse_spconv_bn: bool = False,
    ) -> None:
        """Export voxel-based subgraph to ONNX (full main_body or sparse-only)."""
        model_device = next(model.parameters()).device
        trace_dev = self._resolve_trace_device(model_device, onnx_cfg, warn_on_cpu=True)

        with self._model_on_trace_device(model, model_device, trace_dev):
            orig_sparse_encoder: Optional[nn.Module] = None
            try:
                orig_sparse_encoder = self._maybe_swap_in_float_shadow_encoder(
                    model, trace_dev, fuse_spconv_bn=fuse_spconv_bn
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

                self._torch_onnx_export_module(wrapper_mod, model_inputs, output_path, onnx_cfg)
            finally:
                if orig_sparse_encoder is not None:
                    model.pts_middle_encoder = orig_sparse_encoder

        self.logger.info("Exported ONNX to %s", output_path)

    def _maybe_swap_in_float_shadow_encoder(
        self,
        model: torch.nn.Module,
        trace_dev: torch.device,
        *,
        fuse_spconv_bn: bool,
    ) -> Optional[nn.Module]:
        """Swap ``pts_middle_encoder`` for a fused FP32 shadow used only during tracing.

        Returns the original encoder (to restore after export) or ``None`` if no swap was
        needed. The shadow lets BN be folded (``fuse_spconv_bn``) into a clean sparse ONNX
        graph without mutating the runtime model.
        """
        enc = getattr(model, "pts_middle_encoder", None)
        if enc is None:
            return None

        from deployment.projects.bevfusion.export.sparse_encoder_float_shadow import (
            build_float_sparse_encoder_shadow,
            resolve_sparse_onnx_shadow,
        )

        gm_src, cfg_ov = resolve_sparse_onnx_shadow(enc, model)
        if gm_src is None:
            return None

        self.logger.info(
            "Sparse tower: using fused FP32 shadow encoder for ONNX export "
            "(weights copied from source sparse encoder)."
        )
        if cfg_ov:
            self.logger.info(
                "Shadow rebuild merges %d attribute(s) from model.cfg pts_middle_encoder.",
                len(cfg_ov),
            )

        model.pts_middle_encoder = build_float_sparse_encoder_shadow(
            gm_src,
            trace_dev,
            cfg_overrides=cfg_ov if cfg_ov else None,
            fuse_spconv_bn=bool(fuse_spconv_bn),
        )
        return enc

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
