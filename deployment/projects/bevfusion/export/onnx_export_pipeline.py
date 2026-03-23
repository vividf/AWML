"""BEVFusion ONNX export pipeline.

Exports the BEVFusion main_body to a single ONNX file, including the TopK fix.
Replicates the logic from projects/BEVFusion/deploy/ within the new deployment framework.
"""

from __future__ import annotations

import logging
import os
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
from deployment.projects.bevfusion.io.model_loader import setup_quantization_for_onnx_export

logger = logging.getLogger(__name__)


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

        if coors.shape[1] == 3:
            num_points = coors.shape[0]
            batch_coors = torch.zeros(num_points, 1).to(coors.device)
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
    """ONNX export pipeline for BEVFusion (single-file export).

    Exports the full BEVFusion main_body (sparse encoder → backbone → neck → head → postprocess)
    as a single ONNX file and applies the TopK fix for TensorRT compatibility.
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

        onnx_cfg = self._get_onnx_config(config)
        self.logger.info(
            f"ONNX config: opset={onnx_cfg['opset_version']}, inputs={onnx_cfg['input_names']}, outputs={onnx_cfg['output_names']}"
        )

        # Use QuantizeLinear/DequantizeLinear in ONNX (same as CenterPoint). Without this,
        # pytorch_quantization TensorQuantizer exports as primitive ops (Mul, Round, Clip, Div).
        setup_quantization_for_onnx_export()
        self.logger.info("Running torch.onnx.export...")
        self._export_to_onnx(model, voxels, coors, num_points_per_voxel, str(temp_path), onnx_cfg)

        num_proposals = self._get_num_proposals(model)
        self._fix_topk(str(temp_path), str(output_path), num_proposals)

        self.logger.info("=" * 80)
        self.logger.info(f"BEVFusion ONNX export successful: {output_path}")
        self.logger.info("=" * 80)

        return Artifact(path=str(output_dir_path))

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
            coords = coords[:, :]  # [M, 3] (z, y, x)

        return feats, coords, sizes

    def _get_onnx_config(self, config: BaseDeploymentConfig) -> Dict[str, Any]:
        """Build ONNX export configuration from the deployment config."""
        component_cfg = config.components_cfg.get_component("bevfusion_main_body")
        io_cfg = component_cfg.io

        input_names = [inp.name for inp in io_cfg.inputs]
        output_names = [out.name for out in io_cfg.outputs]

        dynamic_axes = {}
        if hasattr(io_cfg, "dynamic_axes") and io_cfg.dynamic_axes:
            dynamic_axes = dict(io_cfg.dynamic_axes)

        onnx_settings = config.onnx_config
        opset_version = getattr(onnx_settings, "opset_version", 17)
        # BEVFusion sparse encoder calls SparseConvTensor.dense() → huge dense grid; small GPUs OOM.
        # Default trace on CPU (host RAM); override with onnx_config trace_device or BEVFUSION_ONNX_TRACE_DEVICE.
        trace_device = onnx_settings.trace_device or os.environ.get("BEVFUSION_ONNX_TRACE_DEVICE", "cpu")

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

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        output_path: str,
        onnx_cfg: Dict[str, Any],
    ) -> None:
        """Export the wrapped model to ONNX."""
        model_device = next(model.parameters()).device
        trace_dev = torch.device(onnx_cfg.get("trace_device", "cpu"))

        moved_for_trace = trace_dev != model_device
        if moved_for_trace:
            self.logger.info(
                "Moving model to %s for ONNX tracing (model was on %s; avoids GPU OOM from sparse dense()).",
                trace_dev,
                model_device,
            )
            model.to(trace_dev)

        try:
            wrapper = BEVFusionMainBodyWrapper(model)
            model_inputs = (
                voxels.to(trace_dev),
                coors.to(trace_dev),
                num_points_per_voxel.to(trace_dev),
            )
            wrapper.eval()
            wrapper.to(trace_dev)

            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    model_inputs,
                    output_path,
                    export_params=onnx_cfg["export_params"],
                    input_names=onnx_cfg["input_names"],
                    output_names=onnx_cfg["output_names"],
                    opset_version=onnx_cfg["opset_version"],
                    dynamic_axes=onnx_cfg["dynamic_axes"],
                    keep_initializers_as_inputs=onnx_cfg["keep_initializers_as_inputs"],
                    verbose=onnx_cfg["verbose"],
                )
        finally:
            if moved_for_trace:
                model.to(model_device)
                if model_device.type == "cuda":
                    torch.cuda.empty_cache()

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
