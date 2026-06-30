"""BEVFusion ONNX Pipeline Implementation."""

from __future__ import annotations

import logging
import os.path as osp
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion.inference.bevfusion_inference_pipeline import BEVFusionDeploymentPipeline
from deployment.projects.bevfusion.io.component_utils import has_component, is_split_bevfusion_components

logger = logging.getLogger(__name__)


def _normalize_coors_for_legacy_main_body_contract(coors: torch.Tensor) -> torch.Tensor:
    """``[x, y, z]`` from voxelization → ``[z, y, x]`` for legacy ONNX/TRT graph inputs."""
    from deployment.projects.bevfusion.io.coors_contract import voxel_indices_xyz_to_graph_input_zyx

    if coors.ndim == 2 and coors.shape[1] == 3:
        return voxel_indices_xyz_to_graph_input_zyx(coors)
    return coors


class BEVFusionONNXPipeline(BEVFusionDeploymentPipeline):
    """ONNXRuntime-based BEVFusion pipeline.

    Single ONNX: voxels/coors/num_points → bbox_pred/score/label_pred.

    Split ONNX: sparse session → ``lidar_bev``, then dense session → outputs.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        onnx_dir: str,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.ONNX, device=device)

        self.onnx_dir = onnx_dir
        self._components_cfg = components_cfg
        split_layout = is_split_bevfusion_components(components_cfg)
        merged_model_available = False
        if split_layout and has_component(components_cfg, "bevfusion_main_body"):
            merged_path = resolve_artifact_path(
                base_dir=self.onnx_dir,
                components_cfg=self._components_cfg,
                component_name="bevfusion_main_body",
                file_key="onnx_file",
            )
            merged_model_available = osp.exists(merged_path)
        self._split = split_layout and not merged_model_available
        self.session: Optional[ort.InferenceSession] = None
        self._session_sparse: Optional[ort.InferenceSession] = None
        self._session_dense: Optional[ort.InferenceSession] = None
        self._load_onnx_model()
        logger.info(f"BEVFusion ONNX pipeline initialized from: {onnx_dir} (split={self._split})")

    def _load_onnx_model(self) -> None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = self.device.to_ort_provider()

        if self._split:
            sparse_path = resolve_artifact_path(
                base_dir=self.onnx_dir,
                components_cfg=self._components_cfg,
                component_name="bevfusion_sparse",
                file_key="onnx_file",
            )
            dense_path = resolve_artifact_path(
                base_dir=self.onnx_dir,
                components_cfg=self._components_cfg,
                component_name="bevfusion_dense",
                file_key="onnx_file",
            )
            if not osp.exists(sparse_path):
                raise FileNotFoundError(f"Sparse ONNX not found: {sparse_path}")
            if not osp.exists(dense_path):
                raise FileNotFoundError(f"Dense ONNX not found: {dense_path}")
            self._session_sparse = ort.InferenceSession(sparse_path, sess_options=so, providers=providers)
            self._session_dense = ort.InferenceSession(dense_path, sess_options=so, providers=providers)
            logger.info("Loaded split ONNX: %s , %s", sparse_path, dense_path)
            return

        model_path = resolve_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component_name="bevfusion_main_body",
            file_key="onnx_file",
        )
        if not osp.exists(model_path):
            raise FileNotFoundError(f"BEVFusion ONNX not found: {model_path}")

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        logger.info(f"Loaded BEVFusion ONNX: {model_path}")

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        if self._split:
            return self._run_bevfusion_split(voxels, coors, num_points_per_voxel)

        assert self.session is not None
        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(_normalize_coors_for_legacy_main_body_contract(coors), dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)

        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]

        feed_dict = {}
        for name in input_names:
            if "voxel" in name.lower() and "num" not in name.lower():
                feed_dict[name] = voxels_np
            elif "coor" in name.lower():
                feed_dict[name] = coors_np
            elif "num" in name.lower():
                feed_dict[name] = num_points_np

        outputs = self.session.run(output_names, feed_dict)
        return [torch.from_numpy(out).to(self.torch_device) for out in outputs]

    def _run_bevfusion_split(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        assert self._session_sparse is not None and self._session_dense is not None

        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(_normalize_coors_for_legacy_main_body_contract(coors), dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)

        s_in = [inp.name for inp in self._session_sparse.get_inputs()]
        s_out = [out.name for out in self._session_sparse.get_outputs()]

        sparse_feed = {}
        for name in s_in:
            ln = name.lower()
            if "voxel" in ln and "num" not in ln:
                sparse_feed[name] = voxels_np
            elif "coor" in ln:
                sparse_feed[name] = coors_np
            elif "num" in ln:
                sparse_feed[name] = num_points_np

        sparse_ort_outs = self._session_sparse.run(s_out, sparse_feed)
        if len(sparse_ort_outs) != 1:
            raise RuntimeError(f"Expected 1 sparse output, got {len(sparse_ort_outs)}")
        lidar_bev_np = np.ascontiguousarray(sparse_ort_outs[0].astype(np.float32))

        d_in = [inp.name for inp in self._session_dense.get_inputs()]
        d_out = [out.name for out in self._session_dense.get_outputs()]
        dense_feed = {d_in[0]: lidar_bev_np}

        dense_ort_outs = self._session_dense.run(d_out, dense_feed)
        return [torch.from_numpy(out).to(self.torch_device) for out in dense_ort_outs]
