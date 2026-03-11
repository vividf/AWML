"""BEVFusion ONNX Pipeline Implementation."""

from __future__ import annotations

import logging
import os.path as osp
from typing import List

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.configs import ComponentsConfig
from deployment.core.artifacts import resolve_artifact_path
from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

logger = logging.getLogger(__name__)


class BEVFusionONNXPipeline(BEVFusionDeploymentPipeline):
    """ONNXRuntime-based BEVFusion pipeline.

    Loads a single ONNX model that takes voxels/coors/num_points_per_voxel
    and outputs bbox_pred/score/label_pred.
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
        self._load_onnx_model()
        logger.info(f"BEVFusion ONNX pipeline initialized from: {onnx_dir}")

    def _load_onnx_model(self) -> None:
        model_path = resolve_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component_name="bevfusion_main_body",
            file_key="onnx_file",
        )
        if not osp.exists(model_path):
            raise FileNotFoundError(f"BEVFusion ONNX not found: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = self.device.to_ort_provider()

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        logger.info(f"Loaded BEVFusion ONNX: {model_path}")

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(coors, dtype=np.int32)
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
