"""BEVFusion Deployment Pipeline Base Class.

Provides common preprocessing, postprocessing, and inference logic
shared by PyTorch, ONNX, and TensorRT backend implementations.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.structures import Det3DDataSample
from typing_extensions import override

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class BEVFusionDeploymentPipeline(BaseDeploymentPipeline):
    """Base pipeline for BEVFusion inference.

    Handles voxelization in preprocessing and bbox decoding in postprocessing.
    The model (ONNX/TensorRT) takes voxels/coors/num_points_per_voxel and
    outputs bbox_pred/score/label_pred directly.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        backend_type: Backend,
        device: DeviceSpec,
    ) -> None:
        cfg = getattr(pytorch_model, "cfg", None)

        class_names = getattr(cfg, "class_names", None)
        point_cloud_range = getattr(cfg, "point_cloud_range", None)
        voxel_size = getattr(cfg, "voxel_size", None)

        if class_names is None:
            raise ValueError("class_names not found in pytorch_model.cfg")

        super().__init__(
            model=pytorch_model,
            backend_type=backend_type,
            device=device,
        )

        self.pytorch_model: torch.nn.Module = pytorch_model
        self.num_classes: int = len(class_names)
        self.class_names: List[str] = class_names
        self.point_cloud_range: Optional[List[float]] = point_cloud_range
        self.voxel_size: Optional[List[float]] = voxel_size

    def to_device_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.torch_device)

    def to_numpy(self, data: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
        arr = data.cpu().numpy().astype(dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    @override
    def preprocess(
        self,
        points: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """Voxelize point cloud into voxels/coors/num_points_per_voxel.

        Uses the BEVFusion model's voxelization layer (outside the ONNX graph).

        Args:
            points: Point cloud tensor [N, point_features].

        Returns:
            Tuple of (preprocessed_dict, metadata_dict).
        """
        points_tensor = self.to_device_tensor(points).float()

        with torch.no_grad():
            feats, coords, sizes = [], [], []
            ret = self.pytorch_model.pts_voxel_layer(points_tensor)
            if len(ret) == 3:
                f, c, n = ret
            else:
                f, c = ret
                n = None
            feats.append(f)
            coords.append(c)
            if n is not None:
                sizes.append(n)

            voxels = torch.cat(feats, dim=0)
            coors = torch.cat(coords, dim=0)
            num_points_per_voxel = (
                torch.cat(sizes, dim=0) if sizes else torch.ones(voxels.shape[0], device=voxels.device)
            )

        preprocessed_dict = {
            "voxels": voxels,
            "coors": coors,
            "num_points_per_voxel": num_points_per_voxel,
        }
        return preprocessed_dict, {}

    @override
    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run the BEVFusion model and return raw outputs with latency.

        Args:
            preprocessed_input: Dict with voxels, coors, num_points_per_voxel.

        Returns:
            Tuple of ([bbox_pred, score, label_pred], stage_latencies).
        """
        stage_latencies: Dict[str, float] = {}

        start = time.perf_counter()
        outputs = self.run_bevfusion(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
        )
        stage_latencies["bevfusion_ms"] = (time.perf_counter() - start) * 1000

        return outputs, stage_latencies

    @override
    def postprocess(
        self,
        model_outputs: List[torch.Tensor],
        sample_meta: Dict[str, object],
    ) -> List[Dict[str, Union[List[float], float, int]]]:
        """Decode bbox_pred/score/label_pred into detection dicts.

        The BEVFusion ONNX model already includes postprocessing (sigmoid, TopK,
        Transformer Decoder), so outputs are:
        - bbox_pred: [10, num_proposals] → (center_x, center_y, height, w, l, h, sin, cos, vx, vy)
        - score: [num_proposals]
        - label_pred: [num_proposals]

        Args:
            model_outputs: [bbox_pred, score, label_pred] tensors.
            sample_meta: Sample metadata.

        Returns:
            List of detection dicts with bbox_3d, score, label.
        """
        bbox_pred, score, label_pred = [self.to_device_tensor(o) for o in model_outputs]

        results: List[Dict[str, Union[List[float], float, int]]] = []

        num_proposals = score.shape[0]
        for i in range(num_proposals):
            s = float(score[i].item())
            if s < 1e-6:
                continue

            bbox = bbox_pred[:, i].cpu().numpy()
            cx, cy = float(bbox[0]), float(bbox[1])
            z = float(bbox[2])
            w, l, h = float(bbox[3]), float(bbox[4]), float(bbox[5])
            sin_yaw, cos_yaw = float(bbox[6]), float(bbox[7])
            yaw = float(np.arctan2(sin_yaw, cos_yaw))
            vx, vy = float(bbox[8]), float(bbox[9])

            results.append(
                {
                    "bbox_3d": [cx, cy, z, w, l, h, yaw, vx, vy],
                    "score": s,
                    "label": int(label_pred[i].item()),
                }
            )

        return results

    @abstractmethod
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the BEVFusion model.

        Args:
            voxels: [M, max_points, C]
            coors: [M, 3] (z, y, x)
            num_points_per_voxel: [M]

        Returns:
            [bbox_pred, score, label_pred]
        """
        raise NotImplementedError
