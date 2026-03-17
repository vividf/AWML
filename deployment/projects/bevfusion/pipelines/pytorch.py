"""BEVFusion PyTorch Pipeline Implementation."""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from typing_extensions import override

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

logger = logging.getLogger(__name__)


class BEVFusionPyTorchPipeline(BEVFusionDeploymentPipeline):
    """PyTorch-based BEVFusion pipeline.

    Runs the full model natively, structured to match the ONNX/TensorRT
    staged inference for output consistency.
    """

    def __init__(self, pytorch_model: torch.nn.Module, device: DeviceSpec) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.PYTORCH, device=device)
        logger.info("BEVFusion PyTorch pipeline initialized")

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run BEVFusion via PyTorch, replicating the ONNX container logic.

        Performs: voxel mean reduction → sparse encoder → backbone → neck → head → postprocess.
        """
        model = self.pytorch_model
        model.eval()
        device = self.torch_device

        voxels = voxels.to(device)
        coors = coors.to(device)
        num_points_per_voxel = num_points_per_voxel.to(device)

        with torch.no_grad():
            # Match the export wrapper path exactly:
            # (voxels, coors, num_points_per_voxel) -> model._forward(...)
            if coors.shape[1] == 3:
                num_points = coors.shape[0]
                batch_coors = torch.zeros(num_points, 1, device=device, dtype=coors.dtype)
                coors = torch.cat([batch_coors, coors], dim=1).contiguous()

            batch_inputs_dict = {
                "voxels": {
                    "voxels": voxels,
                    "coors": coors,
                    "num_points_per_voxel": num_points_per_voxel,
                }
            }
            preds = model._forward(batch_inputs_dict, using_image_features=True)

            # Replicate the TrtBevFusionMainContainer postprocessing
            import torch.nn.functional as F

            score = preds["heatmap"].sigmoid()
            one_hot = F.one_hot(preds["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
            score = score * preds["query_heatmap_score"] * one_hot
            score = score[0].max(dim=0)[0]

            bbox_pred = torch.cat(
                [preds["center"][0], preds["height"][0], preds["dim"][0], preds["rot"][0], preds["vel"][0]],
                dim=0,
            )
            label_pred = preds["query_labels"][0]

        return [bbox_pred, score, label_pred]
