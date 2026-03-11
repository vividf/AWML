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
        device = self.torch_device

        voxels = voxels.to(device)
        coors = coors.to(device)
        num_points_per_voxel = num_points_per_voxel.to(device)

        with torch.no_grad():
            # Voxel mean reduction (same as inside ONNX graph)
            feats = voxels.sum(dim=1, keepdim=False) / num_points_per_voxel.float().view(-1, 1)

            # Add batch index (flip z,y,x → x,y,z then prepend batch=0)
            coors_flipped = coors.flip(dims=[-1]).contiguous()
            batch_coors = torch.zeros(coors.shape[0], 1, device=device)
            coors_with_batch = torch.cat([batch_coors, coors_flipped], dim=1).contiguous()

            # Sparse 3D encoder → BEV feature
            x = model.pts_middle_encoder(feats, coors_with_batch, batch_size=1)

            # Backbone + Neck
            if model.pts_backbone is not None:
                x = model.pts_backbone(x)
            if model.pts_neck is not None:
                x = model.pts_neck(x)

            # Head forward
            outputs = model.bbox_head(x, [])

            # Extract first layer, first batch
            preds = outputs[0][0]

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
