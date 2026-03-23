"""BEVFusion PyTorch Pipeline Implementation with per-block latency."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from typing_extensions import override

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

logger = logging.getLogger(__name__)


class BEVFusionPyTorchPipeline(BEVFusionDeploymentPipeline):
    """PyTorch-based BEVFusion pipeline with per-block latency breakdown.

    Runs the full model natively, structured to match the ONNX/TensorRT
    staged inference for output consistency. Reports latency for each block:
    - voxel_encoder_ms: voxel mean reduction
    - sparse_encoder_ms: pts_middle_encoder (spconv)
    - backbone_ms: pts_backbone (SECOND)
    - neck_ms: pts_neck (SECONDFPN)
    - head_ms: bbox_head + postprocess scoring
    """

    def __init__(self, pytorch_model: torch.nn.Module, device: DeviceSpec) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.PYTORCH, device=device)
        logger.info("BEVFusion PyTorch pipeline initialized (per-block latency enabled)")

    @override
    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run BEVFusion with per-block latency measurement.

        Breaks the model into stages and measures each one independently.
        """
        stage_latencies: Dict[str, float] = {}

        total_start = time.perf_counter()
        outputs = self._run_bevfusion_with_breakdown(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
            stage_latencies,
        )
        stage_latencies["bevfusion_ms"] = (time.perf_counter() - total_start) * 1000

        return outputs, stage_latencies

    def _run_bevfusion_with_breakdown(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        stage_latencies: Dict[str, float],
    ) -> List[torch.Tensor]:
        """Run BEVFusion stage by stage, collecting per-block latencies."""
        model = self.pytorch_model
        model.eval()
        device = self.torch_device

        voxels = voxels.to(device)
        coors = coors.to(device)
        num_points_per_voxel = num_points_per_voxel.to(device)

        with torch.no_grad():
            # --- Stage 1: Voxel Encoder (mean reduction) ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if coors.shape[1] == 3:
                num_points = coors.shape[0]
                batch_coors = torch.zeros(num_points, 1, device=device, dtype=coors.dtype)
                coors = torch.cat([batch_coors, coors], dim=1).contiguous()

            if getattr(model, "voxelize_reduce", True):
                voxel_features = voxels.sum(dim=1, keepdim=False) / num_points_per_voxel.type_as(voxels).view(-1, 1)
            else:
                voxel_features = voxels

            torch.cuda.synchronize()
            stage_latencies["voxel_encoder_ms"] = (time.perf_counter() - t0) * 1000

            # --- Stage 2: Sparse Encoder (pts_middle_encoder / spconv) ---
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size=1)

            torch.cuda.synchronize()
            stage_latencies["sparse_encoder_ms"] = (time.perf_counter() - t1) * 1000

            # --- Stage 3: Backbone (pts_backbone / SECOND) ---
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            backbone_out = spatial_features
            if hasattr(model, "pts_backbone") and model.pts_backbone is not None:
                backbone_out = model.pts_backbone(spatial_features)

            torch.cuda.synchronize()
            stage_latencies["backbone_ms"] = (time.perf_counter() - t2) * 1000

            # --- Stage 4: Neck (pts_neck / SECONDFPN) ---
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            neck_out = backbone_out
            if hasattr(model, "pts_neck") and model.pts_neck is not None:
                neck_out = model.pts_neck(backbone_out)

            torch.cuda.synchronize()
            stage_latencies["neck_ms"] = (time.perf_counter() - t3) * 1000

            # --- Stage 5: Detection Head (bbox_head) ---
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            preds = model.bbox_head(neck_out, [])

            torch.cuda.synchronize()
            stage_latencies["head_ms"] = (time.perf_counter() - t4) * 1000

            # --- Stage 6: Post-scoring ---
            torch.cuda.synchronize()
            t5 = time.perf_counter()

            preds = preds[0][0]

            score = preds["heatmap"].sigmoid()
            one_hot = F.one_hot(preds["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
            score = score * preds["query_heatmap_score"] * one_hot
            score = score[0].max(dim=0)[0]

            bbox_pred = torch.cat(
                [preds["center"][0], preds["height"][0], preds["dim"][0], preds["rot"][0], preds["vel"][0]],
                dim=0,
            )
            label_pred = preds["query_labels"][0]

            torch.cuda.synchronize()
            stage_latencies["post_scoring_ms"] = (time.perf_counter() - t5) * 1000

        return [bbox_pred, score, label_pred]

    @override
    def run_bevfusion(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Fallback: run BEVFusion without per-block breakdown."""
        model = self.pytorch_model
        model.eval()
        device = self.torch_device

        voxels = voxels.to(device)
        coors = coors.to(device)
        num_points_per_voxel = num_points_per_voxel.to(device)

        with torch.no_grad():
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
