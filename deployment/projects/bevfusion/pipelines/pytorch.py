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
from deployment.projects.bevfusion.debug.sparse_encoder_hooks import (
    try_register_sparse_encoder_sparse_conv_hooks,
)
from deployment.projects.bevfusion.pipelines.bevfusion_pipeline import BEVFusionDeploymentPipeline

try:
    from projects.BEVFusion.bevfusion.bevfusion import _ensure_float_for_pts_pipeline as _ensure_float_for_pts_impl
except Exception:
    _ensure_float_for_pts_impl = None

logger = logging.getLogger(__name__)


_PYTORCH_TENSOR_LOG_PREFIX = "[BEVFUSION][PyTorch][tensors]"


def _ensure_float_for_pts_pipeline(tensor: torch.Tensor) -> torch.Tensor:
    """Best-effort compatibility wrapper for BEVFusion sparse feature dtype normalization."""
    if _ensure_float_for_pts_impl is not None:
        return _ensure_float_for_pts_impl(tensor)
    return tensor.float() if tensor.dtype != torch.float32 else tensor


def _tensor_stats(t: torch.Tensor, name: str) -> str:
    """Return a compact string with tensor statistics for debugging."""
    t_f = t.float()
    return (
        f"{_PYTORCH_TENSOR_LOG_PREFIX} {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t_f.min().item():.4f} max={t_f.max().item():.4f} "
        f"mean={t_f.mean().item():.4f} std={t_f.std().item():.4f} "
        f"abs_mean={t_f.abs().mean().item():.4f} "
        f"nonzero={t_f.count_nonzero().item()}/{t_f.numel()}"
    )


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

    _debug_frame_count = 0

    def __init__(self, pytorch_model: torch.nn.Module, device: DeviceSpec) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.PYTORCH, device=device)
        try_register_sparse_encoder_sparse_conv_hooks(pytorch_model)
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
                npt = num_points_per_voxel.type_as(voxels).view(-1, 1).clamp(min=1.0)
                voxel_features = voxels.sum(dim=1, keepdim=False) / npt
            else:
                voxel_features = voxels

            torch.cuda.synchronize()
            stage_latencies["voxel_encoder_ms"] = (time.perf_counter() - t0) * 1000

            # --- Stage 2: Sparse Encoder (pts_middle_encoder / spconv) ---
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            _dbg = BEVFusionPyTorchPipeline._debug_frame_count < 2
            BEVFusionPyTorchPipeline._debug_frame_count += 1
            if _dbg:
                print(
                    f"{_PYTORCH_TENSOR_LOG_PREFIX} frame={BEVFusionPyTorchPipeline._debug_frame_count}/2 "
                    f"(native pts_middle_encoder → backbone → neck → head)"
                )
                print(_tensor_stats(voxel_features, "voxel_features_input"))

            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size=1)
            spatial_features = _ensure_float_for_pts_pipeline(spatial_features)

            if _dbg:
                print(_tensor_stats(spatial_features, "sparse_encoder_output"))

            torch.cuda.synchronize()
            stage_latencies["sparse_encoder_ms"] = (time.perf_counter() - t1) * 1000

            # --- Stage 3: Backbone (pts_backbone / SECOND) ---
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            backbone_out = spatial_features
            if hasattr(model, "pts_backbone") and model.pts_backbone is not None:
                backbone_out = model.pts_backbone(_ensure_float_for_pts_pipeline(spatial_features))

            if _dbg:
                if isinstance(backbone_out, (list, tuple)):
                    for bi, bo in enumerate(backbone_out):
                        print(_tensor_stats(bo, f"backbone_out[{bi}]"))
                else:
                    print(_tensor_stats(backbone_out, "backbone_out"))

            torch.cuda.synchronize()
            stage_latencies["backbone_ms"] = (time.perf_counter() - t2) * 1000

            # --- Stage 4: Neck (pts_neck / SECONDFPN) ---
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            neck_out = backbone_out
            if hasattr(model, "pts_neck") and model.pts_neck is not None:
                neck_out = model.pts_neck(backbone_out)

            # Match ``BEVFusion.extract_feat``: head ``bev_pos`` is built for
            # ``grid_size // out_size_factor`` (e.g. 180×180) while SECOND/FPN can
            # yield full voxel BEV (e.g. 1440×1440). Skipping this pools causes
            # ``key`` vs ``key_pos`` length mismatch in the transformer decoder.
            align_fn = getattr(model, "_align_lidar_bev_to_head_grid", None)
            if callable(align_fn):
                neck_out = align_fn(neck_out)

            if _dbg:
                if isinstance(neck_out, (list, tuple)):
                    for ni, no in enumerate(neck_out):
                        print(_tensor_stats(no, f"neck_out[{ni}]"))
                else:
                    print(_tensor_stats(neck_out, "neck_out"))

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

            if _dbg:
                print(_tensor_stats(preds["heatmap"], "head_heatmap_raw"))
                print(_tensor_stats(preds["center"][0], "head_center"))
                print(_tensor_stats(preds["dim"][0], "head_dim"))

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
