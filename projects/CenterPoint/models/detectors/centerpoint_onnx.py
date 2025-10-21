import os
from typing import Callable, Dict, List, Tuple

import torch
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.registry import MODELS
from mmengine.logging import MMLogger, print_log
from torch import nn


class CenterPointHeadONNX(nn.Module):
    """Head module for centerpoint with BACKBONE, NECK and BBOX_HEAD"""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module):
        super(CenterPointHeadONNX, self).__init__()
        self.backbone: nn.Module = backbone
        self.neck: nn.Module = neck
        self.bbox_head: nn.Module = bbox_head
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointHeadONNX!")

    def forward(self, x: torch.Tensor) -> Tuple[List[Dict[str, torch.Tensor]]]:
        """
        Note:
            torch.onnx.export() doesn't support triple-nested output

        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            tuple[list[dict[str, any]]]:
                (num_classes x [num_detect x {'reg', 'height', 'dim', 'rot', 'vel', 'heatmap'}])
        """
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.bbox_head(x)

        return x


@MODELS.register_module()
class CenterPointONNX(CenterPoint):
    """onnx support impl of mmdet3d.models.detectors.CenterPoint"""

    def __init__(self, point_channels: int = 5, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._point_channels = point_channels
        self._device = device
        # Handle both "cuda:0" and "gpu" device strings
        if self._device.startswith("cuda") or self._device == "gpu":
            self._torch_device = torch.device(self._device if self._device.startswith("cuda") else "cuda:0")
        else:
            self._torch_device = torch.device("cpu")
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointONNX!")

    def _get_random_inputs(self):
        """
        Generate random inputs and preprocess it to feed it to onnx.
        """
        # Input channels - create tensors directly on the target device
        points = [
            torch.rand(1000, self._point_channels, device=self._torch_device),
            # torch.rand(1000, self._point_channels, device=self._torch_device),
        ]
        # We only need lidar pointclouds for CenterPoint.
        return {"points": points, "data_samples": None}

    def _extract_random_features(self):
        assert self.data_preprocessor is not None and hasattr(self.data_preprocessor, "voxelize")

        # Ensure data preprocessor is on the correct device
        if hasattr(self.data_preprocessor, 'to'):
            self.data_preprocessor.to(self._torch_device)

        # Get inputs
        inputs = self._get_random_inputs()
        voxel_dict = self.data_preprocessor.voxelize(points=inputs["points"], data_samples=inputs["data_samples"])
        
        # Ensure all voxel tensors are on the correct device
        for key in ["voxels", "num_points", "coors"]:
            if key in voxel_dict and isinstance(voxel_dict[key], torch.Tensor):
                voxel_dict[key] = voxel_dict[key].to(self._torch_device)
        
        assert self.pts_voxel_encoder is not None and hasattr(self.pts_voxel_encoder, "get_input_features")
        input_features = self.pts_voxel_encoder.get_input_features(
            voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"]
        )
        return input_features, voxel_dict

    def save_onnx(
        self,
        save_dir: str,
        verbose=False,
        onnx_opset_version=13,
    ):
        """Save onnx model
        Args:
            batch_dict (dict[str, any])
            save_dir (str): directory path to save onnx models
            verbose (bool, optional)
            onnx_opset_version (int, optional)
        """
        print_log(f"Running onnx_opset_version: {onnx_opset_version}")
        # Get features
        input_features, voxel_dict = self._extract_random_features()

        # === pts_voxel_encoder ===
        pth_onnx_pve = os.path.join(save_dir, "pts_voxel_encoder.onnx")
        torch.onnx.export(
            self.pts_voxel_encoder,
            (input_features,),
            f=pth_onnx_pve,
            input_names=("input_features",),
            output_names=("pillar_features",),
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
            verbose=verbose,
            opset_version=onnx_opset_version,
        )
        print_log(f"Saved pts_voxel_encoder onnx model: {pth_onnx_pve}")
        voxel_features = self.pts_voxel_encoder(input_features)
        voxel_features = voxel_features.squeeze(1)

        # Note: pts_middle_encoder isn't exported
        coors = voxel_dict["coors"]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x (torch.tensor): (batch_size, num_pillar_features, W, H)

        # === pts_backbone ===
        assert self.pts_bbox_head is not None and hasattr(self.pts_bbox_head, "output_names")
        pts_backbone_neck_head = CenterPointHeadONNX(
            self.pts_backbone,
            self.pts_neck,
            self.pts_bbox_head,
        )
        # pts_backbone_neck_head = torch.jit.script(pts_backbone_neck_head)
        pth_onnx_backbone_neck_head = os.path.join(save_dir, "pts_backbone_neck_head.onnx")

        # TODO(vividf): add option for do constant folding to True
        # True cause the numerical consistency issue but better for deploymnet?
        torch.onnx.export(
            pts_backbone_neck_head,
            (x,),
            f=pth_onnx_backbone_neck_head,
            input_names=("spatial_features",),
            output_names=tuple(self.pts_bbox_head.output_names),
            dynamic_axes={
                name: {0: "batch_size", 2: "H", 3: "W"}
                for name in ["spatial_features"] + self.pts_bbox_head.output_names
            },
            verbose=verbose,
            opset_version=onnx_opset_version,
            do_constant_folding=False,  # Disable constant folding for numerical consistency
        )
        print_log(f"Saved pts_backbone_neck_head onnx model: {pth_onnx_backbone_neck_head}")

    def save_torchscript(
        self,
        save_dir: str,
        verbose: bool = False,
    ):
        """Save torchscript model
        Args:
            batch_dict (dict[str, any])
            save_dir (str): directory path to save onnx models
            verbose (bool, optional)
        """
        # Get features
        input_features, voxel_dict = self._extract_random_features()

        pth_pt_pve = os.path.join(save_dir, "pts_voxel_encoder.pt")
        traced_pts_voxel_encoder = torch.jit.trace(self.pts_voxel_encoder, (input_features,))
        traced_pts_voxel_encoder.save(pth_pt_pve)

        voxel_features = traced_pts_voxel_encoder(input_features)
        voxel_features = voxel_features.squeeze()

        # Note: pts_middle_encoder isn't exported
        coors = voxel_dict["coors"]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        pts_backbone_neck_head = CenterPointHeadONNX(
            self.pts_backbone,
            self.pts_neck,
            self.pts_bbox_head,
        )
        pth_pt_head = os.path.join(save_dir, "pts_backbone_neck_head.pt")
        traced_pts_backbone_neck_head = torch.jit.trace(pts_backbone_neck_head, (x))
        traced_pts_backbone_neck_head.save(pth_pt_head)

    # TODO(vividf): this can be removed after the numerical consistency issue is resolved
    def save_onnx_with_intermediate_outputs(self, save_dir: str, onnx_opset_version: int = 13, verbose: bool = False):
        """Export CenterPoint model to ONNX format with intermediate outputs for debugging."""
        import os
        import torch.onnx
        
        print_log(f"Running onnx_opset_version: {onnx_opset_version}")
        print_log("Exporting with intermediate outputs for debugging...")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get features
        input_features, voxel_dict = self._extract_random_features()
        
        # === pts_voxel_encoder ===
        pth_onnx_pve = os.path.join(save_dir, "pts_voxel_encoder.onnx")
        torch.onnx.export(
            self.pts_voxel_encoder,
            (input_features,),
            f=pth_onnx_pve,
            input_names=("input_features",),
            output_names=("pillar_features",),
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
            verbose=verbose,
            opset_version=onnx_opset_version,
        )
        print_log(f"Saved pts_voxel_encoder onnx model: {pth_onnx_pve}")
        voxel_features = self.pts_voxel_encoder(input_features)
        voxel_features = voxel_features.squeeze(1)

        # Note: pts_middle_encoder isn't exported
        coors = voxel_dict["coors"]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        # === Create backbone with intermediate outputs ===
        class BackboneWithIntermediateOutputs(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                outs = []
                for i in range(len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x)
                    outs.append(x)
                return tuple(outs)
        
        backbone_with_outputs = BackboneWithIntermediateOutputs(self.pts_backbone)
        
        # Export backbone with intermediate outputs
        pth_onnx_backbone = os.path.join(save_dir, "pts_backbone_with_intermediate.onnx")
        torch.onnx.export(
            backbone_with_outputs,
            (x,),
            f=pth_onnx_backbone,
            input_names=("spatial_features",),
            output_names=("stage_0", "stage_1", "stage_2"),
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "H", 3: "W"},
                "stage_0": {0: "batch_size", 2: "H", 3: "W"},
                "stage_1": {0: "batch_size", 2: "H", 3: "W"},
                "stage_2": {0: "batch_size", 2: "H", 3: "W"},
            },
            verbose=verbose,
            opset_version=onnx_opset_version,
            do_constant_folding=False,  # Disable constant folding for numerical consistency
        )
        print_log(f"Saved pts_backbone with intermediate outputs: {pth_onnx_backbone}")
        
        return save_dir
