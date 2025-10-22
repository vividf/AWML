import os
from typing import Callable, Dict, List, Tuple

import numpy as np
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

    def _get_real_inputs(self, data_loader=None, sample_idx=0):
        """
        Generate real inputs from data loader instead of random inputs.
        This ensures ONNX export uses realistic data distribution.
        """
        if data_loader is None:
            # Fallback to random inputs if no data loader provided
            return self._get_random_inputs()
        
        try:
            # Get real sample from data loader
            sample = data_loader.load_sample(sample_idx)
            
            # Check if sample has lidar_points
            if 'lidar_points' in sample:
                lidar_path = sample['lidar_points'].get('lidar_path')
                if lidar_path and os.path.exists(lidar_path):
                    # Load point cloud from file
                    points = self._load_point_cloud(lidar_path)
                    # Convert to torch tensor
                    points = torch.from_numpy(points).to(self._torch_device)
                    # Convert to list format expected by voxelize
                    points = [points]
                    return {"points": points, "data_samples": None}
                else:
                    self._logger.warning(f"Lidar path not found or file doesn't exist: {lidar_path}")
            else:
                self._logger.warning(f"Sample doesn't contain lidar_points: {sample.keys()}")
            
            # Fallback to random inputs if real data loading fails
            self._logger.warning("Failed to load real data, falling back to random inputs")
            return self._get_random_inputs()
            
        except Exception as e:
            self._logger.warning(f"Failed to load real data, falling back to random inputs: {e}")
            return self._get_random_inputs()
    
    def _load_point_cloud(self, lidar_path: str) -> np.ndarray:
        """
        Load point cloud from file.
        
        Args:
            lidar_path: Path to point cloud file (.bin or .pcd)
            
        Returns:
            Point cloud array (N, 5) where 5 = (x, y, z, intensity, ring_id)
        """
        if lidar_path.endswith(".bin"):
            # Load binary point cloud (KITTI/nuScenes format)
            # T4 dataset has 5 features: x, y, z, intensity, ring_id
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            
            # Don't pad here - let the voxelization process handle feature expansion
            # The voxelization process will add cluster_center (+3) and voxel_center (+3) features
            # So 5 + 3 + 3 = 11 features total
            
        elif lidar_path.endswith(".pcd"):
            # Load PCD format (placeholder - would need pypcd or similar)
            raise NotImplementedError("PCD format loading not implemented yet")
        else:
            raise ValueError(f"Unsupported point cloud format: {lidar_path}")
        
        return points

    def _extract_features(self, data_loader=None, sample_idx=0):
        """
        Extract features using real data if available, otherwise fallback to random data.
        """
        assert self.data_preprocessor is not None and hasattr(self.data_preprocessor, "voxelize")

        # Ensure data preprocessor is on the correct device
        if hasattr(self.data_preprocessor, 'to'):
            self.data_preprocessor.to(self._torch_device)

        # Get inputs (real data if available, otherwise random)
        inputs = self._get_real_inputs(data_loader, sample_idx)
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
        data_loader=None,
        sample_idx=0,
    ):
        """Save onnx model
        Args:
            save_dir (str): directory path to save onnx models
            verbose (bool, optional)
            onnx_opset_version (int, optional)
            data_loader: Optional data loader to use real data for export
            sample_idx: Index of sample to use for export
        """
        print_log(f"Running onnx_opset_version: {onnx_opset_version}")
        # Get features using real data if available
        input_features, voxel_dict = self._extract_features(data_loader, sample_idx)

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
            do_constant_folding=True,  # Disable constant folding for numerical consistency
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
            do_constant_folding=True,
        )
        print_log(f"Saved pts_backbone with intermediate outputs: {pth_onnx_backbone}")
        
        return save_dir
