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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Note:
            torch.onnx.export() doesn't support triple-nested output

        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            tuple[torch.Tensor, ...]: Direct tensor outputs for ONNX export
        """
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        head_outputs = self.bbox_head(x)
        
        # Extract tensors from head_outputs and return as tuple
        # head_outputs is a tuple of lists, each list contains dicts with head outputs
        if isinstance(head_outputs, (list, tuple)) and len(head_outputs) > 0:
            head_list = head_outputs[0]
            if isinstance(head_list, (list, tuple)) and len(head_list) > 0:
                head_dict = head_list[0]
                if isinstance(head_dict, dict):
                    # Get output names from bbox_head
                    if hasattr(self.bbox_head, 'output_names'):
                        output_names = self.bbox_head.output_names
                    else:
                        # Fallback: get from task_heads
                        output_names = list(self.bbox_head.task_heads[0].heads.keys())
                    
                    # Return tensors in the correct order
                    return tuple(head_dict[name] for name in output_names)
        
        # Fallback: return as is
        return head_outputs


@MODELS.register_module()
class CenterPointONNX(CenterPoint):
    """onnx support impl of mmdet3d.models.detectors.CenterPoint"""

    def __init__(self, point_channels: int = 5, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._point_channels = point_channels
        self._device = device
        self._torch_device = torch.device("cuda:0") if self._device == "gpu" else torch.device("cpu")
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointONNX!")
        
        # Ensure all model components are on the correct device
        self._move_to_device()

    def _move_to_device(self):
        """Move all model components to the correct device."""
        if hasattr(self, 'data_preprocessor') and self.data_preprocessor is not None:
            self.data_preprocessor = self.data_preprocessor.to(self._torch_device)
        if hasattr(self, 'pts_voxel_encoder') and self.pts_voxel_encoder is not None:
            self.pts_voxel_encoder = self.pts_voxel_encoder.to(self._torch_device)
        if hasattr(self, 'pts_middle_encoder') and self.pts_middle_encoder is not None:
            self.pts_middle_encoder = self.pts_middle_encoder.to(self._torch_device)
        if hasattr(self, 'pts_backbone') and self.pts_backbone is not None:
            self.pts_backbone = self.pts_backbone.to(self._torch_device)
        if hasattr(self, 'pts_neck') and self.pts_neck is not None:
            self.pts_neck = self.pts_neck.to(self._torch_device)
        if hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None:
            self.pts_bbox_head = self.pts_bbox_head.to(self._torch_device)
        
        # Also move the entire model to device
        self.to(self._torch_device)

    def _get_random_inputs(self):
        """
        Generate random inputs and preprocess it to feed it to onnx.
        """
        # Input channels
        points = [
            torch.rand(1000, self._point_channels).to(self._torch_device),
            # torch.rand(1000, self._point_channels).to(self._torch_device),
        ]
        # We only need lidar pointclouds for CenterPoint.
        return {"points": points, "data_samples": None}

    def _extract_random_features(self):
        assert self.data_preprocessor is not None and hasattr(self.data_preprocessor, "voxelize")

        # Get inputs
        inputs = self._get_random_inputs()
        voxel_dict = self.data_preprocessor.voxelize(points=inputs["points"], data_samples=inputs["data_samples"])
        assert self.pts_voxel_encoder is not None and hasattr(self.pts_voxel_encoder, "get_input_features")
        input_features = self.pts_voxel_encoder.get_input_features(
            voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"]
        )
        
        # Ensure all tensors are on the same device
        input_features = input_features.to(self._torch_device)
        voxel_dict = {k: v.to(self._torch_device) if isinstance(v, torch.Tensor) else v 
                     for k, v in voxel_dict.items()}
        
        return input_features, voxel_dict

    def _extract_real_features(self, data_loader, sample_index=0):
        """Extract features using real data from data loader - matching PyTorchBackend flow."""
        # Use the specified sample from data loader
        sample = data_loader.load_sample(sample_index)
        input_data = data_loader.preprocess(sample)
        
        # Extract points from input data
        if isinstance(input_data, dict) and 'points' in input_data:
            points = input_data['points']
            if isinstance(points, list):
                points = points[0]  # Take first point cloud
        else:
            raise ValueError("Input data must contain 'points' key")
        
        # Ensure points are on the correct device
        points = points.to(self._torch_device)
        
        # Use the same flow as PyTorchBackend
        from mmdet3d.structures import Det3DDataSample
        data_samples = [Det3DDataSample()]
        
        # Use model's data_preprocessor (same as PyTorchBackend)
        batch_inputs = self.data_preprocessor(
            {'inputs': {'points': [points]}, 'data_samples': data_samples}
        )
        
        # Extract voxel_dict from inputs (same as PyTorchBackend)
        if 'voxels' in batch_inputs['inputs']:
            voxel_dict = batch_inputs['inputs']['voxels']
            
            # Get input features (same as PyTorchBackend)
            input_features = self.pts_voxel_encoder.get_input_features(
                voxel_dict['voxels'], 
                voxel_dict['num_points'], 
                voxel_dict['coors']
            )
            
            # Ensure all tensors are on the same device
            input_features = input_features.to(self._torch_device)
            voxel_dict = {k: v.to(self._torch_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in voxel_dict.items()}
            
            return input_features, voxel_dict
        else:
            raise ValueError("No voxels found in batch_inputs")

    def save_onnx(
        self,
        save_dir: str,
        verbose=False,
        onnx_opset_version=13,
        data_loader=None,
        sample_index=0,
    ):
        """Save onnx model
        Args:
            save_dir (str): directory path to save onnx models
            verbose (bool, optional)
            onnx_opset_version (int, optional)
            data_loader: Data loader for real data (optional)
            sample_index: Index of sample to use for export (default: 0)
        """
        print_log(f"Running onnx_opset_version: {onnx_opset_version}")
        
        # Ensure all model components are on the correct device before export
        self._move_to_device()
        
        # Set model to eval mode for consistent inference
        self.eval()
        
        # Get features - use real data if available, otherwise random
        if data_loader is not None:
            input_features, voxel_dict = self._extract_real_features(data_loader, sample_index)
        else:
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
        assert self.pts_bbox_head is not None
        pts_backbone_neck_head = CenterPointHeadONNX(
            self.pts_backbone,
            self.pts_neck,
            self.pts_bbox_head,
        )
        # pts_backbone_neck_head = torch.jit.script(pts_backbone_neck_head)
        pth_onnx_backbone_neck_head = os.path.join(save_dir, "pts_backbone_neck_head.onnx")
        
        # Get output names from bbox_head
        if hasattr(self.pts_bbox_head, 'output_names'):
            output_names = self.pts_bbox_head.output_names
        else:
            # Fallback: get from task_heads
            output_names = list(self.pts_bbox_head.task_heads[0].heads.keys())
        
        torch.onnx.export(
            pts_backbone_neck_head,
            (x,),
            f=pth_onnx_backbone_neck_head,
            input_names=("spatial_features",),
            output_names=tuple(output_names),
            dynamic_axes={
                name: {0: "batch_size", 2: "H", 3: "W"}
                for name in ["spatial_features"] + output_names
            },
            verbose=verbose,
            opset_version=onnx_opset_version,
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
