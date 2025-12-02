"""CenterPoint-specific exporter workflows and constants."""

from deployment.exporters.centerpoint.onnx_workflow import CenterPointONNXExportWorkflow
from deployment.exporters.centerpoint.tensorrt_workflow import CenterPointTensorRTExportWorkflow
from projects.CenterPoint.deploy.configs.deploy_config import model_io, onnx_config

# Re-export structured configs for direct access
OUTPUT_NAMES = model_io["head_output_names"]

# Component configs
_voxel_cfg = onnx_config["components"]["voxel_encoder"]
_backbone_cfg = onnx_config["components"]["backbone_head"]

VOXEL_ENCODER_NAME = _voxel_cfg["name"]
VOXEL_ENCODER_ONNX = _voxel_cfg["onnx_file"]
VOXEL_ENCODER_ENGINE = _voxel_cfg["engine_file"]

BACKBONE_HEAD_NAME = _backbone_cfg["name"]
BACKBONE_HEAD_ONNX = _backbone_cfg["onnx_file"]
BACKBONE_HEAD_ENGINE = _backbone_cfg["engine_file"]

__all__ = [
    # Workflows
    "CenterPointONNXExportWorkflow",
    "CenterPointTensorRTExportWorkflow",
    # Model architecture constants
    "OUTPUT_NAMES",
    # Export file structure constants
    "VOXEL_ENCODER_NAME",
    "BACKBONE_HEAD_NAME",
    "VOXEL_ENCODER_ONNX",
    "BACKBONE_HEAD_ONNX",
    "VOXEL_ENCODER_ENGINE",
    "BACKBONE_HEAD_ENGINE",
]
