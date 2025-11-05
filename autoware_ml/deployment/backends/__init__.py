"""
Legacy backend module - DEPRECATED.

This module previously contained PyTorchBackend, ONNXBackend, and TensorRTBackend classes.
These have been replaced by the new pipeline architecture.

For inference, use the pipeline classes instead:
- autoware_ml.deployment.pipelines.yolox for YOLOX
- autoware_ml.deployment.pipelines.centerpoint for CenterPoint  
- autoware_ml.deployment.pipelines.calibration for CalibrationStatusClassification

Each pipeline module provides:
- {Model}PyTorchPipeline
- {Model}ONNXPipeline
- {Model}TensorRTPipeline
"""

__all__ = []
