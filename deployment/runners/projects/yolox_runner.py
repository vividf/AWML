"""
YOLOX-specific deployment runner.

This module provides a specialized runner for YOLOX models.
"""

from __future__ import annotations

from typing import Any, Optional

from deployment.core.contexts import ExportContext, YOLOXExportContext
from deployment.runners.common.deployment_runner import BaseDeploymentRunner


class YOLOXOptElanDeploymentRunner(BaseDeploymentRunner):
    """
    YOLOX-specific deployment runner.

    Handles YOLOX-specific requirements:
    - ReLU6 to ReLU replacement for ONNX compatibility
    """

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        context: ExportContext,
    ) -> Any:
        """
        Load YOLOX PyTorch model from checkpoint.

        Performs YOLOX-specific preprocessing:
        - Replaces ReLU6 with ReLU for better ONNX compatibility

        Args:
            checkpoint_path: Path to checkpoint file
            context: Export context. Use YOLOXExportContext for type-safe access
                     to model_cfg_path. Falls back to context.extra for compatibility.

        Returns:
            Loaded PyTorch model
        """
        import torch
        from mmdet.apis import init_detector

        # Extract model_cfg_path from typed context or extra dict
        model_cfg_path: Optional[str] = None
        if isinstance(context, YOLOXExportContext):
            model_cfg_path = context.model_cfg_path
        else:
            raise ValueError("context must be a YOLOXExportContext")

        if model_cfg_path is None:
            # Try to get from model_cfg if it's a file path
            if hasattr(self.model_cfg, "filename"):
                model_cfg_path = self.model_cfg.filename
            else:
                raise ValueError(
                    "model_cfg_path is required for YOLOX model loading. "
                    "Use YOLOXExportContext(model_cfg_path='path/to/config.py') "
                    "or ensure model_cfg has a 'filename' attribute."
                )

        model = init_detector(model_cfg_path, checkpoint_path, device="cpu")
        model.eval()

        # Replace ReLU6 with ReLU for better ONNX compatibility
        def replace_relu6_with_relu(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.ReLU6):
                    setattr(module, name, torch.nn.ReLU(inplace=child.inplace))
                else:
                    replace_relu6_with_relu(child)

        replace_relu6_with_relu(model)

        # Inject model to evaluator via setter (single-direction injection)
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

        return model
