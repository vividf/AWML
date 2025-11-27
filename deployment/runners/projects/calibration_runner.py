"""
CalibrationStatusClassification-specific deployment runner.

This module provides a specialized runner for CalibrationStatusClassification models.
"""

from __future__ import annotations

from typing import Any

import torch
from mmpretrain.apis import get_model

from deployment.core.contexts import ExportContext
from deployment.runners.common.deployment_runner import BaseDeploymentRunner


class CalibrationDeploymentRunner(BaseDeploymentRunner):
    """
    CalibrationStatusClassification-specific deployment runner.

    Handles CalibrationStatusClassification model loading.
    """

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        context: ExportContext,
    ) -> Any:
        """
        Load CalibrationStatusClassification PyTorch model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            context: Export context (currently unused for calibration models,
                     but included for interface consistency)

        Returns:
            Loaded PyTorch model
        """
        # context is available for future extensions
        _ = context

        torch_device = torch.device("cpu")
        model = get_model(self.model_cfg, checkpoint_path, device=torch_device)
        model.eval()

        # Inject model to evaluator via setter (single-direction injection)
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

        return model
