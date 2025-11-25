"""
CalibrationStatusClassification-specific deployment runner.

This module provides a specialized runner for CalibrationStatusClassification models.
"""

from typing import Any, Optional

import torch
from mmpretrain.apis import get_model

from deployment.runners.core.deployment_runner import BaseDeploymentRunner


class CalibrationDeploymentRunner(BaseDeploymentRunner):
    """
    CalibrationStatusClassification-specific deployment runner.

    Handles CalibrationStatusClassification model loading.
    """

    def load_pytorch_model(
        self,
        checkpoint_path: str,
        **kwargs: Any,
    ) -> Any:
        """
        Load CalibrationStatusClassification PyTorch model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device string (e.g., "cpu", "cuda:0"). Defaults to config device.
            **kwargs: Additional arguments

        Returns:
            Loaded PyTorch model
        """
        torch_device = torch.device("cpu")
        model = get_model(self.model_cfg, checkpoint_path, device=torch_device)
        model.eval()

        # Inject model to evaluator via setter (single-direction injection)
        if hasattr(self.evaluator, "set_pytorch_model"):
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

        return model
