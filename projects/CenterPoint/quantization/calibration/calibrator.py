# Copyright (c) OpenMMLab. All rights reserved.
"""Calibration manager for PTQ."""

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from pytorch_quantization import calib
    from pytorch_quantization.nn import TensorQuantizer

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False
    TensorQuantizer = None
    calib = None


def _check_pytorch_quantization():
    """Check if pytorch-quantization is available."""
    if not PYTORCH_QUANTIZATION_AVAILABLE:
        raise ImportError(
            "pytorch-quantization is required for quantization support. "
            "Install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
        )


class CalibrationManager:
    """
    Manages PTQ calibration for CenterPoint model.

    This class handles the complete calibration workflow:
    1. Enable calibration mode on all quantizers
    2. Feed calibration data through the model
    3. Compute optimal amax values from collected statistics
    4. Enable quantization mode with computed amax values

    Args:
        model: PyTorch model with quantization modules

    Example:
        >>> model = CenterPoint(...)
        >>> quant_model(model)  # Insert Q/DQ nodes
        >>> calibrator = CalibrationManager(model)
        >>> calibrator.calibrate(dataloader, num_batches=100)
    """

    def __init__(self, model: nn.Module):
        _check_pytorch_quantization()
        self.model = model
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_quantizer_fast(self):
        """
        Enable fast histogram computation using PyTorch.

        This significantly speeds up calibration by using PyTorch's
        native histogram implementation instead of numpy.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if hasattr(module, "_calibrator") and module._calibrator is not None:
                    if isinstance(module._calibrator, calib.HistogramCalibrator):
                        module._calibrator._torch_hist = True

    def _enable_calibration_mode(self):
        """Enable calibration mode on all TensorQuantizers."""
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()  # Disable fake quantization
                    module.enable_calib()  # Enable statistics collection
                else:
                    module.disable()

    def _disable_calibration_mode(self):
        """Disable calibration mode and enable quantization."""
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()  # Enable fake quantization
                    module.disable_calib()  # Disable statistics collection
                else:
                    module.enable()

    def collect_stats(
        self,
        dataloader: Any,
        num_batches: int = 100,
        forward_fn: Optional[Callable] = None,
    ):
        """
        Collect activation statistics for calibration.

        This method feeds calibration data through the model while
        collecting statistics (min, max, histogram) for each quantizer.

        Args:
            dataloader: DataLoader providing calibration samples
            num_batches: Number of batches to use for calibration
            forward_fn: Optional custom forward function. If provided,
                        called as forward_fn(model, batch). Otherwise,
                        uses model.test_step(batch) for MMDet3D models.
        """
        self.model.eval()
        self._enable_calibration_mode()

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Calibrating")
            for i, batch in pbar:
                if i >= num_batches:
                    break

                try:
                    if forward_fn is not None:
                        forward_fn(self.model, batch)
                    elif hasattr(self.model, "test_step"):
                        # MMDet3D / MMEngine model
                        self.model.test_step(batch)
                    elif hasattr(self.model, "forward"):
                        # Standard PyTorch model
                        if isinstance(batch, dict):
                            self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            self.model(*batch)
                        else:
                            self.model(batch)
                    else:
                        raise ValueError("Cannot determine how to call the model")
                except Exception as e:
                    pbar.write(f"Warning: Batch {i} failed with error: {e}")
                    continue

        self._disable_calibration_mode()

    def compute_amax(self, method: str = "mse"):
        """
        Compute amax values from collected statistics.

        The amax value determines the quantization scale. Different methods
        trade off between clipping error and rounding error:
        - "max": Use maximum observed value (no clipping, higher rounding error)
        - "mse": Minimize mean squared error (balanced)
        - "entropy": Minimize KL divergence (preserve distribution)
        - "percentile": Use percentile of distribution (robust to outliers)

        Args:
            method: Method for computing amax. One of:
                    "max", "mse", "entropy", "percentile"
        """
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(method=method)

                    # Move amax to model device
                    if module._amax is not None:
                        module._amax = module._amax.to(self.device)

    def calibrate(
        self,
        dataloader: Any,
        num_batches: int = 100,
        method: str = "mse",
        forward_fn: Optional[Callable] = None,
    ):
        """
        Run full calibration pipeline.

        This is the main entry point for calibration. It:
        1. Enables fast histogram mode
        2. Collects statistics from calibration data
        3. Computes optimal amax values

        Args:
            dataloader: DataLoader providing calibration samples
            num_batches: Number of batches to use for calibration
            method: Method for computing amax ("max", "mse", "entropy", "percentile")
            forward_fn: Optional custom forward function

        Example:
            >>> calibrator = CalibrationManager(model)
            >>> calibrator.calibrate(val_dataloader, num_batches=100, method="mse")
        """
        print(f"Starting calibration with {num_batches} batches, method={method}")

        self.set_quantizer_fast()
        self.collect_stats(dataloader, num_batches, forward_fn)
        self.compute_amax(method)

        # Print summary
        num_quantizers = sum(1 for _, m in self.model.named_modules() if isinstance(m, TensorQuantizer))
        print(f"Calibration complete. {num_quantizers} quantizers calibrated.")

    def save_calib_cache(self, path: str):
        """
        Save calibration cache (amax values) to file.

        Args:
            path: Path to save calibration cache
        """
        calib_cache = {}
        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._amax is not None:
                    calib_cache[name] = module._amax.cpu()

        torch.save(calib_cache, path)
        print(f"Saved calibration cache to {path}")

    def load_calib_cache(self, path: str):
        """
        Load calibration cache (amax values) from file.

        Args:
            path: Path to load calibration cache from
        """
        calib_cache = torch.load(path, map_location=self.device)

        for name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if name in calib_cache:
                    module._amax = calib_cache[name].to(self.device)

        print(f"Loaded calibration cache from {path}")
