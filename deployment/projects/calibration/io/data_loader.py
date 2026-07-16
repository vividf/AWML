"""Calibration-classifier data loader for deployment.

The calibration classifier has **synthetic** ground truth: the input is a 5-channel fused image
(BGR + projected-LiDAR depth + intensity), and a sample is labelled *miscalibrated* by perturbing
the LiDAR→camera extrinsic before projection, else *calibrated*. The old deployment evaluated each
base sample twice (once calibrated, once miscalibrated) by overriding the evaluation loop; the new
framework owns that loop, so the two variants are expressed here as **sample indexing** instead:

- ``num_samples`` = ``2 × num_base_samples``,
- even index → calibrated (label 1), odd index → miscalibrated (label 0),
- each index is **seeded** (``seed + index``) so the synthetic perturbation is reproducible — the
  same index yields the same fused image across evaluation and cross-backend verification.

This keeps the evaluator a pure metrics adapter (no ``evaluate`` override) while preserving the old
balanced-evaluation behaviour.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import random
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np
import torch
from mmengine.config import Config
from typing_extensions import override

from deployment.io.base_data_loader import BaseDataLoader, SampleData

if TYPE_CHECKING:
    from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
        CalibrationClassificationTransform,
    )

logger = logging.getLogger(__name__)

# Ground-truth label indices (must match the classifier's training label order and the deploy
# config's class_names): 0 = miscalibrated, 1 = calibrated.
_LABEL_MISCALIBRATED = 0
_LABEL_CALIBRATED = 1


class CalibrationDataLoader(BaseDataLoader):
    """DataLoader that expands each base sample into a calibrated + miscalibrated frame.

    ``load_sample`` returns a :class:`SampleData` with:
    - ``input``: the fused ``(H, W, 5)`` image from the transform,
    - ``ground_truth``: ``{gt_label}`` (0 miscalibrated / 1 calibrated),
    - ``metadata``: ``{variant, base_index}`` (diagnostic only).
    """

    def __init__(
        self,
        model_cfg: Config,
        info_file: str,
        class_names: Sequence[str],
        seed: int = 0,
    ) -> None:
        """Load the info file and build the calibrated/miscalibrated transforms.

        Args:
            model_cfg: MMEngine model config; must define ``data_root`` and ``transform_config``
                (and optionally ``max_depth`` / ``dilation_size``).
            info_file: Path to the T4 calibration info ``.pkl`` (a dict with a ``data_list``).
            class_names: Class names in label-index order (index 0 miscalibrated, 1 calibrated).
            seed: Base seed; sample ``index`` is generated with ``seed + index`` for reproducibility.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.info_file = info_file
        self.class_names = list(class_names)
        self._seed = seed
        self._base_samples = self._load_info(info_file)
        # Two fixed-probability transforms: 0.0 always calibrated, 1.0 always miscalibrated.
        self._transform_calibrated = self._build_transform(0.0)
        self._transform_miscalibrated = self._build_transform(1.0)

    @staticmethod
    def _load_info(info_file: str) -> List[Dict[str, Any]]:
        """Load the calibration info ``.pkl`` into its ``data_list``."""
        if not info_file or not os.path.exists(info_file):
            raise FileNotFoundError(f"Calibration info file not found: '{info_file}'.")
        with open(info_file, "rb") as f:
            info_data = pickle.load(f)
        if not isinstance(info_data, dict) or "data_list" not in info_data:
            raise ValueError(f"Info file '{info_file}' must be a dict with a 'data_list' key.")
        samples = info_data["data_list"]
        if not samples:
            raise ValueError(f"No samples found in '{info_file}'.")
        return samples

    def _build_transform(self, miscalibration_probability: float) -> "CalibrationClassificationTransform":
        """Build a test-mode ``CalibrationClassificationTransform`` at a fixed miscalibration prob.

        The transform is imported lazily: it pulls in mmpretrain / mmcv / cv2 / matplotlib, and a
        missing training dependency must not stop the ``calibration`` project from *registering* on
        the CLI — it should surface only when a run actually builds the loader.
        """
        from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
            CalibrationClassificationTransform,
        )

        data_root = self.model_cfg.get("data_root")
        transform_config = self.model_cfg.get("transform_config")
        if data_root is None or transform_config is None:
            raise ValueError("model_cfg must define 'data_root' and 'transform_config' for calibration.")
        return CalibrationClassificationTransform(
            transform_config=transform_config,
            mode="test",
            max_depth=self.model_cfg.get("max_depth", 128.0),
            dilation_size=self.model_cfg.get("dilation_size", 1),
            undistort=True,
            miscalibration_probability=miscalibration_probability,
            enable_augmentation=False,
            data_root=data_root,
            projection_vis_dir=None,
            results_vis_dir=None,
            binary_save_dir=None,
        )

    @override
    def load_sample(self, index: int) -> SampleData:
        if index >= self.num_samples:
            raise IndexError(f"Sample index {index} out of range (0-{self.num_samples - 1}).")

        base_index, variant = divmod(index, 2)
        is_miscalibrated = variant == 1

        # Seed per-index so the synthetic (randomised) miscalibration is reproducible across the
        # evaluation and verification passes, which both call load_sample(index).
        random.seed(self._seed + index)
        np.random.seed((self._seed + index) % (2**32))

        transform = self._transform_miscalibrated if is_miscalibrated else self._transform_calibrated
        result = transform.transform(copy.deepcopy(self._base_samples[base_index]))
        fused_img = result["fused_img"]  # (H, W, 5) float32, normalized to ~[0, 1]

        gt_label = _LABEL_MISCALIBRATED if is_miscalibrated else _LABEL_CALIBRATED
        return SampleData(
            input=fused_img,
            ground_truth={"gt_label": gt_label},
            metadata={"variant": "miscalibrated" if is_miscalibrated else "calibrated", "base_index": base_index},
        )

    @override
    def preprocess(self, sample: SampleData) -> torch.Tensor:
        """Convert the fused ``(H, W, 5)`` image into a ``(1, 5, H, W)`` float32 tensor."""
        fused_img = sample["input"]
        if isinstance(fused_img, np.ndarray):
            tensor = torch.from_numpy(fused_img)
        else:
            tensor = fused_img
        return tensor.permute(2, 0, 1).float().unsqueeze(0)

    @property
    @override
    def num_samples(self) -> int:
        """Two frames (calibrated + miscalibrated) per base sample."""
        return 2 * len(self._base_samples)
