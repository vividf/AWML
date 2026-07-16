"""YOLOX 2D-detection data loader for deployment.

Runs the model config's mmdet ``test_pipeline`` once per sample (load → resize keep-ratio → pad to
square) to produce the model input tensor, and reads ground-truth boxes in **original image space**
straight from the info file. Predictions are rescaled back to original space in postprocess (via the
pipeline's ``scale_factor``), so ground truth and predictions are compared in the same coordinate
frame — matching how mmdet evaluates 2D detection.

Unlike the old loader, ``scale_factor`` is taken from the pipeline's own output metadata rather than
recomputed, and each sample carries its own metadata (the old code reused sample 0's for every
frame).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from typing_extensions import override

from deployment.io.base_data_loader import BaseDataLoader, SampleData
from deployment.projects.yolox.io.model_loader import import_custom_modules

logger = logging.getLogger(__name__)


class YOLOXDataLoader(BaseDataLoader):
    """DataLoader that runs the mmdet test pipeline once per sample.

    ``load_sample`` returns a :class:`SampleData` with:
    - ``input``: the preprocessed image tensor (CHW, BGR, 0-255) from the pipeline,
    - ``metadata``: ``{scale_factor, input_shape, original_shape}`` for decode/rescale,
    - ``ground_truth``: ``{gt_bboxes, gt_labels}`` in original image space.
    """

    def __init__(self, model_cfg: Config, info_file: str = "") -> None:
        """Build the test pipeline and load the info file.

        Args:
            model_cfg: MMEngine model config; must define ``test_pipeline`` (or
                ``test_dataloader.dataset.pipeline``) and, unless ``info_file`` is given, an
                ``ann_file`` under ``test_dataloader.dataset``.
            info_file: Optional T4Dataset info file (JSON or PKL) overriding the dataset's
                ``ann_file``; empty resolves it from the model config.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.info_file = info_file or self._resolve_info_file(model_cfg)
        self._data_list, self._classes = self._load_info(self.info_file)
        self._pipeline = self._build_pipeline(model_cfg)

    @staticmethod
    def _resolve_info_file(model_cfg: Config) -> str:
        """Resolve the eval info file from ``test_dataloader.dataset`` (ann_file, joined to data_root)."""
        try:
            dataset_cfg = model_cfg.test_dataloader.dataset
        except AttributeError as exc:
            raise ValueError("model_cfg must define 'test_dataloader.dataset' or an explicit info_file.") from exc
        ann_file = dataset_cfg.get("ann_file", "")
        data_root = dataset_cfg.get("data_root", "") or ""
        if ann_file and data_root and not os.path.isabs(ann_file):
            return os.path.join(data_root, ann_file)
        return ann_file

    @staticmethod
    def _load_info(info_file: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load a T4Dataset info file (JSON or PKL) into ``(data_list, class_names)``."""
        if not info_file or not os.path.exists(info_file):
            raise FileNotFoundError(f"YOLOX info file not found: '{info_file}'.")
        if info_file.endswith(".pkl"):
            with open(info_file, "rb") as f:
                info_data = pickle.load(f)
        else:
            with open(info_file, "r") as f:
                info_data = json.load(f)
        if not isinstance(info_data, dict) or "data_list" not in info_data:
            raise ValueError(f"Info file '{info_file}' must be a dict with a 'data_list' key.")
        metainfo = info_data.get("metainfo", {}) or {}
        return info_data["data_list"], list(metainfo.get("classes", []))

    def _build_pipeline(self, model_cfg: Config) -> Compose:
        """Build the mmdet test pipeline as an mmengine ``Compose`` (transforms registered first)."""
        init_default_scope("mmdet")
        import mmdet.datasets.transforms  # noqa: F401 - registers standard mmdet transforms

        import_custom_modules(model_cfg)
        test_pipeline = model_cfg.get("test_pipeline")
        if not test_pipeline:
            test_pipeline = model_cfg.test_dataloader.dataset.pipeline
        return Compose(test_pipeline)

    @override
    def load_sample(self, index: int) -> SampleData:
        if index >= len(self._data_list):
            raise IndexError(f"Sample index {index} out of range (0-{len(self._data_list) - 1}).")

        data_info = self._data_list[index]
        # The pipeline mutates its input dict, so hand it a copy and keep the original for GT.
        results = self._pipeline(copy.deepcopy(data_info))
        if "inputs" not in results:
            raise ValueError(f"Pipeline output missing 'inputs' (MMDet 3.x format). Keys: {list(results.keys())}")

        img_tensor = results["inputs"]
        data_samples = results.get("data_samples")
        metainfo = dict(getattr(data_samples, "metainfo", {})) if data_samples is not None else {}

        scale_factor = tuple(metainfo.get("scale_factor", (1.0, 1.0)))
        original_shape = tuple(metainfo.get("ori_shape", (data_info.get("height", 0), data_info.get("width", 0))))
        input_shape = tuple(int(s) for s in img_tensor.shape[-2:])

        gt_bboxes, gt_labels = self._parse_instances(data_info)

        return SampleData(
            input=img_tensor,
            metadata={
                "scale_factor": [float(s) for s in scale_factor],
                "input_shape": input_shape,
                "original_shape": original_shape,
                "img_id": data_info.get("img_id", index),
            },
            ground_truth={"gt_bboxes": gt_bboxes, "gt_labels": gt_labels},
        )

    @staticmethod
    def _parse_instances(data_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Read non-ignored ``instances`` into original-space ``(gt_bboxes [N,4], gt_labels [N])``."""
        gt_bboxes: List[List[float]] = []
        gt_labels: List[int] = []
        for inst in data_info.get("instances", []):
            if inst.get("ignore_flag", 0):
                continue
            gt_bboxes.append([float(v) for v in inst["bbox"]])
            gt_labels.append(int(inst["bbox_label"]))
        bboxes = np.asarray(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.asarray(gt_labels, dtype=np.int64) if gt_labels else np.zeros((0,), dtype=np.int64)
        return bboxes, labels

    @override
    def preprocess(self, sample: SampleData) -> torch.Tensor:
        """Return the pipeline's image tensor as a ``(1, C, H, W)`` float32 tensor."""
        tensor = sample["input"]
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(tensor.shape)}.")
        return tensor.float()

    @property
    @override
    def num_samples(self) -> int:
        return len(self._data_list)

    @property
    def classes(self) -> List[str]:
        """Class names read from the info file's ``metainfo`` (may be empty)."""
        return self._classes
