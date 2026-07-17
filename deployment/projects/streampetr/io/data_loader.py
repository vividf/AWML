"""StreamPETR multi-view camera data loader for deployment.

Wraps the model config's ``StreamPETRDataset`` test dataset (which sorts frames by
``(scene_token, timestamp)``, so **index order is clip order** — the order a temporal model
needs) and returns per-frame multi-view images, camera geometry, ego pose, timestamps, and
the ``prev_exists`` sequence-boundary flag alongside the 3D ground truth.

Unlike the shared ``PointCloudDataLoader``, samples here are the raw ``StreamPETRDataset``
dicts (the dataset does not emit the mmdet3d ``inputs``/``data_samples`` structure).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict

import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, init_default_scope
from typing_extensions import override

from deployment.io.base_data_loader import BaseDataLoader, SampleData
from deployment.projects.streampetr.io.model_loader import import_custom_modules

logger = logging.getLogger(__name__)

#: Tensor keys every loaded sample must carry (produced by the dataset's collect_keys).
_REQUIRED_KEYS = (
    "img",
    "intrinsics",
    "lidar2img",
    "timestamp",
    "ego_pose",
    "ego_pose_inv",
    "prev_exists",
)


class StreamPETRDataLoader(BaseDataLoader):
    """DataLoader over the StreamPETR test dataset (clip-ordered temporal frames).

    ``load_sample`` returns a :class:`SampleData` with:

    - ``input``: dict of per-frame tensors (``img``, ``intrinsics``, ``lidar2img``,
      ``timestamp``, ``ego_pose``, ``ego_pose_inv``, ``prev_exists``),
    - ``metadata``: ``img_metas`` plus ``is_sequence_start`` (True when the temporal memory
      queue must be reset),
    - ``ground_truth``: ``gt_bboxes_3d`` (LiDARInstance3DBoxes) and ``gt_labels_3d``.
    """

    def __init__(self, model_cfg: Config, info_file: str = "") -> None:
        """Build the StreamPETR test dataset used for deployment.

        Args:
            model_cfg: MMEngine model config; must have ``test_dataloader.dataset``.
            info_file: Optional eval info file overriding the dataset's ``ann_file``; empty
                keeps the model config's own ``ann_file``.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.info_file = info_file
        self.dataset = self._build_dataset(model_cfg, info_file)

    @staticmethod
    def _build_dataset(model_cfg: Config, info_file: str) -> Any:
        init_default_scope("mmdet3d")
        import_custom_modules()
        if not hasattr(model_cfg, "test_dataloader"):
            raise ValueError("model_cfg must have 'test_dataloader' with dataset config")
        dataset_cfg = copy.deepcopy(model_cfg.test_dataloader.dataset)
        if info_file:
            dataset_cfg["ann_file"] = info_file
        dataset_cfg["test_mode"] = True
        return DATASETS.build(dataset_cfg)

    @override
    def load_sample(self, index: int) -> SampleData:
        if index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.dataset) - 1})")

        data = self.dataset[index]
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"StreamPETR dataset sample missing keys: {missing} (got {sorted(data.keys())})")

        inputs: Dict[str, torch.Tensor] = {key: data[key] for key in _REQUIRED_KEYS}

        # Index 0 is always a sequence start: the dataset computes prev_exists as
        # ``flag[index-1] == flag[index]``, which at index 0 wraps around to the *last*
        # frame's flag and reports True when the set holds a single scene.
        prev_exists = index > 0 and bool(data["prev_exists"].reshape(-1)[0].item())
        img_metas = data.get("img_metas", [{}])
        metadata = {
            "img_metas": img_metas[0] if isinstance(img_metas, list) and img_metas else img_metas,
            "is_sequence_start": not prev_exists,
            "sample_idx": index,
        }

        ground_truth = self._parse_ground_truth(data)

        return SampleData(input=inputs, metadata=metadata, ground_truth=ground_truth)

    @staticmethod
    def _parse_ground_truth(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert per-frame GT into the numpy layout ``Detection3DEvaluator`` consumes.

        The dataset yields ``gt_bboxes_3d`` as a one-element list holding a
        ``LiDARInstance3DBoxes`` (9-dim: x,y,z,l,w,h,yaw,vx,vy — same convention the
        training-time T4 metric consumes) and ``gt_labels_3d`` as a one-element list of
        label tensors.
        """

        def _first(value: Any) -> Any:
            return value[0] if isinstance(value, list) and value else value

        gt_bboxes_3d = _first(data.get("gt_bboxes_3d"))
        gt_labels_3d = _first(data.get("gt_labels_3d"))
        if hasattr(gt_bboxes_3d, "tensor"):  # LiDARInstance3DBoxes -> [N, 9] numpy
            gt_bboxes_3d = gt_bboxes_3d.tensor.cpu().numpy()
        if isinstance(gt_labels_3d, torch.Tensor):
            gt_labels_3d = gt_labels_3d.cpu().numpy()
        return {"gt_bboxes_3d": gt_bboxes_3d, "gt_labels_3d": gt_labels_3d}

    @override
    def preprocess(self, sample: SampleData) -> Dict[str, torch.Tensor]:
        """Return the per-frame input tensors for the inference pipelines."""
        return sample["input"]

    @property
    @override
    def num_samples(self) -> int:
        return len(self.dataset)
