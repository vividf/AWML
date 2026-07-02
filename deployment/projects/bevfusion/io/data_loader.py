"""BEVFusion DataLoader for deployment.

Wraps MMDet3D Dataset to load point cloud data for BEVFusion inference.
Pipeline runs once per sample in load_sample(), avoiding redundant computation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, init_default_scope
from typing_extensions import override

from deployment.io.base_data_loader import BaseDataLoader


class BEVFusionDataLoader(BaseDataLoader):
    """Deployment dataloader for BEVFusion using MMDet3D Dataset.

    Wraps the same Dataset used by training/testing, ensuring identical
    GT and pipeline processing.
    """

    def __init__(self, info_file: str, model_cfg: Config) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.info_file = info_file
        self.dataset = self._build_dataset(model_cfg, info_file)

    def _build_dataset(self, model_cfg: Config, info_file: str) -> torch.utils.data.Dataset:
        init_default_scope("mmdet3d")
        if not hasattr(model_cfg, "test_dataloader"):
            raise ValueError("model_cfg must have 'test_dataloader' with dataset config")
        dataset_cfg = copy.deepcopy(model_cfg.test_dataloader.dataset)
        dataset_cfg["ann_file"] = info_file
        dataset_cfg["test_mode"] = True
        return DATASETS.build(dataset_cfg)

    @override
    def load_sample(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, object]]]:
        if index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.dataset)-1})")

        data = self.dataset[index]
        pipeline_inputs = data["inputs"]
        points_tensor = pipeline_inputs["points"].to("cpu")

        data_samples = data["data_samples"]
        metainfo = getattr(data_samples, "metainfo", None)
        eval_ann_info = getattr(data_samples, "eval_ann_info", None)
        ground_truth = dict(eval_ann_info) if eval_ann_info else {}

        return {
            "points": points_tensor,
            "metainfo": dict(metainfo) if metainfo else {},
            "ground_truth": ground_truth,
        }

    @override
    def preprocess(
        self, sample: Dict[str, Union[torch.Tensor, Dict[str, object]]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, object]]]:
        return {
            "points": sample["points"],
            "metainfo": sample["metainfo"],
        }

    @property
    @override
    def num_samples(self) -> int:
        return len(self.dataset)

    @property
    def class_names(self) -> List[str]:
        if hasattr(self.dataset, "metainfo") and "classes" in self.dataset.metainfo:
            return list(self.dataset.metainfo["classes"])
        if hasattr(self.model_cfg, "class_names"):
            return list(self.model_cfg.class_names)
        raise ValueError("class_names not found in dataset.metainfo or model_cfg")
