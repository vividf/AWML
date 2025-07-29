import gc

import psutil
from mmcv.transforms import Compose

from autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset import (
    T4CalibrationClassificationDataset,
)
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


# 自訂一個只回傳 inputs 的 transform
class OnlyInputsTransform:
    def __init__(self, data_root=None):
        self.inner = CalibrationClassificationTransform(
            mode="train", debug=False, enable_augmentation=False, data_root=data_root
        )

    def __call__(self, results):
        results = self.inner(results)
        return {"inputs": results["img"]}


DATA_ROOT = "/workspace/data/t4dataset"
pipeline = Compose(
    [
        OnlyInputsTransform(data_root=DATA_ROOT),
    ]
)

ann_file = "/workspace/data/t4dataset/calibration_info/t4dataset_x2_calib_infos_train.pkl"
data_root = DATA_ROOT

dataset = T4CalibrationClassificationDataset(ann_file=ann_file, pipeline=pipeline, data_root=data_root)


def print_mem():
    import os

    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")


for i in range(1000):
    sample = dataset[i]
    del sample
    gc.collect()
    if i % 10 == 0:
        print_mem()
