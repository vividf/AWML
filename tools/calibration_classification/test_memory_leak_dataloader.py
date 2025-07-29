import gc
import os

import psutil
from mmcv.transforms import Compose
from mmpretrain.datasets.transforms import PackInputs
from torch.utils.data import DataLoader

from autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset import (
    T4CalibrationClassificationDataset,
)
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# 準備 pipeline，明確傳入 data_root
DATA_ROOT = "/workspace/data/t4dataset"
pipeline = Compose(
    [
        CalibrationClassificationTransform(mode="train", debug=False, enable_augmentation=False, data_root=DATA_ROOT),
        PackInputs(input_key="img"),
    ]
)

# 準備 dataset
ann_file = "/workspace/data/t4dataset/calibration_info/t4dataset_x2_calib_infos_train.pkl"
data_root = DATA_ROOT

dataset = T4CalibrationClassificationDataset(ann_file=ann_file, pipeline=pipeline, data_root=data_root)

# 準備 DataLoader，加入 collate_fn=lambda x: x
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    collate_fn=lambda x: x,
)


def print_mem():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")


for i, batch in enumerate(loader):
    sample = batch[0]
    del sample
    gc.collect()
    if i % 10 == 0:
        print_mem()
    if i > 1000:
        break
