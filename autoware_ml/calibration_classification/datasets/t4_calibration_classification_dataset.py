import mmengine
from mmpretrain.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class T4CalibrationClassificationDataset(Dataset):
    """
    Dataset for T4 Calibration Classification using the full info.pkl structure.

    Each sample in the annotation file (info.pkl) is a dictionary containing:
        - 'images': dict of camera channels, each with calibration and image info
        - 'lidar_points': dict with lidar path and calibration
        - 'ego2global': 4x4 transformation matrix
        - 'sample_idx', 'token', 'timestamp', etc.

    This class allows downstream transforms and models to access all calibration and sensor data
    for advanced tasks such as projection, sensor fusion, and geometric reasoning.

    Why do we need this class?
    -------------------------
    Many tasks in autonomous driving and sensor fusion require not only the image or point cloud,
    but also the full set of calibration parameters and poses for each sample. By loading the
    complete info.pkl structure, this dataset enables flexible and powerful data processing pipelines
    that can perform projection, coordinate transformation, and multi-modal fusion.

    Role in resnet18_5ch_1xb8-25e_t4base.py:
    -----------------------------------------
    In the config file (resnet18_5ch_1xb8-25e_t4base.py), this dataset is used as the data source
    for training, validation, and testing. It provides each sample with all necessary calibration
    and sensor information, so that the pipeline and model can access raw images, lidar, and all
    geometric relationships for classification or projection-based tasks.
    """

    def __init__(self, ann_file, pipeline=None):
        """
        Args:
            ann_file (str): Path to the annotation file (info.pkl) containing a list of sample dicts.
            pipeline (callable or list, optional): Data processing pipeline to apply to each sample.
        """
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.samples = mmengine.load(self.ann_file)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample by index. The sample contains all calibration and sensor info.
        The pipeline can access and process any part of the sample dict.

        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Sample dictionary with all calibration, image, and lidar info.
        """
        sample = self.samples[idx]
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        return sample
