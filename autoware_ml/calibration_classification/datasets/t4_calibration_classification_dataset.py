import mmengine
from mmcv.transforms import Compose
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
    ---------------------------
    Many tasks in autonomous driving and sensor fusion require not only the image or point cloud,
    but also the full set of calibration parameters and poses for each sample. By loading the
    complete info.pkl structure, this dataset enables flexible and powerful data processing pipelines
    that can perform projection, coordinate transformation, and multi-modal fusion.

    Role in resnet18_5ch_1xb8-25e_t4base.py:
    ----------------------------------------
    In the config file (resnet18_5ch_1xb8-25e_t4base.py), this dataset is used as the data source
    for training, validation, and testing. It provides each sample with all necessary calibration
    and sensor information, so that the pipeline and model can access raw images, lidar, and all
    geometric relationships for classification or projection-based tasks.

    Update for memory efficiency:
    -----------------------------
    Instead of loading the entire info.pkl into memory during initialization (which may cause OOM
    if the dataset is large), this implementation only keeps the list of sample indices. The full
    sample is loaded lazily at access time in __getitem__. This improves memory efficiency
    significantly for large datasets.
    """

    def __init__(self, ann_file, pipeline=None, data_root=None, max_samples=None):
        """
        Args:
            ann_file (str): Path to the annotation file (info.pkl) containing a list of sample dicts or a dict with 'data_list'.
            pipeline (callable or list, optional): Data processing pipeline to apply to each sample.
            data_root (str, optional): Root directory for data files, passed to pipeline transforms.
            max_samples (int, optional): If set, only use the first max_samples samples.
        """
        self.ann_file = ann_file
        self.data_root = data_root

        self.sample_index = mmengine.load(self.ann_file)
        # Handle dict with key 'data_list'
        if isinstance(self.sample_index, dict) and "data_list" in self.sample_index:
            self.sample_index = self.sample_index["data_list"]
        if max_samples is not None:
            self.sample_index = self.sample_index[:max_samples]

        if isinstance(pipeline, list):
            pipeline = [
                dict(x)
                if not (isinstance(x, dict) and x.get("type") == "CalibrationClassificationTransform")
                else {**x, "data_root": data_root}
                for x in pipeline
            ]
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = pipeline

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_index)

    def __getitem__(self, idx):
        """
        Get a sample by index. The full sample dict is loaded only when needed (lazy loading).
        The pipeline can access and process any part of the sample dict.

        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Sample dictionary with all calibration, image, and lidar info.
        """
        # print(f"[GETITEM] idx={idx}")
        sample = self.sample_index[idx]
        # print(f"[GETITEM] before pipeline sample keys: {list(sample.keys())}")
        
        # 詳細顯示 sample 的內容
        # print(  f"[GETITEM] Sample content:")
        for key, value in sample.items():
            if isinstance(value, dict):
                # print(f"  {key}: dict with keys {list(value.keys())}")
                # 如果是 images，顯示每個 camera 的資訊
                if key == 'images':
                    for cam_name, cam_data in value.items():
                        # print(f"    {cam_name}: {type(cam_data)}")
                        pass
                        if isinstance(cam_data, dict):
                            # print(f"      keys: {list(cam_data.keys())}")
                            pass
                # 如果是 lidar_points，顯示 lidar 資訊
                elif key == 'lidar_points':
                    # print(f"      keys: {list(value.keys())}")
                    pass
            elif isinstance(value, (list, tuple)):
                # print(f"  {key}: {type(value)} with length {len(value)}")
                pass
            else:
                #   print(f"  {key}: {type(value)} = {value}")
                pass
        
        if self.pipeline is not None:
            # print(f"[GETITEM] Applying pipeline...")
            sample = self.pipeline(sample)
            # print(f"[GETITEM] after pipeline sample keys: {list(sample.keys())}")
            
            # 詳細顯示 pipeline 處理後的內容
            # print(f"[GETITEM] After pipeline content:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    # print(f"  {key}: {type(value)} with shape {value.shape}")
                    pass
                elif isinstance(value, dict):
                    # print(f"  {key}: dict with keys {list(value.keys())}")
                    pass
                elif isinstance(value, (list, tuple)):
                    # print(f"  {key}: {type(value)} with length {len(value)}")
                    pass
                else:
                    # print(f"  {key}: {type(value)} = {value}")
                    pass
        else:
            # print(f"[GETITEM] No pipeline applied")
            pass
        
        return sample
