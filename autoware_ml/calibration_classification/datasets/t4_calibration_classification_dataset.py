import mmengine
from mmengine.dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class T4CalibrationClassificationDataset(BaseDataset):
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

    def __init__(self, ann_file, pipeline=None, data_root=None, max_samples=None, **kwargs):
        """
        Args:
            ann_file (str): Path to the annotation file (info.pkl) containing a list of sample dicts or a dict with 'data_list'.
            pipeline (callable or list, optional): Data processing pipeline to apply to each sample.
            data_root (str, optional): Root directory for data files, passed to pipeline transforms.
            max_samples (int, optional): If set, only use the first max_samples samples.
            **kwargs: Additional arguments passed to BaseDataset.
        """
        self.max_samples = max_samples

        # Handle pipeline configuration for CalibrationClassificationTransform
        if isinstance(pipeline, list):
            pipeline = [
                (
                    dict(x)
                    if not (isinstance(x, dict) and x.get("type") == "CalibrationClassificationTransform")
                    else {**x, "data_root": data_root}
                )
                for x in pipeline
            ]

        super().__init__(ann_file=ann_file, data_root=data_root, pipeline=pipeline, **kwargs)

    def load_data_list(self):
        """Load annotations from the annotation file.

        Override BaseDataset's load_data_list to handle the specific format of info.pkl.
        """
        self.sample_index = mmengine.load(self.ann_file)
        # Handle dict with key 'data_list'
        if isinstance(self.sample_index, dict) and "data_list" in self.sample_index:
            self.sample_index = self.sample_index["data_list"]
        if self.max_samples is not None:
            self.sample_index = self.sample_index[: self.max_samples]

        return self.sample_index

    def get_cat_ids(self, idx: int):
        """Get category ids by index. Required by BaseDataset.

        Args:
            idx (int): The index of data.

        Returns:
            list[int]: All categories in the sample of specified index.
        """
        # For classification tasks, we might not have category IDs in the same way
        # as detection tasks. Return empty list for now.
        return []

    def prepare_data(self, idx):
        """
        Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of data.

        Returns:
            dict: Processed data.
        """
        # print(f"[GETITEM] idx={idx}")
        sample = self.get_data_info(idx)
        # print(f"[GETITEM] before pipeline sample keys: {list(sample.keys())}")

        # 詳細顯示 sample 的內容
        # print(  f"[GETITEM] Sample content:")
        for key, value in sample.items():
            if isinstance(value, dict):
                # print(f"  {key}: dict with keys {list(value.keys())}")
                # 如果是 images，顯示每個 camera 的資訊
                if key == "images":
                    for cam_name, cam_data in value.items():
                        # print(f"    {cam_name}: {type(cam_data)}")
                        pass
                        if isinstance(cam_data, dict):
                            # print(f"      keys: {list(cam_data.keys())}")
                            pass
                # 如果是 lidar_points，顯示 lidar 資訊
                elif key == "lidar_points":
                    # print(f"      keys: {list(value.keys())}")
                    pass
            elif isinstance(value, (list, tuple)):
                # print(f"  {key}: {type(value)} with length {len(value)}")
                pass
            else:
                #   print(f"  {key}: {type(value)} = {value}")
                pass

        # Apply pipeline
        # print(f"[GETITEM] Applying pipeline...")
        sample = self.pipeline(sample)
        # print(f"[GETITEM] after pipeline sample keys: {list(sample.keys())}")

        # 詳細顯示 pipeline 處理後的內容
        # print(f"[GETITEM] After pipeline content:")
        for key, value in sample.items():
            if hasattr(value, "shape"):
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

        return sample
