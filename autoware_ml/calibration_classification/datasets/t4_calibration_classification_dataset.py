import mmengine
from mmengine.dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class T4CalibrationClassificationDataset(BaseDataset):
    """Dataset for T4 Calibration Classification using the info.pkl structure.

    This dataset loads calibration classification data from T4 dataset format.
    Each sample contains comprehensive sensor and calibration information including
    camera images, lidar point clouds, and geometric transformations.

    Args:
        ann_file (str): Path to the annotation file (info.pkl) containing sample data.
        pipeline (callable or list, optional): Data processing pipeline to apply to each sample.
        data_root (str, optional): Root directory for data files.
        max_samples (int, optional): If set, only use the first max_samples samples.
        **kwargs: Additional arguments passed to BaseDataset.
    """

    def __init__(self, ann_file, pipeline=None, data_root=None, max_samples=None, **kwargs):
        """Initialize the T4 Calibration Classification Dataset.

        Args:
            ann_file (str): Path to the annotation file (info.pkl) containing a list of sample dicts or a dict with 'data_list'.
            pipeline (callable or list, optional): Data processing pipeline to apply to each sample.
            data_root (str, optional): Root directory for data files, passed to pipeline transforms.
            max_samples (int, optional): If set, only use the first max_samples samples.
            **kwargs: Additional arguments passed to BaseDataset.
        """
        # Handle max_samples by setting indices parameter for BaseDataset
        if max_samples is not None:
            kwargs["indices"] = max_samples

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

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format for T4 dataset.

        T4 dataset has a different structure where image paths are nested under
        'images' dictionary for each camera, rather than at the root level.

        Args:
            raw_data_info (dict): Raw data information from T4 dataset.

        Returns:
            dict: Parsed data information.
        """
        # For T4 dataset, we don't need to process img_path at root level
        # since the structure is different. Just return the raw data as is.
        return raw_data_info

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
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of data.

        Returns:
            dict: Processed data.
        """
        sample = self.get_data_info(idx)
        sample = self.pipeline(sample)
        return sample
