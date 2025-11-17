"""
Abstract base class for data loading in deployment.

Each task (classification, detection, segmentation, etc.) must implement
a concrete DataLoader that extends this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict

import torch


class SampleData(TypedDict, total=False):
    """
    Typed representation of a data sample handled by data loaders.

    Attributes:
        input: Raw input data such as images or point clouds.
        ground_truth: Labels or annotations if available.
        metadata: Additional information required for evaluation.
    """

    input: Any
    ground_truth: Any
    metadata: Dict[str, Any]


class BaseDataLoader(ABC):
    """
    Abstract base class for task-specific data loaders.

    This class defines the interface that all task-specific data loaders
    must implement. It handles loading raw data from disk and preprocessing
    it into a format suitable for model inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary containing task-specific settings
        """
        self.config = config

    @abstractmethod
    def load_sample(self, index: int) -> SampleData:
        """
        Load a single sample from the dataset.

        Args:
            index: Sample index to load

        Returns:
            Dictionary containing raw sample data. Structure is task-specific,
            but should typically include:
            - Raw input data (image, point cloud, etc.)
            - Ground truth labels/annotations (if available)
            - Any metadata needed for evaluation

        Raises:
            IndexError: If index is out of range
            FileNotFoundError: If sample data files don't exist
        """
        pass

    @abstractmethod
    def preprocess(self, sample: SampleData) -> torch.Tensor:
        """
        Preprocess raw sample data into model input format.

        Args:
            sample: Raw sample data returned by load_sample()

        Returns:
            Preprocessed tensor ready for model inference.
            Shape and format depend on the specific task.

        Raises:
            ValueError: If sample format is invalid
        """
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        """
        Get total number of samples in the dataset.

        Returns:
            Total number of samples available
        """
        pass
