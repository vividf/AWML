import os
from typing import List, Optional

import cv2
import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


@HOOKS.register_module()
class ResultVisualizationHook(Hook):
    """
    Hook for visualizing the results of calibration classification during training, validation, and testing.

    This hook saves visualizations of the original and undistorted images, along with prediction results, to a specified directory.

    Args:
        data_root (str, optional): Root directory for dataset images. Used to resolve relative image paths.
        results_vis_dir (str, optional): Directory to save results visualization.
    """

    def __init__(self, data_root: Optional[str] = None, results_vis_dir: Optional[str] = None):
        """
        Initialize the ResultVisualizationHook.

        Args:
            data_root (str, optional): Root directory for dataset images.
            results_vis_dir (str, optional): Directory to save results visualization.
        """
        self.transform = CalibrationClassificationTransform(data_root=data_root, results_vis_dir=results_vis_dir)
        self.data_root = data_root
        self.results_vis_dir = results_vis_dir

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[List] = None):
        """
        Called after each training iteration to visualize and save results.

        Args:
            runner: The runner object handling the training loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        self._process_iteration_outputs(outputs, "train")

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[List] = None):
        """
        Called after each validation iteration to visualize and save results.

        Args:
            runner: The runner object handling the validation loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        self._process_iteration_outputs(outputs, "val")

    def after_test_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[List] = None):
        """
        Called after each test iteration to visualize and save results.

        Args:
            runner: The runner object handling the test loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        self._process_iteration_outputs(outputs, "test")

    def _process_iteration_outputs(self, outputs: Optional[List], phase: str):
        """
        Process outputs from any iteration phase (train/val/test).

        Args:
            outputs (list): List of DataSample objects, one per batch element.
            phase (str): Current phase ('train', 'val', or 'test').
        """
        if outputs is None:
            return

        for output in outputs:
            self._process_single_output(output, phase)

    def _process_single_output(self, output, phase: str):
        """Process a single output for visualization."""
        try:
            # Extract prediction label
            pred_label = self._extract_prediction_label(output)

            # Get image path and validate
            img_path = self._get_image_path(output)
            if not img_path:
                return

            # Load and validate original image
            original_image = self._load_original_image(img_path)
            if original_image is None:
                return

            # Create undistorted image
            undistorted_image = self._create_undistorted_image(output, original_image)

            # Visualize results
            self._visualize_results(output, pred_label, original_image, undistorted_image, img_path, phase)

        except Exception as e:
            print(f"[ResultVisualizationHook] Error processing output in {phase} phase: {e}")

    def _extract_prediction_label(self, output) -> int:
        """Extract prediction label from output."""
        return int(output.pred_label.item())

    def _get_image_path(self, output) -> Optional[str]:
        """Get and validate image path from output metadata."""
        if "images" not in output.metainfo or "CAM_FRONT" not in output.metainfo["images"]:
            print("[ResultVisualizationHook] CAM_FRONT image not found in metainfo, skipping visualization.")
            return None

        img_path = output.metainfo["images"]["CAM_FRONT"].get("img_path")
        if not img_path:
            print("[ResultVisualizationHook] img_path not found for CAM_FRONT, skipping visualization.")
            return None

        return img_path

    def _load_original_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and validate original image from path."""
        # Resolve full image path
        img_path_full = self._resolve_image_path(img_path)

        if not os.path.exists(img_path_full):
            print(f"[ResultVisualizationHook] img_path does not exist on disk: {img_path_full}")
            return None

        original_image = cv2.imread(img_path_full)
        if original_image is None:
            print(f"[ResultVisualizationHook] cv2.imread failed to load image: {img_path_full}")
            return None

        return original_image

    def _resolve_image_path(self, img_path: str) -> str:
        """Resolve relative image path to absolute path."""
        if not os.path.isabs(img_path) and getattr(self.transform, "data_root", None):
            return os.path.join(self.transform.data_root, img_path)
        return img_path

    def _create_undistorted_image(self, output, original_image: np.ndarray) -> np.ndarray:
        """Create undistorted image using camera calibration parameters."""
        cam_info = output.metainfo["images"]["CAM_FRONT"]
        camera_matrix = np.array(cam_info["cam2img"])
        distortion_coefficients = np.zeros(5, dtype=np.float32)  # Use zeros if not available

        return cv2.undistort(
            original_image,
            camera_matrix,
            distortion_coefficients,
            newCameraMatrix=camera_matrix,
        )

    def _visualize_results(
        self,
        output,
        pred_label: int,
        original_image: np.ndarray,
        undistorted_image: np.ndarray,
        img_path: str,
        phase: str,
    ):
        """Visualize and save results."""
        input_data = output.metainfo.get("img", None)
        if input_data is None:
            print(f"[ResultVisualizationHook] img data not found for {img_path}, skipping visualization.")
            return

        # Extract image index and sample index
        img_path_full = self._resolve_image_path(img_path)
        img_index = os.path.splitext(os.path.basename(img_path_full))[0]
        sample_idx = output.metainfo["sample_idx"]

        # Perform visualization with phase information
        self.transform.visualize_results(
            input_data,
            pred_label,
            original_image,
            undistorted_image,
            img_index=img_index,
            sample_idx=sample_idx,
            phase=phase,  # Pass phase information to the transform
        )
