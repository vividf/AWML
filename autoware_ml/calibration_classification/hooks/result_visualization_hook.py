import os

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
    Hook for visualizing the results of calibration classification after each test iteration.

    This hook saves visualizations of the original and undistorted images, along with prediction results, to a specified directory.

    Args:
        data_root (str, optional): Root directory for dataset images. Used to resolve relative image paths.
    """

    def __init__(self, data_root=None, results_vis_dir=None):
        """
        Initialize the ResultVisualizationHook.

        Args:
            data_root (str, optional): Root directory for dataset images.
            results_vis_dir (str, optional): Directory to save results visualization.
        """
        self.transform = CalibrationClassificationTransform(
            debug=False, data_root=data_root, results_vis_dir=results_vis_dir
        )
        self.data_root = data_root
        self.results_vis_dir = results_vis_dir

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Called after each test iteration to visualize and save results.

        Args:
            runner: The runner object handling the test loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        # outputs: list of DataSample, one per batch element
        for i, output in enumerate(outputs):
            pred_label = int(output.pred_label.item())
            img_path = None
            if "images" in output.metainfo and "CAM_FRONT" in output.metainfo["images"]:
                img_path = output.metainfo["images"]["CAM_FRONT"].get("img_path")
            if not img_path:
                print("[ResultVisualizationHook] img_path not found for CAM_FRONT, skipping visualization.")
                continue
            # If img_path is not absolute, prepend data_root if available
            if not os.path.isabs(img_path) and getattr(self.transform, "data_root", None):
                img_path_full = os.path.join(self.transform.data_root, img_path)
            else:
                img_path_full = img_path

            if not os.path.exists(img_path_full):
                print(f"[ResultVisualizationHook] img_path does not exist on disk: {img_path_full}")
                continue
            original_image = cv2.imread(img_path_full)
            if original_image is None:
                print(f"[ResultVisualizationHook] cv2.imread failed to load image: {img_path_full}")
                continue
            cam_info = output.metainfo["images"]["CAM_FRONT"]
            sample_idx = output.metainfo["sample_idx"]
            camera_matrix = np.array(cam_info["cam2img"])
            distortion_coefficients = np.zeros(5, dtype=np.float32)  # Use zeros if not available
            undistorted_image = cv2.undistort(
                original_image,
                camera_matrix,
                distortion_coefficients,
                newCameraMatrix=camera_matrix,
            )
            # Get input_data from the img field in metainfo
            input_data = output.metainfo.get("img", None)
            if input_data is not None:
                img_index = os.path.splitext(os.path.basename(img_path_full))[0]
                self.transform.visualize_results(
                    input_data,
                    pred_label,
                    original_image,
                    undistorted_image,
                    img_index=img_index,
                    sample_idx=sample_idx,
                )
            else:
                print(f"[ResultVisualizationHook] img data not found for {img_path_full}, skipping visualization.")
