import os

import cv2
import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from autoware_ml.classification2d.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


@HOOKS.register_module()
class ResultVisualizationHook(Hook):
    def __init__(self, save_dir="./projection_vis_origin/"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.transform = CalibrationClassificationTransform(debug=False)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # outputs: list of DataSample, one per batch element
        # data_batch: dict with input info, e.g., img_path, etc.
        # Adjust keys as needed for your pipeline
        for i, output in enumerate(outputs):
            pred_label = int(output.pred_label.item())
            img_path = output.metainfo["img_path"]
            original_image = cv2.imread(img_path)
            data_dir = os.path.dirname(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0].split("_")[0]
            calibration_data_npz = np.load(os.path.join(data_dir, f"{base_name}_calibration.npz"))
            camera_matrix = calibration_data_npz["camera_matrix"]
            distortion_coefficients = calibration_data_npz["distortion_coefficients"]
            undistorted_image = cv2.undistort(
                original_image,
                camera_matrix,
                distortion_coefficients,
                newCameraMatrix=camera_matrix,
            )
            # Try to get the 5ch input_data from output or data_batch if available
            # If not available, skip visualization for this sample
            input_data = output.metainfo.get("input_data", None)
            if input_data is not None:
                img_index = os.path.splitext(os.path.basename(img_path))[0]
                self.transform.visualize_results(
                    input_data, pred_label, original_image, undistorted_image, img_index=img_index
                )
            else:
                print(f"[ResultVisualizationHook] input_data not found for {img_path}, skipping visualization.")
