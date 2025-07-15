import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample

from autoware_ml.classification2d.datasets.transforms.camera_lidar_augmentation import alter_calibration


@TRANSFORMS.register_module()
class CalibrationClassificationTransform(BaseTransform):
    """Transforms image, pointcloud, and calibration data into inputs for a calibration classifier."""

    def __init__(
        self, validation=False, test=False, debug=False, undistort=True, enable_augmentation=False, save_vis_dir=None
    ):
        super().__init__()
        self.validation = validation
        self.test = test
        self.debug = debug
        self.undistort = undistort
        self.enable_augmentation = enable_augmentation
        self.save_vis_dir = save_vis_dir

    def transform(self, results, force_generate_miscalibration=False):
        # Set random seeds for reproducibility during validation
        if self.validation:
            random.seed(results.get("sample_idx", 0))
            np.random.seed(results.get("sample_idx", 0))
        else:
            random.seed(None)
            np.random.seed(None)

        # print("in transform, results:", results)

        # Loading and preparing data
        camera_data, lidar_data, calibration_data = self.load_data(results)

        # Image undistortion if necessary
        if self.undistort:
            undistorted_data, calibration_data = self.undistort_image(camera_data, calibration_data, alpha=0.0)
        else:
            undistorted_data, calibration_data["new_camera_matrix"] = camera_data, calibration_data["camera_matrix"]

        # Label generation
        if self.test and force_generate_miscalibration:
            raise ValueError("force_generate_miscalibration is not supported in test mode")
        if self.test:  # in test case, label is not used but must be registered
            generate_miscalibration = False
        else:
            generate_miscalibration = force_generate_miscalibration or random.choice([True, False])  # 50/50 split

        if generate_miscalibration:
            calibration_data["camera_to_lidar_pose"] = alter_calibration(
                calibration_data["camera_to_lidar_pose"],
                min_augmentation_angle=1.0,
                max_augmentation_angle=10.0,
                min_augmentation_radius=0.05,
                max_augmentation_radius=0.2,
            )
            label = 0  # Miscalibrated
        else:
            label = 1  # Correctly calibrated

        # Data augmentation: cropping and affine transformations
        if self.enable_augmentation and not self.validation:
            augmented_image, augmented_calibration, augmentation_tf = self.apply_augmentations(
                undistorted_data, calibration_data
            )
        else:
            augmented_image, augmented_calibration, augmentation_tf = undistorted_data, calibration_data, None

        # Generate LiDAR image data
        input_data = self.generate_input_data(
            augmented_image, lidar_data, augmented_calibration, augmentation_tf=augmentation_tf
        )

        # Visualize the results
        if self.save_vis_dir is not None:
            # Create overlay image
            camera_data_vis = input_data[:, :, :3]
            intensity_image = input_data[:, :, 4:5]
            overlay_image = self.create_overlay_image(camera_data_vis, intensity_image)

            os.makedirs(self.save_vis_dir, exist_ok=True)
            img_path = results.get("img_path", None)
            if img_path is not None:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(self.save_vis_dir, f"{base_name}_overlay.jpg")
                cv2.imwrite(out_path, overlay_image)

        # Final results
        results["img"] = input_data
        results["gt_label"] = label

        print("label: ", label)
        results["data_samples"] = DataSample().set_gt_label(label)
        return results

    def load_data(self, results):
        """Loads camera, LiDAR, and calibration data from t4dataset sample dict."""
        img_path = results["img_path"]
        pointcloud_path = results["pointcloud_path"]
        calibration = results["calibration"]

        # 讀取影像
        camera_data = cv2.imread(img_path)

        pc_raw = np.fromfile(pointcloud_path, dtype=np.float32)
        n = 5
        if pc_raw.size % n != 0:
            raise ValueError(f"Pointcloud size {pc_raw.size} is not divisible by {n}")
        print(f"[INFO] {pointcloud_path} has {n} fields per point, total points: {pc_raw.size // n}")
        pc = pc_raw.reshape(-1, n)

        lidar_data = {
            "pointcloud": pc[:, :3],
            "intensities": self.normalize_intensity(pc[:, 3]),
        }

        # Use new calibration structure
        calibration_data = {
            "camera_matrix": calibration["camera_matrix"],
            "distortion_coefficients": calibration["distortion_coefficients"],
            "camera": calibration["camera"],
            "lidar": calibration["lidar"],
            "new_camera_matrix": calibration["camera_matrix"],
        }
        return camera_data, lidar_data, calibration_data

    def normalize_intensity(self, intensities):
        """Normalizes LiDAR intensity values to [0, 1]."""
        min_intensity = intensities.min()
        max_intensity = intensities.max()
        if min_intensity == max_intensity:
            return np.zeros_like(intensities)
        return (intensities - min_intensity) / (max_intensity - min_intensity)

    def undistort_image(self, image, calibration_data, alpha=0.0):
        """Undistorts the image and updates the calibration data."""
        if np.any(calibration_data["distortion_coefficients"]):
            distortion_coefficients = calibration_data["distortion_coefficients"]
            h, w = image.shape[:2]
            calibration_data["new_camera_matrix"], _ = cv2.getOptimalNewCameraMatrix(
                calibration_data["camera_matrix"], distortion_coefficients, (w, h), alpha, (w, h)
            )
            image = cv2.undistort(
                image,
                calibration_data["camera_matrix"],
                distortion_coefficients,
                newCameraMatrix=calibration_data["new_camera_matrix"],
            )
            calibration_data["distortion_coefficients"] = np.zeros_like(calibration_data["distortion_coefficients"])
        return image, calibration_data

    def signed_random(self, min_value, max_value):
        sg = 1 if random.random() < 0.5 else -1
        return sg * random.uniform(min_value, max_value)

    def scale_and_crop_image(self, image, calibration_data, crop_ratio=0.6):
        """Scales and crops the image, updating the camera matrix accordingly"""
        h, w = image.shape[:2]

        # Random noise for crop center offsets
        crop_center_noise = [self.signed_random(0, crop_ratio / 2), self.signed_random(0, crop_ratio / 2)]
        crop_center = np.array([h * (1 + crop_center_noise[0]) / 2, w * (1 + crop_center_noise[1]) / 2])

        # Determine scaled dimensions
        scale_noise = np.random.uniform(crop_ratio, 1 - np.max(np.abs(crop_center_noise)))
        scaled_h, scaled_w = h * scale_noise, w * scale_noise

        # Determine crop region from the center
        start_h, end_h = int(crop_center[0] - scaled_h / 2), int(crop_center[0] + scaled_h / 2)
        start_w, end_w = int(crop_center[1] - scaled_w / 2), int(crop_center[1] + scaled_w / 2)

        # Enforce bounds
        start_h, end_h = max(0, start_h), min(h, end_h)
        start_w, end_w = max(0, start_w), min(w, end_w)

        # Crop and resize
        cropped_image = image[start_h:end_h, start_w:end_w]
        resized_image = cv2.resize(cropped_image, (w, h))

        # Update the undistorted calibration matrix
        scale_factor = w / (end_w - start_w)
        calibration_data["new_camera_matrix"][0, 0] *= scale_factor  # fx
        calibration_data["new_camera_matrix"][1, 1] *= scale_factor  # fy
        crop_offset_w = start_w
        crop_offset_h = start_h
        calibration_data["new_camera_matrix"][0, 2] = (
            calibration_data["new_camera_matrix"][0, 2] - crop_offset_w
        ) * scale_factor
        calibration_data["new_camera_matrix"][1, 2] = (
            calibration_data["new_camera_matrix"][1, 2] - crop_offset_h
        ) * scale_factor

        return resized_image, calibration_data

    def apply_affine_transformation(self, image, calibration_data, max_distortion=0.001):
        """
        Applies a controlled affine transformation to the image and updates the calibration matrix.

        Args:
            image (np.ndarray): Input image.
            calibration_data (dict): Camera calibration data.
            max_distortion (float): Maximum allowable distortion as a fraction of image dimensions.

        Returns:
            tuple: Transformed image, updated calibration data, and the 3x3 affine transformation matrix.
        """
        h, w = image.shape[:2]

        # Limit distortions to a fraction of image dimensions
        max_offset_x = max_distortion * w
        max_offset_y = max_distortion * h

        # Source points (corners of the image)
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])

        # Destination points with bounded distortion
        dst_pts = src_pts + np.random.uniform(
            low=[[-max_offset_x, -max_offset_y], [-max_offset_x, -max_offset_y], [-max_offset_x, -max_offset_y]],
            high=[[max_offset_x, max_offset_y], [max_offset_x, max_offset_y], [max_offset_x, max_offset_y]],
        ).astype(np.float32)

        # Compute affine transformation matrix
        affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)

        # Apply affine transformation to the image
        transformed_image = cv2.warpAffine(image, affine_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)

        # Update the calibration matrix
        affine_transform_3x3 = np.eye(3)
        affine_transform_3x3[:2, :3] = affine_matrix

        return transformed_image, calibration_data, affine_transform_3x3

    def apply_augmentations(self, image, calibration_data):
        """Applies cropping, scaling, and affine transformations."""
        # Scaling and cropping
        if random.random() > 0.5:
            image, calibration_data = self.scale_and_crop_image(image, calibration_data, crop_ratio=0.6)

        # Affine transformation
        affine_matrix = None
        if random.random() > 0.5:
            image, calibration_data, affine_matrix = self.apply_affine_transformation(
                image, calibration_data, max_distortion=0.02
            )

        return image, calibration_data, affine_matrix

    def generate_input_data(self, image, lidar_data, calibration_data, augmentation_tf=None):
        """Generates depth and intensity images using augmented calibration."""
        from pyquaternion import Quaternion

        # Get calibration dicts
        lidar_calib = calibration_data["lidar"]
        cam_calib = calibration_data["camera"]
        camera_matrix = calibration_data["new_camera_matrix"]
        distortion_coefficients = calibration_data["distortion_coefficients"][:8]

        # Transform LiDAR points to the camera coordinate system (direct)
        pointcloud = lidar_data["pointcloud"]
        lidar_R = Quaternion(lidar_calib["rotation"]).rotation_matrix
        lidar_t = np.array(lidar_calib["translation"])
        cam_R = Quaternion(cam_calib["rotation"]).rotation_matrix
        cam_t = np.array(cam_calib["translation"])

        # Match the direct_lidar_to_camera logic from visualization script
        points = pointcloud @ lidar_R.T + lidar_t
        points = points - cam_t
        points = points @ cam_R
        pointcloud_ccs = points
        valid_points = pointcloud_ccs[:, 2] > 0.0
        pointcloud_ccs = pointcloud_ccs[valid_points]
        lidar_data["intensities"] = lidar_data["intensities"][valid_points]

        # Project 3D points into the image plane
        pointcloud_ics, _ = cv2.projectPoints(
            pointcloud_ccs, np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients
        )
        pointcloud_ics = pointcloud_ics.reshape(-1, 2)

        # Apply the affine transformation to the 2D lidar image points
        if augmentation_tf is not None:
            num_points = pointcloud_ics.shape[0]
            homogeneous_ics = np.hstack([pointcloud_ics, np.ones((num_points, 1))])
            transformed_ics = (augmentation_tf @ homogeneous_ics.T).T[:, :2]
        else:
            transformed_ics = pointcloud_ics

        # Generate depth and intensity images
        return self.create_lidar_images(image, transformed_ics, pointcloud_ccs, lidar_data["intensities"])

    def create_lidar_images(self, image, pointcloud_ics, pointcloud_ccs, intensities):
        """Creates depth and intensity images."""
        h, w = image.shape[:2]
        depth_image = np.zeros((h, w), dtype=np.uint8)
        intensity_image = np.zeros((h, w), dtype=np.uint8)

        for point3d, intensity, point2d in zip(pointcloud_ccs, intensities, pointcloud_ics):
            if np.any(np.abs(point2d) > (2**31 - 1)):
                continue
            distance_color = int(np.clip(255 * point3d[2] / 80.0, 0, 255))
            cv2.circle(depth_image, tuple(point2d.astype(int)), 3, distance_color, -1)
            cv2.circle(intensity_image, tuple(point2d.astype(int)), 3, int(intensity * 255), -1)

        depth_image = np.expand_dims(depth_image, axis=2)
        intensity_image = np.expand_dims(intensity_image, axis=2)
        return np.concatenate([image, depth_image, intensity_image], axis=2)

    def create_overlay_image(self, rgb_image, feature_image, rgb_weight=0.3):
        """Created colored overlay image"""
        overlay_image = rgb_image.copy()
        intensity_colormap = cv2.applyColorMap((feature_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        intensity_mask = (feature_image > 0).astype(np.uint8).squeeze(-1)  # Remove singleton dimension
        masked_colormap = intensity_colormap * intensity_mask[:, :, None]  # Apply mask to colormap
        overlay_image[intensity_mask > 0] = cv2.addWeighted(
            overlay_image[intensity_mask > 0], rgb_weight, masked_colormap[intensity_mask > 0], 1 - rgb_weight, 0
        )
        return overlay_image

    def visualize_projection(self, input_data, label, original_image, undistorted_image):
        camera_data = input_data[:, :, :3]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self.create_overlay_image(camera_data, intensity_image)

        # Title
        if label == 1:
            title = "Calibration Params = Correct"
        elif label == 0:
            title = "Calibration Params = Wrong"

        # Visualize results
        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        # plt.subplot(2, 2, 1)
        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # plt.title("Original RGB Image")
        # plt.axis("off")

        # plt.subplot(2, 2, 2)
        # plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
        # plt.title("Undistorted RGB Image (If Applicable)")
        # plt.axis("off")

        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title("LiDAR Overlay on Image")
        plt.axis("off")

        # plt.subplot(2, 2, 4)
        # plt.imshow(intensity_image[:, :, 0], cmap='jet')
        # plt.title("Intensity Image")
        # plt.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_results(self, input_data, label, original_image, undistorted_image):
        camera_data = input_data[:, :, :3]
        depth_image = input_data[:, :, 3:4]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self.create_overlay_image(camera_data, intensity_image)

        # Title
        if label == 1:
            title = "Calibration Correct"
        elif label == 0:
            title = "Calibration Error"

        # Visualize results
        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original RGB Image")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
        plt.title("Undistorted RGB Image (If Applicable)")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(camera_data, cv2.COLOR_BGR2RGB))
        plt.title("Augmented RGB Image")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title("LiDAR Overlay on Image")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(depth_image[:, :, 0], cmap="jet")
        plt.title("Depth Image")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(intensity_image[:, :, 0], cmap="jet")
        plt.title("Intensity Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    results = dict()
    results["img_path"] = "data/calibrated_data/training_set/data/250_image.jpg"
    tf = CalibrationClassificationTransform(debug=True, enable_augmentation=False)
    results = tf(results)
