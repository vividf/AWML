import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from autoware_ml.calibration_classification.datasets.transforms.camera_lidar_augmentation import alter_calibration


@dataclass
class CalibrationData:
    """Structured representation of camera calibration data."""

    camera_matrix: np.ndarray  # Original camera intrinsic matrix (3x3)
    distortion_coefficients: np.ndarray  # Camera distortion coefficients
    # Updated camera matrix after image processing (undistortion, cropping, scaling)
    # This matrix should be used for 3D->2D projection after any image transformations
    # to ensure geometric consistency between the processed image and 3D point projections
    new_camera_matrix: Optional[np.ndarray] = None
    lidar_to_camera_transformation: np.ndarray = None
    lidar_to_ego_transformation: np.ndarray = None
    camera_to_ego_transformation: np.ndarray = None
    lidar_pose: np.ndarray = None
    camera_pose: np.ndarray = None

    def __post_init__(self):
        """Initialize new_camera_matrix if not provided."""
        if self.new_camera_matrix is None:
            self.new_camera_matrix = self.camera_matrix.copy()


@TRANSFORMS.register_module()
class CalibrationClassificationTransform(BaseTransform):
    """Transforms image, pointcloud, and calibration data into inputs for a calibration classifier."""

    def __init__(
        self,
        validation: bool = False,
        test: bool = False,
        debug: bool = False,
        undistort: bool = True,
        enable_augmentation: bool = False,
        use_scipy_quat: bool = False,  # Added option for scipy quaternion conversion
        data_root: str = None,  # 新增
    ):
        """Initialize the CalibrationClassificationTransform.

        Args:
            validation (bool): Whether this is for validation mode. Defaults to False.
            test (bool): Whether this is for test mode. Defaults to False.
            debug (bool): Whether to enable debug visualization. Defaults to False.
            undistort (bool): Whether to undistort images. Defaults to True.
            enable_augmentation (bool): Whether to enable data augmentation. Defaults to False.
            use_scipy_quat (bool): Whether to use scipy for quaternion to rotation matrix conversion. Defaults to False.
        """
        super().__init__()
        self.validation = validation
        self.test = test
        self.debug = debug
        self.undistort = undistort
        self.enable_augmentation = enable_augmentation
        self.use_scipy_quat = use_scipy_quat
        self.data_root = data_root

    def transform(self, results: Dict[str, Any], force_generate_miscalibration: bool = False) -> Dict[str, Any]:
        """Transform input data for calibration classification.

        Args:
            results (Dict[str, Any]): Input data dictionary containing all info.pkl fields.
            force_generate_miscalibration (bool): Whether to force generation of miscalibration. Defaults to False.

        Returns:
            Dict[str, Any]: Transformed data dictionary with processed images and labels.
        """

        # print(f"[DEBUG] Start transform for sample_idx={results.get('sample_idx', 'unknown')}")
        # Set random seeds for reproducibility during validation
        if self.validation:
            random.seed(results["sample_idx"])
            np.random.seed(results["sample_idx"])
        else:
            random.seed(None)
            np.random.seed(None)

        # Loading and preparing data
        camera_data, lidar_data, calibration_data = self.load_data(results, camera_channel="CAM_FRONT")

        # Image undistortion if necessary
        if self.undistort:
            undistorted_data, calibration_data = self.undistort_image(camera_data, calibration_data, alpha=0.0)
        else:
            undistorted_data, calibration_data.new_camera_matrix = camera_data, calibration_data.camera_matrix

        # Label generation
        if self.test and force_generate_miscalibration:
            raise ValueError("force_generate_miscalibration is not supported in test mode")
        if self.test:  # in test case, label is not used but must be registered
            generate_miscalibration = False
        else:
            generate_miscalibration = force_generate_miscalibration or random.choice([True, False])  # 50/50 split

        if generate_miscalibration:
            # TODO(vividf): probably we don't want to put this back?
            calibration_data.lidar_to_camera_transformation = alter_calibration(
                calibration_data.lidar_to_camera_transformation,
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
        if self.debug:
            print("debug is true")

            # Set a unique index for saving visualizations
            img_path = (
                results["images"]["CAM_FRONT"]["img_path"]
                if "images" in results and "CAM_FRONT" in results["images"]
                else None
            )
            if img_path is not None:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                self._current_img_index = base_name
            else:
                self._current_img_index = "unknown"
            self.visualize_projection(
                input_data, label, img_index=self._current_img_index, sample_idx=results["sample_idx"]
            )

        return {"img": input_data, "gt_label": label}
        # Final results

        # dummy = np.zeros((1860, 2880, 5), dtype=np.float32)
        # results["img"] = dummy  # ❗只要這行，就會 leak，即使 return {}
        # # del results["img"]
        # # return {}
        # return {"img": dummy}

        # import torch
        # dummy = np.zeros((1860, 2880, 5), dtype=np.float32)

        # # Convert to torch tensor immediately
        # img_tensor = torch.from_numpy(dummy).permute(2, 0, 1)  # shape (5, H, W)

        # return {
        #     "img": img_tensor,  # Safe: PyTorch handles tensor memory
        #     "gt_label": 0
        # }
        # results["img"] = input_data
        # results["gt_label"] = label

        # 🧹 強制清除所有其他欄位
        allowed_keys = {"img", "gt_label"}
        for k in list(results.keys()):
            if k not in allowed_keys:
                del results[k]

        # TODO(vividf): only for debug
        return {}

        # print(f"[TRANSFORM] results keys: {list(results.keys())}")

        # 移除 DataSample 與 metainfo 相關程式碼，避免 memory leak
        # meta = {"input_data": input_data}
        # if "images" in results:
        #     meta["images"] = results["images"]
        # if "img_path" in results:
        #     meta["img_path"] = results["img_path"]
        # if "img_path" in results:
        #     meta["sample_idx"] = results["sample_idx"]
        # sample = DataSample().set_gt_label(label)
        # sample.set_metainfo(meta)
        # results["data_samples"] = sample
        return results

    def load_data(
        self, sample: dict, camera_channel: str = "CAM_FRONT"
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], CalibrationData]:
        """
        Loads camera, LiDAR, and calibration data from a sample dict (from info.pkl).

        Args:
            sample (dict): One sample from info.pkl.
            camera_channel (str): Which camera channel to use (default: 'CAM_FRONT').

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], CalibrationData]:
                Camera image, LiDAR data dictionary, and calibration data.
        """
        # Load image
        img_path = sample["images"][camera_channel]["img_path"]
        if img_path is not None and self.data_root is not None:
            img_path = os.path.join(self.data_root, img_path)
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Load lidar
        lidar_path = sample["lidar_points"]["lidar_path"]
        if lidar_path is not None and self.data_root is not None:
            lidar_path = os.path.join(self.data_root, lidar_path)
        if lidar_path is None or not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Lidar file not found: {lidar_path}")
        num_pts_feats = sample["lidar_points"].get("num_pts_feats", 5)
        pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_pts_feats)
        # x, y, z, intensity, ring
        lidar_data = {
            "pointcloud": pointcloud[:, :3],
            "intensities": self.normalize_intensity(pointcloud[:, 3]),
        }  # xyz

        # Calibration info extraction from info.pkl
        cam_info = sample["images"][camera_channel]
        lidar_info = sample["lidar_points"]

        camera_matrix = np.array(cam_info["cam2img"]) if cam_info["cam2img"] is not None else None
        distortion_coefficients = np.zeros(5, dtype=np.float32)  # Placeholder, update if available in info.pkl

        lidar_to_camera_transformation = cam_info.get(
            "lidar2cam"
        )  # Not directly in info.pkl, can be computed if needed
        lidar_to_ego_transformation = lidar_info.get("lidar2ego")
        camera_to_ego_transformation = cam_info.get("cam2ego")
        camera_pose = cam_info.get("cam_pose")
        lidar_pose = lidar_info.get("lidar_pose")

        required_fields = {
            "lidar_to_ego_transformation": lidar_to_ego_transformation,
            "camera_to_ego_transformation": camera_to_ego_transformation,
            "camera_pose": camera_pose,
            "lidar_pose": lidar_pose,
        }

        for name, value in required_fields.items():
            if value is None:
                raise ValueError(f"{name} is None")

        # Compose CalibrationData
        calibration_data = CalibrationData(
            camera_matrix=np.array(camera_matrix),
            distortion_coefficients=np.array(distortion_coefficients),
            lidar_to_camera_transformation=np.array(lidar_to_camera_transformation),
            lidar_to_ego_transformation=np.array(lidar_to_ego_transformation),
            camera_to_ego_transformation=np.array(camera_to_ego_transformation),
            lidar_pose=np.array(lidar_pose),
            camera_pose=np.array(camera_pose),
        )

        return image, lidar_data, calibration_data

    def normalize_intensity(self, intensities: np.ndarray) -> np.ndarray:
        """Normalizes LiDAR intensity values to [0, 1] using min-max normalization.

        Args:
            intensities (np.ndarray): Raw intensity values.

        Returns:
            np.ndarray: Normalized intensity values in range [0, 1].
        """
        min_intensity = intensities.min()
        max_intensity = intensities.max()
        if min_intensity == max_intensity:
            return np.zeros_like(intensities)
        return (intensities - min_intensity) / (max_intensity - min_intensity)

    def undistort_image(
        self, image: np.ndarray, calibration_data: CalibrationData, alpha: float = 0.0
    ) -> Tuple[np.ndarray, CalibrationData]:
        """Undistorts the image and updates the calibration data.

        Args:
            image (np.ndarray): Input image to undistort.
            calibration_data (CalibrationData): Camera calibration parameters.
            alpha (float): Free scaling parameter. Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, CalibrationData]: Undistorted image and updated calibration data.
        """
        if np.any(calibration_data.distortion_coefficients):
            distortion_coefficients = calibration_data.distortion_coefficients
            h, w = image.shape[:2]
            # Compute optimal camera matrix for undistortion to maintain geometric accuracy
            calibration_data.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                calibration_data.camera_matrix, distortion_coefficients, (w, h), alpha, (w, h)
            )
            image = cv2.undistort(
                image,
                calibration_data.camera_matrix,
                distortion_coefficients,
                newCameraMatrix=calibration_data.new_camera_matrix,
            )
            calibration_data.distortion_coefficients = np.zeros_like(calibration_data.distortion_coefficients)
        return image, calibration_data

    def signed_random(self, min_value: float, max_value: float) -> float:
        """Generates a random value with random sign.

        Args:
            min_value (float): Minimum absolute value.
            max_value (float): Maximum absolute value.

        Returns:
            float: Random value with random sign.
        """
        sg = 1 if random.random() < 0.5 else -1
        return sg * random.uniform(min_value, max_value)

    def scale_and_crop_image(
        self, image: np.ndarray, calibration_data: CalibrationData, crop_ratio: float = 0.6
    ) -> Tuple[np.ndarray, CalibrationData]:
        """Scales and crops the image, updating the camera matrix accordingly.

        Args:
            image (np.ndarray): Input image to scale and crop.
            calibration_data (CalibrationData): Camera calibration parameters.
            crop_ratio (float): Ratio for cropping. Defaults to 0.6.

        Returns:
            Tuple[np.ndarray, CalibrationData]: Scaled and cropped image with updated calibration data.
        """
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

        # Update camera matrix to account for cropping and scaling transformations
        # This ensures 3D->2D projection remains accurate after image transformations
        scale_factor = w / (end_w - start_w)
        calibration_data.new_camera_matrix[0, 0] *= scale_factor  # fx (focal length x)
        calibration_data.new_camera_matrix[1, 1] *= scale_factor  # fy (focal length y)
        crop_offset_w = start_w
        crop_offset_h = start_h
        # Update principal point coordinates (cx, cy) to account for cropping offset
        calibration_data.new_camera_matrix[0, 2] = (
            calibration_data.new_camera_matrix[0, 2] - crop_offset_w
        ) * scale_factor
        calibration_data.new_camera_matrix[1, 2] = (
            calibration_data.new_camera_matrix[1, 2] - crop_offset_h
        ) * scale_factor

        return resized_image, calibration_data

    def apply_affine_transformation(
        self, image: np.ndarray, max_distortion: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies a controlled affine transformation to the image and updates the calibration matrix.

        Args:
            image (np.ndarray): Input image.
            max_distortion (float): Maximum allowable distortion as a fraction of image dimensions. Defaults to 0.001.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Transformed image and the 3x3 affine transformation matrix.
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

        return transformed_image, affine_transform_3x3

    def apply_augmentations(
        self, image: np.ndarray, calibration_data: CalibrationData
    ) -> Tuple[np.ndarray, CalibrationData, Optional[np.ndarray]]:
        """Applies cropping, scaling, and affine transformations.

        Args:
            image (np.ndarray): Input image to augment.
            calibration_data (CalibrationData): Camera calibration parameters.

        Returns:
            Tuple[np.ndarray, CalibrationData, Optional[np.ndarray]]:
                Augmented image, updated calibration data, and optional affine transformation matrix.
        """
        # Scaling and cropping
        if random.random() > 0.5:
            image, calibration_data = self.scale_and_crop_image(image, calibration_data, crop_ratio=0.6)

        # Affine transformation
        affine_matrix = None
        if random.random() > 0.5:
            image, affine_matrix = self.apply_affine_transformation(image, max_distortion=0.02)

        return image, calibration_data, affine_matrix

    def generate_input_data(
        self,
        image: np.ndarray,
        lidar_data: Dict[str, np.ndarray],
        calibration_data: CalibrationData,
        augmentation_tf: Optional[np.ndarray] = None,
        use_lidar2cam: bool = True,
    ) -> np.ndarray:
        """Generates depth and intensity images using augmented calibration.

        Args:
            image (np.ndarray): Input camera image.
            lidar_data (Dict[str, np.ndarray]): LiDAR point cloud and intensity data.
            calibration_data (CalibrationData): Camera calibration parameters.
            augmentation_tf (Optional[np.ndarray]): Optional augmentation transformation matrix. Defaults to None.

        Returns:
            np.ndarray: Combined image with RGB, depth, and intensity channels.
        """

        points = lidar_data["pointcloud"]  # (N, 3)
        N = points.shape[0]

        points_hom = np.concatenate([points, np.ones((N, 1), dtype=points.dtype)], axis=1)  # (N, 4)

        if use_lidar2cam and calibration_data.lidar_to_camera_transformation is not None:
            # USe lidar2cam directly
            lidar2cam = np.array(calibration_data.lidar_to_camera_transformation)
            points_hom = (lidar2cam @ points_hom.T).T
        else:
            # Step 1: LiDAR to baselink
            lidar_to_ego = np.array(calibration_data.lidar_to_ego_transformation)
            points_hom = (lidar_to_ego @ points_hom.T).T  # (N, 4)

            # Step 2: baselink (LiDAR time) to global
            lidar_pose = np.array(calibration_data.lidar_pose)
            points_hom = (lidar_pose @ points_hom.T).T

            # Step 3: global to baselink (Camera time)
            camera_pose_inv = np.linalg.inv(np.array(calibration_data.camera_pose))
            points_hom = (camera_pose_inv @ points_hom.T).T

            # Step 4: baselink to camera
            camera_to_ego_inv = np.linalg.inv(np.array(calibration_data.camera_to_ego_transformation))
            points_hom = (camera_to_ego_inv @ points_hom.T).T

        pointcloud_ccs = points_hom[:, :3]

        valid_points = pointcloud_ccs[:, 2] > 0.0
        pointcloud_ccs = pointcloud_ccs[valid_points]
        lidar_data["intensities"] = lidar_data["intensities"][valid_points]
        # Project 3D points into the image plane using updated camera matrix
        # This matrix accounts for any image transformations (undistortion, cropping, scaling)
        # to ensure accurate projection of LiDAR points onto the processed image
        camera_matrix = calibration_data.new_camera_matrix
        distortion_coefficients = calibration_data.distortion_coefficients[:8]
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

    def create_lidar_images(
        self, image: np.ndarray, pointcloud_ics: np.ndarray, pointcloud_ccs: np.ndarray, intensities: np.ndarray
    ) -> np.ndarray:
        """Creates depth and intensity images.

        Args:
            image (np.ndarray): Base camera image.
            pointcloud_ics (np.ndarray): Point cloud in image coordinate system.
            pointcloud_ccs (np.ndarray): Point cloud in camera coordinate system.
            intensities (np.ndarray): LiDAR intensity values.

        Returns:
            np.ndarray: Combined image with RGB, depth, and intensity channels.
        """
        h, w = image.shape[:2]
        depth_image = np.zeros((h, w), dtype=np.uint8)
        intensity_image = np.zeros((h, w), dtype=np.uint8)

        valid_mask = (
            (pointcloud_ics[:, 0] >= 0)
            & (pointcloud_ics[:, 0] < w)
            & (pointcloud_ics[:, 1] >= 0)
            & (pointcloud_ics[:, 1] < h)
        )

        valid_ics = pointcloud_ics[valid_mask]
        valid_ccs = pointcloud_ccs[valid_mask]
        valid_intensities = intensities[valid_mask]

        if valid_ics.size > 0:
            radius = 2

            y_offsets, x_offsets = np.mgrid[-radius : radius + 1, -radius : radius + 1]
            y_offsets = y_offsets.flatten()
            x_offsets = x_offsets.flatten()

            center_rows = valid_ics[:, 1].astype(np.int32)
            center_cols = valid_ics[:, 0].astype(np.int32)

            patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
            patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

            in_bounds_mask = (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)

            center_depths = np.clip(255 * valid_ccs[:, 2] / 80.0, 0, 255)
            center_intensities = (valid_intensities * 255).squeeze()

            broadcasted_depths = np.broadcast_to(center_depths[:, np.newaxis], patch_rows.shape)
            broadcasted_intensities = np.broadcast_to(center_intensities[:, np.newaxis], patch_rows.shape)

            final_rows = patch_rows[in_bounds_mask]
            final_cols = patch_cols[in_bounds_mask]
            final_depths = broadcasted_depths[in_bounds_mask]
            final_intensities = broadcasted_intensities[in_bounds_mask]

            sort_indices = np.argsort(final_depths)[::-1]

            sorted_rows = final_rows[sort_indices]
            sorted_cols = final_cols[sort_indices]
            sorted_depths = final_depths[sort_indices]
            sorted_intensities = final_intensities[sort_indices]

            depth_image[sorted_rows, sorted_cols] = sorted_depths.astype(np.uint8)
            intensity_image[sorted_rows, sorted_cols] = sorted_intensities.astype(np.uint8)

        depth_image = np.expand_dims(depth_image, axis=2)
        intensity_image = np.expand_dims(intensity_image, axis=2)
        return np.concatenate([image, depth_image, intensity_image], axis=2)

    def create_overlay_image(
        self, rgb_image: np.ndarray, feature_image: np.ndarray, rgb_weight: float = 0.3
    ) -> np.ndarray:
        """Creates colored overlay image.

        Args:
            rgb_image (np.ndarray): Base RGB image.
            feature_image (np.ndarray): Feature image to overlay.
            rgb_weight (float): Weight for RGB component in overlay. Defaults to 0.3.

        Returns:
            np.ndarray: Overlaid image with features visualized.
        """
        overlay_image = rgb_image.copy()
        intensity_colormap = cv2.applyColorMap((feature_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        intensity_mask = (feature_image > 0).astype(np.uint8).squeeze(-1)  # Remove singleton dimension
        masked_colormap = intensity_colormap * intensity_mask[:, :, None]  # Apply mask to colormap
        overlay_image[intensity_mask > 0] = cv2.addWeighted(
            overlay_image[intensity_mask > 0], rgb_weight, masked_colormap[intensity_mask > 0], 1 - rgb_weight, 0
        )
        return overlay_image

    def visualize_projection(
        self, input_data: np.ndarray, label: int, img_index: str = None, sample_idx: int = None
    ) -> None:
        """Visualizes LiDAR projection results.

        Args:
            input_data (np.ndarray): Combined input data with RGB, depth, and intensity channels.
            label (int): Classification label (0 for miscalibrated, 1 for correct).
            img_index (str, optional): Unique index for filename. Defaults to None.
            sample_idx (int, optional): Sample id to include in filename. Defaults to None.
        """
        import os

        camera_data = input_data[:, :, :3]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self.create_overlay_image(camera_data, intensity_image)

        # Save the overlay image to a directory
        save_dir = "./projection_vis_t4dataset/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"projection_sample_{sample_idx}_{img_index}_label_{label}.png")
        # Convert BGR to RGB for saving with cv2
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Saved projection visualization to {save_path}")

    def visualize_results(
        self,
        input_data: np.ndarray,
        label: int,
        original_image: np.ndarray,
        undistorted_image: np.ndarray,
        img_index: str = None,
        sample_idx: int = None,
    ) -> None:
        """Visualizes comprehensive results including all image types.

        Args:
            input_data (np.ndarray): Combined input data with RGB, depth, and intensity channels.
            label (int): Classification label (0 for miscalibrated, 1 for correct).
            original_image (np.ndarray): Original camera image.
            undistorted_image (np.ndarray): Undistorted camera image.
            img_index (str, optional): Unique index for filename. Defaults to None.
            sample_idx (int, optional): Sample id to include in filename. Defaults to None.
        """
        import os

        print("visualize resuls is called")

        camera_data = input_data[:, :, :3]
        depth_image = input_data[:, :, 3:4]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self.create_overlay_image(camera_data, intensity_image)

        # Title
        if label == 1:
            title = "Calibration Correct"
        elif label == 0:
            title = "Calibration Error"
        else:
            title = f"Calibration Label {label}"

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

        # Save the figure to a directory
        save_dir = "./projection_vis_t4dataset/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"results_sample_{sample_idx}_{img_index}_label_{label}.png")
        plt.savefig(save_path)
        print(f"Saved results visualization to {save_path}")
        plt.close()


if __name__ == "__main__":
    results = dict()
    # TODO(vividf): change this
    results["img_path"] = "data/calibrated_data/training_set/data/250_image.jpg"
    tf = CalibrationClassificationTransform(debug=True, enable_augmentation=False)
    results = tf(results)
