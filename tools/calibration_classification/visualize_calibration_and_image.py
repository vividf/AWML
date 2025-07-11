#!/usr/bin/env python3
import argparse
import json
import os
import re
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from t4_devkit import Tier4
from transforms3d.quaternions import quat2mat


def get_scene_root_dir_path(root_path: str, dataset_version: str, scene_id: str) -> str:
    """Get the scene root directory path, handling version directories."""
    version_pattern = re.compile(r"^\d+$")
    scene_root_dir_path = os.path.join(root_path, dataset_version, scene_id)

    version_dirs = [d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. Please update to the latest version.",
            DeprecationWarning,
        )
        return scene_root_dir_path


def find_sensor_calibration(annotation_dir, channel_name):
    calib_json = os.path.join(annotation_dir, "calibrated_sensor.json")
    sensor_json = os.path.join(annotation_dir, "sensor.json")
    with open(calib_json, "r") as f:
        calib_data = json.load(f)
    with open(sensor_json, "r") as f:
        sensor_data = json.load(f)
    token = None
    for s in sensor_data:
        if s.get("channel", "") == channel_name:
            token = s["token"]
            break
    if token is None:
        raise RuntimeError(f"No {channel_name} found in sensor.json")
    calib = None
    for c in calib_data:
        if c["sensor_token"] == token:
            calib = c
            break
    if calib is None:
        raise RuntimeError(f"No calibration for {channel_name} found in calibrated_sensor.json")
    return {
        "rotation": np.array(calib["rotation"], dtype=np.float32),
        "translation": np.array(calib["translation"], dtype=np.float32),
        "camera_matrix": (
            np.array(calib.get("camera_intrinsic", []), dtype=np.float32) if "camera_intrinsic" in calib else None
        ),
        "distortion_coefficients": (
            np.array(calib.get("camera_distortion", []), dtype=np.float32) if "camera_distortion" in calib else None
        ),
    }


def load_pointcloud_bin(bin_path, show_details=False):
    pc_raw = np.fromfile(bin_path, dtype=np.float32)

    # 直接使用5個fields
    n = 5
    if pc_raw.size % n != 0:
        raise ValueError(f"Pointcloud size {pc_raw.size} is not divisible by {n}")

    print(f"[INFO] {bin_path} has {n} fields per point, total points: {pc_raw.size // n}")
    pc = pc_raw.reshape(-1, n)

    if show_details:
        # 顯示前幾個點的詳細信息
        print(f"[DEBUG] First 5 points with {n} fields each:")
        for i in range(min(5, len(pc))):
            print(
                f"  Point {i}: x={pc[i,0]:.3f}, y={pc[i,1]:.3f}, z={pc[i,2]:.3f}, intensity={pc[i,3]:.3f}, ring={pc[i,4]:.0f}"
            )

        # 顯示統計信息
        print(f"[DEBUG] Point cloud statistics:")
        print(f"  X range: {pc[:,0].min():.3f} ~ {pc[:,0].max():.3f}")
        print(f"  Y range: {pc[:,1].min():.3f} ~ {pc[:,1].max():.3f}")
        print(f"  Z range: {pc[:,2].min():.3f} ~ {pc[:,2].max():.3f}")
        print(f"  Intensity range: {pc[:,3].min():.3f} ~ {pc[:,3].max():.3f}")
        print(f"  Ring range: {pc[:,4].min():.0f} ~ {pc[:,4].max():.0f}")

    return pc[:, :3], pc[:, 3], n  # 回傳 n 讓你知道 field 數


def visualize_calibration_and_image(image_path, annotation_dir, pointcloud_path, output_path=None, show_details=False):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 讀取 calibration
    cam_calib = find_sensor_calibration(annotation_dir, "CAM_FRONT")
    lidar_calib = find_sensor_calibration(annotation_dir, "LIDAR_CONCAT")
    camera_matrix = cam_calib["camera_matrix"]
    distortion_coefficients = cam_calib["distortion_coefficients"]
    q_cam = cam_calib["rotation"]
    t_cam = cam_calib["translation"]
    q_lidar = lidar_calib["rotation"]
    t_lidar = lidar_calib["translation"]

    # 讀取點雲
    pointcloud, intensities, num_fields = load_pointcloud_bin(pointcloud_path, show_details)
    print(f"[INFO] Detected {num_fields} fields per point in {pointcloud_path}")
    min_intensity = intensities.min()
    max_intensity = intensities.max()
    if min_intensity == max_intensity:
        intensities = np.ones_like(intensities)
    else:
        intensities = (intensities - min_intensity) / (max_intensity - min_intensity)

    # 1. LIDAR座標 → ego座標
    R_lidar = quat2mat(q_lidar)
    pc_ego = pointcloud @ R_lidar.T + t_lidar
    # 2. ego座標 → camera座標
    R_cam = quat2mat(q_cam)
    pc_cam = (pc_ego - t_cam) @ R_cam

    # 投影
    rvec = np.zeros((3, 1))
    tvec_cv = np.zeros((3, 1))
    dcoeff = distortion_coefficients[:8]
    img_points, _ = cv2.projectPoints(pc_cam, rvec, tvec_cv, camera_matrix, dcoeff)
    img_points = img_points.reshape(-1, 2)
    h, w = image.shape[:2]
    mask = (
        (img_points[:, 0] > 0)
        & (img_points[:, 0] < w)
        & (img_points[:, 1] > 0)
        & (img_points[:, 1] < h)
        & (pc_cam[:, 2] > 0)
    )
    n_in = np.count_nonzero(mask)
    print(f"[DEBUG] nuScenes標準流程，投影在畫面內的點數: {n_in} / {img_points.shape[0]}")
    print(f"img_points x range: {img_points[:,0].min():.1f} ~ {img_points[:,0].max():.1f}")
    print(f"img_points y range: {img_points[:,1].min():.1f} ~ {img_points[:,1].max():.1f}")
    print(f"pc_cam Z range: {pc_cam[:,2].min():.2f} ~ {pc_cam[:,2].max():.2f}")
    img_points = img_points[mask]
    intensities = intensities[mask]

    # 畫在影像上
    vis_img = image_rgb.copy()
    intensities = np.asarray(intensities).astype(float).flatten()

    # 使用距離來決定顏色，讓近處為紅色，遠處為藍色
    if len(intensities) > 0:
        # 計算每個點到相機的距離
        distances = np.sqrt(pointcloud[mask, 0] ** 2 + pointcloud[mask, 1] ** 2 + pointcloud[mask, 2] ** 2)

        # 設定距離範圍 (可以根據需要調整)
        min_dist = distances.min()
        max_dist = distances.max()

        # 反轉顏色映射：近處為紅色，遠處為藍色
        dist_normalized = (distances - min_dist) / (max_dist - min_dist + 1e-8)
        colors = plt.cm.jet(1.0 - dist_normalized)  # 反轉 colormap

        print(f"[DEBUG] Distance range: {min_dist:.2f} ~ {max_dist:.2f} meters")

        for i, (pt, inten) in enumerate(zip(img_points, intensities)):
            # 獲取顏色
            color = colors[i][:3]  # RGB, 0~1

            # 確保顏色值在有效範圍內
            color = np.clip(color, 0, 1)
            color = tuple(int(255 * float(c)) for c in color)

            # 根據強度調整點的大小
            point_size = max(1, int(2 * inten + 1))

            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), point_size, color, -1)

    # 顯示或保存
    if output_path:
        # Save the image
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_img_bgr)
        print(f"[INFO] Saved visualization to: {output_path}")
    else:
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_img)
        plt.title("LiDAR points projected on image (intensity as color)")
        plt.axis("off")
        plt.show()


def visualize_scene_scene_id(root_path, dataset_version, scene_id, output_dir=None, show_details=False):
    """Visualize all images in a scene given scene ID."""
    # Get scene root directory
    scene_root_dir_path = get_scene_root_dir_path(root_path, dataset_version, scene_id)

    if not os.path.isdir(scene_root_dir_path):
        raise ValueError(f"Scene directory does not exist: {scene_root_dir_path}")

    print(f"[INFO] Processing scene: {scene_id}")
    print(f"[INFO] Scene directory: {scene_root_dir_path}")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir}")

    # Get data directory paths
    data_path = os.path.join(scene_root_dir_path, "data")
    annotation_dir = os.path.join(scene_root_dir_path, "annotation")

    if not os.path.exists(data_path):
        raise ValueError(f"Data directory does not exist: {data_path}")

    if not os.path.exists(annotation_dir):
        raise ValueError(f"Annotation directory does not exist: {annotation_dir}")

    # Get camera and lidar directories
    cam_dir = os.path.join(data_path, "CAM_FRONT")
    lidar_dir = os.path.join(data_path, "LIDAR_CONCAT")

    if not os.path.exists(cam_dir):
        raise ValueError(f"Camera directory does not exist: {cam_dir}")

    if not os.path.exists(lidar_dir):
        raise ValueError(f"Lidar directory does not exist: {lidar_dir}")

    # Get all image files
    image_files = []
    for fname in os.listdir(cam_dir):
        if fname.endswith(".jpg"):
            frame_id = os.path.splitext(fname)[0]
            img_path = os.path.join(cam_dir, fname)
            pc_path = os.path.join(lidar_dir, f"{frame_id}.pcd.bin")

            # Check if corresponding pointcloud exists
            if os.path.exists(pc_path):
                image_files.append((frame_id, img_path, pc_path))

    # Sort by frame_id to maintain order
    image_files.sort(key=lambda x: int(x[0]))

    print(f"[INFO] Found {len(image_files)} image-pointcloud pairs")

    # Process all images
    for i, (frame_id, image_path, pointcloud_path) in enumerate(image_files):
        print(f"[INFO] Processing frame {i+1}/{len(image_files)}: {frame_id}")

        # Check if files exist
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue

        if not os.path.exists(pointcloud_path):
            print(f"[WARNING] Pointcloud file not found: {pointcloud_path}")
            continue

        # Generate output path if saving
        output_path = None
        if output_dir:
            output_filename = f"frame_{i:06d}_{frame_id}.jpg"
            output_path = os.path.join(output_dir, output_filename)

        try:
            visualize_calibration_and_image(image_path, annotation_dir, pointcloud_path, output_path, show_details)
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_id}: {e}")
            continue

    print(f"[INFO] Finished processing scene {scene_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file (jpg)")
    parser.add_argument(
        "--annotation_dir",
        help="Path to annotation directory (should contain calibrated_sensor.json and sensor.json)",
    )
    parser.add_argument("--pointcloud", help="Path to pointcloud bin file")

    # New arguments for scene-based processing
    parser.add_argument("--scene_id", help="Scene ID to process (e.g., e6d0237c-274c-4872-acc9-dc7ea2b77943)")
    parser.add_argument("--root_path", default="./data/t4dataset", help="Root path to T4Dataset")
    parser.add_argument("--dataset_version", help="Dataset version (e.g., db_jpntaxi_v2)")
    parser.add_argument("--output_dir", help="Output directory for saving visualizations")
    parser.add_argument(
        "--show_point_details", action="store_true", help="Show detailed point cloud field information"
    )

    args = parser.parse_args()

    # Check if we're doing scene-based processing or single file processing
    if args.scene_id:
        if not args.dataset_version:
            raise ValueError("--dataset_version is required when using --scene_id")

        visualize_scene_scene_id(
            root_path=args.root_path,
            dataset_version=args.dataset_version,
            scene_id=args.scene_id,
            output_dir=args.output_dir,
            show_details=args.show_point_details,
        )
    else:
        # Original single file processing
        if not all([args.image, args.annotation_dir, args.pointcloud]):
            raise ValueError("--image, --annotation_dir, and --pointcloud are required for single file processing")

        visualize_calibration_and_image(
            args.image, args.annotation_dir, args.pointcloud, show_details=args.show_point_details
        )


if __name__ == "__main__":
    main()
