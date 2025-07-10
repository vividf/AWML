#!/usr/bin/env python3
import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.quaternions import quat2mat


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


def load_pointcloud_bin(bin_path):
    pc_raw = np.fromfile(bin_path, dtype=np.float32)
    if pc_raw.size % 4 == 0:
        pc = pc_raw.reshape(-1, 4)
    elif pc_raw.size % 5 == 0:
        pc = pc_raw.reshape(-1, 5)[:, :4]
    elif pc_raw.size % 6 == 0:
        pc = pc_raw.reshape(-1, 6)[:, :4]
    else:
        raise ValueError(f"Unexpected pointcloud shape for {bin_path}, size={pc_raw.size}")
    return pc[:, :3], pc[:, 3]


def visualize_calibration_and_image(image_path, annotation_dir, pointcloud_path):
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
    pointcloud, intensities = load_pointcloud_bin(pointcloud_path)
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
    for pt, inten in zip(img_points, intensities):
        inten = float(inten)
        color = plt.cm.jet(inten)[:3]  # RGB, 0~1
        color = tuple(int(255 * float(c)) for c in color)
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, color, -1)

    # 顯示
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.title("LiDAR points projected on image (intensity as color)")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file (jpg)")
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Path to annotation directory (should contain calibrated_sensor.json and sensor.json)",
    )
    parser.add_argument("--pointcloud", required=True, help="Path to pointcloud bin file")
    args = parser.parse_args()
    visualize_calibration_and_image(args.image, args.annotation_dir, args.pointcloud)


if __name__ == "__main__":
    main()
