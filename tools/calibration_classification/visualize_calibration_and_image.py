#!/usr/bin/env python3
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_calibration_and_image(image_path, calibration_path, pointcloud_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 讀取 calibration
    calib = np.load(calibration_path)
    print("Calibration keys:", calib.files)
    for key in calib.files:
        print(f"{key}: {calib[key].shape}\n{calib[key]}\n")

    camera_matrix = calib["camera_matrix"]
    camera_to_lidar_pose = calib["camera_to_lidar_pose"]
    distortion_coefficients = calib["distortion_coefficients"]

    # 取 rotation/translation
    rotation_matrix = camera_to_lidar_pose[:3, :3]
    translation_vector = camera_to_lidar_pose[:3, 3]

    # 讀取點雲
    pc = np.load(pointcloud_path)
    pointcloud = pc["pointcloud"]
    intensities = pc["intensities"]
    # 強度正規化
    min_intensity = intensities.min()
    max_intensity = intensities.max()
    if min_intensity == max_intensity:
        intensities = np.ones_like(intensities)
    else:
        intensities = (intensities - min_intensity) / (max_intensity - min_intensity)

    # 投影到影像
    pointcloud_camera = pointcloud @ rotation_matrix.T + translation_vector
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    dcoeff = distortion_coefficients[:8]
    img_points, _ = cv2.projectPoints(pointcloud_camera, rvec, tvec, camera_matrix, dcoeff)
    img_points = img_points.reshape(-1, 2)

    # 遮罩: 只顯示在影像範圍內且 Z>0 的點
    h, w = image.shape[:2]
    mask = (
        (img_points[:, 0] > 0)
        & (img_points[:, 0] < w)
        & (img_points[:, 1] > 0)
        & (img_points[:, 1] < h)
        & (pointcloud_camera[:, 2] > 0)
    )
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
    parser.add_argument("--calibration", required=True, help="Path to calibration npz file")
    parser.add_argument("--pointcloud", required=True, help="Path to pointcloud npz file")
    args = parser.parse_args()
    visualize_calibration_and_image(args.image, args.calibration, args.pointcloud)


if __name__ == "__main__":
    main()
