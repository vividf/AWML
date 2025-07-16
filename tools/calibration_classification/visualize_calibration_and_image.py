#!/usr/bin/env python3
import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion


def load_json(path):
    """Load a JSON file from the given path and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def get_sample_data_info(annotation_dir, channel, frame_id):
    """Retrieve sample data information for a specific channel and frame ID from sample_data.json."""
    sample_data = load_json(os.path.join(annotation_dir, "sample_data.json"))
    # Directly match filename
    if "CAM" in channel:
        target = f"data/{channel}/{frame_id}.jpg"
    else:
        target = f"data/{channel}/{frame_id}.pcd.bin"
    for s in sample_data:
        if s.get("filename", "") == target:
            return s
    raise RuntimeError(f"No sample_data for {channel} {frame_id} (tried filename: {target})")


def get_calibrated_sensor(annotation_dir, token):
    """Retrieve calibrated sensor information for a given token from calibrated_sensor.json."""
    calibrated_sensor = load_json(os.path.join(annotation_dir, "calibrated_sensor.json"))
    for c in calibrated_sensor:
        if c["token"] == token:
            return c
    raise RuntimeError(f"No calibrated_sensor for token {token}")


def get_ego_pose(annotation_dir, token):
    """Retrieve ego pose information for a given token from ego_pose.json."""
    ego_pose = load_json(os.path.join(annotation_dir, "ego_pose.json"))
    for e in ego_pose:
        if e["token"] == token:
            return e
    raise RuntimeError(f"No ego_pose for token {token}")


def load_pointcloud_bin(bin_path, show_details=False):
    """Load a point cloud from a binary file. Optionally print details about the points."""
    pc_raw = np.fromfile(bin_path, dtype=np.float32)
    n = 5
    if pc_raw.size % n != 0:
        raise ValueError(f"Pointcloud size {pc_raw.size} is not divisible by {n}")
    print(f"[INFO] {bin_path} has {n} fields per point, total points: {pc_raw.size // n}")
    pc = pc_raw.reshape(-1, n)
    if show_details:
        print(f"[DEBUG] First 5 points with {n} fields each:")
        for i in range(min(5, len(pc))):
            print(
                f"  Point {i}: x={pc[i,0]:.3f}, y={pc[i,1]:.3f}, z={pc[i,2]:.3f}, intensity={pc[i,3]:.3f}, ring={pc[i,4]:.0f}"
            )
        print(f"[DEBUG] Point cloud statistics:")
        print(f"  X range: {pc[:,0].min():.3f} ~ {pc[:,0].max():.3f}")
        print(f"  Y range: {pc[:,1].min():.3f} ~ {pc[:,1].max():.3f}")
        print(f"  Z range: {pc[:,2].min():.3f} ~ {pc[:,2].max():.3f}")
        print(f"  Intensity range: {pc[:,3].min():.3f} ~ {pc[:,3].max():.3f}")
        print(f"  Ring range: {pc[:,4].min():.0f} ~ {pc[:,4].max():.0f}")
    return pc[:, :3], pc[:, 3], n


def project_lidar_to_image_t4(
    lidar_points,  # (N, 3)
    lidar_calib,
    lidar_pose,
    cam_calib,
    cam_pose,
    cam_intrinsic,
    cam_distortion,
    ignore_distortion=False,
    direct_lidar_to_camera=False,
):
    """
    Project LiDAR points to the image plane using calibration and pose information.
    If direct_lidar_to_camera is True, only apply LiDAR and camera calibration (skip ego/global transforms).
    Otherwise, use the full chain: LiDAR -> Ego (LiDAR time) -> Global -> Ego (Camera time) -> Camera.
    """
    print("direct_lidar_to_camera: ", direct_lidar_to_camera)
    # if direct_lidar_to_camera:
    #     # LiDAR → Camera (skip ego/global)
    #     print("lidar_calib:", lidar_calib)
    #     print("cam_calib:", cam_calib)
    #     print("lidar_points shape:", lidar_points.shape)
    #     print("lidar_points sample:", lidar_points[:5])
    #     points = lidar_points @ Quaternion(lidar_calib["rotation"]).rotation_matrix.T + np.array(
    #         lidar_calib["translation"]
    #     )
    #     print("After lidar extrinsic transform, points shape:", points.shape)
    #     print("points sample:", points[:5])
    #     points = points - np.array(cam_calib["translation"])
    #     print("After translation to camera, points shape:", points.shape)
    #     print("points sample:", points[:5])
    #     points = points @ Quaternion(cam_calib["rotation"]).rotation_matrix
    #     print("After camera rotation, points shape:", points.shape)
    #     print("points sample:", points[:5])
    #     # Additional testing point
    #     test_lidar_points = np.array(
    #         [
    #             [10.157314, 15.85494, 7.982893],
    #             [10.486984, 15.857815, 8.009439],
    #             [10.817074, 15.850803, 8.034791],
    #             [11.156458, 15.859216, 8.066649],
    #             [11.504626, 15.883657, 8.105005],
    #         ]
    #     )
    #     print("\n[TEST] lidar_points sample:", test_lidar_points)
    #     test_points = test_lidar_points @ Quaternion(lidar_calib["rotation"]).rotation_matrix.T + np.array(
    #         lidar_calib["translation"]
    #     )
    #     print("[TEST] After lidar extrinsic transform, points:", test_points)
    #     test_points = test_points - np.array(cam_calib["translation"])
    #     print("[TEST] After translation to camera, points:", test_points)
    #     test_points = test_points @ Quaternion(cam_calib["rotation"]).rotation_matrix
    #     print("[TEST] After camera rotation, points:", test_points)
    # else:
    # Full chain: LiDAR -> Ego (LiDAR time) -> Global -> Ego (Camera time) -> Camera

    print("lidar_calib[rotation]:", lidar_calib["rotation"])
    print("lidar_pose[rotation]:", lidar_pose["rotation"])

    print("Before LiDAR extrinsic:", lidar_points[:5])
    points = lidar_points @ Quaternion(lidar_calib["rotation"]).rotation_matrix.T + np.array(
        lidar_calib["translation"]
    )
    print("After LiDAR extrinsic:", points[:5])
    print("pyquaternion rotmat lidar_calib:", Quaternion(lidar_calib["rotation"]).rotation_matrix)
    points = points @ Quaternion(lidar_pose["rotation"]).rotation_matrix.T + np.array(lidar_pose["translation"])
    print("After LiDAR ego pose:", points[:5])
    print("pyquaternion rotmat lidar_pose:", Quaternion(lidar_pose["rotation"]).rotation_matrix)
    points = points - np.array(cam_pose["translation"])
    print("After subtracting camera ego translation:", points[:5])
    points = points @ Quaternion(cam_pose["rotation"]).rotation_matrix
    print("After camera ego rotation:", points[:5])
    print("pyquaternion rotmat cam_pose:", Quaternion(cam_pose["rotation"]).rotation_matrix)
    points = points - np.array(cam_calib["translation"])
    print("After subtracting camera extrinsic translation:", points[:5])
    points = points @ Quaternion(cam_calib["rotation"]).rotation_matrix
    print("After camera extrinsic rotation:", points[:5])
    print("pyquaternion rotmat cam_calib:", Quaternion(cam_calib["rotation"]).rotation_matrix)
    points_3d = points.T  # (3, N)
    if not ignore_distortion and cam_distortion is not None and len(cam_distortion) > 0:
        img_points, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), cam_intrinsic, cam_distortion)
        img_points = img_points.reshape(-1, 2)
    else:
        img_points = (cam_intrinsic @ points_3d).T
        img_points = img_points[:, :2] / img_points[:, 2:3]
    return img_points, points


def visualize_calibration_and_image(
    image_path,
    annotation_dir,
    pointcloud_path,
    output_path=None,
    show_details=False,
    direct_lidar_to_camera=False,
    camera_id="CAM_FRONT",
):
    """
    Visualize the projection of LiDAR points onto the camera image.
    Args:
        image_path (str): Path to the image file.
        annotation_dir (str): Path to the annotation directory.
        pointcloud_path (str): Path to the point cloud binary file.
        output_path (str, optional): Path to save the output visualization. If None, display the image.
        show_details (bool, optional): Whether to print detailed point cloud information.
        direct_lidar_to_camera (bool, optional): If True, skip ego/global transforms.
        camera_id (str, optional): Camera channel name (e.g., 'CAM_FRONT', 'CAM_LEFT').
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = image.shape[:2]
    pointcloud, intensities, num_fields = load_pointcloud_bin(pointcloud_path, show_details)
    frame_id = os.path.splitext(os.path.basename(image_path))[0]
    # Read metadata
    lidar_info = get_sample_data_info(annotation_dir, "LIDAR_CONCAT", frame_id)
    cam_info = get_sample_data_info(annotation_dir, camera_id, frame_id)
    lidar_calib = get_calibrated_sensor(annotation_dir, lidar_info["calibrated_sensor_token"])
    cam_calib = get_calibrated_sensor(annotation_dir, cam_info["calibrated_sensor_token"])
    lidar_pose = get_ego_pose(annotation_dir, lidar_info["ego_pose_token"])
    cam_pose = get_ego_pose(annotation_dir, cam_info["ego_pose_token"])
    cam_intrinsic = np.array(cam_calib["camera_intrinsic"], dtype=np.float32)
    cam_distortion = (
        np.array(cam_calib.get("camera_distortion", []), dtype=np.float32)
        if "camera_distortion" in cam_calib
        else None
    )
    print("\n===== Calibration Information =====")
    print("[LiDAR Calibration]:\n", json.dumps(lidar_calib, indent=2))
    print("[Camera Calibration]:\n", json.dumps(cam_calib, indent=2))
    print("[LiDAR Ego Pose]:\n", json.dumps(lidar_pose, indent=2))
    print("[Camera Ego Pose]:\n", json.dumps(cam_pose, indent=2))
    print("[Camera Intrinsic]:\n", cam_intrinsic)
    print("[Camera Distortion]:\n", cam_distortion)
    print("===== End Calibration Information =====\n")
    # Project points
    img_points, points_cam = project_lidar_to_image_t4(
        pointcloud,
        lidar_calib,
        lidar_pose,
        cam_calib,
        cam_pose,
        cam_intrinsic,
        cam_distortion,
        direct_lidar_to_camera=direct_lidar_to_camera,
    )
    mask = (
        (img_points[:, 0] > 0)
        & (img_points[:, 0] < w)
        & (img_points[:, 1] > 0)
        & (img_points[:, 1] < h)
        & (points_cam[:, 2] > 0)
    )
    img_points = img_points[mask]
    intensities = intensities[mask]
    # Draw points on the image
    vis_img = image.copy()
    intensities = np.asarray(intensities).astype(float).flatten()
    if len(img_points) > 0:
        # Compute distance from camera for each point
        distances = np.linalg.norm(points_cam[mask], axis=1)
        min_dist = distances.min()
        max_dist = distances.max()
        # Reverse colormap: red for near, blue for far
        dist_normalized = (distances - min_dist) / (max_dist - min_dist + 1e-8)
        colors = plt.cm.jet(1.0 - dist_normalized)
        for i, (pt, color) in enumerate(zip(img_points, colors)):
            bgr = tuple(int(255 * c) for c in color[:3][::-1])  # jet is RGB, cv2 needs BGR
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 1, bgr, -1)
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"[INFO] Saved visualization to: {output_path}")
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"LiDAR points projected on image ({camera_id})")
        plt.axis("off")
        plt.show()


def visualize_scene_scene_id(
    root_path,
    dataset_version,
    scene_id,
    output_dir=None,
    show_details=False,
    direct_lidar_to_camera=False,
    camera_id="CAM_FRONT",
):
    """
    Visualize all images in a scene given a scene ID.
    Args:
        root_path (str): Root path to the dataset.
        dataset_version (str): Dataset version name.
        scene_id (str): Scene ID to process.
        output_dir (str, optional): Directory to save visualizations.
        show_details (bool, optional): Whether to print detailed point cloud information.
        direct_lidar_to_camera (bool, optional): If True, skip ego/global transforms.
        camera_id (str, optional): Camera channel name (e.g., 'CAM_FRONT', 'CAM_LEFT').
    """
    scene_root_dir_path = os.path.join(root_path, dataset_version, scene_id)
    # Automatically enter the only subdirectory (e.g., "0") if present
    subdirs = [d for d in os.listdir(scene_root_dir_path) if os.path.isdir(os.path.join(scene_root_dir_path, d))]
    if len(subdirs) == 1:
        scene_root_dir_path = os.path.join(scene_root_dir_path, subdirs[0])
    if not os.path.isdir(scene_root_dir_path):
        raise ValueError(f"Scene directory does not exist: {scene_root_dir_path}")
    print(f"[INFO] Processing scene: {scene_id}")
    print(f"[INFO] Scene directory: {scene_root_dir_path}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir}")
    data_path = os.path.join(scene_root_dir_path, "data")
    annotation_dir = os.path.join(scene_root_dir_path, "annotation")
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory does not exist: {data_path}")
    if not os.path.exists(annotation_dir):
        raise ValueError(f"Annotation directory does not exist: {annotation_dir}")
    cam_dir = os.path.join(data_path, camera_id)
    lidar_dir = os.path.join(data_path, "LIDAR_CONCAT")
    if not os.path.exists(cam_dir):
        raise ValueError(f"Camera directory does not exist: {cam_dir}")
    if not os.path.exists(lidar_dir):
        raise ValueError(f"Lidar directory does not exist: {lidar_dir}")
    image_files = []
    for fname in os.listdir(cam_dir):
        if fname.endswith(".jpg"):
            frame_id = os.path.splitext(fname)[0]
            img_path = os.path.join(cam_dir, fname)
            pc_path = os.path.join(lidar_dir, f"{frame_id}.pcd.bin")
            if os.path.exists(pc_path):
                image_files.append((frame_id, img_path, pc_path))
    image_files.sort(key=lambda x: int(x[0]))
    print(f"[INFO] Found {len(image_files)} image-pointcloud pairs")
    for i, (frame_id, image_path, pointcloud_path) in enumerate(image_files):
        print(f"[INFO] Processing frame {i+1}/{len(image_files)}: {frame_id}")
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue
        if not os.path.exists(pointcloud_path):
            print(f"[WARNING] Pointcloud file not found: {pointcloud_path}")
            continue
        output_path = None
        if output_dir:
            output_filename = f"frame_{i:06d}_{frame_id}.jpg"
            output_path = os.path.join(output_dir, output_filename)
        try:
            visualize_calibration_and_image(
                image_path,
                annotation_dir,
                pointcloud_path,
                output_path,
                show_details,
                direct_lidar_to_camera,
                camera_id=camera_id,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_id}: {e}")
            continue
    print(f"[INFO] Finished processing scene {scene_id}")


def main():
    """Main entry point for the script. Parses arguments and runs the appropriate visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file (jpg)")
    parser.add_argument(
        "--annotation_dir",
        help="Path to annotation directory (should contain calibrated_sensor.json and sensor.json)",
    )
    parser.add_argument("--pointcloud", help="Path to pointcloud bin file")
    parser.add_argument("--scene_id", help="Scene ID to process (e.g., e6d0237c-274c-4872-acc9-dc7ea2b77943)")
    parser.add_argument("--root_path", default="./data/t4dataset", help="Root path to T4Dataset")
    parser.add_argument("--dataset_version", help="Dataset version (e.g., db_jpntaxi_v2)")
    parser.add_argument("--output_dir", help="Output directory for saving visualizations")
    parser.add_argument(
        "--show_point_details", action="store_true", help="Show detailed point cloud field information"
    )
    parser.add_argument(
        "--direct_lidar_to_camera",
        action="store_true",
        help="Directly transform LiDAR points to camera frame (skip ego/global transforms)",
    )
    parser.add_argument(
        "--camera_id", default="CAM_FRONT", help="Camera channel to use (e.g., CAM_FRONT, CAM_LEFT, CAM_RIGHT, etc.)"
    )
    args = parser.parse_args()
    if args.scene_id:
        if not args.dataset_version:
            raise ValueError("--dataset_version is required when using --scene_id")
        visualize_scene_scene_id(
            root_path=args.root_path,
            dataset_version=args.dataset_version,
            scene_id=args.scene_id,
            output_dir=args.output_dir,
            show_details=args.show_point_details,
            direct_lidar_to_camera=args.direct_lidar_to_camera,
            camera_id=args.camera_id,
        )
    else:
        if not all([args.image, args.annotation_dir, args.pointcloud]):
            raise ValueError("--image, --annotation_dir, and --pointcloud are required for single file processing")
        visualize_calibration_and_image(
            args.image,
            args.annotation_dir,
            args.pointcloud,
            show_details=args.show_point_details,
            direct_lidar_to_camera=args.direct_lidar_to_camera,
            camera_id=args.camera_id,
        )


if __name__ == "__main__":
    main()
