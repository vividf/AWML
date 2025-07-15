import numpy as np
from scipy.spatial.transform import Rotation as R

# Given data
rotation = np.array([0.5001285, -0.50323653, 0.50007385, -0.49653864], dtype=np.float32)  # (x, y, z, w)
translation = np.array([5.314823, 0.05013099, 2.8034], dtype=np.float32)

# Convert quaternion (x, y, z, w) to rotation matrix
rot = R.from_quat(rotation)
rotation_matrix = rot.as_matrix()  # shape (3, 3)

# Build the 4x4 transformation matrix
camera_to_lidar_pose = np.eye(4)
camera_to_lidar_pose[:3, :3] = rotation_matrix
camera_to_lidar_pose[:3, 3] = translation

# Example camera matrix and distortion coefficients
camera_matrix = np.array(
    [
        [8.7886743e02, 0.0000000e00, 1.4020576e03],
        [0.0000000e00, 1.2558944e03, 9.4504761e02],
        [0.0000000e00, 0.0000000e00, 1.0000000e00],
    ]
)
distortion_coefficients = np.zeros(14)  # Example: 14 zeros

# Save them into an npz file
np.savez(
    "00000.npz",
    camera_to_lidar_pose=camera_to_lidar_pose,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
)

print("Variables saved to 00000.npz")
