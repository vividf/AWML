import numpy as np
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# Compute LiDAR → Camera extrinsics as follows:
#
# Given:
#   - LiDAR is at base_link origin, with identity rotation
#   - Camera pose is expressed in base_link frame:
#
#       p_base = R_cb ⋅ p_cam + t_cb
#
# To transform a LiDAR point into camera coordinates:
#
#   p_cam = R_cb.T ⋅ (p_base - t_cb)
#
# Since p_base = p_lidar (because lidar is at base_link origin):
#
#   p_cam = R_cb.T ⋅ p_lidar - R_cb.T ⋅ t_cb
#
# Therefore:
#   R_lc = R_cb.T
#   t_lc = - R_cb.T ⋅ t_cb
#
# So the transform from LiDAR → Camera is:
#
#   p_cam = R_lc ⋅ p_lidar + t_lc
# ------------------------------------------------------------


# ------------------------------------------------------------
# Compute LiDAR → Camera extrinsics when LiDAR is not
# directly mounted at base_link origin.
#
# Given:
#   - Camera → base_link: (R_cb, t_cb)
#   - LiDAR → base_link: (R_lb, t_lb)
#
# LiDAR → Camera:
#   R_lc = R_cb.T ⋅ R_lb
#   t_lc = R_cb.T ⋅ (t_lb - t_cb)
# ------------------------------------------------------------


# ----------------------------------------------------
# Camera → base_link extrinsic parameters
# Quaternion format is [x, y, z, w]
# ----------------------------------------------------
camera_to_baselink_rotation = np.array(
    [0.5001285, -0.50323653, 0.50007385, -0.49653864], dtype=np.float32
)
camera_to_baselink_translation = np.array(
    [5.314823, 0.05013099, 2.8034], dtype=np.float32
)

# ----------------------------------------------------
# LiDAR → base_link extrinsics
# (LiDAR is at the base_link origin, no rotation)
# ----------------------------------------------------
lidar_to_baselink_rotation = np.eye(3)
lidar_to_baselink_translation = np.array([0.0, 0.0, 0.0])

# ----------------------------------------------------
# Compute rotation matrix for Camera → base_link
# ----------------------------------------------------
rot_cb = R.from_quat(camera_to_baselink_rotation)
R_cb = rot_cb.as_matrix()   # shape (3, 3)

# ----------------------------------------------------
# Compute LiDAR → Camera extrinsics
# ----------------------------------------------------

# The correct rotation from LiDAR to Camera
R_lc = R_cb.T

# The correct translation from LiDAR to Camera
t_cb = camera_to_baselink_translation
t_lc = - R_lc @ t_cb

# ----------------------------------------------------
# Build the 4x4 homogeneous transformation matrix
# for LiDAR → Camera
# ----------------------------------------------------
lidar_to_camera_pose = np.eye(4)
lidar_to_camera_pose[:3, :3] = R_lc
lidar_to_camera_pose[:3, 3] = t_lc

# ----------------------------------------------------
# Camera intrinsics matrix
# ----------------------------------------------------
camera_matrix = np.array(
    [
        [878.86743,     0.0,       1402.0576],
        [0.0,       1255.8944,     945.04761],
        [0.0,          0.0,           1.0],
    ]
)

# Example distortion coefficients (assuming none)
distortion_coefficients = np.zeros(14)

# ----------------------------------------------------
# Print results
# ----------------------------------------------------
print("LiDAR to Camera rotation matrix:\n", R_lc)
print("LiDAR to Camera translation vector:\n", t_lc)
print("LiDAR to Camera 4x4 pose matrix:\n", lidar_to_camera_pose)

# ----------------------------------------------------
# Save results into an .npz file
# ----------------------------------------------------
np.savez(
    "00000_calibration.npz",
    lidar_to_camera_pose=lidar_to_camera_pose,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
)

print("Variables saved to 00000_calibration.npz")



# ------------------------------------------------------------
# Compute LiDAR → Camera extrinsics when LiDAR is not
# directly mounted at base_link origin.
#
# Full Derivation:
#
# Definitions:
#   - base_link frame: common vehicle frame
#   - Camera → base_link:
#         p_base = R_cb ⋅ p_cam + t_cb
#
#     This means:
#       A point expressed in camera frame (p_cam)
#       can be transformed to base_link frame as:
#           p_base = R_cb ⋅ p_cam + t_cb
#
#   - LiDAR → base_link:
#         p_base = R_lb ⋅ p_lidar + t_lb
#
#     Similarly, a point in LiDAR frame (p_lidar)
#     is expressed in base_link as:
#           p_base = R_lb ⋅ p_lidar + t_lb
#
# Goal:
#   We want LiDAR → Camera:
#         p_cam = R_lc ⋅ p_lidar + t_lc
#
# Derivation:
#
# Step 1:
#   Start from LiDAR coordinates:
#         p_lidar
#
# Step 2:
#   Transform LiDAR → base_link:
#         p_base = R_lb ⋅ p_lidar + t_lb
#
# Step 3:
#   Transform base_link → camera:
#         p_cam = R_cb.T ⋅ (p_base - t_cb)
#
#   Because:
#         p_base = R_cb ⋅ p_cam + t_cb
#
#   Rearranged:
#         p_cam = R_cb.T ⋅ (p_base - t_cb)
#
# Step 4:
#   Substitute p_base:
#
#     p_cam = R_cb.T ⋅ ( R_lb ⋅ p_lidar + t_lb - t_cb )
#
# Step 5:
#   Distribute R_cb.T:
#
#     p_cam = R_cb.T ⋅ R_lb ⋅ p_lidar + R_cb.T ⋅ ( t_lb - t_cb )
#
# Therefore:
#     R_lc = R_cb.T ⋅ R_lb
#     t_lc = R_cb.T ⋅ ( t_lb - t_cb )
#
# So the transformation from LiDAR → Camera is:
#
#     p_cam = R_lc ⋅ p_lidar + t_lc
#
# ------------------------------------------------------------