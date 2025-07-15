import numpy as np

# Path to the input and output files
input_bin = "data/calibrated_data/validataion_t4/data/00000.pcd.bin"
output_npz = "00000_pointcloud.npz"

# Read the binary file
point_data = np.fromfile(input_bin, dtype=np.float32)

# Reshape to (N, 5) where N is the number of points
if point_data.size % 5 != 0:
    raise ValueError("File size is not a multiple of 5 floats. Check the file format.")

point_data = point_data.reshape((-1, 5))

# Extract x, y, z and intensity
pointcloud = point_data[:, :3]  # shape (N, 3)
intensities = point_data[:, 3]  # shape (N,)

# Save to npz
np.savez(output_npz, pointcloud=pointcloud, intensities=intensities)

print(f"Saved {pointcloud.shape[0]} points to {output_npz}")
