import numpy as np

# Path to your .npz file
# npz_path = "data/calibrated_data/validation_set/data/0_pointcloud.npz"
npz_path = "00000_pointcloud.npz"

# Load the .npz file
data = np.load(npz_path)

# Check and print pointcloud
if "pointcloud" in data:
    pc = data["pointcloud"]
    print(f"pointcloud shape: {pc.shape}")
    print(f"pointcloud dtype: {pc.dtype}")
    print("First 5 points (x, y, z):")
    print(pc[6000:6005])
else:
    print("'pointcloud' not found in the npz file.")

# Check and print intensities
if "intensities" in data:
    intensities = data["intensities"]
    print(f"intensities shape: {intensities.shape}")
    print(f"intensities dtype: {intensities.dtype}")
    print("First 5 intensities:")
    print(intensities[6000:6005])
else:
    print("'intensities' not found in the npz file.")

# List all available keys for reference
print("Available keys in the npz file:", data.files)
