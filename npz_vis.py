import numpy as np

# Path to your .npz file
# npz_path = "data/calibrated_data/validation_set/data/0_calibration.npz"
npz_path = "00000.npz"

# Load the .npz file
data = np.load(npz_path)

# Print everything for each array
for key in data.files:
    print(f"Array name: {key}")
    print(f"  shape: {data[key].shape}")
    print(f"  dtype: {data[key].dtype}")
    print(f"  values:\n{data[key]}\n")
