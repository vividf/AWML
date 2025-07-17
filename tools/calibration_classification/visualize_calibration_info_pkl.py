"""
This script loads and visualizes calibration info from a .pkl file for T4dataset calibration classification.
It prints the total number of samples and details for the first 5 samples, including calibration keys and paths.
"""

import pprint
import sys

import mmengine

if len(sys.argv) > 1:
    info_path = sys.argv[1]
else:
    info_path = "/workspace/data/t4dataset/calibration_info_new/t4dataset_x2_calib_infos_test.pkl"

samples = mmengine.load(info_path)

data_list = samples["data_list"] if isinstance(samples, dict) and "data_list" in samples else samples
print(f"Total samples: {len(data_list)}")
for i, sample in enumerate(data_list):  # Show first 5 samples
    print(f"\nSample {i}:")
    pprint.pprint(sample)
