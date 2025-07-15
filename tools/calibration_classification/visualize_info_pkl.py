import pprint
import sys

import mmengine

if len(sys.argv) > 1:
    info_path = sys.argv[1]
else:
    info_path = "/workspace/data/t4dataset/calibration_info/t4dataset_x2_calib_infos_test.pkl"

samples = mmengine.load(info_path)

print(f"Total samples: {len(samples)}")
for i, sample in enumerate(samples[:15]):  # Show first 5 samples
    print(f"\nSample {i}:")
    pprint.pprint(sample)
    calib = sample.get("calibration", {})
    print("Calibration keys:", calib.keys())
    print("Camera calib:", calib.get("camera", None))
    print("Lidar calib:", calib.get("lidar", None))
    print("Image path:", sample.get("img_path", None))
    print("Pointcloud path:", sample.get("pointcloud_path", None))
