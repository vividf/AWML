import gc
import psutil
import os
from mmcv.transforms import Compose
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import CalibrationClassificationTransform
from mmpretrain.datasets.transforms import PackInputs  # 新增這行

# 準備一個 info.pkl 格式的 sample dict
results = {
    'frame_idx': '00000',
    'images': {
        'CAM_FRONT': {
            'cam2ego': [[0.006751065445899984, -0.006644674898190983, 0.9999551346989739, 5.314823352615826],
                        [-0.9999771102392598, 0.00040471033338729256, 0.006753903100116143, 0.05013098725079912],
                        [-0.0004495696663305848, -0.9999778420070335, -0.006641790577225082, 2.8033999279803035],
                        [0.0, 0.0, 0.0, 1.0]],
            'cam2img': [[878.86743, 0.0, 1402.0576], [0.0, 1255.89441, 945.04759], [0.0, 0.0, 1.0]],
            'cam_pose': [[-0.08892765430455203, 0.9957189918915034, 0.02521034483022576, 94538.61005861571],
                         [-0.9960040185093805, -0.08868678164185302, -0.0105190244403595, 42957.48131541199],
                         [-0.008238168064362795, -0.026045036927965455, 0.9996268246892761, 42.444423002164214],
                         [0.0, 0.0, 0.0, 1.0]],
            'height': 1860,
            'img_path': '/workspace/data/t4dataset/db_j6gen2_v3/104113be-6d9f-40dc-bbbf-1be9f772ccac/0/data/CAM_FRONT/00000.jpg',
            'lidar2cam': [[0.00716945048421609, -0.9999737571205187, -0.001041177231576467, 0.013260173218441196],
                          [-0.006540037873994502, 0.000994292065383099, -0.9999781194045672, 2.843243887990411],
                          [0.999952912333556, 0.007176102950898632, -0.006532737716839118, -5.791093642525084],
                          [0.0, 0.0, 0.0, 1.0]],
            'sample_data_token': '66edd5d825691448d7bc6b9b0f451c1e',
            'timestamp': 1749704568092388,
            'width': 2880
        }
    },
    'lidar_points': {
        'lidar2ego': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        'lidar_path': '/workspace/data/t4dataset/db_j6gen2_v3/104113be-6d9f-40dc-bbbf-1be9f772ccac/0/data/LIDAR_CONCAT/00000.pcd.bin',
        'lidar_pose': [[-0.08934684545290333, 0.9956665989306409, 0.025790792572267116, 94538.65294183002],
                       [-0.9959657214540635, -0.08909714162669084, -0.010676190446073087, 42957.97426218123],
                       [-0.0083320403325015, -0.026640629268921166, 0.9996103510748846, 42.447204569052765],
                       [0.0, 0.0, 0.0, 1.0]],
        'sample_data_token': '4eef0531f88ba3c2c736c665cb0ef80f',
        'timestamp': 1749704568047560
    },
    'sample_idx': 0,
    'scene_id': '104113be-6d9f-40dc-bbbf-1be9f772ccac'
}

# 準備 pipeline，直接傳 instance
pipeline = Compose([
    CalibrationClassificationTransform(debug=False, enable_augmentation=False),
    PackInputs(input_key="img"),
])

def print_mem():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

for i in range(1000):
    results["sample_idx"] = i
    out = pipeline(results.copy())
    del out
    gc.collect()
    if i % 10 == 0:
        print_mem() 