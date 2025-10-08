# Tutorial: Calibration Status Classification

## 1. Setup environment

See [tutorial_installation](/docs/tutorial/tutorial_installation.md) to set up the environment.

### Build Docker for CalibrationStatusClassification

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-calib projects/CalibrationStatusClassification/
```

## 2. Train and evaluation

In this tutorial, we use [CalibrationStatusClassification](/projects/CalibrationStatusClassification/).
If you want to know the tools in detail, please see [calibration_classification](/tools/calibration_classification/) and [CalibrationStatusClassification](/projects/CalibrationStatusClassification/).

### 2.1. Prepare T4dataset

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml-calib -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-calib
```

- Create info files for T4dataset

```sh
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/
```

- (Optional) Process only specific cameras

```sh
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/ --target_cameras CAM_FRONT CAM_LEFT CAM_RIGHT
```

### 2.2. Visualization (Optional)

Visualize calibration data before training to verify sensor alignment:

```sh
python tools/calibration_classification/toolkit.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl --data_root data/t4dataset --output_dir ./work_dirs/calibration_visualization
```

### 2.3. Train

- Single GPU:

```sh
python tools/calibration_classification/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```

- Multi GPU (example with 2 GPUs):

```sh
./tools/calibration_classification/dist_train.sh projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py 2
```

### 2.4. Evaluation

```sh
python tools/calibration_classification/test.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py epoch_25.pth --out results.pkl
```

### 2.5. Deploy the ONNX file

Export model to ONNX and TensorRT:

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    checkpoint.pth
```

For advanced deployment options, see [Deployment guide](/tools/calibration_classification/README.md#6-deployment).
