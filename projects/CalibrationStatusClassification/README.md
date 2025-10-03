# CalibrationStatusClassification
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [package_name](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/)
- Supported dataset
  - [ ] NuScenes
  - [x] T4dataset
- Supported model
  - [x] ResNet18
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md)to set up the environment.
- Docker build for CalibrationStatusClassification

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-calib projects/CalibrationStatusClassification/
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml-calib -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-calib
```

### 2. Config

- Change parameters for your environment by changing [base config file](configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py).

```py
batch_size = 16
max_epochs = 50
```

### 3. Dataset

- [Create dataset](../../tools/calibration_classification/README.md)


### 4. Train

Run training:

- Single GPU:

```sh
python tools/calibration_classification/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```

- Multi GPU (example with 2 GPUs):

```sh
./tools/calibration_classification/dist_train.sh projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py 2
```


### 5. Test
Run testing
```sh
python tools/calibration_classification/test.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py  epoch_25.pth --out {output_file}
```


### 6. Deploy

The deployment script provides model export, verification, and evaluation support:

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    checkpoint.pth
```

For more details on configuration options such as INT8 quantization and custom settings, please refer to the [Deployment guide](../../tools/calibration_classification/README.md#6-deployment)


## Troubleshooting

## Reference
