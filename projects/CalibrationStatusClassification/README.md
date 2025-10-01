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

### 5. Deploy

The deployment script supports model export, verification, and evaluation.

#### 5.1 Basic Export (ONNX and TensorRT)

Export model to ONNX and TensorRT formats:

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
  projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
  projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
  checkpoint.pth \
  --info-pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl \
  --device cuda:0 \
  --work-dir ./work_dirs/
```

#### 5.2 Export with Verification

Verify exported models against PyTorch reference on single sample:

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
  projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
  projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
  checkpoint.pth \
  --info-pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl \
  --sample-idx 0 \
  --device cuda:0 \
  --work-dir ./work_dirs/ \
  --verify
```

#### 5.3 Full Model Evaluation

Evaluate exported models on multiple samples with comprehensive metrics:

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
  projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
  projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
  checkpoint.pth \
  --info-pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl \
  --device cuda:0 \
  --work-dir ./work_dirs/ \
  --evaluate \
  --num-samples 50 \
  --verbose
```

**Evaluation output includes:**
- Overall accuracy and per-class accuracy
- Confusion matrix
- Latency statistics (mean, median, P95, P99, min, max, std)
- Throughput (samples/sec)

#### 5.4 Evaluate Existing Models (without re-export)

To evaluate previously exported models without re-exporting:

```sh
# Evaluate ONNX model
python projects/CalibrationStatusClassification/deploy/main.py \
  projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
  projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
  checkpoint.pth \
  --info-pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl \
  --device cuda:0 \
  --work-dir ./work_dirs/ \
  --evaluate \
  --num-samples 100
```

**Note:** The script will automatically detect and evaluate existing ONNX/TensorRT models in the work directory.

## Troubleshooting

## Reference
