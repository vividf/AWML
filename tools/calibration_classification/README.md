# tools/calibration_classification

The pipeline to make the model.
It contains training, evaluation, and visualization for Calibration classification.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- Supported dataset
  - [] NuScenes
  - [x] T4dataset
- Other supported feature
  - [ ] Add unit test

## 1. Setup environment

See [setting environment](/tools/setting_environment/)

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1. T4dataset

- Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for T4dataset X2 Gen2

```sh
python3 tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/x2.py --version x2 --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/
```

## 3. Train
### 3.1. Change config

- You can change batchsize by file name.
  - For example, 1×b1 -> 2×b8
- If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.

### 3.2. Train

- Train in general by below command.

```sh
python tools/calibration_classification/train.py {config_file}
```

- You can use docker command for training as below.

```sh
docker run -it --rm --gpus '"device=0"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/calibration_classification/train.py {config_file}'
```

### 3.3. Log analysis by Tensorboard

- Run the TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

## 4. Analyze
### 4.1. Evaluation

- Evaluation

```sh
python tools/calibration_classification/test.py {config_file} {checkpoint_file}
```

- After training, you can test for each epoch checkpoints as below.

```sh
python tools/calibration_classification/test.py {config_file} {checkpoint_file} --out {output_file}
```

### 4.2. Visualization
#### 4.2.1 Calibration Visualization

- Visualization for calibration and image view
  - This tool visualizes the calibration between LiDAR and camera sensors
  - It shows the projection of LiDAR points onto camera images

```sh
python tools/calibration_classification/visualize_calibration_and_image.py --image-path {image_path} --annotation-dir {annotation_dir} --pointcloud-path {pointcloud_path} --output-path {output_path}
```

- For visualizing a specific scene:

```sh
python tools/calibration_classification/visualize_calibration_and_image.py --root-path {root_path} --dataset-version {version} --scene-id {scene_id} --output-dir {output_dir}
```

#### 4.2.2 Calibration Info Visualization

- Visualization for calibration info pickle files
  - This tool helps debug and visualize the calibration information stored in pickle files

```sh
python tools/calibration_classification/visualize_calibration_info_pkl.py {pkl_file_path}
```

## 5. Deploy

See each projects
