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

**Available command line arguments:**

- `--config` (required): Path to the T4dataset configuration file
- `--root_path` (required): Root path of the dataset
- `--version` (required): Product version (e.g., x2)
- `-o, --out_dir` (required): Output directory for info files
- `--lidar_channel` (optional): Lidar channel name (default: LIDAR_CONCAT)
- `--target_cameras` (optional): List of target cameras to process (default: all cameras)

**Examples:**

```sh
# Basic usage with default settings (all cameras)
python3 tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/x2.py --version x2 --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/

# Process only specific cameras
python3 tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/x2.py --version x2 --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/ --target_cameras CAM_FRONT CAM_LEFT CAM_RIGHT
```

**Output files:**
The script generates three pickle files for train/val/test splits:
- `t4dataset_{version}_calib_infos_train.pkl`
- `t4dataset_{version}_calib_infos_val.pkl`
- `t4dataset_{version}_calib_infos_test.pkl`

Each file contains calibration information including:
- Camera and LiDAR data paths
- Transformation matrices (lidar2cam, cam2ego, etc.)
- Timestamps and metadata

### 2.2. Calibration Visualization (Before training)

- Visualization for calibration and image view
  - This tool visualizes the calibration between LiDAR and camera sensors
  - It shows the projection of LiDAR points onto camera images using info.pkl files

```sh
# Process all samples from info.pkl
python tools/calibration_classification/visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir ./calibration_visualization

# Process specific sample
python tools/calibration_classification/visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir ./calibration_visualization --sample_idx 0

# Process specific indices
python tools/calibration_classification/visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir ./calibration_visualization --indices 0 1 2
```

- For T4dataset visualization:

```sh
python tools/calibration_classification/visualize_calibration_and_image.py --info_pkl data/t4dataset/calibration_info/t4dataset_x2_calib_infos_test.pkl --data_root data/t4dataset --output_dir ./calibration_visualization
```

## 3. Visualization Settings (During training, validation, testing)

Understanding visualization configuration is crucial for calibration classification tasks. The `CalibrationClassificationTransform` controls two main visualization types:

### 3.1. Visualization Types

**Projection Visualization** (`projection_vis_dir`):
- Shows LiDAR points projected onto camera images
- Helps verify calibration accuracy between sensors

**Results Visualization** (`results_vis_dir`):
- Displays model predictions
- Shows classification results overlaid on images

### 3.2. Configuration Example

```python
# In config file
projection_vis_dir = "./projection_vis_t4dataset/"
results_vis_dir = "./results_vis_t4dataset/"

# In transform pipeline
dict(
    type="CalibrationClassificationTransform",
    mode="validation",
    undistort=True,
    enable_augmentation=False,
    data_root=data_root,
    projection_vis_dir=projection_vis_dir,  # LiDAR projection visualization
    results_vis_dir=results_vis_dir,        # Model prediction visualization
),
```

### 3.3. Usage Strategy

**Training Phase:**
- Disable visualization for efficiency: `projection_vis_dir=None`, `results_vis_dir=None`
- Focus on model training performance

**Validation/Testing Phase:**
- Enable visualization for analysis: Set both directories to desired paths
- Use projection visualization to verify calibration quality
- Use results visualization to understand model predictions

**Output Files:**
- `projection_vis_dir/`: LiDAR points overlaid on camera images
- `results_vis_dir/`: Classification results with predicted labels

## 4. Train
### 4.1. Change config

- You can change batchsize by file name.
  - For example, 1×b1 -> 2×b8
- If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.

### 4.2. Train

- Train in general by below command.

```sh
python tools/calibration_classification/train.py {config_file}
```

- You can use docker command for training as below.

```sh
docker run -it --rm --gpus '"device=0"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/calibration_classification/train.py {config_file}'
```

### 4.3. Log analysis by Tensorboard

- Run the TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

## 5. Analyze
### 5.1. Evaluation

- Evaluation

```sh
python tools/calibration_classification/test.py {config_file} {checkpoint_file}
```

example
```sh
python tools/calibration_classification/test.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py  epoch_25.pth --out {output_file}
```
