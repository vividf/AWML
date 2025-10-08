# tools/calibration_classification

The pipeline to develop the calibration classification model.
It contains training, evaluation, and visualization for Calibration classification.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- Supported dataset
  - [ ] NuScenes
  - [x] T4dataset
- Other supported feature
  - [ ] Add unit test

## 1. Setup environment

Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1. T4dataset

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml-calib -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-calib
```

- Create info files for T4dataset X2 Gen2

```sh
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/
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
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/

# Process only specific cameras
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/ --target_cameras CAM_FRONT CAM_LEFT CAM_RIGHT
```

**Output files:**
The script generates three pickle files for train/val/test splits:
- `t4dataset_{version}_infos_train.pkl`
- `t4dataset_{version}_infos_val.pkl`
- `t4dataset_{version}_infos_test.pkl`

Each file contains calibration information including:
- Camera and LiDAR data paths
- Transformation matrices (lidar2cam, cam2ego, etc.)
- Timestamps and metadata

**Example data structure:**
```python
{
    'frame_id': 'CAM_FRONT',
    'frame_idx': '00003',
    'image': {
        'cam2ego': [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]],
        'cam2img': [[1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]],
        'cam_pose': [[1.0, 0.0, 0.0, 100.0],
                     [0.0, 1.0, 0.0, 200.0],
                     [0.0, 0.0, 1.0, 1.5],
                     [0.0, 0.0, 0.0, 1.0]],
        'height': 1080,
        'img_path': 'data/camera/front/00003.jpg',
        'lidar2cam': [[1.0, 0.0, 0.0, 0.1],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]],
        'distortion_coefficients': [0.0, 0.0, 0.0, 0.0, 0.0],
        'sample_data_token': 'sample_token_123',
        'timestamp': 1234567890,
        'width': 1920
    },
    'lidar_points': {
        'lidar2ego': [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]],
        'lidar_path': 'data/lidar/00003.pcd.bin',
        'lidar_pose': [[1.0, 0.0, 0.0, 100.0],
                       [0.0, 1.0, 0.0, 200.0],
                       [0.0, 0.0, 1.0, 1.5],
                       [0.0, 0.0, 0.0, 1.0]],
        'sample_data_token': 'lidar_token_456',
        'timestamp': 1234567890
    },
    'sample_idx': 3,
    'scene_id': 'scene_001'
}
```

**Key fields explained:**
- `frame_id`: Camera identifier (e.g., CAM_FRONT, CAM_LEFT)
- `frame_idx`: Frame index in the sequence
- `image`: Camera-specific data including intrinsic/extrinsic parameters
- `lidar_points`: LiDAR data with transformation matrices
- `sample_idx`: Sample index in the dataset
- `scene_id`: Scene identifier

### 2.2. Calibration Visualization (Before training)

- Visualization for calibration and image view
  - This tool visualizes the calibration between LiDAR and camera sensors
  - It shows the projection of LiDAR points onto camera images using info.pkl files

```sh
# Process all samples from info.pkl
python tools/calibration_classification/toolkit.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization

# Process specific sample
python tools/calibration_classification/toolkit.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization --sample_idx 0

# Process specific indices
python tools/calibration_classification/toolkit.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization --indices 0 1 2
```

- For T4dataset visualization:

```sh
python tools/calibration_classification/toolkit.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl --data_root data/t4dataset --output_dir ./work_dirs/calibration_visualization
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
test_projection_vis_dir = "./test_projection_vis_t4dataset/"
test_results_vis_dir = "./test_results_vis_t4dataset/"
```

### 3.3. Usage Strategy

**Training/Validation Phase:**
- Disable visualization for efficiency: `projection_vis_dir=None`, `results_vis_dir=None`
- Focus on model training performance

**Testing Phase:**
- Enable visualization for analysis: Set both directories to desired paths
- Use projection visualization to verify calibration quality
- Use results visualization to understand model predictions

**Output Files:**
- `projection_vis_dir/`: LiDAR points overlaid on camera images
- `results_vis_dir/`: Classification results with predicted labels

## 4. Training
### 4.1. Environment Setup
Set `CUBLAS_WORKSPACE_CONFIG` for deterministic behavior, please check this [nvidia doc](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) for more info

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 4.2. Run Training

- Run training with the following command:

```sh
python tools/calibration_classification/train.py {config_file}
```

* Example
```sh
python tools/calibration_classification/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```

- Alternatively, you can use Docker:

```sh
docker run -it --rm --gpus --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/calibration_classification/train.py {config_file}'
```

### 4.3. Log Analysis with TensorBoard

- Run TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

## 5. Evaluation
PyTorch Model Evaluation

```sh
python tools/calibration_classification/test.py {config_file} {checkpoint_file}
```
* Example
```sh
python tools/calibration_classification/test.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py  epoch_25.pth --out {output_file}
```

For ONNX and TensorRT Evaluation check [Section 6.4](#64-evaluate-onnx-and-tensorrt-models).


## 6. Deployment

The deployment system supports exporting models to ONNX and TensorRT formats with a unified configuration approach.

### 6.1. Deployment Configuration


#### Configuration Structure
The config has four main sections:

```python
# Export configuration - controls export behavior
export = dict(
    mode="both",                       # "onnx", "trt", or "both"
    verify=True,                       # Run verification comparing outputs
    device="cuda:0",                   # Device for export
    work_dir="/workspace/work_dirs",   # Output directory
)

# Runtime I/O configuration - data paths
runtime_io = dict(
    info_pkl="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,                      # Sample for export/verification
    onnx_file=None,                    # Optional: existing ONNX path
)

# TensorRT backend configuration
backend_config = dict(
    type="tensorrt",
    common_config=dict(
        max_workspace_size=1 << 30,    # 1 GiB
        precision_policy="auto",        # Precision policy (see below)
    ),
    model_inputs=[...],                 # Dynamic shape configuration
)

# Evaluation configuration - for model evaluation
evaluation = dict(
    enabled=False,                     # Enable evaluation by default
    num_samples=100,                    # Number of samples to evaluate
    verbose=False,                     # Enable detailed logging
    models=dict(
        onnx=None,                    # Optional: path to existing ONNX model
        tensorrt=None,                # Optional: path to existing TensorRT engine
    ),
)
```

#### Precision Policies

Control TensorRT inference precision with the `precision_policy` parameter:

| Policy | Description | Use Case |
|--------|-------------|----------|
| `auto` | TensorRT decides automatically | Default, FP32 |
| `fp16` | Half-precision floating point | 2x faster, small accuracy loss |
| `fp32_tf32` | Tensor Cores for FP32 | Ampere+ GPUs, FP32 speedup |
| `strongly_typed` | Enforces explicit type constraints, preserves QDQ (Quantize-Dequantize) nodes | INT8 quantized models from QAT or PTQ with explicit Q/DQ ops |

### 6.3. Export Modes

#### Export from checkpoint to Both ONNX and TensorRT
In config file, set `export.mode = "both"`.

#### Export from checkpoint to ONNX Only

In config file, set `export.mode = "onnx"`.

#### Convert Existing ONNX to TensorRT

In config file:
- Set `export.mode = "trt"`
- Set `runtime_io.onnx_file = "/path/to/model.onnx"`

**Note:** The checkpoint is used for verification. To skip verification, set `export.verify = False` in config.

#### Export
```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    checkpoint.pth
```


### 6.4. Evaluate ONNX and TensorRT Models

#### Evaluate Exported Models

```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    checkpoint.pth
```


## 7. INT8 Quantization Guide

INT8 quantization reduces model size and improves inference speed by converting 32-bit floating-point weights to 8-bit integers. This guide covers Post-Training Quantization (PTQ) using NVIDIA's ModelOpt tool.


### 7.1. Generate Calibration Data

Calibration data is used to determine optimal quantization parameters. Use representative samples from your training dataset.

#### 7.1.1. Basic Usage

```sh
python tools/calibration_classification/toolkit.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    --info_pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_train.pkl \
    --data_root data/t4dataset \
    --npz_output_path calibration_file.npz \
    --indices 200
```

#### 7.2.2. Memory Requirements

Calibration data is loaded entirely into GPU memory during quantization. Calculate required memory:
For 32 GB memory, the maximum you can use is approximately: 32 GB / 1860 x 2880 x 5 x 4 Bytes, around 321 images.
Use `--indices 200` to limit to first 200 samples.

Additionally, ensure sufficient disk storage is available on the target device for calibration file output.


### 7.3. Build Optimization Docker Environment

The quantization requires NVIDIA ModelOpt, which is provided in a separate Docker image.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-calib-opt \
    -f projects/CalibrationStatusClassification/DockerfileOpt .
```

**What's included:**
- CUDA 12.6 + cuDNN
- ONNX 1.16.2 + ONNXRuntime 1.23.0
- NVIDIA ModelOpt 0.33.1

### 7.4. Launch Optimization Container

```sh
docker run -it --rm --gpus all --shm-size=32g \
    --name awml-opt -p 6006:6006 \
    -v $PWD:/workspace \
    -v $PWD/data:/workspace/data \
    autoware-ml-calib-opt
```

Remember to put your calibration.npz to the workspace/data

### 7.5. Run INT8 Quantization

Inside the Docker container, run the quantization:

```sh
python3 -m modelopt.onnx.quantization \
    --onnx_path=work_dirs/end2end.onnx \
    --quantize_mode=int8 \
    --calibration_data_path=calibration_file.npz
```

**Output**: `work_dirs/end2end.quant.onnx`


### 7.6. Evaluate Quantized Model

#### 7.6.1. Convert to TensorRT INT8 Engine

For optimal performance, convert the quantized ONNX to TensorRT:

**In deployment config** (`projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py`):
```python
export = dict(
    mode="trt",  # Convert ONNX to TensorRT
    verify=True,
    device="cuda:0",
    work_dir="/workspace/work_dirs",
)

runtime_io = dict(
    onnx_file="work_dirs/end2end.quant.onnx",  # Use quantized ONNX
    info_pkl="data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl",
    sample_idx=0,
)

backend_config = dict(
    type="tensorrt",
    common_config=dict(
        max_workspace_size=1 << 30,
        precision_policy="strongly_typed",  # Preserve INT8 quantization
    ),
    # ... model_inputs configuration
)
```

**Run conversion:**
```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```

**Output**: `work_dirs/end2end.engine` (INT8 TensorRT engine)

#### 7.6.3. Evaluate TensorRT INT8 Engine

In config file, set `export.mode = "none"` and `"evaluation.enabled=True"`.
```python
export = dict(
    mode="none",  # Export mode: "onnx", "trt", "both", or "none"
    ...
)

evaluation = dict(
    enabled=True,
    ...
    onnx_model=None,
    tensorrt_model="/workspace/work_dirs/end2end.engine",
)
```

**Evaluate INT8 engine**
```sh
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```
