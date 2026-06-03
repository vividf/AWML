# deploy_to_autoware

This tools aim to deploy for Autoware.

## Update onnx metadata

This script injects extra meta information.

- Run

```sh
python tools/deploy_to_autoware/update_onnx_metadata.py {path to onnx file} --version 1 --doc_string "Example additional description"
```

By this update, logs in Autoware log files or /rosout topic can have model metadata.

```
[autoware_lidar_centerpoint_node-1] [I] [TRT] Input filename:   /home/autoware/autoware_data/lidar_centerpoint/pts_backbone_neck_head_centerpoint_tiny.onnx
[autoware_lidar_centerpoint_node-1] [I] [TRT] ONNX IR version:  0.0.6
[autoware_lidar_centerpoint_node-1] [I] [TRT] Opset version:    11
[autoware_lidar_centerpoint_node-1] [I] [TRT] Producer name:    AWML
[autoware_lidar_centerpoint_node-1] [I] [TRT] Producer version: v1.0.0
[autoware_lidar_centerpoint_node-1] [I] [TRT] Domain:           ai.onnx
[autoware_lidar_centerpoint_node-1] [I] [TRT] Model version:    1
[autoware_lidar_centerpoint_node-1] [I] [TRT] Doc string:       Example additional description
```

## [TBD] Generate ROS parameter

This script makes [the ROS parameter](https://github.com/autowarefoundation/autoware.universe/blob/be8235d785597c41d01782ec35da862ba0906578/perception/autoware_lidar_transfusion/config/transfusion_ml_package.param.yaml) from config file of autoware-ml.

- Run the script

```sh
python tools/deploy_ros_parameter_file/deploy_ros_parameter_file.py {config file of autoware-ml} {config file of deployment} {target_param_config_file}
```

- Example

```sh
tools/deploy_ros_parameter_file/deploy_ros_parameter_file.py \
projects/TransFusion/config/t4dataset/90m-768grid/transfusion_lidar_90m-768grid-t4xx1.py \
projects/TransFusion/config/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py \
projects/TransFusion/config/deploy/target_param_config_file.yaml
```

- The config file of `projects/TransFusion/config/deploy/target_param_config_file.yaml` is as below.
  - The script fulfill the ROS parameter value of autoware-ml

```yaml
/**:
  ros__parameters:
    class_names: {class_names}
    voxels_num: [{backend_config.model_inputs.input_shapes.voxels.min_shape[0]}, {backend_config.model_inputs.input_shapes.voxels.min_shape[1]}, {backend_config.model_inputs.input_shapes.voxels.min_shape[2]}]
    point_cloud_range: {point_cloud_range}
    voxel_size: {vocel_size}
    num_proposals: {num_proposal}
```

## Compare CenterPoint export equivalence

Compare two CenterPoint work dirs end-to-end:

- ONNX file hash, opset, node count, initializer count/hash.
- ONNXRuntime numerical output difference.
- TensorRT rebuild (FP16, fixed shapes, fixed workspace) and output difference.

```sh
python tools/deploy_to_autoware/compare_centerpoint_exports.py \
  work_dirs/centerpoint_2_6_fp16_copy \
  work_dirs/centerpoint_2_6_fp16 \
  --workspace-mib 1024 \
  --report-json /tmp/centerpoint_compare_report.json
```

If you only need ONNX + ORT comparison:

```sh
python tools/deploy_to_autoware/compare_centerpoint_exports.py \
  work_dirs/centerpoint_2_6_fp16_copy \
  work_dirs/centerpoint_2_6_fp16 \
  --skip-trt
```
