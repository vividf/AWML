# Evaluate the j6gen2 partial-ignore StreamPETR with T4MetricV2 instead of
# T4Metric (v1). This mirrors the BEVFusion *_t4metric_v2.py configs: same
# perception_eval-based evaluator and the same min_num_points=2 annotation GT
# filter, which is what autoware-ml's evaluator applies on its side — the pair
# whose 3D metrics were shown to match exactly for BEVFusion after the
# lidar2ego-info / gt-min-points fixes.
#
# Test (inside the AWML container):
#   python tools/detection3d/test.py \
#     projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_t4metric_v2.py \
#     /path/to/checkpoint.pth
_base_ = [
    "./t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py",
]

experiment_group_name = "streampetr/j6gen2_base"
experiment_name = "t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_t4metric_v2"
work_dir = "work_dirs/" + experiment_name

# Also usable for TRAINING (not just test): T4Metric v1 opens every raw scene
# through the nuscenes devkit at validation time, and scenes that gained an
# annotation/lidarseg.json without the lidarseg/ data dir (db_j6gen2_v9 as of
# 2026-08) crash it. T4MetricV2 reads only the info pkl + data_samples.
# Keep one checkpoint per epoch so epochs can be cross-evaluated later.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=10,
        by_epoch=True,
        save_best="T4MetricV2/T4MetricV2/mAP_center_distance_bev",
    ),
)

perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    evaluation_config_dict=_base_.evaluator_metric_configs,
    load_raw_data=False,
)

frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    confidence_threshold_list=None,
)

# The j6gen2 statistics parquet files live next to the info pkls; the _base_
# chain only names the base-DB ones, so spell out the j6gen2 names here.
training_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + "t4dataset_j6gen2_base_statistics_train.parquet"
)
validation_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + "t4dataset_j6gen2_base_statistics_val.parquet"
)
testing_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + "t4dataset_j6gen2_base_statistics_test.parquet"
)

val_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_val_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="validation",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=32,
    scene_batch_size=-1,
    write_metric_summary=False,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=experiment_group_name,
    min_num_points=2,
)

test_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_test_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="testing",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=32,
    scene_batch_size=-1,
    write_metric_summary=True,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=experiment_group_name,
    min_num_points=2,
)
