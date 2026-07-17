# T4MetricV2 evaluator variant of t4_base_vov_flash_480x640_baseline (same convention as
# the CenterPoint *_t4metric_v2.py configs: inherit the training config, replace only the
# evaluators). Used by tools/detection3d/test.py for training-side evaluation and by
# deployment/projects/streampetr (whose entrypoint extracts the metrics settings from the
# `val_evaluator` declaration via `extract_t4metric_v2_config`).
_base_ = [
    "./t4_base_vov_flash_480x640_baseline.py",
]

experiment_name = "t4_base_vov_flash_480x640_baseline_t4metric_v2"


# TODO(vividf): delete this after update to v2.7.2
# Pin the v2.5 artifact's 5-class layout. The shared t4dataset base has since grown to
# 7 classes (+traffic_cone, +barrier); building the head with 7 classes silently fails to
# load the 5-class checkpoint's cls branches (non-strict load) and destroys accuracy, so
# every class-derived key is overridden here to match the trained artifact.
class_names = ["car", "truck", "bus", "bicycle", "pedestrian"]
metainfo = dict(classes=class_names)

model = dict(
    img_roi_head=dict(num_classes=len(class_names)),
    pts_bbox_head=dict(
        num_classes=len(class_names),
        bbox_coder=dict(num_classes=len(class_names)),
    ),
)

val_dataloader = dict(dataset=dict(class_names=class_names, metainfo=metainfo))
test_dataloader = dict(dataset=dict(class_names=class_names, metainfo=metainfo))

# StreamPETR T4 evaluates every class to 51.2 m BEV (the v1 evaluator's eval_class_range),
# so this defines a single 0-51.2 m bucket instead of reusing the 121 m multi-bucket LiDAR
# default in `_base_.evaluator_metric_configs`.
evaluator_metric_configs = dict(
    evaluation_task="detection",
    # target_labels=_base_.class_names,
    target_labels=class_names,
    center_distance_bev_thresholds=[0.5, 1.0, 2.0, 4.0],
    # plane_distance_thresholds is required for the pass fail evaluation
    plane_distance_thresholds=[2.0, 4.0],
    iou_2d_thresholds=None,
    iou_3d_thresholds=None,
    label_prefix="autoware",
    min_distance=0.0,
    max_distance=51.2,
    min_point_numbers=0,
)

perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    evaluation_config_dict=evaluator_metric_configs,
    load_raw_data=False,
)

frame_pass_fail_config = dict(
    # target_labels=_base_.class_names,
    target_labels=class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0] * len(class_names),
    # matching_threshold_list=[2.0] * len(_base_.class_names),
    confidence_threshold_list=None,
)

training_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_train_statistics_file_name
)
testing_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + _base_.info_test_statistics_file_name
validation_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_val_statistics_file_name
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
    dataset_name="t4_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=8,
    scene_batch_size=-1,
    write_metric_summary=False,
    # class_names=_base_.class_names,
    class_names=class_names,
    name_mapping=_base_.name_mapping,
    experiment_name=experiment_name,
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
    dataset_name="t4_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=8,
    scene_batch_size=-1,
    write_metric_summary=True,
    # class_names=_base_.class_names,
    class_names=class_names,
    name_mapping=_base_.name_mapping,
    experiment_name=experiment_name,
)
