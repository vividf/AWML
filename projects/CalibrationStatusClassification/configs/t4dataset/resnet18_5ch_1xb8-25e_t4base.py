_base_ = [
    "mmpretrain::_base_/default_runtime.py",
    "mmpretrain::_base_/models/resnet18.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
]

custom_imports = dict(
    imports=[
        "autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset",
        "autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform",
        "autoware_ml.calibration_classification.hooks.result_visualization_hook",
    ],
    allow_failed_imports=False,
)


batch_size = 8
num_workers = 8
max_epochs = 25

data_preprocessor = dict()

model = dict(
    type="mmpretrain.ImageClassifier",
    backbone=dict(
        type="ResNet",
        depth=18,
        in_channels=5,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# Visualization directory
projection_vis_dir = "./projection_vis_t4dataset/"
results_vis_dir = "./results_vis_t4dataset/"
data_root = "/workspace/data/t4dataset"


train_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="train",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=projection_vis_dir,
        results_vis_dir=None,
    ),
    dict(type="PackInputs", input_key="fused_img", meta_keys=["img_path", "fused_img", "images", "sample_idx"]),
]


info_directory_path = "/workspace/data/t4dataset/calibration_info/"
train_info_file = f"t4dataset_x2_calib_infos_train.pkl"
val_info_file = f"t4dataset_x2_calib_infos_val.pkl"
test_info_file = f"t4dataset_x2_calib_infos_test.pkl"
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + train_info_file,
        pipeline=train_pipeline,
        data_root=data_root,
    ),
)

val_cfg = dict()

val_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="validation",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=projection_vis_dir,
        results_vis_dir=None,
    ),
    dict(type="PackInputs", input_key="fused_img", meta_keys=["img_path", "fused_img", "images", "sample_idx"]),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + val_info_file,
        pipeline=val_pipeline,
        data_root=data_root,
        indices=100,  # Use first 100 samples for validation
    ),
)

val_evaluator = dict(topk=(1,), type="mmpretrain.evaluation.Accuracy")

test_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="test",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=projection_vis_dir,
        results_vis_dir=results_vis_dir,
    ),
    dict(type="PackInputs", input_key="fused_img", meta_keys=["img_path", "fused_img", "images", "sample_idx"]),
    # dict(type="PackInputs", input_key="img"),
]

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + test_info_file,
        pipeline=test_pipeline,
        data_root=data_root,
        indices=5,  # Use first 5 samples for testing
    ),
)

test_evaluator = val_evaluator


custom_hooks = []
if results_vis_dir is not None:
    hook_config = dict(type="ResultVisualizationHook", results_vis_dir=results_vis_dir)
    if data_root is not None:
        hook_config["data_root"] = data_root
    custom_hooks.append(hook_config)


vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
