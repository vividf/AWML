_base_ = [
    "mmpretrain::_base_/default_runtime.py",
    "mmpretrain::_base_/models/resnet18.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
]


import autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset

custom_imports = dict(
    imports=[
        "autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform",
        "autoware_ml.calibration_classification.hooks.result_visualization_hook",
    ],
    allow_failed_imports=False,
)

batch_size = 8
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

data_root = "/workspace/data/t4dataset"

train_pipeline = [
    dict(type="CalibrationClassificationTransform", data_root=data_root),
    dict(type="PackInputs", input_key="img"),
]


info_directory_path = "/workspace/data/t4dataset/calibration_info_new/"
train_info_file = f"t4dataset_x2_calib_infos_train.pkl"
val_info_file = f"t4dataset_x2_calib_infos_val.pkl"
test_info_file = f"t4dataset_x2_calib_infos_test.pkl"
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
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
    dict(type="CalibrationClassificationTransform", validation=True, debug=True, data_root=data_root),
    dict(type="PackInputs", input_key="img"),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=False,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + val_info_file,
        pipeline=val_pipeline,
        data_root=data_root,
    ),
)

val_evaluator = dict(topk=(1,), type="mmpretrain.evaluation.Accuracy")

test_pipeline = [
    dict(type="CalibrationClassificationTransform", test=True, debug=True, data_root=data_root),
    dict(type="PackInputs", input_key="img", meta_keys=["img_path", "input_data", "images", "sample_idx"]),
]

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=False,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + test_info_file,
        pipeline=test_pipeline,
        data_root=data_root,
        max_samples=5,  # Only use the first 5 samples for testing
    ),
)

test_evaluator = val_evaluator

# Register the custom hook
debug = True

custom_hooks = []
if debug:
    custom_hooks.append(dict(type="ResultVisualizationHook", save_dir="./projection_vis_origin/", data_root=data_root))

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
