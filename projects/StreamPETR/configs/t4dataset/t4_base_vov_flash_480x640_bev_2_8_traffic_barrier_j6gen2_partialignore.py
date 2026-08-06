# Train StreamPETR DIRECTLY on the j6gen2 subset, skipping the two-stage
# base->finetune flow: nuScenes VoV-99 pretrain -> j6gen2, with the same
# 2_8-style recipe (partial-ignore + 2D aux head from the baseline parent,
# batch_size 8, lr 5e-5 auto-scaled by total_batch/8).
#
# Compared to the two-stage flow (base training on the full T4 base DB, then
# j6gen2 fine-tune) this sees far less data variety - expect weaker
# generalization; it is the "quick single-stage j6gen2 experiment" config.
#
# Run (single GPU / multi GPU):
#   python tools/detection3d/train.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py
#   bash tools/detection3d/dist_script.sh projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py 2 train
_base_ = [
    "../default/vov_flash_480x640_baseline.py",
]

# Init from the nuScenes model-zoo pretrain (prefetch once to avoid a long
# 0%-GPU download at startup):
#   mkdir -p pretrained && wget -c -O pretrained/nuscenes_vov99_baseline_320x800.pth \
#     'https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth'
# For a TRUE from-scratch run (random init - NOT recommended: the recipe
# relies on the depth-pretrained VoVNet), set:  load_from = None
load_from = "pretrained/nuscenes_vov99_baseline_320x800.pth"

info_directory_path = "info/kokseang_2_8/"
data_root = "data/t4datasets/"

batch_size = 8
num_workers = 32

num_epochs = 10
val_interval = 2

info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

dataset_test_groups = dict(
    _delete_=True,
    j6gen2=(info_test_file_name, True),
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    dataset=dict(
        ann_file=info_directory_path + info_train_file_name,
        data_root=data_root,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_val_file_name,
        data_root=data_root,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_test_file_name,
        data_root=data_root,
    ),
)

val_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_val_file_name)
test_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_test_file_name)

train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)

lr = 5e-5
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)

optim_wrapper = dict(
    type="NoCacheAmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    loss_scale="dynamic",
    clip_grad=dict(max_norm=1, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-4,
    ),
]

auto_scale_lr = dict(base_batch_size=8, enable=True)
