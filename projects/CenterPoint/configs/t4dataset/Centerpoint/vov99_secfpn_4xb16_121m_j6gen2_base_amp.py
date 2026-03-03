_base_ = ["./second_secfpn_4xb16_121m_j6gen2_base_amp.py"]

# VoVNet V-99-eSE backbone variant
experiment_name = "vov99_secfpn_4xb16_121m_j6gen2_base_amp"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

# VoVNet99 uses smaller batch size due to larger backbone memory
train_batch_size = 2
train_dataloader = dict(batch_size=train_batch_size)

custom_imports = dict(imports=_base_.custom_imports["imports"].copy(), allow_failed_imports=False)
custom_imports["imports"] = [x for x in custom_imports["imports"] if "mlflow" not in x]


lr = 3e-4

# TODO(vividf): modify if needed
# param_scheduler = [
#     # learning rate scheduler
#     # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
#     # during the next epochs, learning rate decreases from lr * 10 to
#     # lr * 1e-4
#     dict(
#         type="CosineAnnealingLR",
#         T_max=8,
#         eta_min=lr * 10,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         type="CosineAnnealingLR",
#         T_max=22,
#         eta_min=lr * 1e-4,
#         begin=8,
#         end=max_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     # momentum scheduler
#     # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
#     # during the next epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type="CosineAnnealingMomentum",
#         T_max=8,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         type="CosineAnnealingMomentum",
#         T_max=22,
#         eta_min=1,
#         begin=8,
#         end=max_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
# ]


# TODO(vividf): modify if needed
# param_scheduler = [
#     # learning rate scheduler
#     # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
#     # during the next epochs, learning rate decreases from lr * 10 to
#     # lr * 1e-4
#     dict(
#         type="CosineAnnealingLR",
#         T_max=8,
#         eta_min=lr * 10,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         type="CosineAnnealingLR",
#         T_max=22,
#         eta_min=lr * 1e-4,
#         begin=8,
#         end=max_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     # momentum scheduler
#     # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
#     # during the next epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type="CosineAnnealingMomentum",
#         T_max=8,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         type="CosineAnnealingMomentum",
#         T_max=22,
#         eta_min=1,
#         begin=8,
#         end=max_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
# ]

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
clip_grad = dict(max_norm=1.0, norm_type=2)  # max norm of gradients upper bound to be 15 since amp is used


# Replace SECOND backbone with VoVNet V-99-eSE (BEVVoVNet)
# Use stage3, stage4, stage5 (last 3 stages) → SECONDFPN; stage5 has no MaxPool.
#   Stage3: 512ch @ 510x510   → upsample_stride=1   → 128ch @ 510x510
#   Stage4: 768ch @ 255x255   → upsample_stride=2   → 128ch @ 510x510
#   Stage5: 1024ch @ 255x255  (no_pool_stages=(5,)) → upsample_stride=2 → 128ch @ 510x510
#   Concat → 384ch @ 510x510 (no alignment needed)
model = dict(
    pts_backbone=dict(
        _delete_=True,
        type="BEVVoVNet",
        spec_name="V-99-eSE",
        input_ch=32,
        stem_strides=(1, 1, 1),
        out_features=("stage3", "stage4", "stage5"),
        frozen_stages=-1,
        norm_eval=False,
        no_pool_stages=(5,),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[512, 768, 1024],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=optimizer,
    clip_grad=clip_grad,
    loss_scale={
        "init_scale": 2.0**8,
        "growth_interval": 2000,
    },
)


load_from = None

activation_checkpointing = None
