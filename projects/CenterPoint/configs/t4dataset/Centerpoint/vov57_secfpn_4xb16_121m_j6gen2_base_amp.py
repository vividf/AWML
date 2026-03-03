_base_ = ["./second_secfpn_4xb16_121m_j6gen2_base_amp.py"]

# VoVNet V-57-eSE backbone variant (last 3 stages: stage3, stage4, stage5)
# Lighter than V-99; uses stages 3–5 only (no high-res stage2) for lower latency.
experiment_name = "vov57_secfpn_4xb16_121m_j6gen2_base_amp"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

# VoVNet57 uses smaller batch size due to backbone memory
train_batch_size = 4
train_dataloader = dict(batch_size=train_batch_size)

custom_imports = dict(imports=_base_.custom_imports["imports"].copy(), allow_failed_imports=False)
custom_imports["imports"] = [x for x in custom_imports["imports"] if "mlflow" not in x]


lr = 3e-4

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
clip_grad = dict(max_norm=1.0, norm_type=2)


# BEVVoVNet V-57-eSE with last 3 stages only (stage3, stage4, stage5):
#   Stage3: 512ch @ 510x510 → upsample_stride=1   → 128ch @ 510x510
#   Stage4: 768ch @ 255x255 → upsample_stride=2   → 128ch @ 510x510
#   Stage5: 1024ch @ 255x255 (no MaxPool) → upsample_stride=2 → 128ch @ 510x510
#   Concat → 384ch @ 510x510 (no alignment needed)
model = dict(
    pts_backbone=dict(
        _delete_=True,
        type="BEVVoVNet",
        spec_name="V-57-eSE",
        input_ch=32,
        stem_strides=(1, 1, 1),
        out_features=("stage3", "stage4", "stage5"),
        frozen_stages=-1,
        norm_eval=False,
        no_pool_stages=(5,),  # stage5 keeps 255x255; neck uses [1,2,2] → 510
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
