_base_ = ["./second_secfpn_4xb16_121m_base_amp.py"]

# ResNet34 backbone variant - only override what differs from SECOND base
experiment_name = "resnet34_secfpn_4xb16_121m_base_amp"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# ResNet34 uses smaller batch size (8 vs 16) due to larger backbone memory
train_batch_size = 8
train_dataloader = dict(batch_size=train_batch_size)

# Disable MLflow backend (optional - uncomment in base if needed)
custom_imports = dict(imports=_base_.custom_imports["imports"].copy(), allow_failed_imports=False)
custom_imports["imports"] = [x for x in custom_imports["imports"] if "mlflow" not in x]

# ResNet34-specific: lower LR and gentler scheduler for mixed precision stability
lr = 0.0001
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
clip_grad = dict(max_norm=1.0, norm_type=2)  # max norm of gradients upper bound to be 15 since amp is used

param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.3),
        eta_min=lr * 10,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=lr * 1e-4,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.3),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=1,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# Replace SECOND backbone with ResNet34 (BEVResNet)
# eps=1e-5 for numerical stability in mixed precision
model = dict(
    pts_backbone=dict(
        _delete_=True,
        type="BEVResNet",  # Use custom BEV-friendly ResNet wrapper (renamed to avoid confusion)
        depth=34,
        num_stages=3,
        strides=(1, 2, 2),  # ResNet stage strides: stage0=1, stage1=2, stage2=2
        dilations=(1, 1, 1),  # Dilation for each stage
        out_indices=(0, 1, 2),  # Get features from res_layers 0, 1, 2
        # BEV-friendly stem configuration: no downsampling at input
        deep_stem=True,  # Use three 3x3 convs instead of 7x7: more efficient and better boundary behavior
        conv1_stride=1,  # First conv stride=1 (no downsampling) - applies to deep_stem's first 3x3 conv
        with_pool=False,  # Disable maxpool (no downsampling)
        # pool_stride is only used when with_pool=True, so omitted here
        frozen_stages=-1,  # Don't freeze any stages initially
        base_channels=64,  # ResNet34 outputs: 64, 128, 256 channels (64*1, 64*2, 64*4)
        # ResNet34 uses BasicBlock (expansion=1), so base_channels=64 gives [64, 128, 256]
        norm_cfg=dict(
            type="BN", eps=1e-5, momentum=0.01
        ),  # Fixed: eps changed from 1e-3 to 1e-5 for numerical stability
        norm_eval=False,  # Keep BN in training mode for better performance
        # Remove pretrained weights due to input channel mismatch (3 vs 32)
        # init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),
        style="pytorch",
        in_channels=32,
        # pretrained=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/resnet_34/resnet34_8xb32_mmcls.pth",
            prefix="backbone.",  # Often needed to map keys correctly
        ),
        with_cp=True,
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[
            64,
            128,
            256,
        ],  # ResNet34 layers 0, 1, 2: 64, 128, 256 channels (base_channels=64 * expansion=1 for BasicBlock)
        # Same as SECOND backbone: [64, 128, 256]
        out_channels=[128, 128, 128],
        # BEV-friendly: With conv1_stride=1 and no maxpool, outputs should be:
        # stage0: (1020, 1020) -> downsample stride=0.5 -> (510, 510)
        # stage1: (510, 510) -> upsample stride=1 -> (510, 510)
        # stage2: (255, 255) -> upsample stride=2 -> (510, 510)
        # Final output: (510, 510) to match target size (grid_size // out_size_factor)
        upsample_strides=[0.5, 1, 2],  # Upsample to match target feature map size (510, 510)
        norm_cfg=dict(
            type="BN", eps=1e-5, momentum=0.01
        ),  # Fixed: eps changed from 0.001 (1e-3) to 1e-5 for numerical stability
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=optimizer,
    clip_grad=clip_grad,
    # Update it accordingly
    loss_scale={
        "init_scale": 2.0**8,  # intial_scale: 256
        "growth_interval": 2000,
    },
)
