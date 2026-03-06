_base_ = [
    "vov57_downsample_stem_secfpn_4xb16_121m_j6gen2_base_amp.py",
]

# user setting
experiment_name = "vov57_downsample_stem_secfpn_4xb16_121m_j6gen2_base_amp_rfs_bf16"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

train_frame_object_sampler = dict(
    type="FrameObjectSampler",
    object_samplers=[
        dict(
            type="ObjectBEVDistanceSampler",
            bev_distance_thresholds=[
                _base_.point_cloud_range[0],
                _base_.point_cloud_range[1],
                _base_.point_cloud_range[3],
                _base_.point_cloud_range[4],
            ],
        ),
        dict(
            type="LowPedestriansObjectSampler",
            height_threshold=1.5,
            bev_distance_thresholds=[
                -50.0,
                -50.0,
                50.0,
                50.0,
            ],
        ),
    ],
)

train_dataloader = dict(
    sampler=dict(type="DistributedWeightedRandomSampler", shuffle=True),
    dataset=dict(
        type="T4FrameSamplerDataset",
        repeat_sampling_factor=0.30,
        frame_object_sampler=train_frame_object_sampler,
    ),
)


# Switch from FP16 to BF16 to avoid loss scaler collapse
# BF16 has larger dynamic range (8-bit exponent vs FP16's 5-bit), preventing overflow/NaN issues
# A100 GPUs natively support BF16
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=_base_.optim_wrapper.optimizer,
    clip_grad=_base_.optim_wrapper.clip_grad,
)
