_base_ = [
    "pillar_020_convnext_small_secfpn_4xb8_121m_j6gen2_base_rfs.py",
]

# user setting
experiment_name = "pillar_020_convnext_small_secfpn_4xb8_121m_j6gen2_base_rfs_bf16"
experiment_group_name = "centerpoint/" + _base_.dataset_type
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# Switch from FP32 to BF16 to avoid overflow/NaN issues while keeping stable training
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=_base_.optim_wrapper.optimizer,
    clip_grad=_base_.optim_wrapper.clip_grad,
)
