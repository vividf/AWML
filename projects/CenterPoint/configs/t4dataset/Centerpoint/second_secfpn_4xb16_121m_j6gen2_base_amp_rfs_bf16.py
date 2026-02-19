_base_ = [
    "second_secfpn_4xb16_121m_j6gen2_base_amp_rfs.py",
]

# user setting
experiment_name = "second_secfpn_4xb16_121m_j6gen2_base_amp_rfs_bf16"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

# Switch from FP16 to BF16 to avoid loss scaler collapse
# BF16 has larger dynamic range (8-bit exponent vs FP16's 5-bit), preventing overflow/NaN issues
# A100 GPUs natively support BF16
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=_base_.optim_wrapper.optimizer,
    clip_grad=_base_.optim_wrapper.clip_grad,
)
