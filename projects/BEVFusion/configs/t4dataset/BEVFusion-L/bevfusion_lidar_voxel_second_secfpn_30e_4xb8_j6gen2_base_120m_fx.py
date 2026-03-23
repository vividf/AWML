# FX-traceable sparse encoder for spconv INT8 (prepare_fx/convert_fx).
# Same as base 120m but pts_middle_encoder uses block_type="basicblock_fx"
# so residual blocks use SparseReLU and (out+identity) and can be traced by torch.fx.
_base_ = "./bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py"

model = dict(
    pts_middle_encoder=dict(
        block_type="basicblock_fx",
    ),
)
