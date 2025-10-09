codebase_config = dict(type="mmdet3d", task="VoxelDetection", model_type="end2end")

custom_imports = dict(
    imports=[
        "projects.BEVFusion.deploy",
        "projects.BEVFusion.bevfusion",
        "projects.SparseConvolution",
    ],
    allow_failed_imports=False,
)

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(min_shape=[1, 10, 4], opt_shape=[64000, 10, 4], max_shape=[256000, 10, 4]),
                coors=dict(min_shape=[1, 3], opt_shape=[64000, 3], max_shape=[256000, 3]),
                num_points_per_voxel=dict(min_shape=[1], opt_shape=[64000], max_shape=[256000]),
                image_feats=dict(min_shape=[80, 180, 180], opt_shape=[80, 180, 180], max_shape=[80, 180, 180]),
            )
        )
    ],
)

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file="main_body.onnx",
    input_names=["voxels", "coors", "num_points_per_voxel", "image_feats"],
    output_names=["bbox_pred", "score", "label_pred"],
    dynamic_axes={
        "voxels": {
            0: "voxels_num",
        },
        "coors": {
            0: "voxels_num",
        },
        "num_points_per_voxel": {
            0: "voxels_num",
        },
    },
    input_shape=None,
    verbose=True,
)
