"""
BEVFusion deploy config — **split ONNX / TensorRT (route 1)**

Use this instead of ``deploy_config.py`` when you want:
  - ``bevfusion_sparse.onnx`` / ``.engine`` — voxelization stays outside; graph is
    ``pts_middle_encoder`` only (spconv / plugins / libspconv).
  - ``bevfusion_dense.onnx`` / ``.engine`` — ``pts_backbone`` + ``pts_neck`` + ``bbox_head``
    (+ same postprocess as single-file export). Suitable for plain TensorRT without spconv ops.

**Requirements**
  - LiDAR-only model: ``fusion_layer is None`` and ``img_backbone is None``.
  - Adjust ``tensorrt_profile`` for ``lidar_bev`` channel count / BEV size to match your checkpoint
    (defaults assume ``C=256``, ``H=W=180`` at neck output before head align).

**PTQ / spconv INT8 checkpoint** 請改用：``deploy_config_split_int8.py``（含 ``quantization`` 區塊）。

CLI::

    python -m deployment.cli.main bevfusion \\
        deployment/projects/bevfusion/config/deploy_config_split.py \\
        <your_model_cfg.py>
"""

# ============================================================================
# Checkpoint Path
# ============================================================================
checkpoint_path = "work_dirs/bevfusion/epoch_30.pth"

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",
    work_dir="work_dirs/bevfusion_deployment_split",
    onnx_path=None,
)

_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Split components (must keep keys ``bevfusion_sparse`` + ``bevfusion_dense``)
# ============================================================================
components = dict(
    bevfusion_sparse=dict(
        onnx_file="bevfusion_sparse.onnx",
        engine_file="bevfusion_sparse.engine",
        io=dict(
            inputs=[
                dict(name="voxels", dtype="float32"),
                dict(name="coors", dtype="int32"),
                dict(name="num_points_per_voxel", dtype="int32"),
            ],
            outputs=[
                dict(name="lidar_bev", dtype="float32"),
            ],
            dynamic_axes={
                "voxels": {0: "voxels_num"},
                "coors": {0: "voxels_num"},
                "num_points_per_voxel": {0: "voxels_num"},
                "lidar_bev": {0: "batch", 2: "bev_h", 3: "bev_w"},
            },
        ),
        tensorrt_profile=dict(
            voxels=dict(
                min_shape=[1, 10, 5],
                opt_shape=[64000, 10, 5],
                max_shape=[256000, 10, 5],
            ),
            coors=dict(
                min_shape=[1, 3],
                opt_shape=[64000, 3],
                max_shape=[256000, 3],
            ),
            num_points_per_voxel=dict(
                min_shape=[1],
                opt_shape=[64000],
                max_shape=[256000],
            ),
        ),
    ),
    bevfusion_dense=dict(
        onnx_file="bevfusion_dense.onnx",
        engine_file="bevfusion_dense.engine",
        io=dict(
            inputs=[
                dict(name="lidar_bev", dtype="float32"),
            ],
            outputs=[
                dict(name="bbox_pred", dtype="float32"),
                dict(name="score", dtype="float32"),
                dict(name="label_pred", dtype="int64"),
            ],
            dynamic_axes={
                "lidar_bev": {0: "batch", 2: "bev_h", 3: "bev_w"},
            },
        ),
        # Channel (axis 1) is STATIC in ONNX (= sparse encoder output = SECOND in_channels).
        # TRT requires min==opt==max for static dims; only H/W may use a range.
        tensorrt_profile=dict(
            lidar_bev=dict(
                min_shape=[1, 256, 32, 32],
                opt_shape=[1, 256, 180, 180],
                max_shape=[1, 256, 2048, 2048],
            ),
        ),
    ),
)

runtime_io = dict(
    info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
    sample_idx=0,
)

onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)

evaluation = dict(
    enabled=True,
    num_samples=5,
    num_warmup_samples=2,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=False,
            device=devices["cuda"],
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

verification = dict(
    enabled=False,
    tolerance=1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[],
        onnx=[],
        trt=[],
        none=[],
    ),
)
