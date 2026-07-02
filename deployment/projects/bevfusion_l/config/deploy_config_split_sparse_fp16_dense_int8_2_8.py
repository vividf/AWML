checkpoint_path = "work_dirs/bevfusion/bevfusion_2_8/best_epoch_25_ptq.pth"
# checkpoint_path = "vivid/bench_comparison/bevfusion_2_7/best_epoch_28.pth"


spconv_do_sort = False

# ============================================================================
# Sparse ONNX transform: fuse ImplicitGemm with trailing Relu / Add(const)+Relu.
# ----------------------------------------------------------------------------
# Consumed by ``sparse_int8_onnx_transform --deploy-cfg ...``.
# - True  (default): run ONNX fusion passes and bake act_type / merged bias.
# - False          : keep explicit Relu/Add nodes (debug / ablation).
# ============================================================================
spconv_fuse_implicit_gemm_relu = True

# Fuse SparseConv + BN in ``pts_middle_encoder`` before ONNX export (same as PTQ load path).
# Aligns FP16 sparse subgraph with INT8 deploy / fair latency vs mAP comparison.
fuse_spconv_bn = True


quantization = dict(
    enabled=True,
    ptq_checkpoint=True,
    fuse_bn=True,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=False,
    spconv_int8=False,
    sensitive_layers=[],
)

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",
    work_dir="work_dirs/bevfusion_deployment_2_8_dense_int8",
    onnx_path="work_dirs/bevfusion_deployment_2_8_dense_int8/onnx",
)


# Optional: keep split component definitions for debugging, but export/eval as one main body.
# - False: split sparse+dense ONNX/engine (default)
# - True : one ONNX + one engine + one backend pipeline
bevfusion_merge = dict(
    enabled=True,
    onnx_file="bevfusion_lidar.onnx",
    engine_file="bevfusion_lidar.engine",
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
                min_shape=[1, 32, 5],
                opt_shape=[64000, 32, 5],
                max_shape=[256000, 32, 5],
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
            # Spatial dims must stay at head BEV resolution; see module docstring.
            dynamic_axes={
                "lidar_bev": {0: "batch"},
            },
        ),
        # H,W fixed to head grid (grid_size // out_size_factor). Widen only batch dim if needed.
        tensorrt_profile=dict(
            lidar_bev=dict(
                min_shape=[1, 256, 180, 180],
                opt_shape=[1, 256, 180, 180],
                max_shape=[1, 256, 180, 180],
            ),
        ),
    ),
)

runtime_io = dict(
    info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
    sample_idx=0,
)

onnx_config = dict(
    # BEVFusion 2.8.x exports at opset 18 (matches projects/BEVFusion/configs/deploy/*_tensorrt_dynamic.py).
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    # Promote Q/DQ scale / zero_point Constants to named initializers and annotate the
    # QuantizeLinear/DequantizeLinear node names with [s=..|z=..] so the INT8 scales are
    # visible in the exported ONNX (runs make_qdq_readable). Without this, do_constant_folding
    # inlines them and the Q/DQ nodes show no x_scale / x_zero_point. Matches the old
    # deploy_config_split_sparse_fp16_dense_int8_merge.py.
    visualize_qdq_values=True,
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
    num_warmup=2,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=False,
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
