"""
BEVFusion-L Deployment Configuration — split sparse (INT8) + dense (INT8), merged (PTQ).

Loads a PTQ checkpoint and deploys BOTH the dense tower (backbone/neck/head, pytorch_quantization)
and the sparse encoder (spconv INT8, ``quantization.spconv_int8=True``) in INT8. Layers listed in
``spconv_int8_fp16_layers`` are kept FP16 for accuracy recovery.

Layout (single file, grouped by concern; mirrors centerpoint/config/deploy_config.py):
  1. SHARED VALUES  - single source of truth reused across sections (paths, devices, shapes).
  2. EXPORT         - spconv/export flags, quantization, ONNX/TensorRT build settings, components.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

Only the top-level names `checkpoint_path`, `devices`, `export`, `components`,
`onnx_config`, `tensorrt_config`, `evaluation`, `verification`, `quantization`,
`spconv_int8_fp16_layers`, plus the BEVFusion-only flags `spconv_do_sort`,
`spconv_fuse_implicit_gemm_relu`, `fuse_spconv_bn`, `bevfusion_merge`, are read (by
`BaseDeploymentConfig` / `BEVFusionDeploymentConfig`). Names prefixed with `_` are local
single-source helpers and are intentionally not consumed directly.

CLI::

    python -m deployment.cli.main bevfusion_l \\
        deployment/projects/bevfusion_l/config/deploy_config_split_sparse_int8_dense_int8_2_8.py \\
        <your_model_cfg.py>
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
# ============================================================================

# Checkpoint - single source of truth for the PyTorch model (PTQ .pth: dense _amax + sparse _amax).
checkpoint_path = "work_dirs/bevfusion/bevfusion_2_8/best_epoch_25_ptq_sparse_int8_dense_int8.pth"
# checkpoint_path = "vivid/bench_comparison/bevfusion_2_7/best_epoch_28.pth"

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
# Alias reused by the per-backend evaluation settings below so the CUDA device is written once.
_CUDA = devices["cuda"]

# Deployment output layout. _ONNX_DIR / _TENSORRT_DIR are the single source for both the export
# outputs and the evaluation backends' model_dir / engine_dir (kept in sync here).
_WORK_DIR = "work_dirs/bevfusion_deployment_2_8_sparse_int8_dense_int8"
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# Dense head BEV feature map: [batch, channels, grid_h, grid_w] = grid_size // out_size_factor.
# Fully static (min == opt == max) so constant folding can drop the head's shape-glue and the
# split graph's node count matches the monolithic export.
_LIDAR_BEV_SHAPE = [1, 256, 180, 180]

# Sparse (spconv) voxel input profile: [num_voxels, max_points_per_voxel, voxel_feature_dim].
_MAX_POINTS_PER_VOXEL = 32
_VOXEL_FEATURE_DIM = 5
_VOXELS_OPT = 64000
_VOXELS_MAX = 256000


# ============================================================================
# 2. EXPORT
# ============================================================================

# Bake the pair-mask argsort into GetIndicePairsImplicitGemm.do_sort_i at ONNX symbolic export.
spconv_do_sort = False

# Sparse ONNX postprocess: fuse each `ImplicitGemm -> Relu` (and `Add(const)+Relu`) into an
# activated ImplicitGemm (post-export transform in export/component_builder.py).
# - True  (default): bake activation into ImplicitGemm (act_type) / merged bias.
# - False          : keep explicit Relu/Add nodes (debug / ablation).
spconv_fuse_implicit_gemm_relu = True

# Fuse SparseConv + BN in `pts_middle_encoder` before ONNX export (same as the PTQ load path).
# Aligns the FP16 sparse subgraph with INT8 deploy for a fair latency-vs-mAP comparison.
fuse_spconv_bn = True

# Quantization: dense tower (backbone/neck/head) INT8 via pytorch_quantization AND sparse encoder
# INT8 via spconv (spconv_int8=True). ptq_checkpoint=True re-attaches the quantizer tree before
# loading the PTQ .pth so the calibrated _amax line up.
quantization = dict(
    enabled=True,
    ptq_checkpoint=True,
    fuse_bn=True,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=False,
    spconv_int8=True,
    sensitive_layers=[],
)

# Sparse-conv layers kept FP16 (skip INT8 replacement) for accuracy recovery. Matched as
# case-insensitive substrings against the exported ImplicitGemm node names. Must match the PTQ run
# so the reloaded module tree lines up with the checkpoint. Uncomment more to widen the FP16 set.
spconv_int8_fp16_layers = [
    "conv_input.0",
    "encoder_layer1/encoder_layer1.0/conv1",
    "encoder_layer1/encoder_layer1.0/conv2",
    #
    "encoder_layer1/encoder_layer1.1/conv1",
    "encoder_layer1/encoder_layer1.1/conv2",
    #
    "encoder_layer1/encoder_layer1.2/encoder_layer1.2.0",
    "encoder_layer2/encoder_layer2.0/conv1",
    "encoder_layer2/encoder_layer2.0/conv2",
    #
    "encoder_layer2/encoder_layer2.1/conv1",
    "encoder_layer2/encoder_layer2.1/conv2",
    # #
    "encoder_layer2/encoder_layer2.2/encoder_layer2.2.0",
    "encoder_layer3/encoder_layer3.0/conv1",
    "encoder_layer3/encoder_layer3.0/conv2",
    # #
    # "encoder_layer3/encoder_layer3.1/conv1",
    # "encoder_layer3/encoder_layer3.1/conv2",
    # #
    # "encoder_layer3/encoder_layer3.2/encoder_layer3.2.0",
    # "encoder_layer4/encoder_layer4.0/conv1",
    # "encoder_layer4/encoder_layer4.0/conv2",
    # #
    # "encoder_layer4/encoder_layer4.1/conv1",
    # "encoder_layer4/encoder_layer4.1/conv2",
    # #
    # "conv_out/conv_out.0",
]

# Export mode: "onnx", "trt", "both", "none".
export = dict(
    mode="none",
    work_dir=_WORK_DIR,
    onnx_path=_ONNX_DIR,
)

# Keep the split component definitions but also emit a single merged full-graph ONNX/engine.
# - enabled=False: split sparse+dense ONNX/engine only.
# - enabled=True : additionally emit one merged ONNX + engine + backend pipeline.
bevfusion_merge = dict(
    enabled=True,
    onnx_file="bevfusion_lidar.onnx",
    engine_file="bevfusion_lidar.engine",
)

# ONNX export settings (shared across all components).
onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    # Promote Q/DQ scale / zero_point Constants to named initializers and annotate the
    # QuantizeLinear/DequantizeLinear node names with [s=..|z=..] so the INT8 scales are
    # visible in the exported ONNX (runs make_qdq_readable). Without this, do_constant_folding
    # inlines them and the Q/DQ nodes show no x_scale / x_zero_point.
    visualize_qdq_values=True,
    simplify=False,
)

# TensorRT build settings (shared across all components). plugin_libraries loads the spconv
# ImplicitGemm[Int8] plugin before engine build/deserialize.
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)

# Split components (must keep keys `bevfusion_sparse` + `bevfusion_dense`; the merged full graph
# is derived from this pair by BEVFusionDeploymentConfig when bevfusion_merge is enabled).
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
                min_shape=[1, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                opt_shape=[_VOXELS_OPT, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                max_shape=[_VOXELS_MAX, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
            ),
            coors=dict(
                min_shape=[1, 3],
                opt_shape=[_VOXELS_OPT, 3],
                max_shape=[_VOXELS_MAX, 3],
            ),
            num_points_per_voxel=dict(
                min_shape=[1],
                opt_shape=[_VOXELS_OPT],
                max_shape=[_VOXELS_MAX],
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
                min_shape=_LIDAR_BEV_SHAPE,
                opt_shape=_LIDAR_BEV_SHAPE,
                max_shape=_LIDAR_BEV_SHAPE,
            ),
        ),
    ),
)

# ============================================================================
# 3. EVALUATION
#    ONNX *inference* is unsupported for BEVFusion (sparse graph needs the TRT plugin).
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=-1,
    num_warmup=2,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=_CUDA,
        ),
        tensorrt=dict(
            enabled=True,
            device=_CUDA,
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

# ============================================================================
# 4. VERIFICATION
# ============================================================================
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
