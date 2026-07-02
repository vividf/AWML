"""Shared base for BEVFusion split-INT8 deploy configs (route 1 + spconv INT8).

Holds the variant-independent blocks (sparse+dense components, tensorrt_config, runtime_io,
spconv graph knobs). Variant files inherit via ``_base_`` and define only what differs
(checkpoint_path, quantization, export, onnx_config, spconv_int8_fp16_layers, devices,
evaluation, verification). Do NOT reference these vars by name from a child config — mmengine
execs each config in its own namespace.
"""

# ============================================================================
# Split components（與 deploy_config_split_fp16_opt_2_8.py 相同結構）
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
                "lidar_bev": {0: "batch"},
            },
        ),
        # H,W must match bbox_head BEV (grid_size // out_size_factor); see deploy_config_split_fp16_opt_2_8.py header.
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


# ============================================================================
# Sparse pair-gen: skip the pair-mask argsort for INT8 inference.
# ----------------------------------------------------------------------------
# Matches New3D's ``bool do_sort = !int8_inference_;`` (sparseConvImplicit.cu:368)
# and traveller59/spconv's INT8 guide. Baked into the ONNX graph as
# ``autoware::GetIndicePairsImplicitGemm.do_sort_i=0`` at export time, so the
# INT8 engine keeps sort **off** regardless of runtime. FP16 deploy_configs
# should leave this unset (default ``True``) so FP16 engines still sort.
# - True  (default) : run sort; required for FP16.
# - False           : skip sort; INT8 recommendation (used here).
# See deployment/projects/bevfusion/docs/15_README_AWML_SPCONV_INT8_ACCEL_PLAN.md §10.9.
# ============================================================================
spconv_do_sort = False


# ============================================================================
# Sparse ONNX transform: fuse ImplicitGemm with trailing Relu / Add(const)+Relu.
# ----------------------------------------------------------------------------
# Consumed by ``sparse_int8_onnx_transform --deploy-cfg ...``.
# - True  (default): run ONNX fusion passes and bake act_type / merged bias.
# - False          : keep explicit Relu/Add nodes (debug / ablation).
# ============================================================================
spconv_fuse_implicit_gemm_relu = True


tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    # INT8 sparse conv now lives in the single Autoware plugin library (ImplicitGemm with
    # precision=1); no separate libimplicit_gemm_int8_plugin.so is needed.
    plugin_libraries=[
        "/opt/plugins/libautoware_tensorrt_plugins.so",
    ],
)
