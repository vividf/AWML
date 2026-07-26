"""
BEVFusion-L Deployment Configuration — split FP16, trainStation/DDS removal.

Same FP16 split (sparse + dense) deployment as ``deploy_config.py``, with the sparse graph's
data-dependent-shape boundaries removed: the 4 down-sampling ``GetIndicePairsImplicitGemm`` nodes
are replaced by 16 precomputed-rulebook graph inputs, so TensorRT no longer has to sync the active
voxel count back to the host and no longer splits the engine into ``[trainStationN]`` segments.

**A/B measurement.** Set ``spconv_remove_trainstation = False`` (and point ``export.work_dir`` at a
second directory) to build the baseline engine. When comparing, remember that the rulebook
precompute moves out of the engine and is therefore *not* counted in the reported ``sparse_ms``
CUDA-event window — see ``io/sparse_rulebook_inputs.py``.

**Runtime requirement.** A stripped engine only runs on a runtime that precomputes and binds the
rulebooks: AWML's TensorRT evaluation (this framework) and autoware_bevfusion. The stage geometry
is embedded in the exported ONNX ``metadata_props["rulebook_stages"]``, so the on-board runtime
needs no hard-coded constants.

Layout (single file, grouped by concern; mirrors deploy_config_split_sparse_fp16_dense_int8_2_8.py):
  1. SHARED VALUES  - single source of truth reused across sections (paths, devices, shapes).
  2. EXPORT         - spconv/export flags, ONNX/TensorRT build settings, components.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

CLI (model config comes from `model_cfg` below; a second positional overrides)::

    python -m deployment.cli.main bevfusion_l deployment/projects/bevfusion_l/config/deploy_config_split_fp16_remove_trainstation.py
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
# ============================================================================

model_cfg = "projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m_t4metric_v2.py"
checkpoint_path = "work_dirs/bevfusion/bevfusion_2_8/best_epoch_25.pth"

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
_CUDA = devices["cuda"]

_WORK_DIR = "work_dirs/bevfusion_deployment_2_8_remove_trainstation"
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# Dense head BEV feature map: [batch, channels, grid_h, grid_w] = grid_size // out_size_factor.
_LIDAR_BEV_SHAPE = [1, 256, 180, 180]

# Sparse (spconv) voxel input profile: [num_voxels, max_points_per_voxel, voxel_feature_dim].
# The voxel-count envelope below also bounds the rulebook inputs' active-voxel count (the
# per-stage profile entries are derived from `coors` by BEVFusionDeploymentConfig).
_MAX_POINTS_PER_VOXEL = 32
_VOXEL_FEATURE_DIM = 5
_VOXELS_OPT = 64000
_VOXELS_MAX = 256000

# ============================================================================
# 2. EXPORT
# ============================================================================

# Bake the pair-mask argsort into GetIndicePairsImplicitGemm.do_sort_i at ONNX symbolic export.
# The runtime rulebook precompute reads the same process-global, so engine and precompute agree.
spconv_do_sort = False

# Remove the sparse graph's data-dependent-shape (trainStation) boundaries: delete the 4
# down-sampling GetIndicePairsImplicitGemm nodes and promote their rulebook outputs to graph inputs
# (post-export transform in export/onnx_remove_trainstation_dds.py). The 16 TensorRT profile
# entries are derived from the `coors` profile, so nothing else in this file changes.
# - True : stripped graph; requires a rulebook-precomputing runtime (see module docstring).
# - False: stock graph with in-graph rulebook generation (the A/B baseline).
spconv_remove_trainstation = True

# Sparse ONNX postprocess: fuse each `ImplicitGemm -> Relu` (and `Add(const)+Relu`) into an
# activated ImplicitGemm.
spconv_fuse_implicit_gemm_relu = True

# Fuse SparseConv + BN in `pts_middle_encoder` before ONNX export.
fuse_spconv_bn = True

# Export mode: "onnx", "trt", "both", "none".
export = dict(
    mode="both",
    work_dir=_WORK_DIR,
    onnx_path=_ONNX_DIR,
)

# Keep the split component definitions but also emit a single merged full-graph ONNX/engine.
# The merged graph inherits the sparse component's TensorRT profile, rulebook entries included.
bevfusion_merge = dict(
    enabled=True,
    onnx_file="bevfusion_lidar.onnx",
    engine_file="bevfusion_lidar.engine",
)

# ONNX export settings (shared across all components).
onnx_config = dict(
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# TensorRT build settings (shared across all components). plugin_libraries loads the spconv
# ImplicitGemm plugin before engine build/deserialize.
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)

# Split components (must keep keys `bevfusion_sparse` + `bevfusion_dense`; the merged full graph
# is derived from this pair by BEVFusionDeploymentConfig when bevfusion_merge is enabled).
#
# The rulebook graph inputs are intentionally absent from `io`: they exist only after the
# post-export transform, so they are neither traced nor named by torch.onnx.export. Their TensorRT
# profile entries are derived in component_layout.add_rulebook_input_profiles().
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
