"""
BEVFusion deploy config — **split ONNX / TensorRT + PTQ INT8** (路線 1 + 量化 checkpoint)

結合：
  - **路線 1**：``bevfusion_sparse`` + ``bevfusion_dense`` 兩段 ONNX / 兩顆 engine（見 ``deploy_config_split.py``）
  - **INT8**：與 ``deploy_config_int8.py`` 相同的 ``quantization``，載入 **已 PTQ 的 checkpoint**（spconv 稀疏塔 INT8 + 稠密端 pytorch_quantization）

**隔離問題時怎麼想（與 ``deploy_config_split.py`` 無關）**

  同一條 CLI、同一個 mmconfig（例如 ``*_120m.py``）、**仍用本檔**（split 路徑與 work_dir 不變），只把
  ``checkpoint_path`` 改成 **訓練 FP32 .pth**，並設 ``quantization = dict(enabled=False)``。
  若此時 **mAP 正常**，則 **split / voxel preprocess / eval 管線沒壞**；mAP 掉在 **PTQ .pth 或
  quantization 載入**（含 dense Q/DQ、spconv INT8、key 與 mmconfig 是否一致）。
  稀疏校準使用 **完整 voxel** 不裁切（與 Lidar AI Solution 一致）。

**前置條件**
  - ``checkpoint_path`` 指向 ``bevfusion_quantization.py ptq`` 產生的 **.pth**；``quantization.ptq_checkpoint=True``（本檔預設已開）。
  - 稀疏塔：**主幹 INT8**（含 ``conv_out``）；本檔與 ``deploy_config_split_int8.py`` 的差異在
    ``spconv_int8_fp16_layers``（此處列多層 FP ``ImplicitGemm`` 做對照實驗）。變更後請 **重跑 PTQ**。
**稀疏段 ONNX**
  - 匯出的 ``bevfusion_sparse.onnx`` 為浮點 ``ImplicitGemm``；Path B 以 ``sparse_int8_onnx_transform.py`` 換成 INT8 plugin 圖。

**CLI**（需在含 pytorch-quantization 的環境；Docker 內用 ``pip install --no-cache-dir --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple pytorch-quantization==2.1.3``，詳見 ``deploy_config_int8.py`` 註解）::

    python -m deployment.cli.main bevfusion \\
        deployment/projects/bevfusion/config/deploy_config_split_int8.py \\
        projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \\
        --module main_body

產生 PTQ checkpoint 範例（與 int8 單檔相同）::

    python deployment/quantization/bevfusion_quantization.py ptq \\
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \\
        --checkpoint work_dirs/bevfusion/epoch_30.pth \\
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \\
        --calibrate-samples 256 --batch-size 1 --calib-seed 0 \\
        --output work_dirs/bevfusion/epoch_30_ptq2.pth
"""

# ============================================================================
# Checkpoint
# ============================================================================
# PTQ：必須為 bevfusion_quantization.py 產物。FP32 對照：改為訓練 .pth 並改用下方 Preset B。
# checkpoint_path = "work_dirs/bevfusion/bevfusion_epoch_30_ptq.pth"

# ============================================================================
# Quantization — Preset A：INT8 PTQ（預設）
# ============================================================================
# quantization = dict(
#     enabled=True,
#     ptq_checkpoint=True,
#     fuse_bn=True,
#     quant_backbone=True,
#     quant_neck=True,
#     quant_head=True,
#     quant_add=False,
#     spconv_int8=True,
#     sensitive_layers=[],
# )

# ============================================================================
# Preset B：同 split_int8 路徑 + mmconfig，僅 FP32 驗證 mAP / 管線（註解掉 Preset A 後啟用）
# ============================================================================
# checkpoint_path = "work_dirs/bevfusion/bevfusion_epoch_30.pth"
# quantization = dict(enabled=False)

# ============================================================================
# Preset C：僅稀疏塔 INT8（隔離 dense Q/DQ 是否導致 mAP≈0）
#
# PTQ（需 deploy 內 spconv_int8=True，並加 --sparse-int8-only）::
#
#   python deployment/quantization/bevfusion_quantization.py ptq \\
#       --config .../bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \\
#       --checkpoint work_dirs/bevfusion/bevfusion_epoch_30.pth \\
#       --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py \\
#       --sparse-int8-only \\
#       --calibrate-samples 256 --batch-size 1 --calib-seed 0 \\
#       --output work_dirs/bevfusion/bevfusion_epoch_30_ptq_sparse_only.pth
#
# 評測：註解 Preset A，改為下方三項（checkpoint 指向上列 output；dense 三關必須 False 與 PTQ 一致）
# ============================================================================
# checkpoint_path = "work_dirs/bevfusion/best_epoch_28_ptq_sparse_int8_dense_int8.pth"
checkpoint_path = "vivid/bench_comparison/bevfusion_2_7/best_epoch_28_ptq_sparse_int8_dense_int8.pth"


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
# Sparse BN fold into spconv at ONNX export (float shadow encoder).
# ----------------------------------------------------------------------------
# The PTQ checkpoint is already BN-folded (no sparse BN running buffers), and the
# INT8 channel_scale is derived from the folded weight _amax. If this is False,
# the export float-shadow rebuilds *fresh, untrained* BatchNorm1d nodes (γ=1,β=0,
# mean=0,var=1 ≈ identity) that the checkpoint cannot fill, leaving 21 useless BN
# nodes in the ONNX that also block the ImplicitGemm→Relu fusion. Set True so the
# shadow folds BN → Identity (matches the folded checkpoint, keeps INT8 scales
# consistent) and ImplicitGemm→Relu becomes fuseable.
# ============================================================================
fuse_spconv_bn = True

# ============================================================================
# Sparse ONNX: fuse ImplicitGemm with trailing Relu (FP export + INT8 transform).
# ----------------------------------------------------------------------------
# - True  (default): bake act_type=ReLU into ImplicitGemm, remove standalone Relu nodes.
# - False          : keep explicit Relu nodes (debug / ablation).
# Requires fuse_spconv_bn=True for INT8 so ImplicitGemm→Relu is directly adjacent.
# ============================================================================
spconv_fuse_implicit_gemm_relu = True

# ============================================================================
# Sparse INT8: per-layer FP16 keep-list (accuracy knob).
# ----------------------------------------------------------------------------
# Entries are **case-insensitive substrings** matched ONLY against each
# ``autoware::ImplicitGemm`` node's ``node.name`` (NOT its inputs / outputs).
# Any node whose name contains one of these substrings is kept as FP16
# ``ImplicitGemm`` instead of being replaced by ``ImplicitGemmInt8``. This is
# the recommended way to exclude individual sparse-conv layers from INT8 for
# accuracy recovery, without touching PTQ or the TensorRT plugins.
#
# IMPORTANT — why we match node.name only (do NOT revert this):
#   PyTorch's ONNX exporter names output tensors with their producer's scope
#   path. The Relu/Add output after ``conv_input.0`` therefore still literally
#   contains ``conv_input.0`` in its tensor name, and that tensor becomes the
#   *input* of the NEXT ``ImplicitGemm``. If the matcher also scanned inputs/
#   outputs, writing ``"conv_input.0"`` here would silently FP16-ify the
#   *downstream* layer as well — a known cause of large mAP regressions. The
#   current implementation matches on ``node.name`` so each entry maps to
#   exactly one (or a well-defined group of) node(s).
#
# Notes / gotchas:
#   - Use the node name as printed by ``sparse_int8_onnx_transform --verbose``.
#     Full path (``/pts_middle_encoder/conv_input/conv_input.0/ImplicitGemm``)
#     or a unique tail (``conv_input.0/ImplicitGemm`` / ``conv_input.0``) both
#     work because match is substring on name.
#   - To keep ``conv_out`` as FP ``ImplicitGemm``, add a matching substring under
#     ``spconv_int8_fp16_layers`` (same as any other layer).
#   - The PTQ checkpoint already contains ``_amax`` for these layers, which is
#     harmless: it will simply not be consumed (expect an
#     ``[int8-audit] WARNING: calibrated stems with no matched ImplicitGemm
#     node`` line for each kept-FP16 stem; this is EXPECTED, not an error).
#   - To verify the list took effect, re-run step 4 with ``--verbose`` and
#     count the ``[int8] Keep FP16 ImplicitGemm per spconv_int8_fp16_layers``
#     lines — there must be **exactly one line per intended layer**. More than
#     expected means the pattern is too loose; zero means typo (and you'll
#     also see ``[int8-audit] WARNING: spconv_int8_fp16_layers patterns did
#     NOT match any node``).
# ============================================================================
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

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",
    work_dir="work_dirs/bevfusion_deployment_2_7_sparse_int8_dense_int8_6_4",
    onnx_path="work_dirs/bevfusion_deployment_2_7_sparse_int8_dense_int8_6_4/onnx",
    # onnx_path=None,
)

_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Split components（與 deploy_config_split.py 相同結構）
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
        # H,W must match bbox_head BEV (grid_size // out_size_factor); see deploy_config_split.py header.
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
    info_file="info/kokseang_2_6_1/t4dataset_j6gen2_base_infos_test.pkl",
    # info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
    sample_idx=0,
)

onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    visualize_qdq_values=True,
    simplify=False,
)

tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    # INT8 sparse conv now lives in the single Autoware plugin library (ImplicitGemm with
    # precision=1). No separate libimplicit_gemm_int8_plugin.so is needed anymore.
    plugin_libraries=[
        "/opt/plugins/libautoware_tensorrt_plugins.so",
    ],
)

evaluation = dict(
    enabled=True,
    num_samples=-1,
    num_warmup_samples=2,
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
