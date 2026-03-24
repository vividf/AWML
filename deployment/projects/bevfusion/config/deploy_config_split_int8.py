"""
BEVFusion deploy config — **split ONNX / TensorRT + PTQ INT8** (路線 1 + 量化 checkpoint)

結合：
  - **路線 1**：``bevfusion_sparse`` + ``bevfusion_dense`` 兩段 ONNX / 兩顆 engine（見 ``deploy_config_split.py``）
  - **INT8**：與 ``deploy_config_int8.py`` 相同的 ``quantization``，載入 **已 PTQ 的 checkpoint**（spconv 稀疏塔 INT8 + 稠密端 pytorch_quantization）

**前置條件**
  - 使用 **FX 可 trace** 的 model config（例如 ``*_120m_fx.py``，`block_type=basicblock_fx`），與 PTQ 產出時一致。
  - ``checkpoint_path`` 指向 ``bevfusion_quantization.py ptq`` 等流程產生的 **.pth**。
  - ``quantization.ptq_checkpoint=True``（本檔預設已開）。

**稀疏段 ONNX 注意**
  - 若 ``pts_middle_encoder`` 已是 **convert_fx 後的 qint8 GraphModule**，``bevfusion_sparse.onnx`` 的
    ``torch.onnx.export`` **可能仍失敗**（標準 ONNX 不支援 ``_empty_affine_quantized`` 等）。
  - 此情況下：稠密 ``bevfusion_dense`` 仍可嘗試匯出／建 TRT；稀疏請改 **PyTorch 推理** 或 **libspconv / 自寫權重載入**。

**CLI**（需在含 pytorch-quantization 的環境；Docker 見 ``deploy_config_int8.py`` 註解）::

    python -m deployment.cli.main bevfusion \\
        deployment/projects/bevfusion/config/deploy_config_split_int8.py \\
        projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \\
        --module main_body

產生 PTQ checkpoint 範例（與 int8 單檔相同）::

    python deployment/quantization/bevfusion_quantization.py ptq \\
        --config projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \\
        --checkpoint work_dirs/bevfusion/epoch_30.pth \\
        --deploy-cfg deployment/projects/bevfusion/config/deploy_config_int8.py \\
        --calibrate-samples 256 --batch-size 1 --calib-seed 0 \\
        --output work_dirs/bevfusion/epoch_30_ptq2.pth
"""

# ============================================================================
# Checkpoint — 必須為 PTQ 產物（含校準後的 dense Q/DQ 與稀疏 FX 權重）
# ============================================================================
checkpoint_path = "work_dirs/bevfusion/epoch_30_ptq2.pth"

# ============================================================================
# Quantization（與 deploy_config_int8.py 對齊；專用於已量化 checkpoint）
# ============================================================================
quantization = dict(
    enabled=True,
    ptq_checkpoint=True,
    fuse_bn=True,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=False,
    spconv_int8=True,
    num_calibration_samples=5,
    sensitive_layers=[],
)

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",
    work_dir="work_dirs/bevfusion_split_int8_deployment",
    onnx_path=None,
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
