"""
BEVFusion deploy config — **split ONNX / TensorRT + PTQ INT8** (路線 1 + 量化 checkpoint)

結合：
  - **路線 1**：``bevfusion_sparse`` + ``bevfusion_dense`` 兩段 ONNX / 兩顆 engine（見 ``deploy_config_split.py``）
  - **INT8**：與 ``deploy_config_int8.py`` 相同的 ``quantization``，載入 **已 PTQ 的 checkpoint**（spconv 稀疏塔 INT8 + 稠密端 pytorch_quantization）

**隔離問題時怎麼想（與 ``deploy_config_split.py`` 無關）**

  同一條 CLI、同一個 ``*_120m_fx.py``、**仍用本檔**（split 路徑與 work_dir 不變），只把
  ``checkpoint_path`` 改成 **訓練 FP32 .pth**，並設 ``quantization = dict(enabled=False)``。
  若此時 **mAP 正常**，則 **split / voxel preprocess / eval 管線沒壞**；mAP 掉在 **PTQ .pth 或
  quantization 載入**（含 dense Q/DQ、spconv INT8、key 與 mmconfig 是否一致）。
  稀疏校準預設 **不裁 voxel**（與 Lidar AI Solution 一致）；僅在顯存不足時可選
  ``spconv_calib_max_voxels`` / ``SPCONV_CALIB_MAX_VOXELS``。

**前置條件**
  - ``checkpoint_path`` 指向 ``bevfusion_quantization.py ptq`` 產生的 **.pth**；``quantization.ptq_checkpoint=True``（本檔預設已開）。
  - **``spconv_ptq_basicblock_fx``** 必須與 **產生該 PTQ .pth 時** 的 sparse 圖一致，否則會出現 ``bn1.weight`` missing / ``bn1_scale_0`` unexpected、mAP≈0。

    - **False**：legacy PTQ（``*_base_120m.py``、舊腳本、未做 block 升級）與 deploy 對齊時。
    - **True**：``*_120m_fx.py`` 且已用 **目前** ``bevfusion_quantization.py`` 重跑 PTQ（腳本內會 ``SparseBasicBlock→FX``）時，建議與 PTQ 一致。

**稀疏段 ONNX**
  - ``convert_fx`` 後的 GraphModule 無法直接匯出（ONNX 不支援 ``_empty_affine_quantized``）。
  - 匯出時 pipeline 會 **暫時**以重建的 **FP32 融合稀疏塔** 取代 GraphModule 僅供 trace，結束後還原；
    產生的 ``bevfusion_sparse.onnx`` 為 **浮點稀疏圖**（與 Lidar ``*.scn.onnx`` + libspconv FP16/FP 路線同類），
    **數值與 PTQ INT8 不完全相同**；真 INT8 稀疏推理請仍用 PyTorch 或依 spconv TENSORRT_INT8_GUIDE 餵權重給 plugin。

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
# Checkpoint
# ============================================================================
# PTQ：必須為 bevfusion_quantization.py 產物。FP32 對照：改為訓練 .pth 並改用下方 Preset B。
checkpoint_path = "work_dirs/bevfusion/bevfusion_epoch_30_ptq.pth"

# ============================================================================
# Quantization — Preset A：INT8 PTQ（預設）
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
    spconv_ptq_basicblock_fx=True,
    # Sparse FX voxel cap (PTQ: bevfusion_quantization.py uses CLI --calibrate-samples only).
    spconv_calib_max_voxels=4096,
    sensitive_layers=[],
)

# ============================================================================
# Preset B：同 split_int8 路徑 + *_fx.py，僅 FP32 驗證 mAP / 管線（註解掉 Preset A 後啟用）
# ============================================================================
# checkpoint_path = "work_dirs/bevfusion/bevfusion_epoch_30.pth"
# quantization = dict(enabled=False)

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="onnx",
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
            enabled=False,
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
