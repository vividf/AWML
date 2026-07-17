"""
StreamPETR Deployment Configuration

Layout (single file, grouped by concern):
  1. SHARED VALUES  - single source of truth reused across sections (paths, devices, shapes).
  2. EXPORT         - export mode, ONNX/TensorRT build settings, component definitions.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

Only the top-level names `checkpoint_path`, `model_cfg`, `deploy_log_path`, `devices`,
`export`, `components`, `onnx_config`, `tensorrt_config`, `evaluation`, `verification` are
read by `BaseDeploymentConfig`. Names prefixed with `_` are local helpers.

FROZEN CONTRACT
---------------
The 3-component split and every tensor name below are consumed by Autoware / the DL4AGX
TensorRT runtime (see the reference artifacts in `work_dirs/streampetr/simplify_*.onnx`).
Do not rename components or tensors.

Note on inputs pruned by the simplifier: the export traces `position_embedding` with an
`img_feats` input and `pts_head_memory` with a `data_timestamp` input (matching the original
`projects/StreamPETR/deploy/torch2onnx.py`), but neither value is used inside the traced
graph, so `onnxsim` removes them — the shipped ONNX has 3 and 10 inputs respectively,
matching the deployed reference. The `io.inputs` lists below describe the *traced* inputs
(they define torch.onnx.export input_names order); the `tensorrt_profile` sections describe
the *simplified* graphs' remaining inputs.
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
# ============================================================================

# Checkpoint + model config - single source of truth for the PyTorch model.
# model_cfg is the T4MetricV2 evaluator variant next to the training config (same
# convention as CenterPoint's *_t4metric_v2.py): metrics settings match training
# (51.2 m eval range), and the deployment entrypoint extracts them from `val_evaluator`.
checkpoint_path = "work_dirs/streampetr/best_NuScenesmetric_T4Metric_mAP_epoch_34.pth"
model_cfg = "projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline_t4metric_v2.py"

# Log file path (relative paths are resolved under export.work_dir). None disables file logging.
deploy_log_path = "deployment.log"

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
_CUDA = devices["cuda"]

# Deployment output layout.
_DEPLOY_WORK_DIR = "work_dirs/streampetr_deployment"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# Model geometry (must match the model config: 5 cameras, 480x640 input, stride 16,
# 256 feature channels, memory_len 1024). Used only for the static TensorRT profiles.
_NUM_CAMERAS = 5
_IMG_H, _IMG_W = 480, 640
_FEAT_C = 256
_FEAT_H, _FEAT_W = _IMG_H // 16, _IMG_W // 16  # 30, 40
_NUM_TOKENS = _NUM_CAMERAS * _FEAT_H * _FEAT_W  # 6000
_MEMORY_LEN = 1024


def _static(shape):
    """min == opt == max: every StreamPETR input shape is fixed."""
    return dict(min_shape=shape, opt_shape=shape, max_shape=shape)


# Eval/export info file override (relative to the dataset's data_root). The dumped model
# config points at a user-scoped info directory; this deploy targets the local test info.
runtime_io = dict(
    info_file="info/t4dataset_base_infos_test.pkl",
)

# ============================================================================
# 2. EXPORT
# ============================================================================

export = dict(
    mode="none",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=0,
)

onnx_config = dict(
    opset_version=18,  # matches the deployed reference artifacts
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=True,  # the deployed contract is the onnxsim-simplified graph
)

tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=2 << 30,
)

components = dict(
    extract_img_feat=dict(
        onnx_file="extract_img_feat.onnx",
        engine_file="extract_img_feat.engine",
        io=dict(
            inputs=[
                dict(name="img", dtype="float32"),
            ],
            outputs=[
                dict(name="img_feats", dtype="float32"),
            ],
        ),
        tensorrt_profile=dict(
            img=_static([1, _NUM_CAMERAS, 3, _IMG_H, _IMG_W]),
        ),
    ),
    position_embedding=dict(
        onnx_file="position_embedding.onnx",
        engine_file="position_embedding.engine",
        io=dict(
            # `img_feats` is traced but unused in the graph; onnxsim prunes it (see header).
            inputs=[
                dict(name="img_metas_pad", dtype="float32"),
                dict(name="img_feats", dtype="float32"),
                dict(name="intrinsics", dtype="float32"),
                dict(name="img2lidar", dtype="float32"),
            ],
            outputs=[
                dict(name="pos_embed", dtype="float32"),
                dict(name="cone", dtype="float32"),
            ],
        ),
        tensorrt_profile=dict(
            img_metas_pad=_static([3]),
            intrinsics=_static([1, _NUM_CAMERAS, 4, 4]),
            img2lidar=_static([1, _NUM_CAMERAS, 4, 4]),
        ),
    ),
    pts_head_memory=dict(
        onnx_file="pts_head_memory.onnx",
        engine_file="pts_head_memory.engine",
        io=dict(
            # `data_timestamp` is traced but unused in the graph; onnxsim prunes it (see header).
            inputs=[
                dict(name="x", dtype="float32"),
                dict(name="pos_embed", dtype="float32"),
                dict(name="cone", dtype="float32"),
                dict(name="data_timestamp", dtype="float64"),
                dict(name="data_ego_pose", dtype="float32"),
                dict(name="data_ego_pose_inv", dtype="float32"),
                dict(name="pre_memory_embedding", dtype="float32"),
                dict(name="pre_memory_reference_point", dtype="float32"),
                dict(name="pre_memory_timestamp", dtype="float32"),
                dict(name="pre_memory_egopose", dtype="float32"),
                dict(name="pre_memory_velo", dtype="float32"),
            ],
            outputs=[
                dict(name="all_cls_scores", dtype="float32"),
                dict(name="all_bbox_preds", dtype="float32"),
                dict(name="post_memory_embedding", dtype="float32"),
                dict(name="post_memory_reference_point", dtype="float32"),
                dict(name="post_memory_timestamp", dtype="float64"),
                dict(name="post_memory_egopose", dtype="float32"),
                dict(name="post_memory_velo", dtype="float32"),
                dict(name="reference_points", dtype="float32"),
                dict(name="tgt", dtype="float32"),
                dict(name="temp_memory", dtype="float32"),
                dict(name="temp_pos", dtype="float32"),
                dict(name="query_pos", dtype="float32"),
                dict(name="query_pos_in", dtype="float32"),
                dict(name="outs_dec", dtype="float32"),
            ],
        ),
        tensorrt_profile=dict(
            x=_static([1, _NUM_CAMERAS, _FEAT_C, _FEAT_H, _FEAT_W]),
            pos_embed=_static([1, _NUM_TOKENS, _FEAT_C]),
            cone=_static([1, _NUM_TOKENS, 8]),
            data_ego_pose=_static([1, 4, 4]),
            data_ego_pose_inv=_static([1, 4, 4]),
            pre_memory_embedding=_static([1, _MEMORY_LEN, _FEAT_C]),
            pre_memory_reference_point=_static([1, _MEMORY_LEN, 3]),
            pre_memory_timestamp=_static([1, _MEMORY_LEN, 1]),
            pre_memory_egopose=_static([1, _MEMORY_LEN, 4, 4]),
            pre_memory_velo=_static([1, _MEMORY_LEN, 2]),
        ),
    ),
)

# ============================================================================
# 3. EVALUATION
#
# Temporal constraints (enforced/required):
# - num_warmup MUST stay 0 (StreamPETRDeploymentConfig rejects otherwise) — warmup replays
#   samples and corrupts the temporal memory queue.
# - Samples are consumed in loader index order, which IS clip order (StreamPETRDataset
#   sorts by scene + timestamp), so the memory queue sees a coherent sequence.
# - The metrics config falls back to defaults while the model config still uses T4Metric
#   (v1); cross-backend mAP consistency is meaningful, absolute values are not yet
#   comparable to the training eval (switch to T4MetricV2 for that).
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=19,  # the full local test clip (single scene, clip-ordered)
    num_warmup=0,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=_CUDA,
        ),
        onnx=dict(
            enabled=True,
            device=_CUDA,
            model_dir=_ONNX_DIR,
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
#
# Verification iterates samples from index 0 — a clip start — and each pipeline instance
# begins with a zeroed memory queue, so ref and test see identical temporal state
# (arbitrary-index comparison would be meaningless for a stateful model).
#
# Measured baselines (5-frame T4 test clip, RTX 3060 Laptop / TRT 10.8):
# - pytorch(cpu) vs onnx(cpu), FP32: max_diff ~3e-4 → PASSES (graph is numerically
#   equivalent to PyTorch).
# - onnx(cuda) vs tensorrt(cuda), FP16: mean_diff ~0.02 but max_diff O(10) → FAILS the
#   element-wise check. Known limitation, not an engine bug: the decoder's top-k proposal
#   selection is discrete, so borderline FP16 score flips reorder individual queries and
#   the per-element diff explodes even though the decoded detection set is unchanged
#   (eval mAP: TRT 0.5491 vs ONNX 0.5440). Judge the FP16 engine by evaluation mAP, or
#   verify with a `fp32_tf32` engine build when element-wise parity is required.
# ============================================================================
verification = dict(
    enabled=True,
    tolerance=1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        # mode="none" (artifacts already exported) still verifies both hops.
        none=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
    ),
)
