# StreamPETR deployment

> 中文版遷移說明（含 reference 比對方法細節）：[README_zh.md](README_zh.md)

Deploys StreamPETR (camera-only, temporally-stateful 3D detection) from a trained checkpoint
to the **three chained ONNX components** consumed by Autoware / the DL4AGX TensorRT runtime:

| Component | Graph | Key I/O |
| --- | --- | --- |
| `extract_img_feat` | image backbone + neck | `img [1,N,3,H,W]` → `img_feats` |
| `position_embedding` | 3D position encoding from camera geometry | `img_metas_pad, intrinsics, img2lidar` → `pos_embed, cone` |
| `pts_head_memory` | decoder head + temporal memory queue as explicit I/O | `x, pos_embed, cone, data_ego_pose*, pre_memory_*` → predictions + `post_memory_*` |

The component split and every tensor name are a **frozen contract** (reference artifacts:
`work_dirs/streampetr/simplify_*.onnx`). The host runtime threads the memory queue between
frames (slicing `post_memory_*[:, :memory_len]` back into the next frame's `pre_memory_*`),
keeping the engines stateless.

## Quick start

```bash
python -m deployment.cli.main streampetr \
    deployment/projects/streampetr/config/deploy_config.py \
    [model_cfg.py]   # optional; defaults to the deploy config's top-level model_cfg
```

## What is project-specific here

- **Three-component chained export**: the sample extractor loads a real clip-start frame and
  runs encoder → position embedding to produce each downstream component's actual inputs
  (the retired `projects/StreamPETR/deploy/torch2onnx.py` traced with random dummies), with
  a zeroed memory queue (the state a clip starts from).
- **Flash-attention surgery**: decoder attention is swapped to `PETRMultiheadAttention` at
  load time (`io/model_loader.py`) — flash attention has no ONNX export path.
- **Temporal constraints**: index order of the data loader **is** clip order
  (`StreamPETRDataset` sorts by scene + timestamp); `evaluation.num_warmup` must be 0
  (enforced by `StreamPETRDeploymentConfig`) because warmup replay corrupts the memory queue.

## Migration status

Implemented per `spec_streampetr_migration.md`:

- [x] Phase 0–2: scaffolding, io, ONNX export (I/O + op-count parity vs the deployed reference)
- [x] Phase 3: TensorRT export (FP16; note: engine builds max out GPU+CPU — on thermally
      constrained laptops cap clocks/power first)
- [x] Phase 4–5: stateful inference pipelines (PyTorch/ONNX/TensorRT) + cross-backend
      verification from clip start
- [x] Phase 6: evaluation with training-aligned metrics —
      `projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline_t4metric_v2.py`
      (the CenterPoint `*_t4metric_v2.py` convention: inherits the training config, replaces
      the evaluators; 51.2 m eval range mirroring the v1 `eval_class_range`). It also **pins
      the v2.5 artifact's 5-class layout** — the shared t4dataset base has since grown to 7
      classes, which silently breaks the 5-class checkpoint load. Full local test clip (19
      frames): PyTorch 0.6320 / ONNX 0.6305 / TensorRT-FP16 0.6366 mAP (BEV-center);
      TRT latency 56.7 ms vs PyTorch 176 ms. Absolute values are clip-local (single scene,
      19 frames) — not comparable to the 8,453-frame model-zoo eval.

## Layout

| Path | Role |
| --- | --- |
| `__init__.py` | Registers the `streampetr` `ProjectAdapter` |
| `entrypoint.py` | Self-wired `run(args)` (camera loader — the shared 3D entrypoint is LiDAR-only) |
| `runner.py` | `StreamPETRDeploymentRunner` |
| `config/` | Deploy config + `StreamPETRDeploymentConfig` (component + temporal validation) |
| `io/` | `StreamPETRDataLoader` (clip-ordered multi-view frames), model loader, sample types |
| `export/` | `StreamPETRSampleExtractor`, `StreamPETRComponentBuilder`, `onnx_models/` (traced graphs) |
| `evaluation/executor.py` | `StreamPETRExecutor` (`create_pipeline` lands in Phase 4) |

See the shared docs for framework structure ([../../docs/architecture.md](../../docs/architecture.md)),
config fields ([../../docs/configuration.md](../../docs/configuration.md)) and run behavior
([../../docs/runbook.md](../../docs/runbook.md)).
