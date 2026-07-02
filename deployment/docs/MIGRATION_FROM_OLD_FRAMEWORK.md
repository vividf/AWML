# Migrating the old deployment framework into `deployment/`

This document records how the **old** deployment framework
(`deployment_old_framework_with_different_centerpoint_bevfusion_quantization (copy)/`,
hereafter **OLD**) was integrated into the refactored **`deployment/`** package (**NEW**),
and is the reference for porting any remaining piece or doing a similar migration again.

## What and why

NEW is a refactor of OLD. The refactor dropped three things that OLD had:
- the **BEVFusion** deployment project,
- a full **quantization** module (PTQ/QAT, spconv INT8, a C++ TensorRT INT8 plugin),
- **~20 extra CenterPoint deploy configs** (fp32 / fp16-by-backbone / int8 variants).

The migration brought all of that back into the refactored structure **without changing
the new architecture**.

## The one architecture change to be aware of

NEW = OLD with directories reorganized **plus one abstraction swap**: the old global
`PipelineRegistry` / `PipelineFactory` (and the evaluator-owned `_create_pipeline` /
`_prepare_input` / `_get_output_names`) was replaced by a **per-project
`BackendExecutor`** (`deployment/evaluation/backend_executor.py`). Evaluators became pure
metrics adapters that receive an `executor`. Everything else maps 1:1 with import-path
rewrites.

So when porting an OLD file: rewrite imports per the table, and if it created/queried
pipelines via the factory, move that into the project's `evaluation/executor.py`.

## Old → New import remap (apply to every ported file)

| OLD import prefix | NEW import prefix |
|---|---|
| `deployment.configs.{base,schema,enums}` | `deployment.config.{base,schema,enums}` |
| `deployment.core.backend` (`Backend`) | `deployment.config.enums` |
| `deployment.core.device` (`DeviceSpec`) | `deployment.primitives.device` |
| `deployment.core.artifacts` | `deployment.primitives.artifacts` |
| `deployment.core.contexts` (`ExportContext`) | `deployment.export.contexts` |
| `deployment.core.tensorrt_plugins` | `deployment.primitives.tensorrt_plugins` |
| `deployment.core.evaluation.*` | `deployment.evaluation.*` |
| `deployment.core.io.*` | `deployment.io.*` |
| `deployment.core.metrics.*` | `deployment.metrics.*` |
| `deployment.exporters.common.*` | `deployment.export.exporters.*` |
| `deployment.exporters.export_pipelines.*` (ABCs) | **drop** — custom pipelines are plain duck-typed classes |
| `deployment.pipelines.base_pipeline` / `gpu_resource_mixin` | `deployment.inference.base_inference_pipeline` / `deployment.inference.gpu_resource_mixin` |
| `deployment.pipelines.{registry,factory,base_factory}` | **deleted** — replaced by `deployment.evaluation.backend_executor.BackendExecutor` |
| `deployment.projects.centerpoint.model_loader` | `deployment.projects.centerpoint.io.model_loader` |
| `deployment.projects.centerpoint.onnx_models` | `deployment.projects.centerpoint.export.onnx_models` |
| `deployment.quantization.*` | **unchanged** (already correctly rooted) |
| `deployment.projects.bevfusion.*` | **unchanged** (intra-project) |
| `projects.BEVFusion.*`, `projects.SparseConvolution.*` | **unchanged** (top-level AWML model code, not part of `deployment/`) |

## What was changed/added, by area

1. **Shared infra & schema (additive, default-disabled — existing CenterPoint unaffected):**
   - `config/schema.py`: added `QuantizationConfig` (typed view of the deploy-config
     `quantization` dict + a `raw` passthrough so the proven OLD loader bodies port
     unchanged) and `TensorRTConfig.plugin_libraries`.
   - `config/base.py`: parses `quantization` (folds top-level `spconv_int8_fp16_layers`);
     added a read-only **`deploy_cfg` property** so project export pipelines can read
     project-only keys (`fuse_spconv_bn`, `spconv_int8_fp16_layers`, `bevfusion_merge`,
     `spconv_do_sort`); `get_tensorrt_settings` now forwards `plugin_libraries`.
   - `primitives/tensorrt_plugins.py`: ported verbatim; `load_tensorrt_plugin_libraries`
     is called in `export/exporters/tensorrt_exporter.py` right before
     `trt.init_libnvinfer_plugins(...)` (no-op when no plugins configured).
   - **INT8 is QDQ-driven** — no INT8 `PrecisionPolicy` was added. OLD INT8 configs keep
     `precision_policy="fp16"`; INT8 comes from Q/DQ nodes baked into the ONNX
     (`TensorQuantizer.use_fb_fake_quant=True`) plus the spconv plugin. Sparse-FP16 /
     dense-INT8 is expressed via **separate `bevfusion_sparse` / `bevfusion_dense`
     components**, not per-component precision.

2. **Shared quantization module** — `OLD/quantization/` copied verbatim to
   `deployment/quantization/` (internal imports already `deployment.quantization.*`).
   PTQ/QAT CLIs live here: `python -m deployment.quantization.centerpoint_quantization`,
   `python -m deployment.quantization.bevfusion_quantization`.

3. **CenterPoint parity** — quant load hook ported into
   `projects/centerpoint/io/model_loader.py` (`build_centerpoint_onnx_model(..., quantization=...)`
   + `_load_quantized_checkpoint`/`_build_skip_layers`/`setup_quantization_for_onnx_export`/…);
   `runner.py` threads `config.quantization_config.raw`. 20 OLD deploy configs translated
   (component outer keys → `pts_voxel_encoder`/`pts_backbone_neck_head`, drop inner `name=`,
   `num_warmup_samples`→`num_warmup`, drop `task_type`/`runtime_io`/`output_path`).

4. **BEVFusion project** — full port under `projects/bevfusion/`. Mechanical files
   (io/inference/export/configs) are import-rewrites of OLD; the **glue** was rewritten to
   the new contracts: `evaluation/executor.py` (`BEVFusionExecutor`, replaces the old
   factory), `evaluation/evaluator.py` (new base ctor), `runner.py` (constructs the custom
   `onnx_pipeline`/`tensorrt_pipeline` **before** `super().__init__`; reads BEVFusion-only
   keys from the raw `deploy_cfg`; no `logger`/`module` ctor args), `entrypoint.py`,
   `__init__.py`. BEVFusion keeps its **own** ONNX/TensorRT export pipelines (wrapper
   modules, TopK fix, coordinate flips, split→merge) injected via the runner's override
   hooks — it does not use the generic `SampleExtractor`/`ComponentBuilder` seam.
   INT8 path (spconv int8, sparse INT8 `sparse_int8_onnx_transform`) is gated behind
   `quantization.enabled`, with all quant imports kept **function-local** so the FP16 path
   never imports a quant module. The C++ plugin (`cpp/`), docs (`docs/`),
   `benchmark/`, `scripts/`, and `projects/BEVFusion/plugins/` build infra are copied.

## How to verify

**Host static checks** (no torch/mmengine/CUDA — the runtime is Docker):
```bash
# syntax
find deployment -name '*.py' -not -path '*__pycache__*' -exec python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" {} \;
# undefined names / removed-abstraction leftovers
python3 -m pyflakes deployment/ | grep -i "undefined name"   # expect none
grep -rnE "deployment\.(core|configs|exporters|pipelines)\b|PipelineFactory|pipeline_registry|ExporterFactory" deployment/ --include='*.py'   # expect none
```
A small AST tool that resolves every module-level `deployment.*` import to an existing
file is the strongest "imports cleanly" proxy on the host.

**Docker e2e** (the real validation):
```bash
pytest deployment/tests/test_centerpoint_configs.py
python -m deployment.cli.main centerpoint <deploy_cfg> <model_cfg>
python -m deployment.cli.main bevfusion deployment/projects/bevfusion/config/deploy_config.py <model_cfg> --module main_body
```

## Known caveats / not done

- **C++ INT8 plugin build is deferred.** `projects/bevfusion/cpp/int8_plugin/` files and the
  Python plugin-loading path (`tensorrt_config.plugin_libraries`) are in place, but the
  CMake build must run in the target Docker image (version-coupled with spconv/cumm/TRT;
  see the plugin README's `output_scale` epilogue-fusion bugfix and
  `projects/BEVFusion/plugins/build_plugin_inside_container.sh`).
- **Dangling config name in benchmark/scripts.** The `benchmark/*.sh` and
  `scripts/*.sh` reference `config/deploy_config_split_int8.py`, which never existed in OLD
  (renamed). Point them at an existing variant, e.g. `deploy_config_split_int8_all.py`.
- **Model-repo evolution was not copied.** `AWML_temp/projects/BEVFusion` has some files the
  current repo lacks (older `8xb8` train configs, Dockerfiles, `sparse_convmodule.py`, some
  docs), but those are intentional model-repo changes on this branch (now `4xb8`) — copying
  them would revert current work.
