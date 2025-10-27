### Unified Deployment Architecture for YOLOX, CenterPoint, and Calibration Status Classification

This document proposes a unified, extensible deployment framework that consolidates the three completed deployments (YOLOX_opt_elan, CenterPoint, CalibrationStatusClassification) into a single architecture. It supports single- and multi-engine models, optional mid-processors (e.g., PyTorch middle encoders), and flexible pre/post-processing pipelines.

The goals are:
- Unify the deployment stack across tasks and models
- Remove duplicated logic in per-project backends/wrappers
- Support multi-engine graphs (e.g., CenterPoint’s dual ONNX/TRT) and mid-processors
- Make adding a new model mostly configuration- and registry-driven


## High-level Pipeline

```
Input → Preprocessor(s) → Engine Graph (1..N engines) → Midprocessor(s) → Postprocessor(s) → Output
```

- Preprocessor: prepares inputs for engines (image normalization, voxelization, tensor formatting)
- Engine Graph: one or more ONNX/TRT engines (or PyTorch) with explicit I/O bindings and routing
- Midprocessor: optional in-graph compute not exported to engines (e.g., PyTorch middle encoder)
- Postprocessor: decoding and formatting (e.g., YOLOX decode/NMS, CenterPoint decode/NMS)


## Core Building Blocks (to be added)

- `autoware_ml/deployment/core/modular_pipeline.py`
  - Orchestrates the full pipeline: preprocess → engine graph → midprocess → postprocess
  - Provides a uniform `run(sample)` entry point returning standardized outputs

- `autoware_ml/deployment/backends/modular_backend.py`
  - A unified backend that executes a configurable Engine Graph
  - Delegates ONNX and TensorRT execution to existing `ONNXBackend` and `TensorRTBackend`
  - Supports mixed backends in one graph (if needed), with runtime routing

- `autoware_ml/deployment/backends/engine_manager.py`
  - Loads and caches multiple engines
  - Manages bindings (input/output tensors), shape updates, and memory reuse
  - Provides a simple `run(node_name, inputs)` interface

- `autoware_ml/deployment/processors/`
  - `base_processor.py`: abstract base for pre/mid/post processors
  - `preprocessors/`: `image_normalization.py`, `voxelization.py`, `point_cloud_format.py`, etc.
  - `midprocessors/`: `pytorch_middle_encoder.py`, `feature_fusion.py`, etc.
  - `postprocessors/`: `yolox_decoder.py`, `centerpoint_decoder.py`, `classification_decoder.py`

- `autoware_ml/deployment/core/processor_registry.py`
  - Registry/factory to instantiate processors by name with config kwargs

- `autoware_ml/deployment/exporters/modular_exporter.py`
  - Unified export entry with per-model recipes but common flow
  - Supports:
    - Single-engine export (YOLOX, classification)
    - Multi-engine export (CenterPoint) with multiple ONNX/TRT outputs
    - Optional ONNX wrapper usage (e.g., YOLOX decode-in-graph) as a pre-export transform


## Configuration Schema (proposal)

All deployments describe their pipeline via a single config block. Example skeleton:

```python
deployment = dict(
    device='cuda:0',
    work_dir='work_dirs/example',
    precision_policy='fp16',  # uses existing backend_config.common_config if present
)

io = dict(
    # Dataset / sample selection for export/verification/eval
    ann_file='...',
    img_prefix='...',
    info_file='...',
    sample_idx=0,
)

engines = dict(
    backend='onnx',  # 'onnx' | 'tensorrt' | 'pytorch' per node override allowed
    nodes=[
        # YOLOX single-engine example
        # dict(name='detector', type='engine', path='.../yolox.onnx', inputs=['images'], outputs=['raw']),

        # CenterPoint two-engine example
        # dict(name='voxel_encoder', type='engine', path='.../pts_voxel_encoder.onnx',
        #      inputs=['voxels', 'num_points', 'coors'], outputs=['voxel_features']),
        # dict(name='backbone_head', type='engine', path='.../pts_backbone_neck_head.onnx',
        #      inputs=['spatial_features'], outputs=['reg', 'height', 'dim', 'rot', 'vel', 'hm']),
    ],
)

processors = dict(
    preprocess=[
        # dict(name='image_normalization', type='ImageNormalization', kwargs={...}),
        # dict(name='voxelization', type='Voxelization', kwargs={...}),
    ],
    midprocess=[
        # dict(name='middle_encoder', type='PyTorchMiddleEncoder', inputs=['voxel_features'],
        #      outputs=['spatial_features'], kwargs={...}),
    ],
    postprocess=[
        # dict(name='yolox_decoder', type='YOLOXDecoder', inputs=['raw'], outputs=['detections'], kwargs={...}),
        # dict(name='centerpoint_decoder', type='CenterPointDecoder', inputs=['reg','height','dim','rot','vel','hm'],
        #      outputs=['detections'], kwargs={...}),
        # dict(name='classification_decoder', type='ClassificationDecoder', inputs=['logits'], outputs=['probs','labels']),
    ],
)

evaluation = dict(
    enabled=True,
    num_samples=100,
    models_to_evaluate=['pytorch', 'onnx', 'tensorrt'],
)
```

Key points:
- Each processor and engine node declares explicit inputs/outputs for routing
- The `modular_backend` executes `engines.nodes` in order, while processors can be placed before, between, or after engines
- Backends can be mixed per node (e.g., first engine in ONNX, second in TRT) if needed


## Model-specific Mappings

### YOLOX_opt_elan (2D detection, optional wrapper)

- Preprocess: `ImageNormalization`
- Engines: single engine `detector` (ONNX/TRT)
- Midprocess: none
- Postprocess: `YOLOXDecoder` (unless using decode-in-ONNX via wrapper during export)

Export options:
- Keep wrapper: `modular_exporter` wraps PyTorch model with `YOLOXONNXWrapper` to export decoded outputs
- No wrapper: export raw outputs; use `YOLOXDecoder` postprocessor at runtime

### CalibrationStatusClassification (binary classification)

- Preprocess: `ImageNormalization` (and any stacking of channels as needed)
- Engines: single engine `classifier`
- Postprocess: `ClassificationDecoder` (softmax + argmax + label mapping)

### CenterPoint (3D detection, multi-engine + midprocessor)

- Preprocess: `Voxelization`
- Engines: two nodes
  1) `voxel_encoder` (ONNX/TRT): outputs `voxel_features`
  2) `backbone_head` (ONNX/TRT): consumes `spatial_features`, outputs head tensors
- Midprocess: `PyTorchMiddleEncoder`: converts `voxel_features` to `spatial_features`
- Postprocess: `CenterPointDecoder`: decode + NMS → final 3D boxes

Notes:
- The midprocessor stays in PyTorch by design; its runtime device is aligned with the pipeline device
- The engine manager maintains bindings for both engines and allows dynamic shapes where needed (e.g., number of voxels)


## Unified Evaluator and Data Loader

Keep existing task-specific `BaseDataLoader` and `BaseEvaluator` subclasses, but remove per-model backend special-casing:
- Data loaders focus on producing standardized input tensors and metadata
- Evaluators consume standardized postprocessed outputs (`detections`, `probs`, `labels`) and compute metrics
- The pipeline interfaces look like:
  - `pipeline.prepare_sample(idx) -> inputs`
  - `pipeline.run(inputs) -> standardized_outputs`
  - Evaluators only depend on `standardized_outputs`


## Export Strategy

`modular_exporter.py` supports three modes:
1) Single-engine export: export ONNX → optional TRT. Example: YOLOX, classification
2) Multi-engine export: export multiple ONNXs (and TRTs). Example: CenterPoint
3) Wrapper-assisted export: apply a temporary PyTorch wrapper (e.g., YOLOX decode-inference) before ONNX export

Verification hooks remain unchanged and can be triggered after export to compare outputs across PyTorch/ONNX/TRT.


## Migration Plan

Phase 1: Introduce core modules (non-breaking)
- Add `modular_pipeline.py`, `modular_backend.py`, `engine_manager.py`, processor base + registry
- Implement key processors: `ImageNormalization`, `Voxelization`, `YOLOXDecoder`, `CenterPointDecoder`, `ClassificationDecoder`
- Provide example unified configs for the three models

Phase 2: Migrate existing deployments to unified configs
- YOLOX_opt_elan: switch to modular pipeline; keep wrapper as an export-time option
- CalibrationStatusClassification: switch evaluator to consume standardized outputs
- CenterPoint: replace `centerpoint_tensorrt_backend.py` with engine graph + midprocessor

Phase 3: Remove duplicated per-model backends/wrappers
- Deprecate `projects/*/deploy/*_backend.py` and ad-hoc wrappers where equivalent processors exist
- Consolidate evaluation scripts to call the modular pipeline

Phase 4: Hardening and extensions
- Add more processors (e.g., image resize/letterbox, dynamic voxel configs)
- Add graph-level optimizations (buffer reuse, pre-allocated device memory)
- Add per-node precision overrides (e.g., FP16 for TRT node 2 only)


## Example Unified Configs

### YOLOX_opt_elan

```python
deployment = dict(device='cuda:0', work_dir='work_dirs/yolox_opt_elan', precision_policy='fp16')
io = dict(ann_file='data/t4dataset/yolox_infos_val.json', img_prefix='data/t4dataset/images', sample_idx=0)
engines = dict(backend='onnx', nodes=[dict(name='detector', type='engine', path='work_dirs/yolox_opt_elan/yolox.onnx', inputs=['images'], outputs=['raw'])])
processors = dict(
    preprocess=[dict(name='image_norm', type='ImageNormalization', kwargs=dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))],
    postprocess=[dict(name='yolox_decoder', type='YOLOXDecoder', inputs=['raw'], outputs=['detections'], kwargs=dict(score_thr=0.01, nms_thr=0.65, max_per_img=300))],
)
evaluation = dict(enabled=True, num_samples=100, models_to_evaluate=['onnx'])
```

### CalibrationStatusClassification

```python
deployment = dict(device='cuda:0', work_dir='work_dirs/calib_cls', precision_policy='fp16')
io = dict(info_pkl='data/t4dataset/calibration_info/infos_test.pkl', sample_idx=0)
engines = dict(backend='onnx', nodes=[dict(name='classifier', type='engine', path='work_dirs/calib/end2end.onnx', inputs=['images'], outputs=['logits'])])
processors = dict(
    preprocess=[dict(name='image_norm', type='ImageNormalization', kwargs=dict(mean=[...], std=[...]))],
    postprocess=[dict(name='classification_decoder', type='ClassificationDecoder', inputs=['logits'], outputs=['probs','labels'])],
)
evaluation = dict(enabled=True, num_samples=500, models_to_evaluate=['onnx'])
```

### CenterPoint

```python
deployment = dict(device='cuda:0', work_dir='work_dirs/centerpoint', precision_policy='fp16')
io = dict(info_file='data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl', sample_idx=0)
engines = dict(
    backend='onnx',
    nodes=[
        dict(name='voxel_encoder', type='engine', path='work_dirs/centerpoint/pts_voxel_encoder.onnx',
             inputs=['voxels','num_points','coors'], outputs=['voxel_features']),
        dict(name='backbone_head', type='engine', path='work_dirs/centerpoint/pts_backbone_neck_head.onnx',
             inputs=['spatial_features'], outputs=['reg','height','dim','rot','vel','hm']),
    ],
)
processors = dict(
    preprocess=[dict(name='voxelization', type='Voxelization', kwargs=dict(max_points=32, max_voxels=...))],
    midprocess=[dict(name='middle_encoder', type='PyTorchMiddleEncoder', inputs=['voxel_features'], outputs=['spatial_features'], kwargs=dict(weight_path='.../middle_encoder.pth'))],
    postprocess=[dict(name='centerpoint_decoder', type='CenterPointDecoder', inputs=['reg','height','dim','rot','vel','hm'], outputs=['detections'], kwargs=dict(nms_iou_thr=0.5, score_thr=0.1))],
)
evaluation = dict(enabled=True, num_samples=50, models_to_evaluate=['onnx'])
```


## Extension Guidelines

- To add a new model:
  1) Define preprocessors and postprocessors in `processors/*`
  2) Export engines via `modular_exporter.py` (single or multi-engine)
  3) Describe the pipeline in the unified config schema
  4) Implement a task-specific `BaseDataLoader` and `BaseEvaluator` if needed

- To support mid-graph custom logic:
  - Implement a `midprocessor` and declare its inputs/outputs in `processors.midprocess`
  - The modular pipeline will route tensors between engines and processors by name


## Expected Impact

- Reduced code duplication across projects
- Clear separation of concerns: I/O, engines, processors, metrics
- Easier onboarding for new models and tasks
- CenterPoint-style multi-engine support becomes a first-class feature


## Next Steps (Action Items)

1) Implement core modules: modular pipeline, modular backend, engine manager, processor base + registry
2) Implement initial processors: `ImageNormalization`, `Voxelization`, `YOLOXDecoder`, `CenterPointDecoder`, `ClassificationDecoder`, `PyTorchMiddleEncoder`
3) Add `modular_exporter.py` with single- and multi-engine flows
4) Migrate YOLOX_opt_elan to unified config (keep wrapper as export-time option)
5) Migrate CalibrationStatusClassification to unified config
6) Migrate CenterPoint to engine graph + middle encoder midprocessor
7) Remove duplicated per-model backends/wrappers after verification


---

This design unifies the three existing deployments and provides a scalable path for future models, while explicitly supporting multi-engine graphs and mid-processors like CenterPoint’s PyTorch middle encoder.



