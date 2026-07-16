# AWML Quantization Pipeline

This note sketches the AWML quantization flows for BEVFusion and CenterPoint.
The diagrams focus on PTQ/QAT preparation and calibration, and point to the
main implementation files in this repository.

## BEVFusion PTQ Pipeline

Entry point:
`deployment/projects/bevfusion_l/quantization/quantize.py`

Shared plan:
`deployment/projects/bevfusion_l/quantization/plan.py`

```mermaid
flowchart TD
    A["Start: bevfusion_l quantize.py ptq"] --> B["Parse args"]
    B --> C["Load deploy quantization config<br/>resolve settings: CLI flags over quantization.ptq block"]
    C --> E["Build BEVFusion model<br/>init_model with config and checkpoint"]

    E --> F{"fuse_bn enabled?"}
    F -->|yes| F1["Fuse SparseConv + BN<br/>pts_middle_encoder (sparse stays FP16)"]
    F -->|no| G
    F1 --> G["build_bevfusion_plan(config).prepare(model)"]

    G --> H{"Dense quant enabled?"}
    H -->|yes| H1["DenseQDQScheme<br/>Fuse BN in pts_backbone, pts_neck, bbox_head<br/>Conv2d to QuantConv2d<br/>optional residual-add quantizers"]
    H -->|no| H2["Dense Q/DQ skipped<br/>optional dense BN fuse only"]

    H1 --> I["Build validation dataloader<br/>batch_size, seed, shuffle"]
    H2 --> I

    I --> P{"Dense quant enabled?"}
    P -->|yes| P1["Dense calibration with CalibrationManager"]
    P -->|no| S["Skip dense calibration"]

    P1 --> P2["Disable fake quant<br/>Enable calib for all TensorQuantizers"]
    P2 --> P3["Run model.test_step over N batches"]
    P3 --> P4["Compute amax with MSE"]
    P4 --> P5["Enable fake quant<br/>Disable calib"]
    P5 --> Q["Disable sensitive layers if configured"]
    Q --> R["Save PTQ checkpoint<br/>Save dense .calib cache"]
    S --> R
    R --> T["Deploy later with deployment.cli.main bevfusion_l"]
```

Key detail: the BEVFusion sparse encoder deploys in **FP16** — PTQ only folds its SparseConv+BN
(so the module tree matches deploy) and never quantizes it. Dense Q/DQ is calibrated against the BEV
distribution produced by the FP16 sparse encoder, which is exactly what the deploy path also runs.

## BEVFusion Calibration Detail

```mermaid
flowchart LR
    subgraph Dense_Calibration[BEVFusion dense QDQ calibration]
        L["Calibration dataloader"] --> M["model.test_step<br/>(FP16 sparse -> dense Q/DQ)"]
        M --> N["TensorQuantizer histograms"]
        N --> O["MSE amax"]
        O --> P["PTQ checkpoint + .calib cache"]
    end
```

Sparse BN fold:
`deployment/quantization/sparse/fusion.py`

Dense calibration implementation:
`deployment/quantization/core/calibration.py`

## CenterPoint PTQ Pipeline

Entry point:
`deployment/projects/centerpoint/quantization/quantize.py`

Shared plan:
`deployment/projects/centerpoint/quantization/plan.py`

```mermaid
flowchart TD
    A["Start: centerpoint quantize.py ptq"] --> B["Parse args"]
    B --> C["Load deploy quantization config<br/>resolve settings: CLI flags over quantization.ptq block"]
    C --> D["calibrate_batches = ceil samples over batch_size"]
    D --> E["Resolve sensitive layers"]
    E --> F["Load CenterPoint model<br/>init_model with config and checkpoint"]

    F --> G["Build CenterPoint QuantizationPlan"]
    G --> H["CenterPointDenseScheme.prepare"]
    H --> H1{"fuse_bn enabled?"}
    H1 -->|yes| H2["Fuse Conv + BN globally"]
    H1 -->|no| H3["Skip BN fusion"]
    H2 --> I
    H3 --> I

    I["quant_model composition"] --> I1["pts_backbone<br/>Conv2d to QuantConv2d"]
    I --> I2["pts_neck<br/>Conv2d to QuantConv2d"]
    I --> I3["pts_bbox_head<br/>Conv2d to QuantConv2d"]
    I --> I4["pts_voxel_encoder<br/>Linear to QuantLinear"]
    I --> I5["Optional recipes<br/>residual add, eSE mul, eSE pool, MaxPool input Q/DQ"]

    I1 --> J["Build validation dataloader<br/>batch_size, seed, shuffle"]
    I2 --> J
    I3 --> J
    I4 --> J
    I5 --> J

    J --> K["CalibrationManager.calibrate"]
    K --> K1["Fast torch histogram mode"]
    K1 --> K2["Disable fake quant<br/>Enable calib"]
    K2 --> K3["Run model.test_step over calibration batches"]
    K3 --> K4["Collect histograms and stats"]
    K4 --> K5["Compute amax with MSE"]
    K5 --> K6["Enable fake quant<br/>Disable calib"]
    K6 --> L["Disable sensitive layers"]
    L --> M["Print quantizer status"]
    M --> N["Save PTQ checkpoint"]
    N --> O["Save .calib cache"]
```

CenterPoint is currently a dense Q/DQ path in this AWML pipeline. The named
components are quantized by `quant_model.py`, while the actual calibration state
machine is shared through `CalibrationManager`.

## QAT Pipeline (CenterPoint + BEVFusion)

QAT is a **frozen-amax STE fine-tune** (calibrated scales stay fixed; only weights train — see
`spec_qat.md` §0/§2 for why this is the production method in both CUDA-CenterPoint and modelopt).
One shared hook body + one shared training driver serve both projects:

Entry points (identical shape; settings come from the deploy config's `quantization.qat` block,
CLI flags override):
`deployment/projects/centerpoint/quantization/quantize.py qat` and
`deployment/projects/bevfusion_l/quantization/quantize.py qat`

Shared machinery:
`deployment/quantization/qat_hook.py` (`QATHookBase`) and
`deployment/quantization/producer.py` (`run_qat_training`, `save_qat_checkpoint`).
Project subclasses supply only the plan (+ BEVFusion's calibration forward):
`centerpoint/quantization/qat_hook.py` (`QATHook`),
`bevfusion_l/quantization/qat_hook.py` (`BEVFusionQATHook`).

```mermaid
flowchart TD
    A["Start: quantize.py qat --deploy-cfg ..."] --> B["Load deploy quantization config<br/>(placement + qat block; CLI overrides)"]
    B --> C["run_qat_training (shared driver)"]
    C --> C1["Force AMP off (fp32 QAT)"]
    C1 --> C2["Strip EMA hooks · refuse resume"]
    C2 --> C3["Override lr, epochs, batch_size, work_dir"]
    C3 --> C4["Append project QATHook to custom_hooks<br/>(fuse_bn, keep_fp16, disable_recipes from deploy cfg)"]
    C4 --> C5["cfg.load_from = FP checkpoint"]
    C5 --> K["Runner.train (single GPU)"]

    K --> L["QATHook.before_train"]
    L --> L1["Build the SAME QuantizationPlan<br/>as PTQ / deploy (tree parity)"]
    L1 --> L2["Fuse BN (fuse_bn)"]
    L2 --> L3["Insert Q/DQ modules"]
    L3 --> L4["model.train"]

    L4 --> M["QATHook.before_train_epoch (epoch 0)"]
    M --> N{"calib cache provided?"}
    N -->|yes| N1["Load amax values from .calib<br/>(reuse the PTQ amax — recommended)"]
    N -->|no| O["CalibrationManager.calibrate<br/>VAL dataloader (clean, un-augmented), MSE amax<br/>(BEVFusion: voxel-dtype forward)"]
    N1 --> P
    O --> P["amax health check + disable keep_fp16 quantizers"]
    P --> Q["Fine-tune weights (STE, frozen amax)"]
    Q --> R["QATHook.after_train<br/>log quantizer counts"]
    R --> S["save_qat_checkpoint:<br/>package best/last as {'state_dict'} + .calib"]
    S --> T["Deploy exactly like a PTQ checkpoint"]
```

The packaged QAT artifact is byte-shape-identical to a PTQ one, so the deploy loaders need no
`mode` branch; `deployment/tests/test_qat_tree_parity.py` pins the tree parity between the hook
path and the plan path.

## Implementation Map

| Area | File |
| --- | --- |
| BEVFusion PTQ/QAT CLI | `deployment/projects/bevfusion_l/quantization/quantize.py` |
| BEVFusion shared quantization plan | `deployment/projects/bevfusion_l/quantization/plan.py` |
| BEVFusion sparse BN-fuse scheme (FP16) | `deployment/projects/bevfusion_l/quantization/schemes.py` |
| BEVFusion calibration forward (voxel dtype) | `deployment/projects/bevfusion_l/quantization/calibration.py` |
| BEVFusion QAT hook | `deployment/projects/bevfusion_l/quantization/qat_hook.py` |
| SparseConv+BN fold | `deployment/quantization/sparse/fusion.py` |
| Generic dense Q/DQ scheme | `deployment/quantization/schemes/dense_qdq.py` |
| CenterPoint PTQ/QAT CLI | `deployment/projects/centerpoint/quantization/quantize.py` |
| CenterPoint shared quantization plan | `deployment/projects/centerpoint/quantization/plan.py` |
| CenterPoint quant module composition | `deployment/projects/centerpoint/quantization/quant_model.py` |
| CenterPoint QAT hook | `deployment/projects/centerpoint/quantization/qat_hook.py` |
| Shared QAT hook body | `deployment/quantization/qat_hook.py` |
| Shared QAT training driver + packaging | `deployment/quantization/producer.py` |
| Shared calibration manager | `deployment/quantization/core/calibration.py` |
| QAT ↔ PTQ tree-parity test | `deployment/tests/test_qat_tree_parity.py` |
