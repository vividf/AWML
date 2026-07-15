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
    B --> C["Load deploy quantization config"]
    C --> E["Build BEVFusion model<br/>init_model with config and checkpoint"]

    E --> F{"fuse_bn enabled?"}
    F -->|yes| F1["Fuse SparseConv + BN<br/>pts_middle_encoder (sparse stays FP16)"]
    F -->|no| G
    F1 --> G["build_bevfusion_plan(config, include_sparse=false).prepare(model)"]

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
    B --> C["calibrate_batches = ceil samples over batch_size"]
    C --> D["Load deploy quantization config"]
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

## CenterPoint QAT Pipeline

Entry point:
`deployment/projects/centerpoint/quantization/quantize.py qat`

Hook:
`deployment/projects/centerpoint/quantization/qat_hook.py`

```mermaid
flowchart TD
    A["Start: centerpoint quantize.py qat"] --> B["Parse args"]
    B --> C["Load train config"]
    C --> D["Register QATHook import"]
    D --> E["Override lr, epochs, batch_size, work_dir"]
    E --> F["Load deploy quantization config"]
    F --> G["Resolve sensitive layers and quant flags"]
    G --> H["Append QATHook to custom_hooks"]
    H --> I["Set cfg.load_from checkpoint"]
    I --> J["Build MMEngine Runner"]
    J --> K["runner.train"]

    K --> L["QATHook.before_train"]
    L --> L1["Build same CenterPoint QuantizationPlan"]
    L1 --> L2["Fuse BN if freeze_bn"]
    L2 --> L3["Insert Q/DQ modules"]
    L3 --> L4["model.train"]

    L4 --> M["QATHook.before_train_epoch at calibration_epoch"]
    M --> N{"ptq_calib_cache provided?"}
    N -->|yes| N1["Load amax values from .calib<br/>Skip new calibration"]
    N -->|no| O["CalibrationManager.calibrate<br/>train dataloader"]
    O --> O1["Enable calib and disable fake quant"]
    O1 --> O2["Run calibration batches<br/>or all train batches"]
    O2 --> O3["Compute amax with MSE"]
    O3 --> O4["Enable fake quant and disable calib"]
    N1 --> P
    O4 --> P["Disable sensitive layers"]
    P --> Q["Continue QAT fine-tuning"]
    Q --> R["QATHook.after_train<br/>log quantizer counts"]
    R --> S["MMEngine saves trained checkpoint in work_dir"]
```

## Implementation Map

| Area | File |
| --- | --- |
| BEVFusion PTQ CLI | `deployment/projects/bevfusion_l/quantization/quantize.py` |
| BEVFusion shared quantization plan | `deployment/projects/bevfusion_l/quantization/plan.py` |
| BEVFusion sparse BN-fuse scheme (FP16) | `deployment/projects/bevfusion_l/quantization/schemes.py` |
| SparseConv+BN fold | `deployment/quantization/sparse/fusion.py` |
| Generic dense Q/DQ scheme | `deployment/quantization/schemes/dense_qdq.py` |
| CenterPoint PTQ/QAT CLI | `deployment/projects/centerpoint/quantization/quantize.py` |
| CenterPoint shared quantization plan | `deployment/projects/centerpoint/quantization/plan.py` |
| CenterPoint quant module composition | `deployment/projects/centerpoint/quantization/quant_model.py` |
| CenterPoint QAT hook | `deployment/projects/centerpoint/quantization/qat_hook.py` |
| Shared calibration manager | `deployment/quantization/core/calibration.py` |
