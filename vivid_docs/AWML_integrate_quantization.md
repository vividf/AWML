# AWML CenterPoint Quantization Integration Plan

## Overview

This document outlines the plan to integrate PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training) pipelines into the AWML CenterPoint deployment framework, based on the approach used in CUDA-CenterPoint (Lidar_AI_Solution).

---

## 1. Analysis of CUDA-CenterPoint Quantization Implementation

### 1.1 Core Components

The CUDA-CenterPoint implementation uses the following key components:

| Component | File | Purpose |
|-----------|------|---------|
| `SparseConvolutionQunat` | `tools/sparseconv_quantization.py` | Quantized sparse convolution wrapper |
| `QuantAdd` | `tools/sparseconv_quantization.py` | Quantized addition operation |
| `quant_sparseconv_module()` | `tools/sparseconv_quantization.py` | Replace SparseConv modules with quantized versions |
| `quant_add_module()` | `tools/sparseconv_quantization.py` | Insert quantized add to residual blocks |
| `calibrate_model()` | `tools/sparseconv_quantization.py` | PTQ calibration with histogram method |
| `layer_fusion_bn()` | `onnx_export/funcs.py` | Fuse BatchNorm into convolutions |
| `layer_fusion_relu()` | `onnx_export/funcs.py` | Fuse ReLU activations |
| `export_onnx()` | `onnx_export/exptool.py` | Export with Q/DQ nodes and dynamic ranges |

### 1.2 Quantization Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PTQ Workflow                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Load FP32 model                                                      │
│  2. Insert Q/DQ nodes (quant_sparseconv_module + quant_add_module)      │
│  3. Fuse BatchNorm layers (layer_fusion_bn)                             │
│  4. Calibrate with histogram method (calibrate_model)                   │
│  5. Disable quantization for sensitive layers                           │
│  6. Export to ONNX with dynamic range attributes                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        QAT Workflow                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Load FP32 model                                                      │
│  2. Insert Q/DQ nodes (quant_sparseconv_module + quant_add_module)      │
│  3. Fuse BatchNorm layers (layer_fusion_bn)                             │
│  4. Calibrate with histogram method (initial calibration)               │
│  5. Disable quantization for sensitive layers                           │
│  6. Fine-tune model with reduced learning rate                          │
│  7. Fuse ReLU layers (layer_fusion_relu)                                │
│  8. Export to ONNX with dynamic range attributes                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Implementation Details

#### Q/DQ Node Insertion Strategy
```python
# For SparseConvolution - replaces module in-place
class SparseConvolutionQunat(SparseConvolution, _utils.QuantMixin):
    default_quant_desc_input  = tensor_quant.QuantDescriptor(num_bits=8, calib_method='histogram')
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def forward(self, input):
        # Quantize input features
        input._features = self._input_quantizer(input._features)
        # Quantize weights
        quant_weight = self._weight_quantizer(self.weight)
        return self._conv_forward(..., quant_weight, ...)

# For Add operations in residual blocks
class QuantAdd(nn.Module, _utils.QuantInputMixin):
    def forward(self, input1, input2):
        quant_input1 = self._input_quantizer(input1)
        quant_input2 = self._input_quantizer(input2)
        return torch.add(quant_input1, quant_input2)
```

#### Layer Sensitivity and Selective Disabling
```python
# First few layers typically don't quantize well
disable_quantization(model.backbone.conv_input).apply()
disable_quantization(model.backbone.conv1[0].conv1).apply()
disable_quantization(model.backbone.conv1[0].conv2).apply()
disable_quantization(model.backbone.conv1[0].quant_add).apply()
```

#### ONNX Export with Quantization Attributes
```python
# When exporting, embed dynamic range info as node attributes
nodes.append(
    helper.make_node(
        "SparseConvolution", inputs, [get_tensor_id(y)], f"conv{ilayer}",
        ...
        input_dynamic_range  = self.input_quantizer.amax.cpu().item(),
        weight_dynamic_ranges = self.weight_quantizer.amax.cpu().view(-1).numpy().tolist(),
        precision = "int8",
        output_precision = "int8"
    )
)
```

---

## 2. AWML CenterPoint Architecture Differences

### 2.1 Architecture Comparison

| Aspect | CUDA-CenterPoint | AWML CenterPoint |
|--------|------------------|------------------|
| **Voxel Encoder** | 3D VoxelNet | Pillar-based (BackwardPillarFeatureNet) |
| **Middle Encoder** | 3D Sparse Convolution | PointPillarsScatter (2D) |
| **Backbone** | SpMiddleResNetFHD (3D SparseConv) | SECOND (2D CNN) / ConvNeXt |
| **Neck** | - | SECONDFPN (2D) |
| **Framework** | Det3D + spconv | MMDet3D + MMEngine |

### 2.2 Modules Requiring Quantization

#### AWML CenterPoint Components:
1. **PillarFeatureNet (pts_voxel_encoder)**
   - PFNLayer: Linear → BN1d → ReLU → Max
   - Standard PyTorch modules (easy to quantize)

2. **SECOND Backbone (pts_backbone)**
   - Conv2d → BN2d → ReLU blocks
   - Standard 2D convolutions

3. **SECONDFPN Neck (pts_neck)**
   - Deconvolution layers
   - Conv2d → BN2d → ReLU

4. **CenterHead (pts_bbox_head)**
   - Shared Conv layers
   - Task-specific heads

---

## 3. Integration Plan

### Phase 1: Infrastructure Setup (Week 1-2)

#### 3.1 Create Quantization Module Structure
```
projects/CenterPoint/
├── quantization/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── quant_conv.py          # Quantized Conv2d wrapper
│   │   ├── quant_linear.py        # Quantized Linear wrapper
│   │   └── quant_add.py           # Quantized Add for skip connections
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrator.py          # Calibration manager
│   │   └── methods.py             # Histogram, Max, MSE methods
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── bn_fusion.py           # BN fusion utilities
│   │   └── relu_fusion.py         # ReLU fusion utilities
│   ├── export/
│   │   ├── __init__.py
│   │   └── onnx_quantized.py      # Export with Q/DQ nodes
│   └── tools/
│       ├── __init__.py
│       ├── sensitivity.py         # Layer sensitivity profiler
│       └── utils.py               # Helper functions
```

#### 3.2 Add Dependencies
```python
# In pyproject.toml or requirements.txt
pytorch-quantization >= 2.1.2  # NVIDIA TensorRT quantization toolkit
```

### Phase 2: Quantization Module Implementation (Week 2-4)

#### 3.3 Implement Core Quantization Classes

**File: `projects/CenterPoint/quantization/modules/quant_conv.py`**
```python
import torch.nn as nn
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils

class QuantConv2d(nn.Conv2d, _utils.QuantMixin):
    """Quantized Conv2d with per-channel weight quantization."""

    default_quant_desc_input = tensor_quant.QuantDescriptor(
        num_bits=8, calib_method='histogram'
    )
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, x):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)
        return self._conv_forward(quant_input, quant_weight, self.bias)
```

**File: `projects/CenterPoint/quantization/modules/quant_linear.py`**
```python
import torch.nn as nn
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils

class QuantLinear(nn.Linear, _utils.QuantMixin):
    """Quantized Linear for PillarFeatureNet."""

    default_quant_desc_input = tensor_quant.QuantDescriptor(
        num_bits=8, calib_method='histogram'
    )
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, x):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)
        return nn.functional.linear(quant_input, quant_weight, self.bias)
```

**File: `projects/CenterPoint/quantization/modules/quant_add.py`**
```python
import torch
import torch.nn as nn
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils

class QuantAdd(nn.Module, _utils.QuantInputMixin):
    """Quantized addition for skip connections in SECOND/FPN."""

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self):
        super().__init__()
        self.init_quantizer(self.default_quant_desc_input)

    def forward(self, x1, x2):
        quant_x1 = self._input_quantizer(x1)
        quant_x2 = self._input_quantizer(x2)
        return quant_x1 + quant_x2
```

#### 3.4 Implement Module Replacement Functions

**File: `projects/CenterPoint/quantization/replace.py`**
```python
import torch.nn as nn
from typing import Type

from .modules import QuantConv2d, QuantLinear, QuantAdd


def transfer_to_quantization(nn_instance: nn.Module, quant_module: Type) -> nn.Module:
    """Transfer weights from original module to quantized version."""
    quant_instance = quant_module.__new__(quant_module)
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)
    quant_instance.init_quantizer(
        quant_module.default_quant_desc_input,
        quant_module.default_quant_desc_weight
    )
    return quant_instance


def quant_conv_module(model: nn.Module):
    """Replace all Conv2d modules with QuantConv2d."""
    def replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            submodule_name = name if prefix == "" else prefix + "." + name
            replace_module(submodule, submodule_name)

            if isinstance(submodule, nn.Conv2d) and not isinstance(submodule, QuantConv2d):
                module._modules[name] = transfer_to_quantization(submodule, QuantConv2d)

    replace_module(model)


def quant_linear_module(model: nn.Module):
    """Replace all Linear modules with QuantLinear."""
    def replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            submodule_name = name if prefix == "" else prefix + "." + name
            replace_module(submodule, submodule_name)

            if isinstance(submodule, nn.Linear) and not isinstance(submodule, QuantLinear):
                module._modules[name] = transfer_to_quantization(submodule, QuantLinear)

    replace_module(model)
```

### Phase 3: Calibration Implementation (Week 4-5)

#### 3.5 Implement Calibration Manager

**File: `projects/CenterPoint/quantization/calibration/calibrator.py`**
```python
import torch
from tqdm import tqdm
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib


class CalibrationManager:
    """Manages PTQ calibration for CenterPoint model."""

    def __init__(self, model: torch.nn.Module, calib_method: str = "histogram"):
        self.model = model
        self.calib_method = calib_method

    def collect_stats(self, dataloader, num_batches: int = 100):
        """Collect statistics for calibration."""
        self.model.eval()

        # Enable calibration mode
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed calibration data
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Calibrating")):
                if i >= num_batches:
                    break
                _ = self.model(**batch)

        # Disable calibration mode
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    def compute_amax(self, method: str = "mse"):
        """Compute amax values from collected statistics."""
        device = next(self.model.parameters()).device

        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(method=method)
                    module._amax = module._amax.to(device)

    def calibrate(self, dataloader, num_batches: int = 100, method: str = "mse"):
        """Run full calibration pipeline."""
        self.collect_stats(dataloader, num_batches)
        self.compute_amax(method)
```

### Phase 4: Layer Fusion (Week 5-6)

#### 3.6 Implement BN Fusion

**File: `projects/CenterPoint/quantization/fusion/bn_fusion.py`**
```python
import torch
import torch.nn as nn


def fuse_bn_weights(conv_weight, conv_bias, bn_mean, bn_var, bn_eps, bn_weight, bn_bias):
    """Fuse BatchNorm parameters into Conv weights."""
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)

    # Fuse weights: W_fused = W * (bn_weight * rsqrt(bn_var))
    conv_weight = conv_weight * (bn_weight * bn_var_rsqrt).reshape(-1, 1, 1, 1)

    # Fuse bias: b_fused = (conv_bias - bn_mean) * rsqrt(bn_var) * bn_weight + bn_bias
    conv_bias = (conv_bias - bn_mean) * bn_var_rsqrt * bn_weight + bn_bias

    return nn.Parameter(conv_weight), nn.Parameter(conv_bias)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fuse Conv2d and BatchNorm2d modules in-place."""
    assert not conv.training and not bn.training, "Fusion only for eval mode"

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight, conv.bias,
        bn.running_mean, bn.running_var, bn.eps,
        bn.weight, bn.bias
    )


def fuse_model_bn(model: nn.Module):
    """Fuse all Conv-BN pairs in the model."""
    prev_name = None
    prev_module = None
    modules_to_fuse = []

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if prev_module is not None and isinstance(prev_module, nn.Conv2d):
                modules_to_fuse.append((prev_name, name))
        prev_name = name
        prev_module = module

    # Perform fusion
    for conv_name, bn_name in modules_to_fuse:
        conv = dict(model.named_modules())[conv_name]
        bn = dict(model.named_modules())[bn_name]
        fuse_conv_bn(conv, bn)
        # Remove BN (replace with identity)
        parent_name = ".".join(bn_name.split(".")[:-1])
        bn_attr = bn_name.split(".")[-1]
        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)
        setattr(parent, bn_attr, nn.Identity())

    return model
```

### Phase 5: Integration with Deployment Pipeline (Week 6-7)

#### 3.7 Create Quantization-Aware Export Pipeline

**File: `deployment/exporters/centerpoint/quantized_export_pipeline.py`**
```python
from pathlib import Path
from typing import Optional
import logging
import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
from deployment.exporters.export_pipelines.base import OnnxExportPipeline


class CenterPointQuantizedONNXExportPipeline(OnnxExportPipeline):
    """ONNX export pipeline with INT8 quantization support."""

    def __init__(
        self,
        exporter_factory,
        component_extractor,
        config: BaseDeploymentConfig,
        quantization_config: dict = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(exporter_factory, component_extractor, config, logger)
        self.quantization_config = quantization_config or {}

    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
        calibration_dataloader=None,
        num_calibration_batches: int = 100,
    ) -> Artifact:
        """Export with optional quantization."""

        if self.quantization_config.get("enabled", False):
            self.logger.info("Applying PTQ quantization...")
            model = self._apply_quantization(
                model,
                calibration_dataloader,
                num_calibration_batches
            )

        return super().export(
            model=model,
            data_loader=data_loader,
            output_dir=output_dir,
            config=config,
            sample_idx=sample_idx,
        )

    def _apply_quantization(self, model, dataloader, num_batches):
        """Apply PTQ to the model."""
        from projects.CenterPoint.quantization.replace import (
            quant_conv_module, quant_linear_module
        )
        from projects.CenterPoint.quantization.calibration import CalibrationManager
        from projects.CenterPoint.quantization.fusion import fuse_model_bn

        # Step 1: Insert Q/DQ nodes
        quant_conv_module(model.pts_backbone)
        quant_conv_module(model.pts_neck)
        quant_conv_module(model.pts_bbox_head)
        quant_linear_module(model.pts_voxel_encoder)

        # Step 2: Fuse BatchNorm
        fuse_model_bn(model)

        # Step 3: Calibrate
        calibrator = CalibrationManager(model)
        calibrator.calibrate(dataloader, num_batches)

        # Step 4: Disable sensitive layers (if configured)
        self._disable_sensitive_layers(model)

        return model

    def _disable_sensitive_layers(self, model):
        """Disable quantization for sensitive layers."""
        from projects.CenterPoint.quantization.utils import disable_quantization

        sensitive_layers = self.quantization_config.get("sensitive_layers", [])
        for layer_name in sensitive_layers:
            try:
                layer = dict(model.named_modules())[layer_name]
                disable_quantization(layer).apply()
                self.logger.info(f"Disabled quantization for: {layer_name}")
            except KeyError:
                self.logger.warning(f"Layer not found: {layer_name}")
```

#### 3.8 Update Deployment Configuration

**File: `projects/CenterPoint/deploy/configs/deploy_config.py` (additions)**
```python
# Quantization Configuration
quantization = dict(
    enabled=False,  # Set to True to enable PTQ/QAT
    mode="ptq",     # 'ptq' or 'qat'
    calibration=dict(
        num_batches=100,
        method="histogram",  # 'histogram', 'max', 'mse'
    ),
    fusion=dict(
        fuse_bn=True,
        fuse_relu=True,
    ),
    sensitive_layers=[
        # Layers to skip quantization (determined by sensitivity analysis)
        # "pts_voxel_encoder.pfn_layers.0",
        # "pts_backbone.blocks.0",
    ],
    precision=dict(
        input="fp16",      # Input precision
        weights="int8",    # Weight precision
        output="fp16",     # Output precision
    ),
)
```

### Phase 6: QAT Training Integration (Week 7-8)

#### 3.9 Create QAT Training Hook

**File: `projects/CenterPoint/quantization/hooks/qat_hook.py`**
```python
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class QATHook(Hook):
    """Hook for QAT training with CenterPoint."""

    def __init__(
        self,
        calibration_batches: int = 100,
        start_epoch: int = 0,
        freeze_bn: bool = True,
    ):
        self.calibration_batches = calibration_batches
        self.start_epoch = start_epoch
        self.freeze_bn = freeze_bn
        self._calibrated = False

    def before_train(self, runner):
        """Insert Q/DQ nodes and calibrate before training."""
        from projects.CenterPoint.quantization.replace import (
            quant_conv_module, quant_linear_module
        )
        from projects.CenterPoint.quantization.calibration import CalibrationManager
        from projects.CenterPoint.quantization.fusion import fuse_model_bn

        model = runner.model

        # Insert quantization modules
        quant_conv_module(model.pts_backbone)
        quant_conv_module(model.pts_neck)
        quant_conv_module(model.pts_bbox_head)
        quant_linear_module(model.pts_voxel_encoder)

        # Fuse BN layers
        if self.freeze_bn:
            model.eval()
            fuse_model_bn(model)
            model.train()

    def before_train_epoch(self, runner):
        """Calibrate at start_epoch."""
        if runner.epoch == self.start_epoch and not self._calibrated:
            model = runner.model
            dataloader = runner.train_dataloader

            calibrator = CalibrationManager(model)
            calibrator.calibrate(dataloader, self.calibration_batches)
            self._calibrated = True
            runner.logger.info("QAT calibration completed")
```

### Phase 7: Tools and Utilities (Week 8-9)

#### 3.10 Sensitivity Profiler

**File: `projects/CenterPoint/quantization/tools/sensitivity.py`**
```python
import torch
from typing import Callable, List
from pytorch_quantization import nn as quant_nn


def build_sensitivity_profile(
    model: torch.nn.Module,
    val_dataloader,
    eval_fn: Callable,
    output_file: str = "sensitivity_profile.csv",
):
    """
    Analyze the impact of each quantized layer on model accuracy.

    Args:
        model: Quantized model
        val_dataloader: Validation dataloader
        eval_fn: Function that returns mAP given model and dataloader
        output_file: Output CSV file path
    """
    # Get all quantizer layer names
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)

    # Disable all quantizers first
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()

    # Get baseline (FP32) accuracy
    baseline_map = eval_fn(model, val_dataloader)
    print(f"Baseline mAP (FP32): {baseline_map:.4f}")

    results = []

    # Enable each layer one at a time and measure impact
    for quant_layer in quant_layer_names:
        # Enable this layer's quantizers
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()

        # Evaluate
        with torch.no_grad():
            layer_map = eval_fn(model, val_dataloader)

        delta = baseline_map - layer_map
        results.append({
            "layer": quant_layer,
            "mAP": layer_map,
            "delta": delta,
        })
        print(f"Layer: {quant_layer}, mAP: {layer_map:.4f}, Delta: {delta:.4f}")

        # Disable this layer's quantizers
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()

    # Save results
    import csv
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mAP", "delta"])
        writer.writeheader()
        writer.writerows(results)

    # Return layers sorted by impact (highest delta first)
    results.sort(key=lambda x: x["delta"], reverse=True)
    return results
```

#### 3.11 CLI Tools

**File: `tools/detection3d/centerpoint_quantization.py`**
```python
"""
CenterPoint Quantization Tools

Usage:
    # PTQ Mode
    python tools/detection3d/centerpoint_quantization.py ptq \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --output work_dirs/centerpoint_ptq.pth

    # Sensitivity Analysis
    python tools/detection3d/centerpoint_quantization.py sensitivity \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --output sensitivity_report.csv

    # QAT Mode
    python tools/detection3d/centerpoint_quantization.py qat \
        --config projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
        --checkpoint work_dirs/centerpoint/best.pth \
        --calibrate-batches 100 \
        --epochs 10 \
        --lr 0.0001 \
        --output work_dirs/centerpoint_qat.pth
"""

import argparse
import sys
sys.path.insert(0, ".")

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="CenterPoint Quantization Tools")
    subparsers = parser.add_subparsers(dest="command")

    # PTQ command
    ptq_parser = subparsers.add_parser("ptq", help="Post-Training Quantization")
    ptq_parser.add_argument("--config", required=True, help="Model config file")
    ptq_parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    ptq_parser.add_argument("--calibrate-batches", type=int, default=100)
    ptq_parser.add_argument("--output", required=True, help="Output checkpoint path")

    # Sensitivity command
    sens_parser = subparsers.add_parser("sensitivity", help="Sensitivity Analysis")
    sens_parser.add_argument("--config", required=True, help="Model config file")
    sens_parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    sens_parser.add_argument("--calibrate-batches", type=int, default=100)
    sens_parser.add_argument("--output", default="sensitivity_report.csv")

    # QAT command
    qat_parser = subparsers.add_parser("qat", help="Quantization-Aware Training")
    qat_parser.add_argument("--config", required=True, help="Model config file")
    qat_parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    qat_parser.add_argument("--calibrate-batches", type=int, default=100)
    qat_parser.add_argument("--epochs", type=int, default=10)
    qat_parser.add_argument("--lr", type=float, default=0.0001)
    qat_parser.add_argument("--output", required=True, help="Output checkpoint path")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ptq":
        run_ptq(args)
    elif args.command == "sensitivity":
        run_sensitivity(args)
    elif args.command == "qat":
        run_qat(args)
    else:
        print("Please specify a command: ptq, sensitivity, or qat")


def run_ptq(args):
    """Run PTQ quantization."""
    from projects.CenterPoint.quantization.ptq import quantize_ptq

    cfg = Config.fromfile(args.config)
    model = quantize_ptq(
        cfg,
        args.checkpoint,
        calibrate_batches=args.calibrate_batches,
    )

    import torch
    torch.save({"state_dict": model.state_dict()}, args.output)
    print(f"PTQ model saved to: {args.output}")


def run_sensitivity(args):
    """Run sensitivity analysis."""
    from projects.CenterPoint.quantization.tools.sensitivity import build_sensitivity_profile

    cfg = Config.fromfile(args.config)
    # ... build model and dataloader

    results = build_sensitivity_profile(
        model, val_dataloader, eval_fn,
        output_file=args.output
    )

    print("\nTop 10 Sensitive Layers:")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['layer']}: delta={r['delta']:.4f}")


def run_qat(args):
    """Run QAT training."""
    cfg = Config.fromfile(args.config)

    # Modify config for QAT
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.train_cfg.max_epochs = args.epochs
    cfg.custom_hooks.append(
        dict(
            type="QATHook",
            calibration_batches=args.calibrate_batches,
        )
    )

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
```

---

## 4. Expected Results and Verification

### 4.1 Performance Targets

Based on CUDA-CenterPoint results:

| Model | Validation mAP | Validation NDS | Notes |
|-------|----------------|----------------|-------|
| FP16 Baseline | ~59.55% | ~66.75% | Original model |
| PTQ INT8 | ~59.08% | ~66.45% | <0.5% mAP drop |
| QAT INT8 | ~59.20% | ~66.53% | <0.35% mAP drop |

### 4.2 Verification Checklist

- [ ] PTQ model produces valid ONNX output
- [ ] Q/DQ nodes are correctly placed in ONNX graph
- [ ] TensorRT can build INT8 engine from quantized ONNX
- [ ] INT8 inference matches FP16 within tolerance
- [ ] mAP degradation is within acceptable range (<1%)
- [ ] Latency improvement on target hardware

### 4.3 TensorRT INT8 Engine Build

```python
# TensorRT config for INT8
backend_config = dict(
    common_config=dict(
        precision_policy="int8",  # Enable INT8
        max_workspace_size=4 << 30,
    ),
    calibration=dict(
        # For TensorRT native calibration (if not using PTQ ONNX)
        calibrator="entropy",  # or "minmax"
        num_calibration_batches=100,
    ),
)
```

---

## 5. Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Infrastructure Setup | Week 1-2 | Directory structure, dependencies |
| Phase 2: Quantization Modules | Week 2-4 | QuantConv2d, QuantLinear, QuantAdd, replace functions |
| Phase 3: Calibration | Week 4-5 | CalibrationManager, histogram/max/mse methods |
| Phase 4: Layer Fusion | Week 5-6 | BN fusion, ReLU fusion utilities |
| Phase 5: Deployment Integration | Week 6-7 | QuantizedONNXExportPipeline, config updates |
| Phase 6: QAT Training | Week 7-8 | QATHook, training integration |
| Phase 7: Tools & Utilities | Week 8-9 | Sensitivity profiler, CLI tools |
| Phase 8: Testing & Documentation | Week 9-10 | End-to-end testing, documentation |

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Accuracy degradation > 1% | Use sensitivity analysis to identify and skip problematic layers |
| TensorRT compatibility issues | Test with multiple TensorRT versions, use standard Q/DQ patterns |
| MMEngine integration complexity | Start with standalone scripts, then integrate with hooks |
| Different spconv versions | AWML uses pillar-based (no 3D spconv), reducing complexity |

---

## 7. References

- [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [pytorch-quantization Documentation](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [CUDA-CenterPoint QAT Implementation](../../../Lidar_AI_Solution/CUDA-CenterPoint/qat/)
- [MMDet3D CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)

---

## 8. Appendix: Key API Reference

### pytorch_quantization Core Classes

```python
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.nn.modules import _utils

# Quantization descriptors
tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

# Calibrators
calib.MaxCalibrator
calib.HistogramCalibrator
calib.EntropyCalibrator

# Mixin classes
_utils.QuantMixin        # For Conv, Linear
_utils.QuantInputMixin   # For custom ops (Add, etc.)
```

### AWML CenterPoint Model Structure

```python
CenterPoint
├── pts_voxel_encoder (BackwardPillarFeatureNet)
│   └── pfn_layers (ModuleList[PFNLayer])
│       └── PFNLayer: Linear → BN1d → ReLU → Max
├── pts_middle_encoder (PointPillarsScatter)
│   └── No learnable parameters
├── pts_backbone (SECOND)
│   └── blocks (ModuleList)
│       └── Sequential: Conv2d → BN2d → ReLU
├── pts_neck (SECONDFPN)
│   └── deblocks (ModuleList)
│       └── Sequential: ConvTranspose2d → BN2d → ReLU
└── pts_bbox_head (CenterHead)
    ├── shared_conv
    └── task_heads (ModuleList)
```
