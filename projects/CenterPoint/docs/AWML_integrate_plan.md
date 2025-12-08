# AWML CenterPoint Quantization Integration Plan

This document outlines the plan to integrate PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training) pipelines into the AWML CenterPoint deployment framework, based on the approach used in CUDA-CenterPoint (Lidar_AI_Solution).

---

## 1. Analysis of CUDA-CenterPoint Q/DQ Node Insertion Mechanism

### 1.1 Overview

The CUDA-CenterPoint implementation uses NVIDIA's `pytorch-quantization` library to insert Quantization/Dequantization (Q/DQ) nodes into the neural network graph. These nodes simulate INT8 quantization during training and inference, enabling:

- **PTQ (Post-Training Quantization)**: Calibrate a pre-trained FP32 model to INT8
- **QAT (Quantization-Aware Training)**: Fine-tune with fake quantization for better accuracy

### 1.2 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `SparseConvolutionQunat` | `tools/sparseconv_quantization.py` | Quantized sparse convolution wrapper |
| `QuantAdd` | `tools/sparseconv_quantization.py` | Quantized addition for residual connections |
| `quant_sparseconv_module()` | `tools/sparseconv_quantization.py` | Replace SparseConv modules with quantized versions |
| `quant_add_module()` | `tools/sparseconv_quantization.py` | Insert quantized add to residual blocks |
| `calibrate_model()` | `tools/sparseconv_quantization.py` | PTQ calibration with histogram method |
| `layer_fusion_bn()` | `onnx_export/funcs.py` | Fuse BatchNorm into convolutions |
| `layer_fusion_relu()` | `onnx_export/funcs.py` | Fuse ReLU activations |
| `export_onnx()` | `onnx_export/exptool.py` | Export with Q/DQ nodes and dynamic ranges |

### 1.3 Q/DQ Node Insertion Pattern

The key insight is that Q/DQ nodes are inserted by **replacing standard PyTorch modules** with quantization-aware versions that inherit from both the original module class and `_utils.QuantMixin`:

```python
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils

class SparseConvolutionQunat(SparseConvolution, _utils.QuantMixin):
    """Quantized SparseConvolution with per-channel weight quantization."""

    # Define quantization parameters
    default_quant_desc_input = tensor_quant.QuantDescriptor(
        num_bits=8,
        calib_method='histogram'
    )
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self, ...):
        super().__init__(...)
        # Initialize quantizers from QuantMixin
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, input):
        # Q/DQ on input activations
        input._features = self._input_quantizer(input._features)

        # Q/DQ on weights
        quant_weight = self._weight_quantizer(self.weight)

        # Forward with quantized tensors
        return self._conv_forward(..., quant_weight, ...)
```

### 1.4 Module Replacement Strategy

The `quant_sparseconv_module()` function recursively traverses the model and replaces modules in-place:

```python
def transfer_spconv_to_quantization(nn_instance, quantmodule):
    """Transfer weights from original module to quantized version."""
    quant_instance = quantmodule.__new__(quantmodule)

    # Copy all attributes from original module
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    # Initialize quantizers
    quant_instance.init_quantizer(
        quantmodule.default_quant_desc_input,
        quantmodule.default_quant_desc_weight
    )
    return quant_instance

def quant_sparseconv_module(model):
    """Replace all SparseConv modules with quantized versions."""
    def replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            replace_module(submodule, f"{prefix}.{name}" if prefix else name)

            if isinstance(submodule, (spconv.SubMConv3d, spconv.SparseConv3d)):
                module._modules[name] = transfer_spconv_to_quantization(
                    submodule, SparseConvolutionQunat
                )
    replace_module(model)
```

### 1.5 QuantAdd for Residual Connections

For residual blocks, the element-wise addition needs separate quantization to handle different input scales:

```python
class QuantAdd(nn.Module, _utils.QuantInputMixin):
    """Quantized addition for skip connections."""

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self):
        super().__init__()
        self.init_quantizer(self.default_quant_desc_input)

    def forward(self, input1, input2):
        # Same quantizer for both inputs (same scale)
        quant_input1 = self._input_quantizer(input1)
        quant_input2 = self._input_quantizer(input2)
        return torch.add(quant_input1, quant_input2)
```

### 1.6 Calibration Process

PTQ calibration collects activation statistics to determine quantization scales (amax values):

```python
def calibrate_model(model, dataloader, device, batch_processor_callback, num_batch=25):

    def collect_stats(model, data_loader, num_batch):
        """Feed data to collect statistics."""
        model.eval()

        # Enable calibration mode on all TensorQuantizers
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()  # Disable fake quantization
                    module.enable_calib()   # Enable statistics collection
                else:
                    module.disable()

        # Feed calibration data
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= num_batch:
                    break
                batch_processor_callback(model, data)

        # Restore quantization mode
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    def compute_amax(model, method="mse"):
        """Compute optimal amax from collected statistics."""
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(method=method)
                    module._amax = module._amax.to(device)

    collect_stats(model, dataloader, num_batch)
    compute_amax(model, method="mse")
```

### 1.7 Layer Fusion

Before quantization, BatchNorm layers are fused into preceding convolutions to reduce quantization error:

```python
def fuse_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """Fuse BatchNorm parameters into convolution weights."""
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    # W_fused = W * (gamma / sqrt(var + eps))
    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (conv_w.ndim - 1))

    # b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return nn.Parameter(conv_w), nn.Parameter(conv_b)
```

### 1.8 Sensitivity Analysis

To identify layers that cause significant accuracy degradation when quantized:

```python
def build_sensitivity_profile(model, val_loader, eval_fn):
    """Analyze quantization impact of each layer."""

    # Disable all quantizers
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)

    # Test each layer individually
    for quant_layer in quant_layer_names:
        # Enable only this layer's quantizers
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()

        # Evaluate impact
        mAP = eval_fn(model, val_loader)
        print(f"Layer: {quant_layer}, mAP: {mAP}")

        # Disable again
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()
```

### 1.9 Selective Layer Disabling

Based on sensitivity analysis, the first few layers are typically kept in FP16:

```python
def disable_quant_layer(model):
    """Disable quantization for sensitive layers."""
    # First conv layer
    disable_quantization(model.backbone.conv_input).apply()

    # First residual block
    disable_quantization(model.backbone.conv1[0].conv1).apply()
    disable_quantization(model.backbone.conv1[0].conv2).apply()
    disable_quantization(model.backbone.conv1[0].quant_add).apply()
```

### 1.10 ONNX Export with Quantization Attributes

When exporting to ONNX, dynamic range information is embedded as node attributes:

```python
nodes.append(
    helper.make_node(
        "SparseConvolution", inputs, [get_tensor_id(y)], f"conv{ilayer}",
        # ... other attributes ...
        input_dynamic_range=self.input_quantizer.amax.cpu().item(),
        weight_dynamic_ranges=self.weight_quantizer.amax.cpu().view(-1).numpy().tolist(),
        precision="int8",
        output_precision="int8"
    )
)
```

---

## 2. AWML CenterPoint Architecture Analysis

### 2.1 Architecture Comparison

| Aspect | CUDA-CenterPoint | AWML CenterPoint |
|--------|------------------|------------------|
| **Voxel Encoder** | 3D VoxelNet | Pillar-based (BackwardPillarFeatureNet) |
| **Middle Encoder** | 3D Sparse Convolution | PointPillarsScatter (2D scatter) |
| **Backbone** | SpMiddleResNetFHD (3D SparseConv) | SECOND (2D CNN) / ConvNeXt |
| **Neck** | - | SECONDFPN (2D) |
| **Head** | CenterHead | CenterHead |
| **Framework** | Det3D + spconv | MMDet3D + MMEngine |

### 2.2 AWML CenterPoint Model Structure

```
CenterPoint
├── data_preprocessor (Det3DDataPreprocessor)
│   └── voxel_layer - Voxelization
│
├── pts_voxel_encoder (BackwardPillarFeatureNet)
│   └── pfn_layers (ModuleList[PFNLayer])
│       └── PFNLayer: Linear → BN1d → ReLU → Max pooling
│
├── pts_middle_encoder (PointPillarsScatter)
│   └── No learnable parameters (scatter operation)
│
├── pts_backbone (SECOND)
│   └── blocks (ModuleList[Sequential])
│       └── Sequential: Conv2d → BN2d → ReLU (repeated)
│
├── pts_neck (SECONDFPN)
│   └── deblocks (ModuleList[Sequential])
│       └── Sequential: ConvTranspose2d → BN2d → ReLU
│
└── pts_bbox_head (CenterHead)
    ├── shared_conv: Conv2d → BN2d → ReLU
    └── task_heads (ModuleList)
        └── SeparateHead: Conv2d layers for each output
```

### 2.3 Modules Requiring Quantization

| Component | Module Type | Quantization Approach |
|-----------|-------------|----------------------|
| `pts_voxel_encoder.pfn_layers` | Linear + BN1d | QuantLinear |
| `pts_backbone.blocks` | Conv2d + BN2d | QuantConv2d |
| `pts_neck.deblocks` | ConvTranspose2d + BN2d | QuantConvTranspose2d |
| `pts_bbox_head.shared_conv` | Conv2d + BN2d | QuantConv2d |
| `pts_bbox_head.task_heads` | Conv2d | QuantConv2d |

### 2.4 Key Simplification

AWML uses **pillar-based encoding** with standard 2D operations, which means:
- No need for custom `SparseConvolutionQunat` class
- Can use standard `pytorch-quantization` QuantConv2d directly
- Simpler module replacement logic
- Better TensorRT compatibility

---

## 3. Implementation Phases

### Phase 1: Create Quantization Module Structure

Create the following directory structure:

```
projects/CenterPoint/quantization/
├── __init__.py
├── modules/
│   ├── __init__.py
│   ├── quant_conv.py          # QuantConv2d, QuantConvTranspose2d
│   ├── quant_linear.py        # QuantLinear for PFNLayer
│   └── quant_add.py           # QuantAdd for skip connections (if any)
├── calibration/
│   ├── __init__.py
│   └── calibrator.py          # CalibrationManager class
├── fusion/
│   ├── __init__.py
│   └── bn_fusion.py           # BN fusion utilities
├── replace.py                 # Module replacement functions
└── utils.py                   # Helper functions (enable/disable quantization)
```

#### `modules/quant_conv.py`

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

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, x):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)
        return self._conv_forward(quant_input, quant_weight, self.bias)


class QuantConvTranspose2d(nn.ConvTranspose2d, _utils.QuantMixin):
    """Quantized ConvTranspose2d for FPN upsample layers."""

    default_quant_desc_input = tensor_quant.QuantDescriptor(
        num_bits=8, calib_method='histogram'
    )
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, x, output_size=None):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,
            num_spatial_dims=2, dilation=self.dilation
        )
        return nn.functional.conv_transpose2d(
            quant_input, quant_weight, self.bias, self.stride,
            self.padding, output_padding, self.groups, self.dilation
        )
```

#### `modules/quant_linear.py`

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

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.init_quantizer(
            self.default_quant_desc_input,
            self.default_quant_desc_weight
        )

    def forward(self, x):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)
        return nn.functional.linear(quant_input, quant_weight, self.bias)
```

#### `replace.py`

```python
import torch.nn as nn
from typing import Type, Set

from .modules import QuantConv2d, QuantConvTranspose2d, QuantLinear


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


def quant_conv_module(model: nn.Module, skip_names: Set[str] = None):
    """Replace Conv2d and ConvTranspose2d with quantized versions."""
    skip_names = skip_names or set()

    def replace_module(module, prefix=""):
        for name in list(module._modules.keys()):
            submodule = module._modules[name]
            full_name = f"{prefix}.{name}" if prefix else name

            replace_module(submodule, full_name)

            if full_name in skip_names:
                continue

            if isinstance(submodule, nn.Conv2d) and not isinstance(submodule, QuantConv2d):
                module._modules[name] = transfer_to_quantization(submodule, QuantConv2d)
            elif isinstance(submodule, nn.ConvTranspose2d) and not isinstance(submodule, QuantConvTranspose2d):
                module._modules[name] = transfer_to_quantization(submodule, QuantConvTranspose2d)

    replace_module(model)


def quant_linear_module(model: nn.Module, skip_names: Set[str] = None):
    """Replace Linear modules with quantized versions."""
    skip_names = skip_names or set()

    def replace_module(module, prefix=""):
        for name in list(module._modules.keys()):
            submodule = module._modules[name]
            full_name = f"{prefix}.{name}" if prefix else name

            replace_module(submodule, full_name)

            if full_name in skip_names:
                continue

            if isinstance(submodule, nn.Linear) and not isinstance(submodule, QuantLinear):
                module._modules[name] = transfer_to_quantization(submodule, QuantLinear)

    replace_module(model)
```

### Phase 2: Implement Calibration

#### `calibration/calibrator.py`

```python
import torch
from tqdm import tqdm
from typing import Callable, Optional
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib


class CalibrationManager:
    """Manages PTQ calibration for CenterPoint model."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def set_quantizer_fast(self):
        """Enable fast histogram computation using PyTorch."""
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module._calibrator._torch_hist = True

    def collect_stats(
        self,
        dataloader,
        num_batches: int = 100,
        forward_fn: Optional[Callable] = None,
    ):
        """Collect activation statistics for calibration."""
        self.model.eval()

        # Enable calibration mode
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Collect statistics
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Calibrating")):
                if i >= num_batches:
                    break

                if forward_fn is not None:
                    forward_fn(self.model, batch)
                else:
                    # Default: assume batch is a dict with standard MMDet3D format
                    if isinstance(batch, dict):
                        self.model.test_step(batch)
                    else:
                        self.model(batch)

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
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(method=method)
                    module._amax = module._amax.to(self.device)

    def calibrate(
        self,
        dataloader,
        num_batches: int = 100,
        method: str = "mse",
        forward_fn: Optional[Callable] = None,
    ):
        """Run full calibration pipeline."""
        self.set_quantizer_fast()
        self.collect_stats(dataloader, num_batches, forward_fn)
        self.compute_amax(method)
```

### Phase 3: Layer Fusion Utilities

#### `fusion/bn_fusion.py`

```python
import torch
import torch.nn as nn
from typing import List, Tuple


def fuse_bn_weights(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
) -> Tuple[nn.Parameter, nn.Parameter]:
    """Fuse BatchNorm parameters into Conv weights."""
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)

    # Reshape for broadcasting
    shape = [-1] + [1] * (conv_weight.ndim - 1)

    # Fuse weights: W_fused = W * (gamma / sqrt(var + eps))
    fused_weight = conv_weight * (bn_weight * bn_var_rsqrt).reshape(shape)

    # Fuse bias: b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    fused_bias = (conv_bias - bn_mean) * bn_var_rsqrt * bn_weight + bn_bias

    return nn.Parameter(fused_weight), nn.Parameter(fused_bias)


def fuse_conv_bn(conv: nn.Module, bn: nn.Module):
    """Fuse Conv and BatchNorm modules in-place."""
    assert not conv.training and not bn.training, "Fusion only for eval mode"

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight, conv.bias,
        bn.running_mean, bn.running_var, bn.eps,
        bn.weight, bn.bias
    )


def find_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """Find all Conv-BN pairs in the model."""
    pairs = []
    prev_name = None
    prev_module = None

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if prev_module is not None and isinstance(prev_module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                pairs.append((prev_name, name))
        prev_name = name
        prev_module = module

    return pairs


def fuse_model_bn(model: nn.Module) -> nn.Module:
    """Fuse all Conv-BN pairs in the model."""
    model.eval()

    pairs = find_conv_bn_pairs(model)
    modules_dict = dict(model.named_modules())

    for conv_name, bn_name in pairs:
        conv = modules_dict[conv_name]
        bn = modules_dict[bn_name]

        # Fuse parameters
        fuse_conv_bn(conv, bn)

        # Replace BN with Identity
        parent_name = ".".join(bn_name.split(".")[:-1])
        bn_attr = bn_name.split(".")[-1]

        if parent_name:
            parent = modules_dict[parent_name]
        else:
            parent = model

        setattr(parent, bn_attr, nn.Identity())

    return model
```

### Phase 4: Integration with Deployment Pipeline

#### Quantization-aware export configuration

Add to `projects/CenterPoint/deploy/configs/deploy_config.py`:

```python
# Quantization Configuration
quantization = dict(
    enabled=False,                    # Set True to enable PTQ
    mode="ptq",                       # 'ptq' or 'qat'
    calibration=dict(
        num_batches=100,              # Number of calibration batches
        method="histogram",           # 'histogram', 'max', 'entropy'
        amax_method="mse",            # Method for computing amax: 'mse', 'entropy', 'percentile'
    ),
    fusion=dict(
        fuse_bn=True,                 # Fuse BatchNorm before quantization
    ),
    sensitive_layers=[
        # Layers to keep in FP16 (determined by sensitivity analysis)
        # Example: "pts_voxel_encoder.pfn_layers.0"
    ],
    precision=dict(
        default_input="int8",
        default_weight="int8",
        first_layer_input="fp16",     # Keep first layer input in FP16
        last_layer_output="fp16",     # Keep last layer output in FP16
    ),
)
```

### Phase 5: CLI Tools

#### `tools/detection3d/centerpoint_quantization.py`

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="CenterPoint Quantization Tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # PTQ command
    ptq_parser = subparsers.add_parser("ptq", help="Post-Training Quantization")
    ptq_parser.add_argument("--config", required=True, help="Model config file")
    ptq_parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    ptq_parser.add_argument("--calibrate-batches", type=int, default=100)
    ptq_parser.add_argument("--amax-method", default="mse", choices=["mse", "entropy", "percentile"])
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


def initialize_quantization():
    """Initialize pytorch-quantization library."""
    from pytorch_quantization import quant_modules
    from absl import logging as quant_logging
    quant_logging.set_verbosity(quant_logging.ERROR)


def main():
    args = parse_args()
    initialize_quantization()

    if args.command == "ptq":
        run_ptq(args)
    elif args.command == "sensitivity":
        run_sensitivity(args)
    elif args.command == "qat":
        run_qat(args)


def run_ptq(args):
    """Run PTQ quantization."""
    from projects.CenterPoint.quantization.replace import quant_conv_module, quant_linear_module
    from projects.CenterPoint.quantization.calibration import CalibrationManager
    from projects.CenterPoint.quantization.fusion import fuse_model_bn
    from mmdet3d.apis import init_model

    print("Loading model...")
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device="cuda:0")
    model.eval()

    print("Fusing BatchNorm layers...")
    fuse_model_bn(model)

    print("Inserting Q/DQ nodes...")
    quant_conv_module(model.pts_backbone)
    quant_conv_module(model.pts_neck)
    quant_conv_module(model.pts_bbox_head)
    quant_linear_module(model.pts_voxel_encoder)

    print("Building calibration dataloader...")
    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    print(f"Calibrating with {args.calibrate_batches} batches...")
    calibrator = CalibrationManager(model)
    calibrator.calibrate(dataloader, args.calibrate_batches, method=args.amax_method)

    print(f"Saving quantized model to {args.output}...")
    torch.save({"state_dict": model.state_dict()}, args.output)
    print("PTQ complete!")


def run_sensitivity(args):
    """Run sensitivity analysis."""
    from projects.CenterPoint.quantization.tools.sensitivity import build_sensitivity_profile
    # Implementation follows the pattern from CUDA-CenterPoint
    print("Sensitivity analysis not yet implemented")


def run_qat(args):
    """Run QAT training."""
    print("QAT training not yet implemented")


if __name__ == "__main__":
    main()
```

### Phase 6: QAT Training Hook

#### `quantization/hooks/qat_hook.py`

```python
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class QATHook(Hook):
    """Hook for QAT training with CenterPoint."""

    def __init__(
        self,
        calibration_batches: int = 100,
        calibration_epoch: int = 0,
        freeze_bn: bool = True,
        sensitive_layers: list = None,
    ):
        self.calibration_batches = calibration_batches
        self.calibration_epoch = calibration_epoch
        self.freeze_bn = freeze_bn
        self.sensitive_layers = sensitive_layers or []
        self._quantized = False
        self._calibrated = False

    def before_train(self, runner):
        """Insert Q/DQ nodes before training starts."""
        from projects.CenterPoint.quantization.replace import quant_conv_module, quant_linear_module
        from projects.CenterPoint.quantization.fusion import fuse_model_bn

        model = runner.model

        if self.freeze_bn:
            model.eval()
            fuse_model_bn(model)

        # Insert quantization modules
        quant_conv_module(model.pts_backbone, skip_names=set(self.sensitive_layers))
        quant_conv_module(model.pts_neck, skip_names=set(self.sensitive_layers))
        quant_conv_module(model.pts_bbox_head, skip_names=set(self.sensitive_layers))
        quant_linear_module(model.pts_voxel_encoder, skip_names=set(self.sensitive_layers))

        self._quantized = True
        runner.logger.info("QAT: Inserted Q/DQ nodes")

    def before_train_epoch(self, runner):
        """Calibrate at the specified epoch."""
        if runner.epoch == self.calibration_epoch and not self._calibrated:
            from projects.CenterPoint.quantization.calibration import CalibrationManager

            model = runner.model
            dataloader = runner.train_dataloader

            runner.logger.info(f"QAT: Starting calibration with {self.calibration_batches} batches")
            calibrator = CalibrationManager(model)
            calibrator.calibrate(dataloader, self.calibration_batches)

            self._calibrated = True
            runner.logger.info("QAT: Calibration completed")
```

---

## 4. Configuration Templates

### 4.1 PTQ Configuration

```python
# projects/CenterPoint/configs/quantization/ptq_config.py

_base_ = ['../t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py']

# PTQ settings
quantization = dict(
    enabled=True,
    mode='ptq',
    calibration=dict(
        num_batches=100,
        method='histogram',
        amax_method='mse',
    ),
    fusion=dict(
        fuse_bn=True,
    ),
    sensitive_layers=[],
)
```

### 4.2 QAT Configuration

```python
# projects/CenterPoint/configs/quantization/qat_config.py

_base_ = ['../t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py']

# Reduce learning rate for QAT fine-tuning
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),  # 10x smaller than original
)

# Shorter training for fine-tuning
train_cfg = dict(max_epochs=10)

# QAT hook
custom_hooks = [
    dict(
        type='QATHook',
        calibration_batches=100,
        calibration_epoch=0,
        freeze_bn=True,
        sensitive_layers=[],
    ),
]
```

### 4.3 Deployment Configuration with Quantization

```python
# projects/CenterPoint/deploy/configs/deploy_config_int8.py

_base_ = ['./deploy_config.py']

# Enable quantization for export
quantization = dict(
    enabled=True,
    mode='ptq',
    calibration=dict(
        num_batches=100,
        method='histogram',
        amax_method='mse',
    ),
)

# TensorRT INT8 settings
backend_config = dict(
    common_config=dict(
        precision_policy='int8',  # Changed from 'auto' to 'int8'
        max_workspace_size=4 << 30,
    ),
)
```

---

## 5. Expected Results

### 5.1 Performance Targets

Based on CUDA-CenterPoint results, expected performance:

| Model | Validation mAP | Validation NDS | Notes |
|-------|----------------|----------------|-------|
| FP16 Baseline | ~59.55% | ~66.75% | Original model |
| PTQ INT8 | ~59.08% | ~66.45% | <0.5% mAP drop |
| QAT INT8 | ~59.20% | ~66.53% | <0.35% mAP drop |

### 5.2 Latency Improvement

Expected TensorRT INT8 vs FP16 speedup: **1.5x - 2x** on modern NVIDIA GPUs (A100, Orin, etc.)

### 5.3 Verification Checklist

- [ ] PTQ model produces valid ONNX output
- [ ] Q/DQ nodes are correctly placed in ONNX graph
- [ ] TensorRT can build INT8 engine from quantized ONNX
- [ ] INT8 inference matches FP16 within tolerance (1e-1)
- [ ] mAP degradation is within acceptable range (<1%)
- [ ] End-to-end latency improvement measured

---

## 6. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
quantization = [
    "pytorch-quantization>=2.1.2",
]
```

Or install directly:

```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

---

## 7. References

- [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [pytorch-quantization Documentation](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- [CUDA-CenterPoint QAT Implementation](../../../Lidar_AI_Solution/CUDA-CenterPoint/qat/)
- [MMDet3D CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)
