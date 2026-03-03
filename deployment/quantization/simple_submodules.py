"""
Minimal networks for testing QDQ insertion: SimpleOSA (one OSA module), SimpleOSA3 (three OSA blocks),
and Simple_eSE (one eSE module).

Used with ptq-simple to calibrate using random tensors (no dataset). Weights are initialized
in a reasonable range so PTQ calibration and export can be tested without full CenterPoint.

Three-branch identity (VoVNet-like)
------------------------------------
In real VoV99, when identity=True the block input (identity_feat) is used in three places:
  1. Input to first conv (layers[0])
  2. First element of concat (output[0] → torch.cat → concat 1x1)
  3. Add at end of eSE (xt + identity_feat)
If we insert a separate Q for each use, TRT does three FP32 reformats. Use a single
block_input_quantizer (one Q) and fan-out to all three (same as eSE single-Q at input).

Recommended QDQ placement (TRT friendly)
----------------------------------------
Use ONNX Q/DQ semantics below; implement with onnx-graphsurgeon or your insertion pass.

1) Module input x: single Quantize, then fork to two consumers
   x_fp32
     └─ Qx (QuantizeLinear, scale=Sx, zp=0)
          ├─ DQx_pool → GlobalAveragePool → ...
          └─ DQx_mul  → (bypass for Mul)
   Do NOT insert separate QDQ for pool path vs mul path. Share the same Qx (same Sx/zp)
   to avoid TRT using two different int8 layouts (e.g. NC/4HW4 vs NHWC16).

2) GAP → 1x1 Conv: QDQ around Conv input; weight per-channel quant.
   DQx_pool → GlobalAveragePool → gate_fp32
   gate_fp32 → Qgate_in → DQgate_in → Conv1x1(weight: Qw→DQw) → ...
   TRT typically runs this block in INT8.

3) After hsigmoid (Add + Clip + Div): re-quantize gate before Mul
   gate_after_hsigmoid_fp32
     └─ Qgate_mul (scale=Sg, zp=0)
          └─ DQgate_mul → Mul(DQx_mul, DQgate_mul)
   Hsigmoid is often expanded to elementwise and stays FP16/FP32. Quantize the gate
   explicitly before Mul so TRT can run Mul in INT8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root so we can import from projects/CenterPoint
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from projects.CenterPoint.models.backbones.vovnet import _OSA_module, eSEModule
except ImportError:
    # Fallback: define minimal eSE and OSA inline if project layout differs
    from collections import OrderedDict

    import torch.nn.functional as F

    class Hsigmoid(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.inplace = inplace

        def forward(self, x):
            return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    class eSEModule(nn.Module):
        def __init__(self, channel, reduction=4):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            self.hsigmoid = Hsigmoid()

        def forward(self, x):
            inp = x
            x = self.avg_pool(x)
            x = self.fc(x)
            x = self.hsigmoid(x)
            return inp * x

    def _conv3x3(in_ch, out_ch, name, postfix, stride=1, padding=1):
        return [
            (f"{name}_{postfix}/conv", nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, bias=False)),
            (f"{name}_{postfix}/norm", nn.BatchNorm2d(out_ch)),
            (f"{name}_{postfix}/relu", nn.ReLU(inplace=True)),
        ]

    def _conv1x1(in_ch, out_ch, name, postfix, stride=1, padding=0):
        return [
            (f"{name}_{postfix}/conv", nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=padding, bias=False)),
            (f"{name}_{postfix}/norm", nn.BatchNorm2d(out_ch)),
            (f"{name}_{postfix}/relu", nn.ReLU(inplace=True)),
        ]

    class _OSA_module(nn.Module):
        def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
        ):
            super().__init__()
            self.identity = identity
            self.depthwise = depthwise
            self.isReduced = False
            self.layers = nn.ModuleList()
            in_channel = in_ch
            for i in range(layer_per_block):
                self.layers.append(nn.Sequential(OrderedDict(_conv3x3(in_channel, stage_ch, module_name, i))))
                in_channel = stage_ch
            in_channel = in_ch + layer_per_block * stage_ch
            self.concat = nn.Sequential(OrderedDict(_conv1x1(in_channel, concat_ch, module_name, "concat")))
            self.ese = eSEModule(concat_ch)

        def forward(self, x):
            identity_feat = x
            output = [x]
            x = self.layers[0](x)
            for layer in self.layers[1:]:
                x = layer(x)
                output.append(x)
            x = torch.cat(output, dim=1)
            xt = self.concat(x)
            xt = self.ese(xt)
            if self.identity:
                xt = xt + identity_feat
            return xt


def _init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights in a reasonable range (Kaiming for conv, scale for BN)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.weight.data is not None:
            module.weight.data.mul_(gain)
        if getattr(module, "bias", None) is not None and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "running_mean") and module.running_mean is not None:
            module.running_mean.zero_()
        if hasattr(module, "running_var") and module.running_var is not None:
            module.running_var.fill_(1.0)


class Simple_eSE(nn.Module):
    """
    Single eSE module for testing QDQ insertion (pool input + mul identity).
    After eSE, a 3x3 conv is applied. Input: (B, channel, H, W), output: (B, channel, H, W).
    Default channel=256 to match VoVNet concat_ch.

    Target QDQ placement: see module docstring "Recommended QDQ placement (TRT friendly)"
    (single Q at input → DQ to pool and to mul; QDQ around GAP→Conv; Q gate before Mul).
    """

    def __init__(self, channel: int = 256):
        super().__init__()
        self.channel = channel
        self.ese = eSEModule(channel)
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
        self.apply(lambda m: _init_weights(m, gain=0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ese(x)
        return self.conv(x)


class SimpleOSA(nn.Module):
    """
    Single OSA module (no MaxPool) for testing QDQ insertion (concat inputs, eSE, residual).
    Uses VoVNet-like channels: in_ch=128, stage_ch=160, concat_ch=256, layer_per_block=5.
    Input: (B, 128, H, W), output: (B, 256, H, W).
    """

    def __init__(
        self,
        in_ch: int = 128,
        stage_ch: int = 160,
        concat_ch: int = 256,
        layer_per_block: int = 5,
    ):
        super().__init__()
        self.osa = _OSA_module(
            in_ch,
            stage_ch,
            concat_ch,
            layer_per_block,
            module_name="OSA2_1",
            SE=True,
            identity=False,
            depthwise=False,
        )
        self.apply(lambda m: _init_weights(m, gain=0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.osa(x)


class SimpleOSA3(nn.Module):
    """
    Three OSA blocks (VoVNet-like stage) for testing shared Q/DQ on identity three-way fork.

    Structure: OSA2_1 (no identity) → OSA2_2 (identity=True) → OSA2_3 (identity=True).
    Same channels as VoVNet stage2: in_ch=128, stage_ch=160, concat_ch=256, 5 layers per block.
    The block input when identity=True is used in three places: first conv, concat branch,
    and Add after eSE. A single block_input_quantizer (one Q) should feed all three to avoid
    three FP32 reformats in TRT.

    Input: (B, 128, H, W), output: (B, 256, H, W).
    """

    def __init__(
        self,
        in_ch: int = 128,
        stage_ch: int = 160,
        concat_ch: int = 256,
        layer_per_block: int = 5,
    ):
        super().__init__()
        self.osa1 = _OSA_module(
            in_ch,
            stage_ch,
            concat_ch,
            layer_per_block,
            module_name="OSA2_1",
            SE=True,
            identity=False,
            depthwise=False,
        )
        self.osa2 = _OSA_module(
            concat_ch,
            stage_ch,
            concat_ch,
            layer_per_block,
            module_name="OSA2_2",
            SE=False,
            identity=True,
            depthwise=False,
        )
        self.osa3 = _OSA_module(
            concat_ch,
            stage_ch,
            concat_ch,
            layer_per_block,
            module_name="OSA2_3",
            SE=False,
            identity=True,
            depthwise=False,
        )
        self.apply(lambda m: _init_weights(m, gain=0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.osa1(x)
        x = self.osa2(x)
        x = self.osa3(x)
        return x


class SimpleWrapper(nn.Module):
    """
    Wraps a submodule as pts_backbone so quant_model() can be used unchanged.
    """

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.pts_backbone = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pts_backbone(x)


def build_simple_model(submodule_type: str, device: str = "cuda:0") -> nn.Module:
    """
    Build SimpleOSA, SimpleOSA3, or Simple_eSE wrapped for quant_model (has pts_backbone).

    Args:
        submodule_type: "osa", "osa3", or "ese"
        device: target device

    Returns:
        Wrapped model on device with random weights.
    """
    if submodule_type.lower() == "ese":
        sub = Simple_eSE(channel=256)
    elif submodule_type.lower() == "osa":
        sub = SimpleOSA(in_ch=128, stage_ch=160, concat_ch=256, layer_per_block=5)
    elif submodule_type.lower() == "osa3":
        sub = SimpleOSA3(in_ch=128, stage_ch=160, concat_ch=256, layer_per_block=5)
    else:
        raise ValueError(f"submodule_type must be 'osa', 'osa3', or 'ese', got {submodule_type!r}")
    model = SimpleWrapper(sub)
    model = model.to(device)
    return model


def get_simple_input_shape(submodule_type: str) -> tuple:
    """Return (C, H, W) for random calibration input."""
    if submodule_type.lower() == "ese":
        return (256, 32, 32)
    if submodule_type.lower() == "osa":
        return (128, 64, 64)
    if submodule_type.lower() == "osa3":
        return (128, 64, 64)
    raise ValueError(f"submodule_type must be 'osa', 'osa3', or 'ese', got {submodule_type!r}")
