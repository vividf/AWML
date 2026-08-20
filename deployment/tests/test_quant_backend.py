# Copyright (c) OpenMMLab. All rights reserved.
"""Backend integrity: the framework's PTQ mini-flow on nvidia-modelopt.

Runs QuantConv2d + QuantLinear + residual recipe → calibrate → ONNX export once and checks the
invariants the deploy pipeline depends on:

- weight amax equals the per-channel |weight| max (deterministic max calibrator),
- activation amax is finite and positive (histogram MSE calibrator — exercises the seam's
  ``_compute_amax_mse`` patch),
- a fresh (uncalibrated) quant tree absorbs the checkpoint's ``_amax`` buffers (deploy-load
  path — exercises the seam's ``_load_from_state_dict`` patch),
- the exported ONNX carries the expected QuantizeLinear/DequantizeLinear pairs.

Docker-only: needs torch + modelopt. The removed pytorch-quantization backend must be rejected
explicitly (see ``test_pytorch_quantization_backend_rejected``); the historical cross-backend
parity evidence lives in ``deployment/centerpoint_tutorial`` (A/B logs, 56/56 identical amax).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

import torch.nn as nn  # noqa: E402

from deployment.quantization.core import backend as quant_backend  # noqa: E402


class BasicBlock(nn.Module):  # name matters: attach_quant_add class-gates on it
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.Identity()  # BasicBlockForwardHook expects norm1/norm2
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.Identity()
        self.relu = nn.ReLU()
        self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(4, 8)
        self.block = BasicBlock(8)

    def forward(self, x):
        # x: [N, HW, 4] -> [N, 8, 4, 4]
        f = self.pre(x)
        f = f.transpose(1, 2).reshape(x.shape[0], 8, 4, 4)
        return self.block(f)


def _swap_to_quant(model):
    from deployment.quantization.core.replace import (
        ensure_quant_descriptors_initialized,
        quant_conv_module,
        quant_linear_module,
    )
    from deployment.quantization.recipes.attach import attach_quant_add

    ensure_quant_descriptors_initialized()
    quant_conv_module(model)
    quant_linear_module(model)
    attach_quant_add(model)


@pytest.fixture(scope="module")
def calibrated():
    from deployment.quantization.core.calibration import CalibrationManager

    torch.manual_seed(0)
    model = Net().eval()
    _swap_to_quant(model)

    data = [torch.randn(2, 16, 4) * 2 for _ in range(6)]
    mgr = CalibrationManager(model)
    mgr.calibrate(data, num_batches=len(data), method="mse", forward_fn=lambda m, b: m(b))
    return model


def test_weight_amax_is_per_channel_max(calibrated):
    """The max calibrator must reproduce |weight| max over the non-axis dims exactly."""
    checked = 0
    for name, module in calibrated.named_modules():
        wq = getattr(module, "_weight_quantizer", None)
        if wq is None:
            continue
        amax = wq.amax.flatten()
        w = module.weight.detach()
        expected = w.abs().amax(dim=tuple(range(1, w.dim()))).flatten()
        assert torch.equal(amax, expected), name
        checked += 1
    assert checked == 3  # pre Linear + conv1 + conv2


def test_activation_amax_finite_positive(calibrated):
    sd = calibrated.state_dict()
    keys = [k for k in sd if k.endswith("_amax") and "_weight_quantizer" not in k]
    assert keys
    for k in keys:
        v = sd[k]
        assert torch.isfinite(v).all() and (v > 0).all(), (k, v)


def test_checkpoint_roundtrip_loads_amax(calibrated):
    """A fresh quant tree must absorb a PTQ checkpoint's _amax buffers (deploy-load path)."""
    sd = calibrated.state_dict()
    amax_keys = [k for k in sd if k.endswith("_amax")]

    model2 = Net().eval()
    _swap_to_quant(model2)
    _, unexpected = model2.load_state_dict(sd, strict=False)
    assert list(unexpected) == []
    sd2 = model2.state_dict()
    for k in amax_keys:
        assert k in sd2 and torch.equal(sd2[k].cpu(), sd[k].cpu()), k


def test_onnx_export_produces_qdq(calibrated):
    import io
    from collections import Counter

    onnx = pytest.importorskip("onnx")
    from deployment.quantization.core.utils import setup_quantization_for_onnx_export

    setup_quantization_for_onnx_export()
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(calibrated, torch.randn(1, 16, 4), buf, opset_version=17, dynamo=False)
    g = onnx.load_from_string(buf.getvalue()).graph
    ops = Counter(n.op_type for n in g.node)
    num_q, num_dq = ops.get("QuantizeLinear", 0), ops.get("DequantizeLinear", 0)
    # pre Linear (in+w) + conv1 (in+w) + conv2 (in+w) = 6 Q/DQ pairs.
    # (residual_quantizer reuses conv1._input_quantizer -> whether the tracer emits an extra
    # pair for the reuse site may vary, so require >= 6.)
    assert num_q >= 6 and num_q == num_dq, (num_q, num_dq)


def test_pytorch_quantization_backend_rejected(monkeypatch):
    """Selecting the removed backend must fail loudly, not silently run on modelopt."""
    monkeypatch.setenv("AWML_QUANT_BACKEND", "pytorch-quantization")
    quant_backend._reset_for_testing()
    try:
        with pytest.raises(ValueError, match="has been removed"):
            quant_backend.resolve()
    finally:
        monkeypatch.delenv("AWML_QUANT_BACKEND", raising=False)
        quant_backend._reset_for_testing()
        assert quant_backend.resolve() == "modelopt"
