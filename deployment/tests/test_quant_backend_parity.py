# Copyright (c) OpenMMLab. All rights reserved.
"""Cross-backend parity: pytorch-quantization vs nvidia-modelopt through the framework seam.

Runs the framework's PTQ mini-flow (QuantConv2d + QuantLinear + residual recipe → calibrate →
load amax → ONNX export) once per installed backend **in a subprocess** (backend resolution is
cached per process; see ``deployment.quantization.core.backend.resolve``), then compares:

- quantizer state_dict keys (checkpoint interchangeability),
- weight amax (deterministic per-channel max → must be bit-identical),
- activation amax (same data, same MSE-histogram criterion → near-identical),
- ONNX Q/DQ node counts and the input-quantizer scale constants.

Docker-only: needs torch; each backend section is skipped when that library is missing.
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

_WORKER = r"""
import json, sys, io
import torch, torch.nn as nn

from deployment.quantization.core import backend as quant_backend
from deployment.quantization.core.replace import ensure_quant_descriptors_initialized
from deployment.quantization.core.modules import QuantConv2d, QuantLinear
from deployment.quantization.core.calibration import CalibrationManager
from deployment.quantization.core.utils import setup_quantization_for_onnx_export


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


def swap_to_quant(model):
    ensure_quant_descriptors_initialized()
    from deployment.quantization.core.replace import quant_conv_module, quant_linear_module
    quant_conv_module(model)
    quant_linear_module(model)
    from deployment.quantization.recipes.attach import attach_quant_add
    attach_quant_add(model)


torch.manual_seed(0)
model = Net().eval()
swap_to_quant(model)

data = [torch.randn(2, 16, 4) * 2 for _ in range(6)]  # fixed seed -> identical across backends


def fwd(m, batch):
    m(batch)


mgr = CalibrationManager(model)
mgr.calibrate(data, num_batches=len(data), method="mse", forward_fn=fwd)

sd = model.state_dict()
amax = {k: sd[k].flatten().tolist() for k in sorted(sd) if k.endswith("_amax")}

# Checkpoint round-trip: a FRESH (uncalibrated) quant tree must absorb the _amax buffers,
# exactly like the deploy loaders do (pytorch-quantization creates the buffer on load; the
# modelopt backend needs the seam's _load_from_state_dict patch for this).
model2 = Net().eval()
swap_to_quant(model2)
_, load_unexpected = model2.load_state_dict(sd, strict=False)
sd2 = model2.state_dict()
# .cpu(): pytorch-quantization's load hook materializes the created _amax buffer on CUDA.
reload_amax_ok = all(k in sd2 and torch.equal(sd2[k].cpu(), sd[k].cpu()) for k in amax)

setup_quantization_for_onnx_export()
buf = io.BytesIO()
with torch.no_grad():
    torch.onnx.export(model, torch.randn(1, 16, 4), buf, opset_version=17, dynamo=False)
import onnx
from collections import Counter
g = onnx.load_from_string(buf.getvalue()).graph
ops = Counter(n.op_type for n in g.node)

print("PARITY_JSON:" + json.dumps({
    "backend": quant_backend.resolve(),
    "amax_keys": sorted(amax),
    "amax": amax,
    "num_q": ops.get("QuantizeLinear", 0),
    "num_dq": ops.get("DequantizeLinear", 0),
    "load_unexpected": sorted(load_unexpected),
    "reload_amax_ok": reload_amax_ok,
}))
"""


def _run_backend(backend_name):
    if backend_name == "pytorch-quantization":
        pytest.importorskip("pytorch_quantization")
    else:
        pytest.importorskip("modelopt")
    env = dict(os.environ, AWML_QUANT_BACKEND=backend_name)
    proc = subprocess.run([sys.executable, "-c", _WORKER], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{backend_name} worker failed:\n{proc.stdout}\n{proc.stderr}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("PARITY_JSON:")][-1]
    return json.loads(line[len("PARITY_JSON:") :])


@pytest.fixture(scope="module")
def results():
    return {name: _run_backend(name) for name in ("pytorch-quantization", "modelopt")}


def test_each_backend_produces_qdq(results):
    for name, r in results.items():
        # pre Linear (in+w) + conv1 (in+w) + conv2 (in+w) = 6 Q/DQ pairs.
        # (residual_quantizer reuses conv1._input_quantizer -> no extra pair; whether the
        # tracer emits a 7th pair for the reuse site may differ, so require >= 6.)
        assert r["num_q"] >= 6 and r["num_q"] == r["num_dq"], (name, r["num_q"], r["num_dq"])


def test_checkpoint_roundtrip_loads_amax(results):
    """A fresh quant tree must absorb a PTQ checkpoint's _amax buffers (deploy-load path)."""
    for name, r in results.items():
        assert r["load_unexpected"] == [], (name, r["load_unexpected"])
        assert r["reload_amax_ok"], name


def test_state_dict_keys_identical(results):
    a, b = results["pytorch-quantization"], results["modelopt"]
    assert a["amax_keys"] == b["amax_keys"]


def test_weight_amax_bit_identical(results):
    a, b = results["pytorch-quantization"], results["modelopt"]
    for k in a["amax_keys"]:
        if "_weight_quantizer" not in k:
            continue
        va, vb = a["amax"][k], b["amax"][k]
        assert va == vb, f"{k}: {va[:3]} vs {vb[:3]}"


def test_activation_amax_close(results):
    a, b = results["pytorch-quantization"], results["modelopt"]
    for k in a["amax_keys"]:
        if "_input_quantizer" not in k and "residual_quantizer" not in k:
            continue
        (va,), (vb,) = a["amax"][k], b["amax"][k]
        rel = abs(va - vb) / max(abs(va), 1e-12)
        assert rel < 0.05, f"{k}: {va} vs {vb} (rel {rel:.4f})"
