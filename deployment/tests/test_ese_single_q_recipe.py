"""Characterization test for the single-Q-at-input eSE recipe (spec.md §4 R1(d)).

Locks the reformat-minimizing INT8 eSE placement that vov99 relies on and that the always-on recipe
must reproduce: exactly ONE quantizer at the eSE input, fanned out to *both* ``Mul`` operands, and NO
legacy ``mul_identity`` second quantizer (that path was deleted in Goal 2). The "identical numerics"
half of R1(d) is covered by the Docker e2e mAP gate; this test covers the *structure*.

Docker-only: needs torch (and, for the attach test, ``pytorch_quantization``); skipped on the
torch-less host, where only ``ast``/pyflakes run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from deployment.quantization.recipes.forward_hooks import eSEModuleForwardHook  # noqa: E402


class _SpyQuantizer(nn.Module):
    """Identity stand-in for ``TensorQuantizer`` that counts how many tensors it quantized."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):  # noqa: D401
        self.calls += 1
        return x


class _FakeESEModule(nn.Module):
    """Minimal module exercising the attrs :class:`eSEModuleForwardHook` reads."""

    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(4, 4, 1)
        self.hsigmoid = nn.Sigmoid()


def test_ese_forward_hook_single_quantization() -> None:
    """With pool_input + mul_gate present, the input is quantized ONCE and fans out to both Mul operands."""
    module = _FakeESEModule()
    module.pool_input_quantizer = _SpyQuantizer()
    module.mul_gate_quantizer = _SpyQuantizer()
    hook = eSEModuleForwardHook(module)

    x = torch.randn(1, 4, 8, 8)
    out = hook(x)

    # One Q at the eSE input; the gate gets its own Q; the bypass reuses qx (no second input Q).
    assert module.pool_input_quantizer.calls == 1
    assert module.mul_gate_quantizer.calls == 1
    assert not hasattr(module, "mul_identity_quantizer"), "legacy two-Q path must be gone"

    expected = x * module.hsigmoid(module.fc(module.avg_pool(x)))
    assert torch.allclose(out, expected)


def test_ese_forward_hook_fp_fallback_without_pool_input() -> None:
    """Without the recipe (no pool_input_quantizer) the hook runs the plain FP path — no quantizers."""
    module = _FakeESEModule()
    hook = eSEModuleForwardHook(module)

    x = torch.randn(1, 4, 8, 8)
    out = hook(x)

    expected = x * module.hsigmoid(module.fc(module.avg_pool(x)))
    assert torch.allclose(out, expected)


def test_ese_attach_produces_single_q_no_mul_identity() -> None:
    """The merged eSE recipe (one attach call, no ordering contract) yields the single-Q structure."""
    from deployment.quantization.core import backend as _quant_backend

    if not _quant_backend.available():
        pytest.skip("no quantization backend installed")
    from deployment.quantization.recipes.attach import attach_ese_quantizers

    # The recipe dispatches on the class *name* "eSEModule".
    class eSEModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(4, 4, 1)
            self.hsigmoid = nn.Sigmoid()

    root = nn.Module()
    root.add_module("ese", eSEModule())

    attach_ese_quantizers(root)

    ese = root.ese
    assert getattr(ese, "pool_input_quantizer", None) is not None
    assert getattr(ese, "mul_gate_quantizer", None) is not None
    assert not hasattr(ese, "mul_identity_quantizer"), "legacy two-Q path must be gone"
