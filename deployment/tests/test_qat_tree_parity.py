"""QAT ↔ PTQ tree-parity test — the sacred invariant, executable (spec_qat.md §4 WP5.1).

The invariant: the PTQ producer, the deploy loader, and the QAT hook all build the quantized tree
via the same ``build_<model>_plan(config).prepare(model)``, so the trees are identical *by
construction*. Producer/loader pairs exercise it implicitly (a mismatched state_dict fails to
load); the QAT hook was the third, untested consumer. This test prepares one model directly
through the plan (the PTQ/deploy path) and one through ``QATHookBase.before_train`` (the QAT
path) and asserts the resulting trees are identical: same ``state_dict`` key set, same
``TensorQuantizer`` module set.

Requires torch + mmengine + nvidia-modelopt (the deployment runtime image); no GPU, no
dataset — calibration is not needed to compare tree *structure*.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmengine")
from deployment.quantization.core import backend as _quant_backend

if not _quant_backend.available():
    pytest.skip("quantization backend (nvidia-modelopt) not installed", allow_module_level=True)

import torch.nn as nn  # noqa: E402

from deployment.config.schema import QuantizationConfig  # noqa: E402


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())


class _DummyCenterPoint(nn.Module):
    """Minimal module exposing the towers ``quant_model`` walks (structure only, never run)."""

    def __init__(self) -> None:
        super().__init__()
        self.pts_voxel_encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.pts_backbone = nn.Sequential(_conv_bn_relu(32, 64), _conv_bn_relu(64, 64))
        self.pts_neck = nn.Sequential(_conv_bn_relu(64, 128))
        self.pts_bbox_head = nn.Sequential(nn.Conv2d(128, 8, 1))


class _DummyBEVFusion(nn.Module):
    """Minimal module exposing BEVFusion's dense towers (sparse encoder absent → scheme skips)."""

    def __init__(self) -> None:
        super().__init__()
        self.pts_backbone = nn.Sequential(_conv_bn_relu(32, 64), _conv_bn_relu(64, 64))
        self.pts_neck = nn.Sequential(_conv_bn_relu(64, 128))
        self.bbox_head = nn.Sequential(nn.Conv2d(128, 8, 1))


def _quantizer_names(model: nn.Module) -> set:
    TensorQuantizer = _quant_backend.get_tensor_quantizer_cls()

    return {n for n, m in model.named_modules() if isinstance(m, TensorQuantizer)}


def _seed_and_build(model_cls: type) -> nn.Module:
    torch.manual_seed(0)
    return model_cls()


_CASES = [
    pytest.param(
        _DummyCenterPoint,
        "deployment.projects.centerpoint.quantization.qat_hook",
        "QATHook",
        "deployment.projects.centerpoint.quantization.plan",
        "build_centerpoint_plan",
        ("pts_voxel_encoder",),
        (),
        id="centerpoint",
    ),
    pytest.param(
        _DummyBEVFusion,
        "deployment.projects.bevfusion_l.quantization.qat_hook",
        "BEVFusionQATHook",
        "deployment.projects.bevfusion_l.quantization.plan",
        "build_bevfusion_plan",
        (),
        ("add",),
        id="bevfusion",
    ),
]


@pytest.mark.parametrize(
    "model_cls, hook_module, hook_name, plan_module, plan_name, keep_fp16, disable_recipes", _CASES
)
def test_qat_hook_builds_the_ptq_tree(
    model_cls, hook_module, hook_name, plan_module, plan_name, keep_fp16, disable_recipes
) -> None:
    import importlib

    build_plan = getattr(importlib.import_module(plan_module), plan_name)
    hook_cls = getattr(importlib.import_module(hook_module), hook_name)

    config = QuantizationConfig(enabled=True, fuse_bn=True, keep_fp16=keep_fp16, disable_recipes=disable_recipes)

    # PTQ/deploy path: the plan, called directly.
    ptq_model = _seed_and_build(model_cls)
    build_plan(config).prepare(ptq_model)

    # QAT path: the hook's before_train on a fresh identical model.
    qat_model = _seed_and_build(model_cls)
    hook = hook_cls(
        calibration_batches=1,
        fuse_bn=config.fuse_bn,
        keep_fp16=list(config.keep_fp16),
        disable_recipes=list(config.disable_recipes),
    )
    runner = SimpleNamespace(model=qat_model, logger=logging.getLogger("test_qat_tree_parity"))
    hook.before_train(runner)

    assert set(ptq_model.state_dict().keys()) == set(qat_model.state_dict().keys()), (
        "QAT hook produced a different state_dict key set than the PTQ plan — "
        "the identical-tree invariant is broken"
    )
    assert _quantizer_names(ptq_model) == _quantizer_names(
        qat_model
    ), "QAT hook produced a different TensorQuantizer set than the PTQ plan"
    # The hook must leave the model in train mode for fine-tuning.
    assert qat_model.training


def test_base_hook_is_not_directly_usable() -> None:
    """QATHookBase without a project build_plan must fail loud at construction."""
    from deployment.quantization.qat_hook import QATHookBase

    with pytest.raises(TypeError):
        QATHookBase()
