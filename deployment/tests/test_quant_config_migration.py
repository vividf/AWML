"""Parity oracle for the Goal-2 quantization config migration (spec.md §3.8(2)).

Locks the *behavior-preserving* migration from the old ~13 boolean flags to the declarative
``default_precision`` / ``keep_fp16`` / ``disable_recipes`` surface. For every migrated deploy config
this test re-derives, from the config's **old** flags, the set of module names that used to be kept in
FP16 — using the (now-deleted) rule ``{towers turned off} ∪ resolved_sensitive_layers()`` — and asserts
it equals the config's new ``keep_fp16``. It also checks that a recipe that used to be off maps to a
``disable_recipes`` entry, and that no legacy flag survived.

This is a *name-level* oracle: pure Python, no torch / no model, so it runs anywhere. It proves the
migration reproduces the same set of kept module names. The subtree-expansion equivalence
(``expand_keep_fp16``) and the final numerics are covered by the Docker e2e mAP check.

.. note:: **Shelf-life — migration oracle, not a live invariant.** This test freezes each config's
   ``keep_fp16`` / ``disable_recipes`` to the values derived from the pre-Goal-2 flags. Once the
   Goal-2 Docker e2e verify has landed, any *deliberate* accuracy retuning of a config's
   ``keep_fp16`` (e.g. widening VoVNet FP16 stages) will — correctly — fail this test; at that
   point update the tuned config's entry here or retire this file. Do not treat a failure after
   intentional tuning as a regression.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set

import pytest

# deployment/tests/ -> deployment/
_DEPLOYMENT = Path(__file__).resolve().parents[1]
_PROJECTS = _DEPLOYMENT / "projects"

# ---------------------------------------------------------------------------------------------------
# OLD (pre-migration) ``quantization`` flags, copied verbatim from each config's pre-Goal-2 revision.
# ---------------------------------------------------------------------------------------------------
_VOV_FLAGS = dict(
    quant_ese_mul_identity=True,
    quant_ese_pool_input=True,
    quant_maxpool_input=True,
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=True,
    quant_linear_backbone=True,
    skip_backbone_first_stages=0,
    skip_backbone_stages=[],
    skip_vovnet_stages=[0, 1],
    sensitive_layers=[],
)
_CONVNEXT_FLAGS = dict(
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=True,
    quant_linear_backbone=True,
    skip_backbone_first_stages=0,
    skip_backbone_stages=[],
    sensitive_layers=[],
)
_RESNET_FLAGS = dict(
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=True,
    skip_backbone_first_stages=0,
    skip_backbone_stages=[],
    sensitive_layers=[],
)
_SECOND_FLAGS = dict(
    quant_voxel_encoder=False,
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    # quant_add unset -> old default False
    skip_backbone_stages=[0],
    sensitive_layers=[],
)
_BEVFUSION_FLAGS = dict(
    quant_backbone=True,
    quant_neck=True,
    quant_head=True,
    quant_add=False,
    sensitive_layers=[],
)

OLD_FLAGS: Dict[str, dict] = {
    "centerpoint/config/deploy_config_int8_vov99.py": _VOV_FLAGS,
    "centerpoint/config/deploy_config_int8_vov57.py": _VOV_FLAGS,
    "centerpoint/config/deploy_config_int8_convnext_small.py": _CONVNEXT_FLAGS,
    "centerpoint/config/deploy_config_int8_resnet.py": _RESNET_FLAGS,
    "centerpoint/config/deploy_config_int8_resnet_base.py": _RESNET_FLAGS,
    "centerpoint/config/deploy_config_int8_second_2_6_quant_release.py": _SECOND_FLAGS,
    "bevfusion_l/config/deploy_config_split_sparse_fp16_dense_int8_2_8.py": _BEVFUSION_FLAGS,
}

# Tower toggle -> dotted module name. (No config turns the head off, so the CenterPoint/BEVFusion head
# naming difference — pts_bbox_head vs bbox_head — never affects the derived set.)
_TOWER_NAME = {
    "quant_voxel_encoder": "pts_voxel_encoder",
    "quant_backbone": "pts_backbone",
    "quant_neck": "pts_neck",
    "quant_head": "pts_bbox_head",
}
_VOVNET_STAGE_NAMES = ("stem", "stage2", "stage3", "stage4")

_LEGACY_KEYS = frozenset(
    {
        "quant_backbone",
        "quant_neck",
        "quant_head",
        "quant_voxel_encoder",
        "quant_add",
        "quant_linear_backbone",
        "quant_ese_mul_identity",
        "quant_ese_pool_input",
        "quant_maxpool_input",
        "skip_backbone_first_stages",
        "skip_backbone_stages",
        "skip_vovnet_stages",
        "sensitive_layers",
    }
)


def _old_resolved_sensitive_layers(flags: dict) -> Set[str]:
    """Reimplements the deleted ``QuantizationConfig.resolved_sensitive_layers()``."""
    skip: Set[str] = set(flags.get("sensitive_layers", []) or [])
    stage_ids: Set[int] = set()
    if int(flags.get("skip_backbone_first_stages", 0) or 0) > 0:
        stage_ids.update(range(int(flags["skip_backbone_first_stages"])))
    stage_ids.update(flags.get("skip_backbone_stages", []) or [])
    for stage_id in stage_ids:
        skip.add(f"pts_backbone.blocks.{stage_id}")
    vov = flags.get("skip_vovnet_stages")
    if vov is not None:
        for idx in vov:
            if 0 <= idx < len(_VOVNET_STAGE_NAMES):
                skip.add(f"pts_backbone.{_VOVNET_STAGE_NAMES[idx]}")
    return skip


def _expected_keep_fp16(flags: dict) -> Set[str]:
    """Migration rule: ``{towers turned off} ∪ resolved_sensitive_layers()``."""
    keep = {name for flag, name in _TOWER_NAME.items() if flag in flags and not flags[flag]}
    return keep | _old_resolved_sensitive_layers(flags)


def _expected_disable_recipes(flags: dict) -> Set[str]:
    """A recipe that was off migrates to a ``disable_recipes`` entry. Old ``quant_add`` default False."""
    return set() if flags.get("quant_add", False) else {"add"}


def _extract_quantization_block(path: Path) -> dict:
    """Evaluate just the ``quantization = dict(...)`` assignment from a deploy config (no import).

    The value is a ``dict(...)`` call over literals (bool/str/list), so it evaluates safely with only
    the ``dict`` builtin exposed — no mmengine / torch / ``_base_`` resolution needed.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "quantization" for t in node.targets
        ):
            return eval(compile(ast.Expression(node.value), str(path), "eval"), {"dict": dict})  # noqa: S307
    raise AssertionError(f"no top-level `quantization = dict(...)` assignment in {path}")


@pytest.mark.parametrize("rel_path", sorted(OLD_FLAGS))
def test_keep_fp16_reproduces_old_resolution(rel_path: str) -> None:
    old = OLD_FLAGS[rel_path]
    block = _extract_quantization_block(_PROJECTS / rel_path)

    assert set(block.get("keep_fp16", [])) == _expected_keep_fp16(
        old
    ), f"{rel_path}: migrated keep_fp16 does not reproduce the old FP16-retained set"
    assert set(block.get("disable_recipes", [])) == _expected_disable_recipes(
        old
    ), f"{rel_path}: migrated disable_recipes does not match the old recipe on/off state"
    assert block.get("default_precision") == "int8", f"{rel_path}: default_precision must be 'int8'"


@pytest.mark.parametrize("rel_path", sorted(OLD_FLAGS))
def test_no_legacy_flags_remain(rel_path: str) -> None:
    block = _extract_quantization_block(_PROJECTS / rel_path)
    leftover = _LEGACY_KEYS & set(block)
    assert not leftover, f"{rel_path}: legacy quantization flags still present: {sorted(leftover)}"
