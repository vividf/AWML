"""Parse-validation for every BEVFusion deploy config (mirror of test_centerpoint_configs.py).

Loads each ``deployment/projects/bevfusion_l/config/deploy_config*.py`` and feeds its sections
through the typed parsers in ``deployment.config.schema``. This catches structural/rename errors
(wrong component keys, invalid precision_policy/scenarios, misspelled quantization keys) WITHOUT
needing a real checkpoint, CUDA, or a full ``BEVFusionDeploymentConfig`` (which validates
checkpoint existence + CUDA availability). Before this file existed, a structural typo in a
BEVFusion deploy config was invisible until the Docker e2e run (spec.md §5.3 4C.3).

Requires CPU torch + mmengine (the deployment runtime image); no GPU needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mmengine.config import Config

from deployment.config.schema import (
    ComponentsConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    OnnxConfig,
    QuantizationConfig,
    TensorRTConfig,
    VerificationConfig,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "projects" / "bevfusion_l" / "config"
_CONFIG_FILES = sorted(_CONFIG_DIR.glob("deploy_config*.py"))


def _config_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize("config_path", _CONFIG_FILES, ids=[_config_id(p) for p in _CONFIG_FILES])
def test_bevfusion_config_parses(config_path: Path) -> None:
    """Every BEVFusion deploy config parses cleanly under the typed schema."""
    cfg = Config.fromfile(str(config_path))

    components_cfg = ComponentsConfig.from_dict(cfg["components"])
    ExportConfig.from_dict(cfg["export"])
    OnnxConfig.from_dict(cfg.get("onnx_config"))
    TensorRTConfig.from_dict(cfg.get("tensorrt_config", {}))
    EvaluationConfig.from_dict(cfg.get("evaluation", {}))
    VerificationConfig.from_dict(cfg.get("verification", {}))
    DeviceConfig.from_dict(cfg.get("devices", {}))
    QuantizationConfig.from_dict(cfg.get("quantization", {}))

    # Layout-aware component check, same rule BEVFusionDeploymentConfig._validate_components
    # enforces at construction: split needs sparse+dense; otherwise a merged graph.
    component_names = set(components_cfg.component_names())
    if {"bevfusion_sparse", "bevfusion_dense"} & component_names:
        missing = {"bevfusion_sparse", "bevfusion_dense"} - component_names
        assert not missing, f"{config_path.name}: split layout missing components {sorted(missing)}"
    else:
        assert "bevfusion_merged" in component_names, (
            f"{config_path.name}: expected split (bevfusion_sparse+bevfusion_dense) or merged "
            f"(bevfusion_merged) layout, got {sorted(component_names)}"
        )


@pytest.mark.parametrize(
    "config_path",
    [p for p in _CONFIG_FILES if "int8" in p.stem],
    ids=[_config_id(p) for p in _CONFIG_FILES if "int8" in p.stem],
)
def test_int8_configs_enable_quantization(config_path: Path) -> None:
    """INT8 configs must carry an enabled quantization block."""
    cfg = Config.fromfile(str(config_path))
    quant = QuantizationConfig.from_dict(cfg.get("quantization", {}))
    assert quant.enabled, f"{config_path.name}: expected quantization.enabled=True"


def test_config_dir_is_nonempty() -> None:
    """Guard against the glob silently matching nothing."""
    assert _CONFIG_FILES, f"No deploy_config*.py found under {_CONFIG_DIR}"
