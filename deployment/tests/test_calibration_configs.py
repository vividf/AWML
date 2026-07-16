"""Parse-validation for every calibration-classifier deploy config.

Feeds each ``deployment/projects/calibration/config/deploy_config*.py`` through the typed parsers in
``deployment.config.schema`` to catch structural/rename errors WITHOUT a real checkpoint, CUDA, or a
full ``BaseDeploymentConfig``. Requires CPU torch + mmengine; no GPU needed. Mirrors
``test_centerpoint_configs.py``.
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

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "projects" / "calibration" / "config"
_CONFIG_FILES = sorted(_CONFIG_DIR.glob("deploy_config*.py"))


def _config_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize("config_path", _CONFIG_FILES, ids=[_config_id(p) for p in _CONFIG_FILES])
def test_calibration_config_parses(config_path: Path) -> None:
    """Every calibration deploy config parses cleanly under the typed schema."""
    cfg = Config.fromfile(str(config_path))

    components_cfg = ComponentsConfig.from_dict(cfg["components"])
    ExportConfig.from_dict(cfg["export"])
    OnnxConfig.from_dict(cfg.get("onnx_config"))
    TensorRTConfig.from_dict(cfg.get("tensorrt_config", {}))
    EvaluationConfig.from_dict(cfg.get("evaluation", {}))
    VerificationConfig.from_dict(cfg.get("verification", {}))
    DeviceConfig.from_dict(cfg.get("devices", {}))
    QuantizationConfig.from_dict(cfg.get("quantization", {}))

    # The classifier is a single whole-model component.
    assert set(components_cfg.component_names()) == {
        "model"
    }, f"{config_path.name}: unexpected component keys {sorted(components_cfg.component_names())}"

    # Class names are calibration-specific deploy metadata (the model config records only num_classes).
    class_names = cfg.get("class_names")
    assert isinstance(class_names, (list, tuple)) and class_names, f"{config_path.name}: class_names must be non-empty"


def test_config_dir_is_nonempty() -> None:
    """Guard against the glob silently matching nothing."""
    assert _CONFIG_FILES, f"No deploy_config*.py found under {_CONFIG_DIR}"
