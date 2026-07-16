# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the shared PTQ-producer settings resolver (``resolve_ptq_settings``).

The resolver is the seam that makes the ``quantize.py ptq`` commands config-driven the same way as
QAT: deploy-config ``quantization.ptq`` block as the single source of truth for the recipe, the
top-level manifest keys (``model_cfg`` / ``checkpoint_path``) for what the artifact is, CLI flags
as overrides. Pure-Python (torch-free), so these run on the host.
"""

import argparse

import pytest

from deployment.config.schema import QuantizationConfig
from deployment.quantization.producer import resolve_ptq_settings


def _args(**overrides) -> argparse.Namespace:
    """Argparse namespace as the ptq subcommand produces it: every flag defaulting to None."""
    ns = dict(
        config=None,
        checkpoint=None,
        calibrate_samples=None,
        batch_size=None,
        calib_seed=None,
        calib_shuffle=None,
        output=None,
    )
    ns.update(overrides)
    return argparse.Namespace(**ns)


def _config_with_block() -> QuantizationConfig:
    return QuantizationConfig.from_dict(
        {
            "enabled": True,
            "mode": "ptq",
            "ptq": {
                "checkpoint": "ckpt/fp.pth",
                "calibrate_samples": 400,
                "batch_size": 4,
                "calib_seed": 0,
                "calib_shuffle": True,
            },
        }
    )


class TestResolvePTQSettings:
    def test_config_supplies_everything(self):
        # Recipe from the ptq block; model_cfg / output from the top-level manifest keys.
        settings = resolve_ptq_settings(
            _args(),
            _config_with_block(),
            "work_dirs/model_ptq.pth",
            "cfg/model.py",
            default_calibrate_samples=100,
        )
        assert settings == dict(
            model_cfg="cfg/model.py",
            checkpoint="ckpt/fp.pth",
            calibrate_samples=400,
            batch_size=4,
            calib_seed=0,
            calib_shuffle=True,
            output="work_dirs/model_ptq.pth",
        )

    def test_cli_flags_override_config_values(self):
        settings = resolve_ptq_settings(
            _args(
                config="cli/model.py",
                checkpoint="cli/fp.pth",
                calibrate_samples=938,
                batch_size=16,
                calib_seed=7,
                output="cli/out_ptq.pth",
            ),
            _config_with_block(),
            "work_dirs/model_ptq.pth",
            "cfg/model.py",
            default_calibrate_samples=100,
        )
        assert settings["model_cfg"] == "cli/model.py"
        assert settings["checkpoint"] == "cli/fp.pth"
        assert settings["calibrate_samples"] == 938
        assert settings["batch_size"] == 16
        assert settings["calib_seed"] == 7
        assert settings["output"] == "cli/out_ptq.pth"

    def test_no_block_uses_project_defaults(self):
        config = QuantizationConfig.from_dict({"enabled": True, "mode": "ptq"})
        settings = resolve_ptq_settings(
            _args(config="cli/model.py", checkpoint="cli/fp.pth"),
            config,
            "work_dirs/model_ptq.pth",
            default_calibrate_samples=256,
        )
        assert settings["calibrate_samples"] == 256
        assert settings["batch_size"] == 1
        assert settings["calib_seed"] is None
        assert settings["calib_shuffle"] is False

    def test_missing_model_cfg_everywhere_exits(self):
        with pytest.raises(SystemExit, match="--config"):
            resolve_ptq_settings(
                _args(checkpoint="cli/fp.pth"),
                _config_with_block(),
                "out.pth",
                None,
                default_calibrate_samples=100,
            )

    def test_missing_output_everywhere_exits(self):
        with pytest.raises(SystemExit, match="--output"):
            resolve_ptq_settings(_args(), _config_with_block(), None, "cfg/model.py", default_calibrate_samples=100)

    def test_cli_seed_zero_beats_block_seed(self):
        # 0 is a valid seed — the CLI-wins check must be `is not None`, not truthiness.
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "ptq": {"calibrate_samples": 400, "calib_seed": 5}}
        )
        settings = resolve_ptq_settings(
            _args(checkpoint="c.pth", calib_seed=0),
            config,
            "out.pth",
            "m.py",
            default_calibrate_samples=100,
        )
        assert settings["calib_seed"] == 0
