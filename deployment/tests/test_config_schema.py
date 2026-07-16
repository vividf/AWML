"""Unit tests for deploy-config schema parsing/validation.

These are pure-Python (no GPU) and exercise the validation branches in
``deployment.config.schema`` that previously had no coverage.
"""

from __future__ import annotations

import pytest

from deployment.config.schema import ComponentsConfig, PTQConfig, QATConfig, QuantizationConfig, VerificationScenario


def _valid_component() -> dict:
    return {
        "onnx_file": "model.onnx",
        "engine_file": "model.engine",
        "io": {
            "inputs": [{"name": "x", "dtype": "float32"}],
            "outputs": [{"name": "y", "dtype": "float32"}],
            "dynamic_axes": {"x": {0: "batch"}},
        },
    }


class TestComponentsConfig:
    def test_parses_valid_component(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        comp = cfg.get_component("model")
        assert comp.name == "model"
        assert comp.onnx_file == "model.onnx"
        assert comp.engine_file == "model.engine"
        assert [i.name for i in comp.io.inputs] == ["x"]
        assert [o.name for o in comp.io.outputs] == ["y"]
        assert comp.io.dynamic_axes == {"x": {0: "batch"}}

    def test_missing_onnx_file_raises(self):
        comp = _valid_component()
        del comp["onnx_file"]
        with pytest.raises(KeyError):
            ComponentsConfig.from_dict({"model": comp})

    def test_empty_outputs_raises(self):
        comp = _valid_component()
        comp["io"]["outputs"] = []
        with pytest.raises(KeyError):
            ComponentsConfig.from_dict({"model": comp})

    def test_bad_dynamic_axes_type_raises(self):
        comp = _valid_component()
        comp["io"]["dynamic_axes"] = {"x": {"not_an_int": "batch"}}
        with pytest.raises(TypeError):
            ComponentsConfig.from_dict({"model": comp})

    def test_unknown_component_lookup_raises(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        with pytest.raises(KeyError):
            cfg.get_component("does_not_exist")

    def test_get_artifact_filename(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        assert cfg.get_artifact_filename("model", "engine_file") == "model.engine"


class TestVerificationScenario:
    def test_parses_valid_scenario(self):
        scenario = VerificationScenario.from_dict(
            {
                "ref_backend": "pytorch",
                "ref_device": "cpu",
                "test_backend": "onnx",
                "test_device": "cpu",
            }
        )
        assert scenario.ref_backend.value == "pytorch"
        assert scenario.test_backend.value == "onnx"

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError):
            VerificationScenario.from_dict({"ref_backend": "pytorch"})


def _valid_qat_block() -> dict:
    return {
        "train_cfg": "projects/CenterPoint/configs/train.py",
        "checkpoint": "work_dirs/epoch_30.pth",
        "epochs": 3,
        "lr": 1e-4,
        "calibrate_samples": 400,
    }


class TestQATConfig:
    def test_parses_valid_block(self):
        qat = QATConfig.from_dict(_valid_qat_block())
        assert qat.epochs == 3
        assert qat.lr == pytest.approx(1e-4)
        assert qat.calibrate_samples == 400
        assert qat.calib_cache is None
        assert qat.work_dir is None

    def test_epochs_and_lr_are_required(self):
        block = _valid_qat_block()
        del block["epochs"]
        del block["lr"]
        with pytest.raises(ValueError, match="epochs"):
            QATConfig.from_dict(block)

    def test_unknown_key_raises(self):
        block = _valid_qat_block()
        block["epoch"] = 3  # typo of "epochs"
        with pytest.raises(ValueError, match="epoch"):
            QATConfig.from_dict(block)

    def test_calibrate_samples_defaults_to_reference(self):
        block = _valid_qat_block()
        del block["calibrate_samples"]
        assert QATConfig.from_dict(block).calibrate_samples == 400

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError):
            QATConfig.from_dict(["not", "a", "dict"])


class TestQuantizationConfigQAT:
    def test_qat_block_parses_under_qat_mode(self):
        cfg = QuantizationConfig.from_dict({"enabled": True, "mode": "qat", "qat": _valid_qat_block()})
        assert cfg.mode == "qat"
        assert cfg.qat is not None
        assert cfg.qat.epochs == 3

    def test_qat_block_under_ptq_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            QuantizationConfig.from_dict({"enabled": True, "mode": "ptq", "qat": _valid_qat_block()})

    def test_qat_mode_without_block_is_provenance_only(self):
        cfg = QuantizationConfig.from_dict({"enabled": True, "mode": "qat"})
        assert cfg.qat is None

    def test_absent_section_has_no_qat(self):
        assert QuantizationConfig.from_dict(None).qat is None


def _valid_ptq_block() -> dict:
    # The model config is deliberately NOT a block key — it lives at the deploy config's top level
    # (`model_cfg`), shared with the deploy CLI.
    return {
        "checkpoint": "work_dirs/epoch_30.pth",
        "calibrate_samples": 400,
        "calib_seed": 0,
    }


class TestPTQConfig:
    def test_parses_valid_block(self):
        ptq = PTQConfig.from_dict(_valid_ptq_block())
        assert ptq.calibrate_samples == 400
        assert ptq.checkpoint == "work_dirs/epoch_30.pth"
        assert ptq.calib_seed == 0

    def test_model_cfg_is_not_a_block_key(self):
        block = _valid_ptq_block()
        block["model_cfg"] = "projects/CenterPoint/configs/model.py"
        with pytest.raises(ValueError, match="model_cfg"):
            PTQConfig.from_dict(block)

    def test_calibrate_samples_is_required(self):
        block = _valid_ptq_block()
        del block["calibrate_samples"]
        with pytest.raises(ValueError, match="calibrate_samples"):
            PTQConfig.from_dict(block)

    def test_unknown_key_raises(self):
        block = _valid_ptq_block()
        block["calibrate_sample"] = 400  # typo of "calibrate_samples"
        with pytest.raises(ValueError, match="calibrate_sample"):
            PTQConfig.from_dict(block)

    def test_defaults(self):
        ptq = PTQConfig.from_dict({"calibrate_samples": 100})
        assert ptq.batch_size == 1
        assert ptq.calib_seed is None
        assert ptq.calib_shuffle is False
        assert ptq.checkpoint is None

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError):
            PTQConfig.from_dict(["not", "a", "dict"])


class TestQuantizationConfigPTQ:
    def test_ptq_block_parses_under_ptq_mode(self):
        cfg = QuantizationConfig.from_dict({"enabled": True, "mode": "ptq", "ptq": _valid_ptq_block()})
        assert cfg.ptq is not None
        assert cfg.ptq.calibrate_samples == 400

    def test_default_mode_is_ptq(self):
        cfg = QuantizationConfig.from_dict({"enabled": True, "ptq": _valid_ptq_block()})
        assert cfg.mode == "ptq"
        assert cfg.ptq is not None

    def test_ptq_block_under_qat_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            QuantizationConfig.from_dict({"enabled": True, "mode": "qat", "ptq": _valid_ptq_block()})

    def test_explicit_none_ptq_block_is_ignored(self):
        # The _base_ inheritance pattern: a mode="qat" child config drops the inherited ptq
        # block with an explicit ptq=None.
        cfg = QuantizationConfig.from_dict({"enabled": True, "mode": "qat", "ptq": None, "qat": _valid_qat_block()})
        assert cfg.ptq is None
        assert cfg.qat is not None

    def test_absent_section_has_no_ptq(self):
        assert QuantizationConfig.from_dict(None).ptq is None
