# Copyright (c) OpenMMLab. All rights reserved.
"""Quantization backend resolver: pytorch-quantization or nvidia-modelopt.

Single seam through which the framework reaches the fake-quant library. Both backends descend
from the same NVIDIA codebase and share the calibration surface the framework relies on
(``TensorQuantizer`` with ``disable_quant``/``enable_calib``/``load_calib_amax``, ``_amax``
buffers with identical state_dict keys, ``MaxCalibrator``/``HistogramCalibrator``), so the rest
of ``deployment.quantization`` stays backend-agnostic and imports everything quantizer-related
from here instead of from ``pytorch_quantization`` directly.

Backend selection (resolved once, then cached):

- ``AWML_QUANT_BACKEND=pytorch-quantization`` — force the classic backend.
- ``AWML_QUANT_BACKEND=modelopt``            — force nvidia-modelopt (``modelopt.torch.quantization``).
- unset / ``auto``                            — pytorch-quantization when installed, else modelopt.

The two backends differ in exactly two places the framework must know about, both wrapped here:

1. **Descriptors**: pytorch-quantization uses ``tensor_quant.QuantDescriptor(calib_method=...)``;
   modelopt uses ``config.QuantizerAttributeConfig(calibrator=...)``. See
   :mod:`.descriptors` for the per-layer choices built on :func:`make_quant_desc`.
2. **ONNX export**: pytorch-quantization needs the global ``TensorQuantizer.use_fb_fake_quant=True``
   switch; modelopt's ``TensorQuantizer`` traces to QuantizeLinear/DequantizeLinear natively
   (:func:`exports_qdq_natively` → :func:`setup_onnx_export` is a no-op there).

Leaf module: imports only stdlib + the backend package (lazily); safe for every core submodule.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_VAR = "AWML_QUANT_BACKEND"
PYTORCH_QUANTIZATION = "pytorch-quantization"
MODELOPT = "modelopt"
_KNOWN = (PYTORCH_QUANTIZATION, MODELOPT)

# Module import names per backend.
_IMPORT_NAME = {PYTORCH_QUANTIZATION: "pytorch_quantization", MODELOPT: "modelopt"}

_resolved: Optional[str] = None
_resolve_done = False


def _installed(backend: str) -> bool:
    return importlib.util.find_spec(_IMPORT_NAME[backend]) is not None


def resolve() -> Optional[str]:
    """Return the active backend name, or ``None`` when neither library is installed.

    Resolution is cached for the life of the process: the quantized modules bind their
    ``TensorQuantizer`` class at attach time, so switching backends mid-process is not supported —
    set ``AWML_QUANT_BACKEND`` before the first quantization import.
    """
    global _resolved, _resolve_done
    if _resolve_done:
        return _resolved

    requested = os.environ.get(_ENV_VAR, "auto").strip().lower() or "auto"
    if requested in ("auto", ""):
        if _installed(PYTORCH_QUANTIZATION):
            _resolved = PYTORCH_QUANTIZATION
        elif _installed(MODELOPT):
            _resolved = MODELOPT
        else:
            _resolved = None
    elif requested in _KNOWN:
        if not _installed(requested):
            raise ImportError(
                f"{_ENV_VAR}={requested} but '{_IMPORT_NAME[requested]}' is not installed. {install_hint()}"
            )
        _resolved = requested
    else:
        raise ValueError(f"{_ENV_VAR}={requested!r} — expected one of {_KNOWN + ('auto',)}")

    _resolve_done = True
    if _resolved is not None:
        logger.info("Quantization backend: %s", _resolved)
    return _resolved


def _reset_for_testing() -> None:
    """Clear the cached resolution (unit tests only — see :func:`resolve` for why not runtime)."""
    global _resolved, _resolve_done
    _resolved = None
    _resolve_done = False


def available() -> bool:
    """Whether any quantization backend is installed."""
    return resolve() is not None


def install_hint(purpose: str = "quantization support") -> str:
    """Standard install message naming both accepted backends."""
    return (
        f"A quantization backend is required for {purpose}. Install one of:\n"
        "  pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com\n"
        "  pip install nvidia-modelopt\n"
        f"(select explicitly with {_ENV_VAR}=pytorch-quantization|modelopt)"
    )


def require(purpose: str = "quantization support") -> str:
    """Return the active backend name, raising the install hint when none is available."""
    name = resolve()
    if name is None:
        raise ImportError(install_hint(purpose))
    return name


# ---------------------------------------------------------------------------
# modelopt bug workarounds
# ---------------------------------------------------------------------------

_modelopt_patched = False


def _ensure_modelopt_patches() -> None:
    """Apply behavior-probed fixes for modelopt gaps vs pytorch-quantization (as of 0.46.0).

    Two seams, both detected behaviorally so a fixed upstream is left untouched, both applied
    once per process:

    1. **Histogram-MSE calibration**: ``modelopt...calib.histogram._compute_amax_mse`` still
       calls ``fake_tensor_quant(centers, amax, num_bits, unsigned)`` with pytorch-quantization's
       positional signature, but modelopt's ``FakeTensorQuantFunction.forward`` takes ``bias``
       as the third argument. ``num_bits=8`` therefore lands in ``bias`` and ``unsigned=False``
       in ``num_bits``, and the MSE search degenerates to near-histogram-max amax (empirically
       2–4x too large on CenterPoint activations; entropy/percentile are unaffected). Replace
       the module-level function with a corrected copy.

    2. **Checkpoint amax load**: pytorch-quantization's ``TensorQuantizer._load_from_state_dict``
       creates the ``_amax`` buffer on the fly when the incoming state_dict carries one; modelopt's
       ``TensorQuantizer`` has no such override, so loading a PTQ checkpoint into a freshly built
       (uncalibrated) quantizer tree silently drops every ``_amax`` as an "unexpected key". Add an
       equivalent override so PTQ checkpoints are interchangeable across backends.
    """
    global _modelopt_patched
    if _modelopt_patched:
        return
    _modelopt_patched = True

    import numpy as np
    import torch
    from modelopt.torch.quantization.calib import histogram as _mo_histogram
    from modelopt.torch.quantization.nn import TensorQuantizer as _MoTensorQuantizer
    from modelopt.torch.quantization.tensor_quant import fake_tensor_quant as _mo_ftq

    # Probe with the exact call shape histogram.py uses; identity-range inputs must survive.
    probe = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    try:
        out = _mo_histogram.fake_tensor_quant(probe, torch.tensor(1.0), 8, False)
        if torch.allclose(out, probe, atol=0.05):
            return  # upstream fixed — nothing to do
    except Exception:
        pass  # broken in a louder way; patch below

    def _fixed_compute_amax_mse(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
        """modelopt's ``_compute_amax_mse`` with the ``fake_tensor_quant`` call corrected."""
        if calib_bin_edges is None and calib_hist is None:
            return None
        if not (isinstance(num_bits, int) and num_bits >= 0):
            raise TypeError("Invalid num_bits. num_bits must be a positive integer.")
        counts = torch.from_numpy(calib_hist[:]).float()
        edges = torch.from_numpy(calib_bin_edges[:]).float()
        device = None
        if torch.cuda.is_available():
            device = counts.device
            counts = counts.cuda()
            edges = edges.cuda()
        centers = (edges[1:] + edges[:-1]) / 2
        mses = []
        arguments = []
        for i in range(start_bin, len(centers), stride):
            amax = centers[i]
            # Positional: (inputs, amax, bias, num_bits, exponent_bits, unsigned)
            quant_centers = _mo_ftq(centers, amax, None, num_bits, 0, unsigned)
            mses.append(((quant_centers - centers) ** 2 * counts).mean().cpu())
            arguments.append(i)
        argmin = int(np.argmin(mses))
        calib_amax = centers[arguments[argmin]]
        if device is not None:
            calib_amax = calib_amax.to(device)
        return calib_amax

    _mo_histogram._compute_amax_mse = _fixed_compute_amax_mse
    logger.warning("Patched modelopt histogram MSE calibration (upstream fake_tensor_quant signature bug)")

    # --- 2. checkpoint amax load -------------------------------------------------------------
    probe_q = _MoTensorQuantizer()
    probe_q._load_from_state_dict({"_amax": torch.tensor(1.5)}, "", {}, True, [], [], [])
    if getattr(probe_q, "_amax", None) is None:
        _orig_load = _MoTensorQuantizer._load_from_state_dict

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            amax = state_dict.get(prefix + "_amax")
            if amax is not None and "_amax" not in self._buffers:
                self.register_buffer("_amax", amax.data.clone())
            _orig_load(self, state_dict, prefix, *args, **kwargs)

        _MoTensorQuantizer._load_from_state_dict = _load_from_state_dict
        logger.warning("Patched modelopt TensorQuantizer._load_from_state_dict to create _amax on load")


# ---------------------------------------------------------------------------
# Class / module accessors
# ---------------------------------------------------------------------------


def get_tensor_quantizer_cls() -> Any:
    """The backend's ``TensorQuantizer`` class (raises when no backend is installed)."""
    name = require("TensorQuantizer")
    if name == PYTORCH_QUANTIZATION:
        from pytorch_quantization.nn import TensorQuantizer
    else:
        from modelopt.torch.quantization.nn import TensorQuantizer

        _ensure_modelopt_patches()
    return TensorQuantizer


def get_tensor_quantizer_cls_or_none() -> Optional[Any]:
    """Like :func:`get_tensor_quantizer_cls`, but ``None`` when no backend is installed."""
    if not available():
        return None
    return get_tensor_quantizer_cls()


def get_calib() -> Any:
    """The backend's ``calib`` module (``MaxCalibrator`` / ``HistogramCalibrator``)."""
    name = require("calibrators")
    if name == PYTORCH_QUANTIZATION:
        from pytorch_quantization import calib
    else:
        from modelopt.torch.quantization import calib

        _ensure_modelopt_patches()
    return calib


def get_calib_or_none() -> Optional[Any]:
    """Like :func:`get_calib`, but ``None`` when no backend is installed."""
    if not available():
        return None
    return get_calib()


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def make_quant_desc(**kwargs: Any) -> Any:
    """Build a quantizer descriptor in the active backend's dialect.

    Accepts pytorch-quantization's vocabulary (``num_bits``, ``axis``, ``calib_method``) and
    translates for modelopt (whose ``QuantizerAttributeConfig`` spells the calibrator field
    ``calibrator`` and takes ``axis`` as an int).
    """
    name = require("quantization descriptors")
    if name == PYTORCH_QUANTIZATION:
        from pytorch_quantization import tensor_quant

        return tensor_quant.QuantDescriptor(**kwargs)

    from modelopt.torch.quantization.config import QuantizerAttributeConfig

    translated = dict(kwargs)
    if "calib_method" in translated:
        translated["calibrator"] = translated.pop("calib_method")
    axis = translated.get("axis")
    if isinstance(axis, tuple) and len(axis) == 1:
        translated["axis"] = axis[0]
    return QuantizerAttributeConfig(**translated)


def desc_calib_method(desc: Any) -> Optional[str]:
    """The calibrator name (``"max"``/``"histogram"``) of a descriptor, whichever dialect it is."""
    method = getattr(desc, "calib_method", None)
    if method is None:
        method = getattr(desc, "calibrator", None)
    return method if isinstance(method, str) else None


def get_preset_desc(preset: str) -> Any:
    """A named ``QUANT_DESC_*`` preset from the backend's ``tensor_quant`` module.

    The preset names are identical in both backends (modelopt kept pytorch-quantization's).
    """
    name = require("quantization descriptors")
    if name == PYTORCH_QUANTIZATION:
        from pytorch_quantization import tensor_quant
    else:
        from modelopt.torch.quantization import tensor_quant
    return getattr(tensor_quant, preset)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def exports_qdq_natively() -> bool:
    """Whether the backend's ``TensorQuantizer`` traces to Q/DQ during ONNX export on its own.

    modelopt registers ONNX symbolics for its fake-quant autograd functions, so a quantized model
    exports QuantizeLinear/DequantizeLinear without any global switch — and the quantizers must NOT
    be bypassed during tracing. pytorch-quantization needs ``use_fb_fake_quant=True`` first.
    """
    return resolve() == MODELOPT


def setup_onnx_export() -> None:
    """Put the backend into ONNX-export mode (idempotent; no-op without a backend).

    pytorch-quantization: sets the global ``TensorQuantizer.use_fb_fake_quant = True`` so
    quantizers trace as Q/DQ pairs. modelopt: nothing to do (see :func:`exports_qdq_natively`).
    """
    name = resolve()
    if name is None:
        return
    if name == PYTORCH_QUANTIZATION:
        from pytorch_quantization.nn import TensorQuantizer

        TensorQuantizer.use_fb_fake_quant = True
        logger.info("Enabled use_fb_fake_quant for ONNX export of quantized models")
    else:
        logger.info("modelopt backend: TensorQuantizer exports Q/DQ natively (no global switch)")
