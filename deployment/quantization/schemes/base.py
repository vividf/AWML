# Copyright (c) OpenMMLab. All rights reserved.
"""Quantization scheme abstraction — the single seam between deployment stages and quantization.

Motivation
----------
Quantization used to be smeared across every deployment stage, and the PTQ producer re-created the
quantized module tree by convention (a code comment saying "must match the loader"). The dense
(pytorch_quantization Q/DQ) tower and the sparse encoder (SparseConv+BN fold for FP16 deploy) had no
shared contract.

A :class:`QuantizationScheme` is a strategy object with one structural step, :meth:`prepare`, that
inserts quantizers / fuses BN. Both the dense and sparse towers implement it; their *internals stay
different* (native TensorRT INT8 Q/DQ vs. a SparseConv+BN fold), but the **seam** is uniform. Because
the loader and the PTQ producer both build their module tree by calling the *same* scheme's
``prepare``, the PTQ ``state_dict`` and the deploy ``load_state_dict`` line up by construction, not by
a "keep these two code paths in sync" comment.

A :class:`QuantizationPlan` composes schemes for a whole model (e.g. a sparse scheme on
``pts_middle_encoder`` + a dense scheme on the backbone/neck/head) and fans ``prepare`` out to every
scheme in order.

Design rule (see ``deployment/docs/architecture.md``): deployment stages (loader / export /
inference) should touch quantization **only through this interface**, and any monkeypatch belongs
inside a scheme's :meth:`~QuantizationScheme.prepare`, never in stage code.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


class QuantizationScheme(abc.ABC):
    """One quantization strategy applied to (part of) a model."""

    #: Stable identifier used in logs. Override in subclasses.
    name: str = "base"

    @abc.abstractmethod
    def prepare(self, model: Any) -> Any:
        """Insert quantizers / fuse BN in place. MUST be identical on PTQ and deploy sides.

        This is the structural step whose result the PTQ ``state_dict`` is saved against and the
        deploy-time ``load_state_dict`` restores onto. Returns ``model`` (mutated in place) for
        chaining convenience.
        """


class QuantizationPlan:
    """An ordered composition of :class:`QuantizationScheme` objects for one model.

    A project declares *which* schemes apply to *which* submodels (composition); the schemes own
    *how* (algorithm). Stage code holds a plan and calls :meth:`prepare` — it never sees
    quantization internals.
    """

    def __init__(self, schemes: Optional[Sequence[QuantizationScheme]] = None) -> None:
        self.schemes: List[QuantizationScheme] = list(schemes or [])

    def __bool__(self) -> bool:
        return bool(self.schemes)

    def __len__(self) -> int:
        return len(self.schemes)

    def add(self, scheme: QuantizationScheme) -> "QuantizationPlan":
        self.schemes.append(scheme)
        return self

    def prepare(self, model: Any) -> Any:
        for scheme in self.schemes:
            logger.info("[quant-plan] prepare: %s", scheme.name)
            scheme.prepare(model)
        return model
