"""Utilities for loading TensorRT custom plugin libraries.

Plugin paths are read only from deploy_config.tensorrt_config.plugin_libraries.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Iterable, List, Tuple

_LOADED_PLUGIN_LIBS: set[str] = set()


def _normalize_libraries(plugin_libraries: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for path in plugin_libraries:
        p = str(path).strip()
        if not p:
            continue
        expanded = os.path.expandvars(os.path.expanduser(p))
        if expanded in seen:
            continue
        seen.add(expanded)
        normalized.append(expanded)
    return normalized


def _get_tensorrt_registries():
    """Best-effort fetch runtime and builder registries across TRT versions."""
    try:
        import tensorrt as trt
    except Exception:
        return []

    regs = []
    try:
        runtime_registry = trt.get_plugin_registry()
        if runtime_registry is not None:
            regs.append(("runtime", runtime_registry))
    except Exception:
        pass

    if hasattr(trt, "get_builder_plugin_registry"):
        builder_registry = None
        try:
            builder_registry = trt.get_builder_plugin_registry()
        except TypeError:
            if hasattr(trt, "EngineCapability"):
                try:
                    builder_registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
                except Exception:
                    builder_registry = None
        except Exception:
            builder_registry = None
        if builder_registry is not None:
            regs.append(("builder", builder_registry))

    uniq = []
    seen = set()
    for name, reg in regs:
        k = id(reg)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((name, reg))
    return uniq


def load_tensorrt_plugin_libraries(
    logger: logging.Logger,
    plugin_libraries: Iterable[str],
) -> Tuple[str, ...]:
    """Load custom TensorRT plugin libraries.

    Paths are taken only from deploy_config.tensorrt_config.plugin_libraries.
    For TensorRT 10+ (Plugin V3), prefer registry-based loading
    (`IPluginRegistry.load_library`) so creators are registered properly.
    Fallback to `ctypes.CDLL(..., RTLD_GLOBAL)` for compatibility.

    Args:
        logger: Logger used for informative diagnostics.
        plugin_libraries: Plugin library paths from deploy config (e.g. tensorrt_config.plugin_libraries).

    Returns:
        Tuple of loaded plugin library identifiers.

    Raises:
        FileNotFoundError: If a configured path includes a slash but does not exist.
        OSError: If dlopen fails for a provided library.
    """
    resolved = _normalize_libraries(plugin_libraries)
    loaded_now: List[str] = []

    for library in resolved:
        if library in _LOADED_PLUGIN_LIBS:
            continue

        if "/" in library and not os.path.exists(library):
            raise FileNotFoundError(f"TensorRT plugin library not found: {library}")

        # Try TensorRT registry loader first (required by many TRT10 V3 plugins).
        loaded_with_registry = False
        for reg_name, registry in _get_tensorrt_registries():
            if not hasattr(registry, "load_library"):
                continue
            try:
                registry.load_library(library)
                loaded_with_registry = True
                logger.info("Loaded TensorRT plugin library via %s registry: %s", reg_name, library)
            except Exception as exc:  # pragma: no cover - best effort fallback path
                logger.debug("%s registry load failed for %s: %s", reg_name, library, exc)

        # Always ensure symbols are globally visible for dependent shared objects.
        if not loaded_with_registry:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            logger.info("Loaded TensorRT plugin library via ctypes: %s", library)
        else:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)

        _LOADED_PLUGIN_LIBS.add(library)
        loaded_now.append(library)

    if not resolved:
        logger.debug("No custom TensorRT plugin libraries configured. Set tensorrt_config.plugin_libraries.")

    return tuple(_LOADED_PLUGIN_LIBS)
