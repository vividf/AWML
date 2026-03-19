"""Check whether required sparse TensorRT plugin creators are available."""

from __future__ import annotations

import argparse
import ctypes
import sys

import tensorrt as trt

REQUIRED_CREATORS = ("ImplicitGemm", "GetIndicePairsImplicitGemm")


def _get_candidate_registries() -> list[tuple[str, object]]:
    registries: list[tuple[str, object]] = []
    runtime_registry = trt.get_plugin_registry()
    if runtime_registry is not None:
        registries.append(("runtime", runtime_registry))

    if hasattr(trt, "get_builder_plugin_registry"):
        builder_registry = None
        try:
            # TRT Python APIs vary by version.
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
            registries.append(("builder", builder_registry))

    # De-duplicate same object
    uniq: list[tuple[str, object]] = []
    seen = set()
    for name, reg in registries:
        k = id(reg)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((name, reg))
    return uniq


def _iter_creators(registry: object):
    # TRT10 may expose V2 creators via plugin_creator_list and
    # V3 creators via all_creators / all_creators_recursive.
    yielded = []
    for attr in ("plugin_creator_list", "all_creators", "all_creators_recursive"):
        if hasattr(registry, attr):
            try:
                for c in list(getattr(registry, attr)):
                    yielded.append(c)
            except Exception:
                pass
    # de-duplicate by (name, version, namespace)
    uniq = []
    seen = set()
    for c in yielded:
        key = (
            getattr(c, "name", ""),
            getattr(c, "plugin_version", ""),
            getattr(c, "plugin_namespace", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TensorRT sparse plugin creators.")
    parser.add_argument(
        "--plugin-so",
        type=str,
        default="",
        help="Optional absolute path to custom TensorRT plugin .so to dlopen before check.",
    )
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    if args.plugin_so:
        loaded_any_registry = False
        for reg_name, registry in _get_candidate_registries():
            if hasattr(registry, "load_library"):
                try:
                    registry.load_library(args.plugin_so)
                    loaded_any_registry = True
                    print(f"Loaded plugin via {reg_name} registry: {args.plugin_so}")
                except Exception as exc:
                    print(f"{reg_name} registry load failed: {exc}")

        ctypes.CDLL(args.plugin_so, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(logger, "")
        if not loaded_any_registry:
            print(f"Loaded plugin via ctypes: {args.plugin_so}")

    creators = []
    for reg_name, registry in _get_candidate_registries():
        reg_creators = _iter_creators(registry)
        print(f"{reg_name} registry creator count: {len(reg_creators)}")
        creators.extend(reg_creators)

    # de-duplicate across registries
    uniq = []
    seen = set()
    for c in creators:
        key = (
            getattr(c, "name", ""),
            getattr(c, "plugin_version", ""),
            getattr(c, "plugin_namespace", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    creators = uniq
    print(f"Total unique TensorRT creators: {len(creators)}")

    missing = []
    for name in REQUIRED_CREATORS:
        hits = [c for c in creators if c.name == name]
        print(f"{name}: {len(hits)}")
        for creator in hits:
            print(f"  - version={creator.plugin_version}, namespace='{creator.plugin_namespace}'")
        if not hits:
            missing.append(name)

    if missing:
        # Helpful debug: show nearby names.
        print("Creators containing 'Gemm'/'Indice' for debugging:")
        for c in creators:
            n = getattr(c, "name", "")
            if "Gemm" in n or "Indice" in n or "Indices" in n:
                print(
                    f"  - {n} (version={getattr(c, 'plugin_version', '?')}, "
                    f"namespace='{getattr(c, 'plugin_namespace', '')}')"
                )
        print(f"Missing required creators: {', '.join(missing)}")
        return 1

    print("All required sparse creators are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
