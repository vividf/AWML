#!/usr/bin/env python3
"""Patch autoware_tensorrt_plugins to disable the pair-mask sort for INT8.

Background
----------
New3D / Lidar_AI_Solution sets `bool do_sort = !int8_inference_;` inside
`sparseConvImplicit.cu`, i.e. when the downstream conv runs in INT8, the
argsort inside `get_indice_pairs_implicit_gemm` is skipped. See:
  https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md#performance-guide
  Lidar_AI_Solution_open_source_spconv/libraries/New3DSparseConvolution/libspconv/src/sparseConvImplicit.cu:368

AWML's INT8 sparse encoder uses `autoware_tensorrt_plugins` for pair-gen
and a separate ImplicitGemmInt8 plugin for the conv. The pair-gen plugin
currently calls `SpconvOps::get_indice_pairs_implicit_gemm(..., use_direct_table)`
which falls back to `do_sort=true` (the default).

This helper patches the cloned plugin source so that the pair-gen plugin:
  1. Reads an `SPCONV_DO_SORT` env var (same name as traveller59 spconv's
     Python flag) at runtime.
  2. Defaults to `do_sort=false` (INT8 deploy behaviour) when the env var
     is not set, matching New3D's `do_sort = !int8_inference_`.
  3. Passes the resolved `do_sort` into the SpconvOps call.

The patch is idempotent: re-running this script on an already-patched file
is a no-op.
"""

from __future__ import annotations

import pathlib
import re
import sys

MARKER = "AWML_PATCH_DO_SORT_FOR_INT8"

HELPER_SNIPPET = f"""
// {MARKER}: disable pair-gen sort for INT8 inference.
// Matches New3D's `bool do_sort = !int8_inference_;`
// (sparseConvImplicit.cu:368). Runtime toggle: SPCONV_DO_SORT=1 re-enables.
namespace {{
inline bool awml_get_do_sort_from_env() {{
  const char * env = std::getenv(\"SPCONV_DO_SORT\");
  if (env == nullptr) {{
    return false;  // AWML INT8 sparse encoder: skip sort by default.
  }}
  std::string v(env);
  if (v == \"0\" || v == \"false\" || v == \"False\" || v == \"FALSE\") {{
    return false;
  }}
  return true;
}}
}}  // namespace
"""


def patch_plugin_source(path: pathlib.Path) -> bool:
    """Patch the plugin .cpp in-place. Returns True if modified."""
    if not path.is_file():
        raise FileNotFoundError(f"plugin source not found: {path}")

    src = path.read_text()
    if MARKER in src:
        print(f"[awml-patch] already patched, skipping: {path}")
        return False

    # 1) Rewrite both SpconvOps::get_indice_pairs_implicit_gemm call sites so
    #    that they pass `do_sort` explicitly. Both calls currently end with
    #    `tv::CUDAKernelTimer(false), use_direct_table);`.
    call_pat = re.compile(
        r"tv::CUDAKernelTimer\(false\),\s*use_direct_table\);",
        flags=re.MULTILINE,
    )
    call_repl = (
        "tv::CUDAKernelTimer(false), use_direct_table, "
        "/*do_sort=*/awml_get_do_sort_from_env());"
    )
    new_src, n_replaced = call_pat.subn(call_repl, src)
    if n_replaced != 2:
        raise RuntimeError(
            f"expected 2 SpconvOps::get_indice_pairs_implicit_gemm call sites "
            f"in {path}, found {n_replaced}. Upstream layout may have changed."
        )

    # 2) Inject the helper namespace after the last top-level #include. This
    #    keeps it before any function definitions while not polluting the
    #    `nvinfer1::plugin` namespace.
    include_iter = list(
        re.finditer(r"^#include[^\n]*\n", new_src, flags=re.MULTILINE)
    )
    if not include_iter:
        raise RuntimeError(f"no #include lines found in {path} to anchor helper")
    anchor = include_iter[-1].end()
    new_src = new_src[:anchor] + HELPER_SNIPPET + new_src[anchor:]

    path.write_text(new_src)
    print(f"[awml-patch] patched {path} (2 call sites + helper)")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: disable_sort_for_int8.py "
            "<path to get_indices_pairs_implicit_gemm_plugin.cpp>",
            file=sys.stderr,
        )
        return 2
    patch_plugin_source(pathlib.Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
