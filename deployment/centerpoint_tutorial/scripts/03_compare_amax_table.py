#!/usr/bin/env python
"""Step 3: Markdown table comparing reproduced amax vs the release checkpoint's amax.

    python work_dirs/centerpoint_tutorial/scripts/03_compare_amax_table.py \
        > work_dirs/centerpoint_tutorial/calib_trace/amax_comparison.md
"""

import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = torch.load(ROOT / "checkpoints" / "original_release_amax.pth", map_location="cpu", weights_only=True)
    with open(ROOT / "calib_trace" / "final_amax_all_quantizers.pkl", "rb") as f:
        ours = pickle.load(f)

    print("| quantizer | release amax (400 samples, full val) | reproduced amax (60 local samples) | rel diff |")
    print("|---|---|---|---|")
    rows = []
    for k in sorted(release.keys()):
        name = k.replace("._amax", "")
        if name not in ours:
            continue
        r = np.asarray(release[k]).flatten()
        o = np.asarray(ours[name]).flatten()
        if r.size == 1:  # activation (per-tensor)
            rel = abs(float(o[0]) - float(r[0])) / max(float(r[0]), 1e-9)
            rows.append((name, f"{float(r[0]):.4f}", f"{float(o[0]):.4f}", f"{rel:.1%}", rel, "act"))
        else:  # weight (per-channel) — weights identical, so this should be ~0
            rel = float(np.max(np.abs(o - r) / np.maximum(np.abs(r), 1e-9)))
            rows.append((name, f"per-ch[{r.size}]", f"per-ch[{o.size}]", f"{rel:.2e}", rel, "wt"))
    for name, a, b, c, _, _ in rows:
        print(f"| `{name}` | {a} | {b} | {c} |")

    act = [r[4] for r in rows if r[5] == "act"]
    wt = [r[4] for r in rows if r[5] == "wt"]
    print()
    print(f"Activation amax rel diff: median {np.median(act):.1%}, max {max(act):.1%} (n={len(act)})")
    if wt:
        print(f"Weight amax rel diff (should be ~0): max {max(wt):.2e} (n={len(wt)})")


if __name__ == "__main__":
    main()
