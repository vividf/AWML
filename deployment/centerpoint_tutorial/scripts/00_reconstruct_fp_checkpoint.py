#!/usr/bin/env python
"""Step 0: Reconstruct an FP (non-quantized) checkpoint from a PTQ checkpoint.

Why this works
--------------
PTQ (post-training quantization, calibration-only) NEVER changes the model
weights. The only thing calibration adds are the ``_amax`` buffers stored
inside every ``TensorQuantizer`` (``..._input_quantizer._amax`` /
``..._weight_quantizer._amax``). So:

    PTQ state_dict  =  BN-fused FP weights  +  amax buffers

Stripping the amax keys gives us back the deployable FP weights. Note that the
weights are the *BN-fused* ones (the PTQ producer runs fuse_bn before
calibration), so when this checkpoint is loaded into a fresh (unfused) model
the backbone/neck BatchNorm keys are simply missing: BN layers stay at their
default init (gamma=1, beta=0, mean=0, var=1), which makes them a numerical
no-op (error ~5e-6 from eps) — the network output is identical to the fused
model for all practical purposes.

Usage:
    python 00_reconstruct_fp_checkpoint.py \
        --ptq-checkpoint /path/epoch_29_ptq.pth \
        --output work_dirs/centerpoint_tutorial/checkpoints/epoch_29_fp_reconstructed.pth
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptq-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ptq_checkpoint, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt)

    fp_sd = {k: v for k, v in sd.items() if "_amax" not in k}
    amax_sd = {k: v for k, v in sd.items() if "_amax" in k}

    print(f"PTQ checkpoint keys : {len(sd)}")
    print(f"  weight/buffer keys: {len(fp_sd)} (kept)")
    print(f"  amax keys         : {len(amax_sd)} (stripped)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fp_sd}, out)
    print(f"FP checkpoint written to {out}")

    # Keep the original amax values next to it: they are the reference we
    # compare our re-calibrated amax against in step 3.
    amax_out = out.with_name("original_release_amax.pth")
    torch.save(amax_sd, amax_out)
    print(f"Original release amax values written to {amax_out}")


if __name__ == "__main__":
    main()
