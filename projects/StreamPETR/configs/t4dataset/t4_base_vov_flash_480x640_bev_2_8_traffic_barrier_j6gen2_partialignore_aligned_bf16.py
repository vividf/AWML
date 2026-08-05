# The autoware-ml-aligned retrain recipe (2026-07-31): identical to the
# fp16 j6gen2 partialignore config except the AMP dtype is bfloat16, matching
# autoware-ml's bf16-mixed training.
#
# Why bf16 again after the 2026-07-29 ablation "showed no benefit": that run
# never actually changed the attention precision. FlashAttention.forward
# called `.half()` unconditionally, which silently downcast bf16 back to fp16
# inside cross-attention, so the ablation compared fp16-attention against
# fp16-attention. attention.py now honors the caller's dtype, so with this
# config the numerics finally match autoware-ml everywhere:
#   backbone/FFN bf16, self-attention fp32 (the stock autocast(enabled=False)
#   island in petr_transformer.py), cross-attention bf16.
# The single-batch overfit probe attributes essentially the whole remaining
# framework fitting gap to attention precision (parity_out, 2026-07-30).
#
# Rides along from the working tree (see git diff):
#   - cv2 INTER_LINEAR resize (pixel parity with autoware-ml)
#   - DN gravity-center z targets
#   - local-count loss normalization (multi-GPU alignment)
#   - StreamPETR-scoped GT hygiene filter (no-op on this pkl, safeguard only)
#   - per-camera augmentation sampling + 1px flip fix (NOTE: tested once as
#     exp2 and scored lowest; revert transform_3d.py's per-camera loop to a
#     single _sample_augmentation draw if this should be excluded)
_base_ = [
    "./t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py",
]

optim_wrapper = dict(
    dtype="bfloat16",
    loss_scale=dict(enabled=False, _delete_=True),
)
