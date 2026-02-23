SECOND + SECONDFPN Architecture Summary
1. Overview

This document summarizes the full tensor shape transformation for:

Backbone: SECOND

Neck: SECONDFPN

Input spatial feature: (B, 32, 1020, 1020)

The goal is to clearly describe:

Each stage output shape

Each kernel’s role

How spatial alignment is achieved

Why torch.cat() works without shape errors

2. Backbone: SECOND
Configuration
in_channels = 32
out_channels = [64, 128, 256]
layer_strides = [2, 2, 2]

Each stage consists of:

One 3×3 Conv (stride=2, padding=1)

Followed by multiple 3×3 Conv (stride=1, padding=1)

BN + ReLU after each conv

2.1 Input
Input spatial feature:
(B, 32, 1020, 1020)
2.2 Stage 0

First Conv:

Conv2d(32 → 64, kernel=3, stride=2, pad=1)

Output size:

1020 → 510

Output:

x0 = (B, 64, 510, 510)

Remaining conv layers in stage 0:

Conv2d(64 → 64, kernel=3, stride=1, pad=1)

No spatial change.

Final Stage 0 output:

x0 = (B, 64, 510, 510)
2.3 Stage 1

First Conv:

Conv2d(64 → 128, kernel=3, stride=2, pad=1)

Spatial:

510 → 255

Output:

x1 = (B, 128, 255, 255)

Remaining conv layers:

Conv2d(128 → 128, kernel=3, stride=1, pad=1)

Final:

x1 = (B, 128, 255, 255)
2.4 Stage 2

First Conv:

Conv2d(128 → 256, kernel=3, stride=2, pad=1)

Spatial:

255 → 128

Output:

x2 = (B, 256, 128, 128)

Remaining conv layers:

Conv2d(256 → 256, kernel=3, stride=1, pad=1)

Final:

x2 = (B, 256, 128, 128)
3. Neck: SECONDFPN
Configuration
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[64, 128, 256],
    out_channels=[128, 128, 128],
    upsample_strides=[0.5, 1, 2],
)

Key behavior:

Branch	Input	Operation	Purpose
0	x0	stride=0.5 → Conv downsample	Reduce large map
1	x1	stride=1 → 1×1 Conv	Keep resolution
2	x2	stride=2 → ConvTranspose	Upsample small map

All branches align to x1 resolution (255×255).

4. Detailed Neck Computation
4.1 Branch 0 (stride = 0.5)

Code converts 0.5 into:

stride = round(1 / 0.5) = 2
Conv2d(kernel=2, stride=2)

Weight seen in ONNX:

conv(128, 64, 2, 2)

Input:

x0 = (B, 64, 510, 510)

Output:

up0 = (B, 128, 255, 255)

This downsamples x0 to match x1 resolution.

4.2 Branch 1 (stride = 1)

Weight in ONNX:

conv(128, 128, 1, 1)

Input:

x1 = (B, 128, 255, 255)

Output:

up1 = (B, 128, 255, 255)

No spatial change.

4.3 Branch 2 (stride = 2)

Weight in ONNX:

ConvTranspose(256, 128, 2, 2)

Input:

x2 = (B, 256, 128, 128)

ConvTranspose parameters (inferred from working concat):

Likely:

stride = 2
kernel = 2
padding = 1
output_padding = 1

Using formula:

H_out = (H_in - 1)*s - 2p + k + op

We get:

128 → 255

Output:

up2 = (B, 128, 255, 255)
5. Final Concatenation

All branches now aligned:

up0 = (B, 128, 255, 255)
up1 = (B, 128, 255, 255)
up2 = (B, 128, 255, 255)

Concatenation:

out = torch.cat([up0, up1, up2], dim=1)

Final output:

(B, 384, 255, 255)

Neck returns:

[(B, 384, 255, 255)]
6. Why Concat Does NOT Crash

Even though original backbone produced:

510 / 255 / 128

The neck performs:

x0 → downsample → 255

x1 → keep → 255

x2 → upsample (carefully padded) → 255

Thus all spatial dimensions match.

No mismatch.
No runtime error.

7. Final Architecture Summary
Input: (B, 32, 1020, 1020)

SECOND Backbone:
    Stage0 → (B, 64, 510, 510)
    Stage1 → (B, 128, 255, 255)
    Stage2 → (B, 256, 128, 128)

SECONDFPN:
    x0 ↓2  → (B, 128, 255, 255)
    x1      → (B, 128, 255, 255)
    x2 ↑2   → (B, 128, 255, 255)

Concat:
    (B, 384, 255, 255)
8. Key Insight

SECONDFPN does NOT always upsample everything to the largest map.

Instead:

It aligns all feature maps to the middle resolution (x1).

This is why:

stride 0.5 exists

stride 2 exists

concat works cleanly
