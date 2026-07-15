# Quantization Refactor Spec (DRAFT — for discussion, no code changes yet)

Status: **Goal 1 implemented** (in the working tree — spconv-INT8 removal done; Docker e2e pending).
**Goal 2 implemented** (host-verified: ast + pyflakes clean, parity-oracle test green over all 7 configs;
Docker e2e mAP pending — see §3.7/§3.8 and the two Docker-verify caveats below).
**Goal 3** is the design-review layer, re-checked against the current tree after Goal 1 (§4).
**Goal 4 implemented (2026-07-16)** — framework-wide cleanliness pass over deployment/ + quantization
(§5); all four work packages landed in the working tree, host-verified (ast 155 files clean, pyflakes
clean, golden `_base_`-merge check green over all 6 INT8 configs, migration-oracle logic green, grep
gates pass). Docker e2e (mAP identical) pending, plus the parked 4B.3 / 4D.3 Docker decisions — see §5.7.

> **Goal 2 Docker-verify caveats** (behavior-preservation assumptions the host cannot check):
> 1. BEVFusion `disable_recipes=["add"]` — confirm whether load-bearing (does the dense backbone contain
>    a block `attach_quant_add` matches?). Either way it preserves behavior; drop it only if confirmed no-op.
> 2. `quant_model` now **always** quantizes `pts_backbone` Linear (was gated by `quant_linear_backbone`).
>    Safe iff ResNet/SECOND backbones have no `nn.Linear` (they are Conv-only); confirm mAP unchanged for
>    the `resnet*` / `second*` configs.

Scope: `deployment/` quantization for CenterPoint and BEVFusion-L.

Four goals:

1. **Remove all spconv INT8 code** — sparse-encoder INT8 gave no meaningful mAP/latency win, so the
   sparse encoder stays FP16 and every INT8-sparse-specific path is deleted.
2. **Redesign the quantization config** — replace the ~13 ad-hoc boolean flags in each
   `deploy_config_*.py` with a small declarative surface, informed by NVIDIA modelopt and
   CUDA-BEVFusion.
3. **Design review & risk hardening** — a Principal-Engineer pass over Goals 1–2:
   challenge the load-bearing assumptions against the *actual* code, surface architectural risks, and
   make the under-specified decisions explicit — **without** redesigning what already works.
4. **Framework cleanliness pass** (§5) — with Goals 1–3 landed, re-review the *whole*
   deployment + quantization framework and remove what the earlier goals left behind: drifted
   copies, decorative typed layers, dead public surface, and config-file cloning. Optimize only for
   simplicity, cognitive load, maintainability, locality of change, information hiding, and clear
   ownership — explicitly **not** for SOLID compliance, design patterns, LOC reduction, or more
   abstraction.

---

## 0. Background — how quantization is wired today

The **engine** is already well-factored (keep it):

- `deployment/quantization/` is model-agnostic: `core/` (Q/DQ modules, replace, fusion,
  calibration, descriptors), `recipes/` (architecture-specific Q/DQ placement), `schemes/` (the
  `QuantizationScheme` / `QuantizationPlan` seam), `sparse/` (spconv).
- Each project has a **single** `build_<model>_plan(config).prepare(model)` that the PTQ producer,
  the deploy loader, and (CenterPoint) the QAT hook all call. This guarantees the PTQ `state_dict`
  and the deploy `load_state_dict` line up **by construction**. **This invariant is sacred — every
  change below must preserve it.**

The **config surface** is the problem. `QuantizationConfig` (`deployment/config/schema.py`) carries
~18 fields; a real config (`centerpoint/config/deploy_config_int8_vov99.py`) sets 13 of them:

```python
quantization = dict(
    enabled=True, mode="ptq", fuse_bn=True,
    quant_ese_mul_identity=True, quant_ese_pool_input=True, quant_maxpool_input=True,
    quant_voxel_encoder=False, quant_backbone=True, quant_neck=True, quant_head=True,
    quant_add=True, quant_linear_backbone=True,
    skip_backbone_first_stages=0, skip_backbone_stages=[],
    skip_vovnet_stages=[0, 1],
    sensitive_layers=[],
)
```

Those 13 flags conflate **three unrelated concerns**:

| Concern | Flags today | Who should decide |
|---|---|---|
| **A. Precision placement** — which parts are INT8 vs kept FP16 | `quant_backbone/neck/head/voxel_encoder`, `sensitive_layers`, `skip_backbone_first_stages`, `skip_backbone_stages`, `skip_vovnet_stages`, (BEVFusion) `spconv_int8_fp16_layers` | **deployment** (per-config tuning) |
| **B. Architecture recipe** — which Q/DQ placement rules apply (residual-add, eSE, maxpool, ConvNeXt Linear) | `quant_add`, `quant_ese_mul_identity`, `quant_ese_pool_input`, `quant_maxpool_input`, `quant_linear_backbone` | **the model architecture** — NOT a per-deploy choice |
| **C. Mechanics** | `enabled`, `mode`, `fuse_bn`, `ptq_checkpoint`, `calib_cache_path` | legitimately config |

Concern **B** is the core of the mess: the recipes attach by **module class name**
(`attach_quant_add` matches `BasicBlock`/`SparseBasicBlock`/`ConvNeXtBlock`/`_OSA_module`;
eSE/maxpool match `eSEModule`/`nn.MaxPool2d`) and are **idempotent no-ops when the architecture
lacks that module**. So a VoVNet model *always* wants the eSE+maxpool+add recipes and a ConvNeXt
model *always* wants add+linear_backbone — this is a property of the backbone, yet today it is
re-declared as booleans in every deploy config. **Architecture knowledge is leaking into config.**

---

## 1. Reference comparison — AWML vs CUDA-BEVFusion vs modelopt

### 1.1 CUDA-BEVFusion (`Lidar_AI_Solution/CUDA-BEVFusion/qat/`)
100% imperative, hard-coded. Layer selection = "which subtree you pass to `quantize_*()`"; skips =
two literal `disable_quantization(model.…deblocks[0][0])` calls; residual/concat handling =
monkey-patched `.forward` + shared `_input_quantizer` objects. **No config file at all.**

- **Good ideas worth keeping:** (a) **shared quantizer for Add/Concat** so TRT fuses Conv+Add
  (AWML already borrowed this — `attach_quant_add` quantizes only the identity branch); (b) explicit
  per-layer precision + skip list; (c) spconv treated as a separate backend (per-channel weight axis,
  custom ONNX op with baked-in ranges).
- **Anti-patterns to avoid:** attribute-path surgery (`layer3[0].conv1`), forward monkey-patching as
  the config mechanism, skip rules scattered across 4 files, precision set in the *export* script
  separately from where quantizers are inserted. Not portable, not declarative.

### 1.2 NVIDIA modelopt (`TensorRT-Model-Optimizer/modelopt/torch/quantization/`)
Fully declarative. One entry point `mtq.quantize(model, config, forward_loop)`; config is a
two-key dict:

```python
INT8_DEFAULT_CFG = {
    "quant_cfg": {                                  # pattern (glob on quantizer name) -> settings
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer":  {"num_bits": 8, "axis": None},
        **_default_disabled_quantizer_cfg,          # "default": {"enable": False} + sensitive globs
    },
    "algorithm": "max",                             # calibration, orthogonal to placement
}
```

- **Key design:** `"default"` disables everything first, then specific glob patterns re-enable /
  reconfigure by exception (later match wins). Config surface grows with the number of **distinct
  policies**, not the number of layers. Calibration (`algorithm`), placement (`quant_cfg`), and data
  (`forward_loop`) are fully orthogonal. Settings are pydantic-validated with sane defaults.
- **Caveats:** order-dependence is implicit; matching hinges on a stable naming convention; a glob
  that matches nothing fails **silently** (no feedback).

### 1.3 Verdict — which is production-grade
**modelopt's config model wins for production** and is the target philosophy: declarative,
composable, "sane default + exception overrides," validated schema. CUDA-BEVFusion contributes the
INT8 *placement* know-how (already in AWML's recipes). **AWML's engine (scheme/plan seam + single
`build_<model>_plan`) is actually stronger than either reference** — modelopt has no
"PTQ and deploy build the identical tree by construction" guarantee, and CUDA-BEVFusion has none of
this structure. So we **keep the AWML engine and re-skin only the config surface** in the modelopt
style. We do NOT adopt modelopt as a dependency and do NOT rewrite the engine.

---

## 2. Goal 1 — Remove spconv INT8

### 2.1 The critical KEEP / REMOVE boundary
The BEVFusion sparse encoder must still run — just in **FP16**. So we remove the INT8-sparse path
but keep the FP16-sparse path. The one shared function that must survive is
`fuse_spconv_bn_in_encoder` (used by both FP16 export and the old INT8 path); it currently lives
inside the file we are gutting, so it must be relocated first.

**KEEP (FP16 sparse path — do not touch):**
- `fuse_spconv_bn_in_encoder` (relocate out of `sparse/spconv_int8.py` → `sparse/fusion.py`, or fold
  into `export/spconv_bn_fusion.py` which is already its designated shim).
- `export/spconv_bn_fusion.py`, `export/onnx_fuse_implicit_gemm_activation.py`.
- Config flags `spconv_do_sort`, `spconv_fuse_implicit_gemm_relu`, `fuse_spconv_bn`.
- The **FP16** ImplicitGemm TRT plugin (in `libautoware_tensorrt_plugins.so`, external).
- `deploy_config_split_sparse_fp16_dense_int8_2_8.py` (this becomes the canonical BEVFusion config).

**REMOVE (INT8-sparse-specific):**

*Engine (`deployment/quantization/sparse/`):*
- `spconv_int8.py` → delete `apply_nvidia_spconv_int8`, `calibrate_spconv_nvidia` (relocate the BN
  fuse first). File likely disappears.
- `spconv_add_patch.py` (`ensure_spconv_quantize_per_tensor_float_activations` — INT8-only).
- `naming.py` (`tail_without_encoder_layers`, `topologically_sorted_sparse_stems`) — **verify** only
  the INT8 ONNX transform imports it (grep says yes) before deleting.
- Trim `sparse/__init__.py` exports; the `sparse/` package may collapse to just the BN-fuse helper.

*BEVFusion export (`projects/bevfusion_l/export/`):*
- `sparse_int8_onnx_transform.py`, `sparse_int8_transform_ops.py`, `sparse_int8_onnx_audit.py`
  (the `ImplicitGemm → ImplicitGemmInt8` graph rewrite).
- `sparse_encoder_float_shadow.py` — **verify** it is INT8-debug-only (imports the shared BN fuse but
  has spconv_int8 refs) before deleting.

*BEVFusion C++ plugin:*
- `projects/bevfusion_l/cpp/int8_plugin/` (9 files — the `ImplicitGemmInt8` TRT plugin + CUDA
  quantize kernels). Removing this means the built `.so` no longer needs the INT8 plugin; the FP16
  ImplicitGemm plugin is separate.

*BEVFusion scripts / experimental / benchmark / debug:*
- `scripts/verify_spconv_int8.py`, `scripts/docker_eval_split_int8.sh`.
- `experimental/export_sparse_encoder_int8.py`, `experimental/benchmark_sparse_int8.sh`.
- `benchmark/profile_sparse_encoder.py`, `debug/sparse_encoder_hooks.py` — trim INT8 refs (keep FP16
  profiling/hooks).

*Config to delete:*
- `projects/bevfusion_l/config/deploy_config_split_sparse_int8_dense_int8_2_8.py`.

**EDIT (remove INT8 branches, keep the file):**
- `quantization/schemes.py` — `SpconvInt8Scheme`: drop the `int8=True` path and scale-buffer
  registration; it reduces to a fuse-only scheme → rename `SpconvBnFuseScheme`.
- `quantization/plan.py` — `build_bevfusion_plan`: drop the `spconv_int8` branch; the sparse scheme
  becomes fuse-only (still added under `fuse_bn`). Reconsider whether `include_sparse` is still needed.
- `quantization/quantize.py` (25 refs) — remove sparse-INT8 calibration; BEVFusion PTQ becomes
  dense-only.
- `io/model_loader.py` (9 refs), `runner.py` (9 refs), `export/component_builder.py` (9 refs),
  `config/bevfusion_deployment_config.py` (5 refs), `inference/pytorch_inference_pipeline.py` (1),
  `quantization/__init__.py` (3) — strip INT8-sparse branches.

*Config schema:*
- `deployment/config/schema.py` — delete fields `spconv_int8`, `spconv_int8_fp16_layers`, and the
  fold in `load_quantization_config`.
- `deployment/config/base.py` (lines ~78–83) — delete the `spconv_int8_fp16_layers` fold.

*Docs:*
- Update `docs/quantization_pipeline.md`, `deployment/quantization/README.md`.
- The ~29 `projects/bevfusion_l/docs/*int8*` notes: mark stale / move to an `archive/` folder (user
  decides; not deleting history unasked).

### 2.2 Verification gates (host has no torch; runs in Docker)
- Host: `python -m pyflakes` / `ast.parse` on every edited file; grep for dangling
  `spconv_int8` / `SpconvInt8` / `ImplicitGemmInt8` references.
- Docker e2e: BEVFusion FP16-sparse + dense-INT8 deploy + eval still matches the reference mAP
  (baseline 0.3228 / 0.3931). CenterPoint deploy unaffected.

---

## 3. Goal 2 — Config redesign

### 3.1 Principle
Split the three conflated concerns (§0):
- **B (architecture recipe)** moves **into code** — each project's scheme *always* attaches its
  architecture's recipe set; the class-name gate in `recipes/attach.py` makes every attach a no-op where
  the module is absent, so **no backbone detection is needed** (R4). It disappears from deploy configs,
  except a per-config `disable_recipes=[...]` opt-out (required, not just for ablation — §3.5(2), R1).
- **A (precision placement)** becomes **one declarative, subtree-glob list** (`keep_fp16`) in the config.
- **C (mechanics)** stays, minimal.

### 3.2 Proposed new `quantization` block

```python
quantization = dict(
    # --- C. mechanics ---
    enabled=True,
    mode="ptq",                 # "ptq" | "qat"
    fuse_bn=True,
    ptq_checkpoint=True,        # checkpoint carries calibrated _amax
    calib_cache_path=None,

    # --- A. precision placement (modelopt-style: default + exceptions) ---
    default_precision="int8",   # everything the plan reaches is INT8 by default
    keep_fp16=[                 # kept FP16; SUBTREE match — a name keeps that module + all its children (§3.3a)
        "pts_voxel_encoder",        # was quant_voxel_encoder=False
        "pts_backbone.stem",        # was skip_vovnet_stages=[0]
        "pts_backbone.stage2",      # was skip_vovnet_stages=[1]
        # "pts_bbox_head",          # was quant_head=False
        # "pts_neck.deblocks.*.0",  # ConvTranspose2d — no TRT INT8
    ],
)
```

The vov99 example collapses **13 flags → 4 keys + a `keep_fp16` list**. All of
`quant_backbone/neck/head/voxel_encoder`, `sensitive_layers`, `skip_backbone_first_stages`,
`skip_backbone_stages`, `skip_vovnet_stages` collapse into `keep_fp16`. All of
`quant_add/quant_ese_*/quant_maxpool_input/quant_linear_backbone` disappear (auto-attached by the
plan builder). For BEVFusion, `spconv_int8` / `spconv_int8_fp16_layers` are already gone (Goal 1).

Before → after (vov99):

```python
# BEFORE (13 flags)                         # AFTER
quant_ese_mul_identity=True,                 default_precision="int8",
quant_ese_pool_input=True,                   keep_fp16=[
quant_maxpool_input=True,                        "pts_voxel_encoder",
quant_voxel_encoder=False,                       "pts_backbone.stem",
quant_backbone=True,                             "pts_backbone.stage2",
quant_neck=True,                             ],
quant_head=True,
quant_add=True,
quant_linear_backbone=True,
skip_backbone_first_stages=0,
skip_backbone_stages=[],
skip_vovnet_stages=[0, 1],
sensitive_layers=[],
```

### 3.3 How it maps onto the engine — concrete semantics
The original draft left two things implicit, and both bite (R3). Pin them down:

**(a) `keep_fp16` is a *subtree* match, not raw `fnmatch`.** Today the skip test is
`full_name in skip_names` (`core/replace.py:213`) and the towers are walked *from the tower root*
(`quant_model.py:64`), so the root's own name is **never tested** — only its descendants are. So the
draft's own mapping `quant_voxel_encoder=False → keep_fp16=["pts_voxel_encoder"]` would **silently fail
to skip the tower**, and the zero-match guard would *not* fire (the root matches exactly one module).
Fix (chosen — **no core change**): `keep_fp16` means "this module **and everything under it**," resolved
**in the `keep_fp16` expansion helper**, not in the engine. The helper — which must exist anyway for the
zero-match warning (§3.4) — expands each pattern against `named_modules()` to {matched modules **∪ their
descendants**} (`n == m or n.startswith(m + ".")`) and passes that concrete set as `skip_names`.
`core/replace.py` stays **byte-frozen**: its existing exact-match test then fires on the first materialized
descendant it reaches. Effect-identical for every existing sub-tower skip (`pts_backbone.stem`). *(Alternative
considered and rejected: make the skip test itself prefix-aware inside `replace.py` — 2 lines × 2 functions.
Rejected because it edits a shared engine primitive for a capability the helper already provides — see §3.8(1).)*

**(b) `keep_fp16` gates Q/DQ reach only — never BN fusion.** Today `sensitive_layers` → `skip_names`
(skips Q/DQ), but `fuse_targets` stays the full dense set (`dense_qdq.py:50`), so a skipped tower is
still BN-fused and its `state_dict` keys stay aligned on both sides. `keep_fp16` inherits this exactly:
it subtracts from the *quantized* set; the *fused* set is untouched. If `keep_fp16` ever changed fusion,
PTQ and deploy trees would diverge.

**Where the work happens:**
- **Build time** (`build_<model>_plan(config)`): construct the scheme(s), passing `keep_fp16`,
  `default_precision`, `disable_recipes` straight through. No model yet → no glob expansion here.
- **Prepare time** (`scheme.prepare(model)`): model exists → expand `keep_fp16` against
  `model.named_modules()`, **log per-pattern match count and warn on zero (§3.4)**, then drive
  Conv/Linear replacement (minus the resolved subtrees) and attach the recipe set (minus
  `disable_recipes`). `resolved_sensitive_layers()`, `skip_*`, and the `quant_*` booleans are gone.
- The recipe set is **per project scheme**, not global: `CenterPointDenseScheme` attaches
  add + eSE + maxpool (VoVNet) via `quant_model`; BEVFusion's `DenseQDQScheme` attaches add only. Each
  attach is class-gated, so unconditional attach is safe *given* `disable_recipes` for exceptions (R1).
- Both PTQ producer and deploy loader call the same builder with the same args → identical-tree invariant
  holds, **once `include_sparse` is closed** (R2, §3.8(3)).

### 3.4 New footgun guard (improves on both references)
When expanding `keep_fp16` globs, **log the match count per pattern and warn on zero matches**. This
fixes modelopt's "silent no-match" weakness and catches typos in `keep_fp16` immediately.

### 3.5 Confirmed decisions
1. **Default-precision model:** ✅ `default_precision="int8"` + `keep_fp16` exceptions (opt-out).
2. **Recipes:** ✅ always-on, auto-attached per project scheme (class-gated; no backbone detection — R4).
   `disable_recipes=[...]` is **required to preserve behavior, not just for ablation**: BEVFusion ships
   `quant_add=False` today, so its migration MUST set `disable_recipes=["add"]` or the always-on add
   recipe changes its tree/mAP (R1, §3.7). Removed from deploy configs otherwise.
3. **Migration:** ✅ **hard-cut** — rewrite every `deploy_config_*.py` to the new schema and delete
   the old fields from `QuantizationConfig`. No back-compat shim.
4. **`keep_fp16` value form:** flat list of globs (binary int8/fp16 for now). A `{pattern: precision}`
   map is a future extension if per-layer bit-width (fp8, etc.) is ever needed — not built now.
5. **Key naming:** `default_precision` + `keep_fp16` + `disable_recipes`.
6. **eSE/maxpool default policy (resolves R1) — the *single-Q-at-input* recipe is canonical, not a choice:**
   the CenterPoint scheme attaches, in order, `attach_quant_add` → `attach_ese_pool_input` →
   `attach_ese_mul_identity` → `attach_maxpool`. With `pool_input` already present, `mul_identity` adds
   **no** second Q (`recipes/attach.py:168`), so `eSEModuleForwardHook` runs the single-Q-at-input path:
   one FP32→INT8 reformat at the eSE input, fanned out to both `Mul` operands (`forward_hooks.py:272-282`).
   This is the *best* fully-INT8, reformat-minimizing design (matches modelopt's per-tensor shared input
   quantizer + CUDA-BEVFusion's shared-Q; R1) — it is what vov99 already runs, so it is behavior-preserving
   by construction. Also **delete the legacy two-Q path** (`mul_identity`-only) — it is dead and strictly
   worse (see Simplicity guardrails). `disable_recipes` remains for genuine opt-outs (e.g. BEVFusion `add`).

### 3.6 Fields deleted from `QuantizationConfig` (hard-cut)
`quant_backbone`, `quant_neck`, `quant_head`, `quant_voxel_encoder`, `quant_add`,
`quant_linear_backbone`, `quant_ese_mul_identity`, `quant_ese_pool_input`, `quant_maxpool_input`,
`skip_backbone_first_stages`, `skip_backbone_stages`, `skip_vovnet_stages`, `sensitive_layers`.
Plus the helpers `resolved_sensitive_layers()`, `dense_quant_enabled()`, and the `_VOVNET_STAGE_NAMES`
expansion. (`spconv_int8` / `spconv_int8_fp16_layers` were never typed fields — already gone with Goal 1.)
Added: `default_precision`, `keep_fp16`, `disable_recipes`.

**Migrate the call sites, don't just delete (verified against the current tree).** `resolved_sensitive_layers()`
has 5 callers (`centerpoint/{io/model_loader.py:173, quantization/quantize.py:225 & 433, quantization/plan.py:39}`
and `bevfusion_l/quantization/quantize.py:198`); `dense_quant_enabled()` now has a live caller
(`bevfusion_l/quantization/quantize.py:199` — this *inverts* R5's original "confirm zero callers"). Each site
moves to the glob-expansion path in the **same** commit, or the build breaks.

### 3.7 Concrete migration (behavior-preserving) — both live configs
Migration rule: `keep_fp16 = {towers where quant_*=False} ∪ resolved_sensitive_layers()`; every recipe that
was **on** stays on by default, every recipe that was **off** becomes a `disable_recipes` entry. This makes
each config's quantized tree identical before/after — the parity oracle in §3.8(2) enforces it.

**CenterPoint `deploy_config_int8_vov99.py`** (13 flags → 4 keys + a list):

```python
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    default_precision="int8",
    keep_fp16=[
        "pts_voxel_encoder",     # was quant_voxel_encoder=False
        "pts_backbone.stem",     # was skip_vovnet_stages=[0]
        "pts_backbone.stage2",   # was skip_vovnet_stages=[1]
    ],
    # disable_recipes omitted: add + eSE + maxpool + backbone-Linear were ALL on today.
)
```

`quant_linear_backbone=True` is reproduced because the CenterPoint scheme quantizes backbone Linear by
default; `pts_voxel_encoder`'s Linear stays FP via the `keep_fp16` **subtree** entry (needs §3.8(1)).

**BEVFusion `deploy_config_split_sparse_fp16_dense_int8_2_8.py`:**

```python
quantization = dict(
    enabled=True,
    mode="ptq",
    fuse_bn=True,
    ptq_checkpoint=True,
    default_precision="int8",
    keep_fp16=[],
    disable_recipes=["add"],   # was quant_add=False — REQUIRED, else the always-on add recipe changes the tree (R1)
)
```

**Docker-verify for BEVFusion:** confirm whether its dense backbone actually contains a block
`attach_quant_add` matches (`BasicBlock` / `ConvNeXtBlock` / `_OSA_module` / `SparseBasicBlock`). If yes,
`disable_recipes=["add"]` is load-bearing (holds current mAP); if no, `add` is already a no-op and the entry
is harmless. Ship it either way — **removing** it can only *change* behavior, never preserve it.

### 3.8 Where the subtree logic lives + verification gates
1. **Subtree expansion in the `keep_fp16` helper — NO core change (chosen).** The new helper (needed anyway
   for the zero-match warning, §3.4) expands each pattern against `named_modules()` to {matched modules ∪
   their descendants} and passes that concrete set as `skip_names`; `core/replace.py` is untouched and its
   exact-match test does the rest (§3.3a). Rationale: (i) the helper must expand globs regardless, so the
   subtree behavior is nearly free; (ii) it keeps **all** `keep_fp16` semantics in **one** new place instead
   of splitting them between a helper and the engine; (iii) it leaves the shared engine primitive frozen,
   honoring the "keep the engine, re-skin the config" non-goal; (iv) it makes the parity oracle (item 2) a
   pure set-equality assertion; (v) it sidesteps a latent inconsistency — `quant_conv_module` skips by exact
   match (`replace.py:213`) but `attach_maxpool_input_quantizer` already skips by loose `startswith`
   (`attach.py:246`) — because concrete names satisfy both checks. *Alternative (rejected): make the skip test
   prefix-aware inside `replace.py`, ~2 lines × 2 functions — smaller diff in isolation, but edits a primitive
   every quant path shares for no capability the helper doesn't already give.*
2. **Parity oracle test — write FIRST, before the schema change (R6).** For each existing config, take its
   *old* flags, compute `{tower opt-outs} ∪ resolved_sensitive_layers()`, and assert it equals the
   subtree-expansion of the migrated `keep_fp16`; also assert the recipe on/off set matches. This turns
   `resolved_sensitive_layers()` into a golden oracle *before* it is deleted. Expect a few configs to expose
   previously-dead skip entries (now a zero-match warning) — fix those as config bugs, not by loosening the guard.
3. **Close the `include_sparse` hole (R2).** With the sparse scheme now fuse-only, make the sparse BN-fold
   unconditional inside `build_bevfusion_plan` (drop the `include_sparse` parameter) and delete the separate
   step-[2] fold in `bevfusion_l/quantization/quantize.py:209`. Then PTQ and deploy call the builder with
   identical args and "identical tree by construction" becomes literally true instead of comment-enforced.
4. **Docker e2e (unchanged acceptance gate).** CenterPoint vov99 and BEVFusion mAP must match the
   pre-refactor baseline (0.3228 / 0.3931) exactly — the concrete test for "numerics unchanged" (Non-goal §5).

---

## 4. Goal 3 — Design review: risks, challenged assumptions & tradeoffs

*Stance: evolve the plan, don't redesign it. The verdict in §1.3 is right — the AWML engine (scheme/plan
seam + single `build_<model>_plan`) is the strong part; keep it. Goals 1–2 are the right scope. This
review reads the plan against the current source and flags where the plan's prose is more confident than
the code warrants. Findings are ordered by severity; each is claim → evidence → risk → recommendation.*

**Bottom line (updated after Goal 1 + eSE re-check):** **R1 is corrected and downgraded** — the eSE re-check
confirms the deploy configs already encode the best fully-INT8, reformat-minimizing setup, so the recipes
are canonical architecture (belong in code, always-on) rather than a per-deploy tuning choice; the residual
action is to bake in the single-Q recipe and *delete* the inferior legacy path. **R3 is now the top open
risk** (a silent glob-skip trap). R5 is mostly *resolved* by Goal 1. Everything is folded into the concrete
Goal 2 design (§3.3, §3.5, §3.7, §3.8); none require redesign.

### 4.0 Re-check after Goal 1 — status & where each finding is now handled
The detailed R1–R6 write-ups below are unchanged and still accurate (`recipes/attach.py`, `core/replace.py`,
`quant_model.py` were **not** touched by Goal 1, so every line reference still holds). This table is the
current status overlay:

| # | Status after Goal 1 | New evidence / where it's resolved |
|---|---|---|
| **R1** | **CORRECTED — risk downgraded** | Re-check (forward hooks + modelopt) shows eSE is **not** a tuning choice: the vov99 setup is the *one* canonical fully-INT8, reformat-minimizing design (single-Q-at-input). Bake it in (§3.5 #6); **delete the legacy two-Q path**. The only real per-config opt-out is BEVFusion `quant_add=False` → `disable_recipes=["add"]` (§3.7). |
| **R2** | **CONFIRMED, still open** | `include_sparse` still varies per caller (`quantize.py:221`=False, `model_loader.py:242`=True). Cheaper to fix now (sparse is fuse-only). Fix: §3.8(3). |
| **R3** | **CONFIRMED, was understated** | Silent trap: the draft's `quant_voxel_encoder=False → keep_fp16=["pts_voxel_encoder"]` does **not** skip the tower under exact-match/`fnmatch`, and the zero-match guard won't catch it. Fix: subtree skip (§3.3a, §3.8(1)) + parity oracle (§3.8(2)). |
| **R4** | **CONFIRMED** | `recipes/attach.py` unchanged & still purely class-gated. "Detected backbone" language removed from §3.1 — no machinery needed. |
| **R5** | **MOSTLY RESOLVED by Goal 1** | Stale `SpconvInt8Scheme` docstring fixed (`centerpoint/quantization/schemes.py:10`); `--sparse-int8-only` gone. Remaining: `dense_quant_enabled()` now *has* a caller (inverts the note) → migrate not delete (§3.6); one cosmetic doc-path comment in `profile_sparse_encoder.py`. |
| **R6** | **CONFIRMED** | Sequencing holds; the parity oracle is now explicitly "write first" — §3.8(2). |

### R1 [CORRECTED after eSE re-check, 2026-07-16] — "recipes are a per-deploy tuning choice" was mostly **wrong**: the eSE/OSA/maxpool set is *one* canonical fully-INT8 design
- **My original claim (now retracted):** that Concern B leaks per-deploy tuning, because eSE "has two
  competing strategies" and always-on might pick the wrong one and move numerics. Reading the *forward
  hooks* (not just the attach functions) and cross-checking modelopt shows this was a misread.
- **What the code actually does (verified):** with `pool_input_quantizer` present — the vov99 setup —
  `eSEModuleForwardHook` (`recipes/forward_hooks.py:272-282`) runs the **single-Q-at-input** path: *one*
  FP32→INT8 reformat at the eSE input, then `qx` fans out to **both** the GAP branch and the bypass, so
  `qx * gate` has both `Mul` operands already in INT8. `attach_ese_mul_identity_quantizer` deliberately
  adds **no** second Q when pool_input is present (`attach.py:168`). The `mul_identity`-only branch
  (`forward_hooks.py:283-292`) is a **legacy two-Q fallback** — a *second* reformat, pool branch left
  un-quantized — and **no shipping config uses it**. The same one-shared-Q-fan-out principle drives
  `OSAModuleForwardHook` (single Q at block input → 3 consumers, `forward_hooks.py:196-251`),
  `QuantBeforePool` (Q before MaxPool), and the residual-add hooks. Together they are precisely the
  "eliminate most reformats, keep the graph fully INT8" design vov99 was tuned to — so what the deploy
  config enables *is* the best fully-INT8 setup, exactly as you said.
- **Reference check (modelopt = correct reference):** `INT8_DEFAULT_CFG` (`config.py:163`) is "disable
  everything, then per-tensor `*input_quantizer` (`axis=None`) + per-channel `*weight_quantizer` on GEMM
  ops." AWML's single-Q-at-eSE-input is the faithful realization of that principle — one per-tensor input
  quantizer, **shared** across the fan-out — and goes one step beyond modelopt's generic default by also
  forcing the element-wise `Mul` to INT8. CUDA-BEVFusion's shared `_input_quantizer` for Add/Concat (§1.1)
  is the identical idea. So the recipe **is** architecture+deployment knowledge with a single best form: it
  belongs in code, always-on, exactly as §3 proposes. This **strengthens** the original design, not weakens it.
- **Corrected risk (downgraded from "highest"):** there is no eSE "fork" to get wrong, provided the
  always-on default is the single-Q path (it reproduces vov99 by construction). The *one* genuine
  per-config exception is unrelated to eSE — BEVFusion ships `quant_add=False`, so always-on `add` would
  change **its** tree; that (not ablation) is the real reason `disable_recipes` must exist.
- **Changes this produces (the "parts to change"):** (a) bake the single-Q-at-input eSE recipe as the
  always-on default — §3.5 #6. (b) **Delete the legacy two-Q eSE path** (`mul_identity`-only branch of
  `eSEModuleForwardHook` + the no-`pool_input` branch of `attach_ese_mul_identity_quantizer`): dead,
  strictly more reformats, and the *only* thing that made the recipe look like a tuning choice — removing
  it removes the ambiguity (Simplicity guardrails). (c) `disable_recipes=["add"]` for BEVFusion (§3.7).
  (d) Keep the numeric characterization test (same calib → identical `_amax` + identical Q/DQ node count)
  as the guard that "always-on == vov99-today." **R3 (not R1) is now the top open risk.**

### R2 — The "identical tree by construction" invariant has a hole: `include_sparse` is a per-caller argument
- **Claim challenged:** §0 / plan.py docstrings — "PTQ and deploy call the same `build_<model>_plan` → identical
  tree **by construction**, not by a keep-in-sync comment."
- **Evidence:** for BEVFusion the callers pass **different** arguments: PTQ producer →
  `build_bevfusion_plan(config, include_sparse=False)` (`quantization/quantize.py:221`); deploy loader →
  `include_sparse=True` (`io/model_loader.py:242`) and `False` (`:270`). The trees line up only because the
  sparse BN-fold is done separately on the PTQ side, reconciled by a **comment** (`quantize.py:216`). So the
  guarantee is really "identical *given identical arguments*," and `include_sparse` is exactly the kind of
  divergent-argument seam the invariant was meant to abolish — it moved from "two functions" to "one function,
  two argument values + a comment."
- **Risk:** low today, but it is a latent trap: any future caller that gets `include_sparse` wrong produces a
  silently mismatched tree, and the invariant's own docstring says that can't happen.
- **Recommendation:** fold this into Goal 1. §2.1 already says "reconsider whether `include_sparse` is still
  needed." Once the sparse scheme is fuse-only (`SpconvBnFuseScheme`), prefer making the sparse BN-fold
  **unconditional inside the plan** (or a property of `config.fuse_bn` alone) so there are **zero** divergent
  arguments and the invariant becomes literally true. If it must stay, rename the guarantee honestly:
  "identical given identical config *and* `include_sparse`."

### R3 — Parity is real, but the mechanism is subtler than §6 states, and two edge cases will bite
- **Claim challenged:** §3.3 — expand `keep_fp16` globs against `named_modules()` and pass the set "exactly as
  `sensitive_layers` does today."
- **Evidence:** today's skip is **exact full-name match on a container node → skip whole subtree**
  (`core/replace.py:213`, `if full_name in skip_names`) — *not* prefix/`startswith`, *not* `fnmatch`. Two
  consequences the plan glosses:
  1. **fnmatch ≠ prefix.** `fnmatch("pts_backbone.blocks.0", "pts_backbone.blocks")` is `False`. A bare
     `keep_fp16=["pts_backbone.stem"]` only "works" because a module node named *exactly* `pts_backbone.stem`
     exists, and the existing container-skip then drops its subtree. Users importing the old "sensitive_layers =
     prefix" mental model will write patterns that match nothing.
  2. **`resolved_sensitive_layers()` can emit names for modules that don't exist** (`config/schema.py:262` builds
     strings like `f"pts_backbone.{name}"` with no existence check). Those were harmless silent no-ops. The new
     zero-match warning (§3.4) will **fire** on them, and the parity set will differ — not a regression, but a
     pre-existing dead-config smell the guard is now exposing.
- **Risk:** medium. If glob semantics don't reproduce `resolved_sensitive_layers()` exactly, PTQ and deploy skip
  sets diverge → the trees diverge → state_dict load breaks or numerics shift.
- **Recommendation:** (a) Make the parity test (§6 Risks) an **oracle test**: for every existing `deploy_config_*.py`,
  assert `expand(keep_fp16) == resolved_sensitive_layers()` *and* that every no-wildcard entry resolves to a real
  module node. (b) **Decision (settled): `keep_fp16` is a subtree match, resolved in the expansion helper, not in
  the engine** — a bare name keeps that module and all descendants; globs still work. Chosen over editing
  `replace.py` because the helper must expand globs anyway (zero-match warning), so subtree behavior comes for
  free while the shared engine primitive stays frozen (§3.3a, §3.8(1)). (c) Expect a handful of existing configs to
  trip the zero-match warning on dead entries — clean those as config bugs, don't "fix" them by loosening the guard.

### R4 — "Auto-attach keyed off the detected backbone" is unnecessary machinery — the class-gate *is* the detection
- **Claim challenged:** §3.3 — "`build_<model>_plan` … always attaches the architecture recipes, **keyed off the
  detected backbone**."
- **Evidence:** every attach function already iterates `named_modules()` and matches by **class name**
  (`attach.py:92`, `:165`, `:211`, `:242`); each is a no-op when the class is absent, and `_install_forward_hook`
  is idempotent (`attach.py:63`). There is nothing to "detect" — attaching unconditionally *is* the detection.
  Also note the recipe set is **per-scheme, not global**: `CenterPointDenseScheme` runs the full eSE/maxpool/add
  set via `quant_model` (`centerpoint/quantization/schemes.py:75`), while BEVFusion's `DenseQDQScheme` attaches
  only conv + add (`schemes/dense_qdq.py:76`).
- **Risk:** low, but "detect the backbone" invites someone to build a backbone-classification registry (a god
  object) that the idempotent class-gate makes pointless — added complexity for zero benefit.
- **Recommendation:** delete the "detected backbone" framing. Say plainly: *each project scheme always attaches
  its architecture's recipe set; the class-name gate makes every attach a no-op where the module is absent.* One
  caveat to verify before declaring "always-on safe": the ConvNeXt/voxel-encoder **Linear** path — confirm it is
  as idempotent and class-gated as the conv/add/eSE/maxpool paths (I did not find an `attach_linear_*` in
  `recipes/attach.py`; it lives in the `quant_model` / `quant_linear_module` path — verify before relying on it).

### R5 — Dead code and doc-rot that Goal 1's grep must also catch
- `centerpoint/quantization/schemes.py:10` still documents the deleted `SpconvInt8Scheme` — stale after the
  `SpconvBnFuseScheme` rename (§2.1). Fix as part of Goal 1.
- `QuantizationConfig.with_overrides()` (`config/schema.py:290`) advertises a `--sparse-int8-only` CLI override;
  that flag and its call path die with Goal 1 — grep for the CLI arg, not just the field.
- §3.6 deletes `dense_quant_enabled()`; confirm zero callers before deleting (grep). Same for the
  `_VOVNET_STAGE_NAMES` expansion.

### R6 — Sequencing: land the parity oracle *before* Goal 2, not alongside it
- §7's order (Goal 1 → Goal 2) is correct. One refinement: write the `resolved_sensitive_layers()`
  characterization test **first**, against the *current* configs, so it is a known-good oracle. Then Goal 2's
  glob expansion is validated against a fixed target instead of being co-developed with the thing that is
  supposed to check it. Cheapest possible insurance against R3.

### Simplicity guardrails — what to deliberately **not** do
- **Keep** the decision (§3.4(4)) to ship binary int8/fp16 `keep_fp16`, not a `{pattern: precision}` DSL. Don't
  pre-build fp8/per-bit-width.
- **Keep** the decision to not adopt modelopt as a dependency (§1.3). Borrow the *shape* (default + exceptions),
  not the library.
- The **single highest-ROI addition** is the zero-match warning (§3.4) — it fixes the one genuine weakness shared
  by both reference designs. Ship it; resist adding anything more to the config surface.
- **Delete, don't keep, the legacy two-Q eSE path** (`mul_identity`-only branch of `eSEModuleForwardHook`,
  `forward_hooks.py:283-292`, + the no-`pool_input` branch of `attach_ese_mul_identity_quantizer`). No shipping
  config uses it, it is strictly more reformats than the single-Q path, and it is the only reason the eSE recipe
  ever *looked* like a per-deploy choice (R1). Removing it makes "recipes = canonical architecture" true in code,
  not just in prose — the highest-ROI *deletion*.

### 4.9 Goal-3 close-out — implementation status (2026-07-16)
Every actionable recommendation is now landed in code (host-verified: ast + pyflakes clean, both oracle/
characterization test files written; Docker e2e pending). Per finding:

| # | Recommendation | Status |
|---|---|---|
| **R1** | single-Q eSE default; delete legacy two-Q; `disable_recipes=["add"]` for BEVFusion; characterization test | **DONE** — two-Q path deleted (`forward_hooks.py`, `attach.py`); `disable_recipes` wired; `deployment/tests/test_ese_single_q_recipe.py` added (structural). Numeric half = Docker mAP. |
| **R2** | make sparse BN-fold unconditional; drop `include_sparse` | **DONE** — `build_bevfusion_plan(config)` folds sparse on `fuse_bn`; both callers identical. |
| **R3** | subtree skip in the `keep_fp16` helper (no core change) + parity oracle test | **DONE** — `expand_keep_fp16` (`core/replace.py`) + `deployment/tests/test_quant_config_migration.py` (green over 7 configs). Zero-match-on-real-model check = Docker. |
| **R4** | delete "detected backbone" framing; recipes always-on + class-gated | **DONE** — schemes attach unconditionally; no detection registry. |
| **R5** | dead-code/doc-rot cleanup | **DONE** — schema helpers/fields removed & callers migrated; canonical `deployment/quantization/README.md` + `docs/quantization_pipeline.md` + `profile_sparse_encoder.py` comment updated. |
| **R6** | parity oracle as a golden test | **DONE** — the migration oracle is committed as a test. |

**Left for the user (not code):** the ~29 `projects/bevfusion_l/docs/*int8*` historical notes still describe
the removed spconv-INT8 subsystem — per §2.1 these are *your* call (archive vs delete; "not deleting history
unasked"), so they are untouched. **Docker-pending** (host can't run torch): e2e mAP == 0.3228 / 0.3931, plus
the two behavior caveats in the top-of-file status (BEVFusion `add`; always-on backbone-Linear).

---

## 5. Goal 4 — Framework cleanliness pass (post-Goal-1–3 re-review, 2026-07-16)

### 5.0 Charter, method, verdict

**Optimize only for:** simplicity · cognitive load · maintainability · locality of change ·
information hiding · clear ownership.
**Never optimize for:** SOLID compliance · design patterns · reducing LOC · maximizing abstraction.
**Hard constraints:** zero numeric change (Docker mAP 0.3228 / 0.3931 stays the acceptance gate);
the sacred invariant (§0 — PTQ producer, deploy loader, QAT hook all call the same
`build_<model>_plan(config).prepare(model)`) is preserved by every item below; deletions, merges,
and inlining beat any new interface.

**Method:** four independent deep reviews of the current tree (quantization engine; per-project
quantization glue; core deployment framework; deploy-config surface), each against the lens above,
cross-checked against Graphify (2026-07-03 report — pre-quantization, so used for the core skeleton
only) and first-hand reads of `config/schema.py` and `quantization/core/replace.py`.

**Verdict:** the *centers* are healthy — the scheme/plan seam, `expand_keep_fp16` (engine holds the
mechanism, config holds the values), `tensorrt_runner`, `OutputComparator`, the verify/eval split,
the `plan.py` chokepoints, and `quant_model.py` all have exactly one owner and should not be
touched. The costs are concentrated at the **edges**: (a) the two PTQ producers and two deploy
loaders have drifted into near-copies with one already-diverged helper and one bespoke re-implementation
that endangers the sacred invariant; (b) the typed `QuantizationConfig` is parsed three times and
then bypassed via `.raw`, so the schema is decorative on the deploy path; (c) the 6 CenterPoint INT8
configs are ~85% verbatim clones and have already started to rot (wrong comments, dead keys, stale
advice); (d) Goals 1–3's deletions left residue — dead public exports, a lying function name, and
docs describing removed knobs. Everything below is a fix to one of those four cost centers.

### 5.1 Work package 4A — one owner for the quant-config path *(correctness-adjacent; do first)*

**4A.1 Reject unknown `quantization` keys.** `QuantizationConfig.from_dict`
(`config/schema.py:217-234`) reads a fixed key set with `.get()` and silently drops the rest, and the
loaders read extra keys (e.g. `calib_cache_path` at `centerpoint/io/model_loader.py:185`) off `raw`.
A typo — `keep_fp16s=[...]`, `disable_recipe=[...]` — parses fine and degrades to "quantize
everything INT8," detectable only by the Docker mAP gate. Fix: after building, compute
`set(raw) - KNOWN_KEYS` and **raise** on non-empty (~4 lines, one place). This is the config-key
sibling of the §3.4 zero-match warning and closes the last silent-misconfig class.
*(Lens: cognitive-load footgun.)*

**4A.2 Parse `QuantizationConfig` once; stop smearing ownership across three files.** Today
`config/base.py:64` builds the typed object, the runners throw it away and forward the raw dict
(`centerpoint/runner.py:97`, `bevfusion_l/runner.py:131` — `dict(...quantization_config.raw)`), and
both loaders **re-parse** it (`centerpoint/io/model_loader.py:172`, `bevfusion_l/io/model_loader.py:234`);
`load_quantization_config` (`schema.py:243`) is a third constructor for the producer CLIs. The typed
fields from the first parse are read by nobody. Fix: runners pass the typed
`self.config.quantization_config` straight through; delete the `from_dict` re-parse in both loaders.
`raw` stays available on the object for the few verbatim-dict consumers (shrink it later if those
disappear). One parse, one owner (`base.py`). *(Lens: clear ownership, information hiding — the
schema docstring at `schema.py:186-189` currently *documents* the bypass as a feature.)*

**4A.3 One spelling of "disable the FP16 layers."** The canonical loop —
`expand_keep_fp16(...)` → `disable_quantization(layer).apply()` — appears 3× (`centerpoint/quantization/quantize.py:296-302`,
`qat_hook.py:172-177`, `bevfusion_l/quantization/quantize.py:129-139`), and the CenterPoint **deploy
loader** re-implements it a 4th way with `name.startswith(...)` matching + raw `module.disable()`
(`centerpoint/io/model_loader.py:246-272`) — despite already computing the correct set via
`expand_keep_fp16` at line 182. Divergent match semantics between producer and deploy side is
exactly the drift the sacred invariant exists to prevent. Fix: one shared 4-line helper (natural
home: next to `expand_keep_fp16`), all four sites call it; delete
`_disable_quantization_for_sensitive_layers`. *(Lens: clear ownership + the invariant.)*

### 5.2 Work package 4B — merge the drifted copies (producers & loaders)

**4B.1 Hoist the three duplicated `TensorQuantizer` helpers.** `_import_tensor_quantizer`,
`_move_quantizer_amax_to_device`, `setup_quantization_for_onnx_export` exist near-verbatim in both
loaders (`centerpoint/io/model_loader.py:34-44, 228-243, 315-327` vs
`bevfusion_l/io/model_loader.py:103-131, 276-291, 323-331`) — and have **already diverged once**:
only the BEVFusion `_import_tensor_quantizer` carries the `restore_deployment_logging()` absl-hijack
fix in a `finally`, so CenterPoint is silently missing a known fix. Move all three into
`deployment/quantization/` (both loaders already import from it), consolidating on the BEVFusion
superset. CenterPoint-only `_validate_quantizer_amax` stays local. *(Lens: locality of change —
proven drift under maintenance.)*

**4B.2 Shared PTQ-producer helpers.** The two `quantize.py` producers duplicate ~60–80 lines:
dataloader seed block **byte-identical** (`centerpoint …/quantize.py:257-266` vs
`bevfusion_l …/quantize.py:214-223`), shuffle block byte-identical (CP:268-272 / BF:225-228),
ceil-batches math (CP:201 / BF:161), checkpoint + `.calib` save (CP:330-336 / BF:257-269), absl
logging init (CP:171-178 / BF:289-294). Move the verbatim blocks to one home —
`deployment/quantization/producer.py`: `init_quant_logging()`,
`build_calib_dataloader(cfg, batch_size, seed, shuffle)`, `save_ptq_checkpoint(...)` — and each
`quantize.py` keeps only its model-specific bits (`run_qat` is genuinely CenterPoint-only; leave it).
This is moving copied code to one home, not adding abstraction — imitate the
`spconv_bn_fusion.py` re-export pattern, which is the house style for exactly this.
Also: the one-off diagnostic dumps (CP residual-status dump `quantize.py:308-327`, BF
`_print_ptq_save_check` `quantize.py:142-150`) — delete or gate behind `--debug`.
*(Lens: locality of change, simplicity.)*

**4B.3 Resolve the loader post-load asymmetry (owner decision).** CenterPoint-only:
`_validate_quantizer_amax` (`model_loader.py:275-312`). BEVFusion-only:
`_set_tensor_quantizers_inference_mode` (`model_loader.py:294-320`), whose docstring says without it
mAP collapses to ~0. Both loaders build the identical tree by construction, so this asymmetry is
either a silently missing step on one side or a model-specific fact that must say *why* in place.
Decide in Docker (does CenterPoint need inference-mode? does BEVFusion want amax validation?), then
either share both via 4B.1's module or write the one-line reason each is per-project.
*(Lens: cognitive load — an unexplained asymmetry at the framework's most safety-critical seam.)*

**4B.4 Merge the eSE attach pair; retire the lying name.** `attach_ese_pool_input_quantizer` +
`attach_ese_mul_identity_quantizer` (`recipes/attach.py:151-205`) are one concept split into two
order-dependent functions ("Assumes … ran first", `attach.py:171`; "Order matters",
`quant_model.py:73-75`) — and after the Goal-3 two-Q deletion the second one no longer touches any
`mul_identity` quantizer at all (it attaches `mul_gate_quantizer`, `attach.py:175`). Merge into a
single `attach_ese_quantizers(model)` that adds `pool_input_quantizer` + `mul_gate_quantizer` and
installs the hook once; update the two call sites (`quant_model.py:74-75`,
`tests/test_ese_single_q_recipe.py:79-80`). *(Lens: hidden ordering contract + a name that lies.)*

### 5.3 Work package 4C — deploy-config de-dup + rot fix

**4C.1 One `_base_` file for the CenterPoint INT8 configs.** The 6 INT8 configs are ~85% verbatim
copies: the `components` IO map + `dynamic_axes` block and the `verification.scenarios` block are
byte-identical across all 6 + `deploy_config.py`; only checkpoint, `keep_fp16`/`disable_recipes`,
TRT profile shapes, opset, work_dir, and eval knobs vary. Renaming one IO tensor is a **7-file
edit** today. Fix with the mechanism the loader already supports (MMEngine `_base_` — currently
used by zero configs): one `_deploy_config_int8_base.py` holding `components`, `onnx_config`,
`evaluation`, `verification`, `devices`; each variant shrinks to `_base_ = [...]` + its ~15 true
diffs. Adding a variant stays one file; changing a shared concern becomes one edit. This is the
idiomatic one-file story, not a new pattern. *(Lens: locality of change.)*

**4C.2 Fix the clone rot in the same stroke** (all direct symptoms of 4C.1):
- `deploy_config_int8_vov57.py:69-71` and `vov99.py:69-71` carry a copy-pasted **ConvNeXt** comment
  ("10 input channels … 1216 grid") contradicting their own values (11 ch, 1020 grid). Delete/correct.
- `vov57.py:21-22` / `vov99.py:21-22` still advise tuning via `quant_head=False` /
  `sensitive_layers` — the exact flags Goal 2 deleted (and `test_no_legacy_flags_remain` forbids) —
  and reference a non-existent `README_PTQ_ACCURACY_VOV99.md` (real doc:
  `quantization/docs/ptq_accuracy_vov99.md`). Rewrite to "add the module to `keep_fp16`" + fix path.
- Every config passes `devices=devices` inside `verification=dict(...)`, but
  `VerificationConfig.from_dict` (`schema.py:550-574`) never reads it — a dead key that looks
  load-bearing. Drop the line everywhere (scenarios already carry per-scenario devices).

**4C.3 Test hardening.**
- `test_centerpoint_configs.py` parse-checks every CenterPoint config; **no equivalent exists for
  BEVFusion** — a structural typo there is invisible until Docker. Parametrize over both config dirs.
- `test_quant_config_migration.py:30-91` pins each config's `keep_fp16` to the value derived from
  the *old* flags — correct as a one-time migration oracle, but any legitimate future `keep_fp16`
  tuning will fail it. Add a header comment: "migration oracle — retire after the Goal-2 Docker
  verify lands," so it is not mistaken for a live invariant.

### 5.4 Work package 4D — post-deletion residue sweep (deletions & docs)

**4D.1 Trim the advertised API to the real one.** Zero external callers (grep-verified) for
`transfer_to_quantization`, `fuse_conv_bn`, `QuantConvTranspose2d`, `QuantLinear` in
`quantization/__init__.py:37-59` — drop them from the root exports (they stay importable at their
definition sites for the internal users). The entire re-export block in `recipes/__init__.py:10-36`
is dead surface — every real importer uses the concrete submodule paths (`quant_model.py:16`,
`test_ese_single_q_recipe.py:19`); delete it, keep the docstring. The remaining root API is what is
actually used: `CalibrationManager`, `expand_keep_fp16`, `quant_conv_module` / `quant_linear_module`,
`fuse_model_bn`, `disable_quantization`, `print_quantizer_status`, the three schemes.
*(Lens: information hiding — a reader must be able to tell load-bearing from decorative.)*

**4D.2 Small engine deletions** (each independently landable):
- `attach_quant_add(model, target_class_names=None)` (`attach.py:71`): both callers pass only
  `model`; drop the param and its two-branch matcher.
- `fuse_model_bn(model, inplace=True)` (`fusion.py:246`): no caller passes `False`; delete the param
  and the dead `deepcopy` branch (`fusion.py:261-264`). *(Contrast: `calibrate(method=...)` is
  genuinely varied — leave it.)*
- Descriptor population is stamped in 3–4 places (`quant_conv.py:80-83,139-142`,
  `quant_linear.py:45-48`, `replace.py:85-108`, `attach.py:46-52`): give it one home (module
  `__init__` calls `ensure_quant_descriptors_initialized()`), and delete the unreachable
  per-channel ConvTranspose guard (`replace.py:95-103`) — `descriptors.py`, the single source,
  can never produce what it guards against.
- Inline `_get_bn_num_features` / `_get_conv_out_channels` into their only caller
  (`fusion.py:205-206`).
- Optional, lowest priority: collapse `sparse/` (one 49-line function + `__init__`) into a single
  `sparse.py`, keeping the import path.

**4D.3 Decide-and-document (needs Docker, not just edits):**
- Two clone mechanisms for one concept: Conv uses the clean rebuild path
  (`replace.py:113` `_rebuild_conv2d_as_quant` — `__init__` + weight copy, hook-safe), Linear uses
  the `vars()` transplant (`replace.py:193` `transfer_to_quantization`). Either move Linear onto a
  `_rebuild_linear_as_quant` for symmetry, or write down in place why the transplant is safe for
  Linear specifically. One mechanism, or one sentence — not two silent ones.
- `_skip_fake_quant_for_export_trace` guards Conv/ConvTranspose forwards
  (`quant_conv.py:99-112,158-169`) but not `QuantLinear.forward` (`quant_linear.py:64-73`). Verify
  in Docker whether Linear export needs it; hoist a shared guard or add the one-line reason.

**4D.4 Doc residue.**
- `quantization/__init__.py:13` still labels `sparse` "the spconv INT8 subsystem" — it is FP16
  BN-fold only. One-line fix.
- `quantization/docs/ese_int8_changes.md:78-96,115,127` documents the deleted two-Q fallback and
  the removed `quant_ese_*` keys. Rewrite to the `disable_recipes` surface or delete (the single-Q
  rationale already lives in `forward_hooks.py:254-269`).
- The ~29 `projects/bevfusion_l/docs/*int8*` notes: unchanged policy — the user's call (§4.9).
- `docs/quantization_pipeline.md` (repo root) and `quantization/README.md` were re-verified: both
  match the current surface; no action.

**4D.5 Core-framework nits (do only these; see 5.6 for what we deliberately skip):**
- Inline the pure pass-through pair in `ExportOrchestrator`: `_run_onnx_export` → `_export_onnx`
  (`runtime/export_orchestrator.py:164-177` / `205-233`) and the TRT mirror (`190-203` / `235-270`).
  The export path is 7 files / ~11 hops end-to-end; the tiers are defensible, this hop is not.

### 5.5 Sequencing & verification gates

Order: **4A → 4B → 4C → 4D** (4A is correctness-adjacent and makes 4B's merges safer; 4C and 4D are
independent of each other and individually landable). Every step is behavior-preserving by intent.

Gates (same regime as Goals 1–2):
- Host: `ast.parse` + `pyflakes` on every touched file; `pytest deployment/tests/` green — including
  the parity oracle (until retired per 4C.3), the eSE characterization test (updated call sites),
  and the new BEVFusion config parse test.
- Grep gates: no remaining `_disable_quantization_for_sensitive_layers`, no
  `attach_ese_mul_identity_quantizer`, no `from deployment.quantization import transfer_to_quantization`.
- Docker e2e: CenterPoint vov99 + BEVFusion mAP **identical** to baseline (0.3228 / 0.3931) — this
  goal changes zero numerics, so *identical* (not "within tolerance") stays the bar. 4B.3 and 4D.3
  explicitly park their behavior questions here.

### 5.6 Deliberately not doing (guardrails against churn)

- **Keep** the scheme/plan seam exactly as is — it is a real deep module backing the invariant
  (3 concrete schemes, one-method interface). Do **not** merge `CenterPointDenseScheme` into
  `DenseQDQScheme`: CenterPoint genuinely needs Conv+Linear+eSE+maxpool, BEVFusion conv+add only.
- **Keep** `expand_keep_fp16` in `core/replace.py` — re-review confirmed ownership is clean
  (engine = mechanism, config = values; all five consumers call the one function).
- **Keep** the orchestrator → pipeline → exporter tiers, the verification/evaluation split (two
  genuinely different loops sharing one `BackendExecutor`), and the two orchestrators unmerged
  (similar shape, different bodies — merging adds branching).
- **Do not** replace `ProjectAdapter`/`ProjectRegistry` (a dict wrapper, but stable and small —
  flagged as evidence, not scheduled) and **do not** build a shared TensorRT-pipeline base class
  now (~30 lines of near-duplicate per project; absorb only if those files are touched anyway).
- **Do not** add any new config surface, DSL, back-compat shim, or "framework utils" grab-bag
  module. Every 4B move goes to a named, single-purpose home.
- Healthy code named healthy stays untouched: `tensorrt_runner.py`, `OutputComparator`,
  `detection3d_entrypoint.py`'s inject-the-three-things pattern, `availability.py`,
  `descriptors.py`, `plan.py` (both), `quant_model.py`, `qat_hook.py`, the exporters' atomic
  staging/publish writes, `primitives/artifacts.py`.

### 5.7 Goal-4 close-out — implementation status (2026-07-16)

All four work packages are landed in the working tree. Per item:

| Item | Status / notes |
|---|---|
| **4A.1** unknown-key guard | **DONE** — `QuantizationConfig.KNOWN_KEYS` + raise in `from_dict`; all 7 configs' literal blocks verified against the guard. |
| **4A.2** single parse | **DONE+** — runners pass the typed object; both loader re-parses deleted; went further than planned: the `raw` escape hatch had **zero** consumers left, so the field itself is deleted (the typed schema is now the only contract). |
| **4A.3** one disable spelling | **DONE** — `disable_quantizers_in` (core/utils.py, next to `disable_quantization`); all 4 sites converted; the bespoke `startswith` loader variant deleted. |
| **4B.1** loader helper hoist | **DONE** — `get_tensor_quantizer_cls` (carries the absl-restore fix CenterPoint was missing), `move_quantizer_amax_to_device`, `setup_quantization_for_onnx_export` in core/utils.py; both loaders + BEVFusion runner import them. |
| **4B.2** producer helpers | **DONE** — `deployment/quantization/producer.py` (`init_quant_logging`, `build_calib_dataloader`, `save_ptq_checkpoint`); both `quantize.py` producers converted; both one-off debug dumps deleted. Two deliberate error-path improvements (not numeric changes): a failed `.calib` save now raises instead of warning (BEVFusion), and CenterPoint's missing-layer disable warning now comes from the shared helper. |
| **4B.3** loader asymmetry | **PARKED for Docker** (as specced) — both functions now document the asymmetry + TODO(Docker) pointer here. |
| **4B.4** eSE attach merge | **DONE** — single `attach_ese_quantizers`; ordering contract and `mul_identity` name gone; `quant_model.py` + characterization test updated. |
| **4C.1** config `_base_` | **DONE** — `_deploy_config_int8_base.py` (underscore keeps it out of the `deploy_config*` glob); 6 variants now hold only checkpoint / quantization / export dirs / profile shapes / opset / eval overrides. **Golden check green**: a pure-python re-implementation of MMEngine's recursive merge proves base+variant == pre-refactor file for every section of all 6 configs (quantization section excluded — its diff vs git HEAD is exactly the already-tested Goal-2 flag migration). |
| **4C.2** rot fixes | **DONE** — wrong ConvNeXt comment now lives only in the ConvNeXt config (with correct 10ch/1216 values); vov advice rewritten to `keep_fp16` + real doc path (`quantization/docs/ptq_accuracy_vov99.md`); dead `verification.devices` key dropped (now one place — the base); second's usage docstring path corrected. |
| **4C.3** test hardening | **DONE** — `tests/test_bevfusion_configs.py` (layout-aware: split vs merged), oracle shelf-life note added to `test_quant_config_migration.py`. |
| **4D.1** export trim | **DONE** — root `__init__` drops `transfer_to_quantization` / `fuse_conv_bn` / `QuantConvTranspose2d` / `QuantLinear` (grep: zero external users) and adds the new utils; `recipes/__init__.py` re-export block deleted (all importers use submodule paths). |
| **4D.2** small deletions | **DONE** — `attach_quant_add` param dropped (module constant `_RESIDUAL_BLOCK_CLASSES`); `fuse_model_bn` `inplace` + dead deepcopy branch deleted; unreachable per-channel ConvTranspose guard deleted; `_get_conv_out_channels`/`_get_bn_num_features` inlined. **Deviation:** descriptor-populate consolidation was *rejected* — `transfer_to_quantization` builds via `__new__` (bypasses `__init__`), and folding the stamps into one call site needs modules→replace imports (a cycle); instead the two deliberate population points are now documented in `ensure_quant_descriptors_initialized`. `sparse/`→`sparse.py` collapse skipped (optional, lowest value). |
| **4D.3** decide-and-document | **DOCUMENTED, decisions parked for Docker** — clone duality (Conv rebuild vs Linear transplant) explained at `transfer_to_quantization` with the unify-option TODO; QuantLinear's missing trace-skip guard explained at its `forward` (no-op in every shipped path since fb_fake_quant is set first). |
| **4D.4** doc residue | **DONE** — `__init__.py` sparse label fixed; `ese_int8_changes.md` rewritten to the current single-attach/`disable_recipes` surface (kept the still-valid TRT reformat rationale); README layering tree updated (producer.py + utils.py scope). |
| **4D.5** orchestrator inline | **DONE** — `_run_onnx_export`/`_run_tensorrt_export` inlined into `run()`; stale-ONNX guard behavior unchanged; control-flow tests updated to stub `_export_onnx`/`_export_tensorrt`. |

**Host verification:** `ast.parse` clean over all 155 deployment `.py` files; pyflakes clean on every
touched file (remaining hits are pre-existing noqa'd side-effect imports); grep gates pass (no
`_disable_quantization_for_sensitive_layers` / old eSE attach names / `.raw` / loader re-parse /
removed-export imports outside intentional docstring history); golden `_base_`-merge check green
(6/6 configs); migration-oracle logic green (7/7 configs, run with a pytest stub — host has no
pytest/mmengine/torch/Docker socket this session).

**Docker-pending:** (a) e2e mAP **identical** to 0.3228 / 0.3931 (the goal's acceptance gate);
(b) 4B.3 — do both loaders need both post-load steps; (c) 4D.3 — unify the Linear clone path /
trace-skip guard or keep the documented reasons; (d) run `tests/test_bevfusion_configs.py`,
`test_centerpoint_configs.py` (now exercising `_base_` resolution), `test_export_orchestrator.py`,
and `test_ese_single_q_recipe.py` in the runtime image.
