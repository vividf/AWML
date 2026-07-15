# Quantization Refactor Spec (DRAFT — for discussion, no code changes yet)

Status: **Goal 1 implemented** (in the working tree — spconv-INT8 removal done; Docker e2e pending).
**Goal 2 designed, not implemented** — §3 is the concrete design to agree on *before* touching code.
**Goal 3** is the design-review layer, re-checked against the current tree after Goal 1 (§4).

Scope: `deployment/` quantization for CenterPoint and BEVFusion-L.

Three goals:

1. **Remove all spconv INT8 code** — sparse-encoder INT8 gave no meaningful mAP/latency win, so the
   sparse encoder stays FP16 and every INT8-sparse-specific path is deleted.
2. **Redesign the quantization config** — replace the ~13 ad-hoc boolean flags in each
   `deploy_config_*.py` with a small declarative surface, informed by NVIDIA modelopt and
   CUDA-BEVFusion.
3. **Design review & risk hardening** (this discussion) — a Principal-Engineer pass over Goals 1–2:
   challenge the load-bearing assumptions against the *actual* code, surface architectural risks, and
   make the under-specified decisions explicit — **without** redesigning what already works.

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
Fix: `keep_fp16` means "this module **and everything under it**." Resolve each pattern with `fnmatch`
against `named_modules()`, then skip any module whose name equals a resolved name *or* is a descendant of
one (`name == s or name.startswith(s + ".")`). This is the ~2-line core change in §3.8(1); it is
**effect-identical** for every existing sub-tower skip (`pts_backbone.stem` already skips its subtree).

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

### 3.8 The one required core change + verification gates
1. **Prefix/subtree skip (required; ~2 lines × 2 functions).** In `quant_conv_module` and
   `quant_linear_module` (`core/replace.py:213` and `:256`) change the skip test from `full_name in
   skip_names` to also skip descendants:
   `full_name in skip_names or any(full_name == s or full_name.startswith(s + ".") for s in skip_names)`.
   Required so a tower-root `keep_fp16` entry actually skips the tower (§3.3a). This is a semantic
   tightening, **not** an engine rewrite, and is effect-identical for every existing sub-tower skip.
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
  module node. (b) Decide the semantics explicitly: either (i) document that `keep_fp16` matches **module node
  names** and a bare name skips that node's subtree via the existing container-skip (simplest — my
  recommendation), or (ii) treat a metacharacter-free pattern as an explicit subtree prefix. Pick one and write
  it down; don't leave it implicit. (c) Expect a handful of existing configs to trip the zero-match warning on
  dead entries — clean those as config bugs, don't "fix" them by loosening the guard.

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

---

## 5. Non-goals
- Not adopting modelopt as a dependency; not rewriting the Q/DQ engine, schemes, recipes, or
  calibration. Only the **config surface** and the **spconv-INT8 removal** are in scope.
- Not changing dense-INT8 numerics for CenterPoint or BEVFusion (mAP must be unchanged).
- Not touching QAT semantics beyond the config-surface rename.

## 6. Risks
- Wide blast radius on the BEVFusion side for Goal 1 (~20 files). Mitigation: phase it, grep for
  dangling refs, Docker e2e after each phase.
- Relocating `fuse_spconv_bn_in_encoder` must not change `state_dict` keys (FP16 deploy still loads
  BN-folded checkpoints). Verify byte-identical fold.
- `keep_fp16` glob expansion must reproduce today's `resolved_sensitive_layers()` results exactly for
  existing configs, or PTQ/deploy trees diverge. Add a parity test.

## 7. Suggested sequencing
1. Goal 1 first (removal) on its own — smaller behavioral surface, easier to verify mAP unchanged.
2. Then Goal 2 (config redesign) on the reduced surface.
3. Rewrite deploy configs + docs last.
