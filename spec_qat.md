# QAT Completion Spec — CenterPoint + BEVFusion (DRAFT — plan only, no code changes yet)

Status: **DRAFT for design review** (2026-07-16). Nothing below is implemented.
Scope: `deployment/` QAT for CenterPoint and BEVFusion-L, on top of the landed Goals 1–4
(see `spec.md`). References: `Lidar_AI_Solution` (CUDA-BEVFusion / CUDA-CenterPoint /
CUDA-V2XFusion) and `TensorRT-Model-Optimizer` (modelopt ~0.35).

Hard constraints inherited from `spec.md`:
- **The sacred invariant** — PTQ producer, deploy loader, and QAT hook all call the same
  `build_<model>_plan(config).prepare(model)`, so the quantized tree is identical by construction.
  Every item below preserves it; several items exist *only* to close remaining holes in it.
- The quantization **engine** (scheme/plan seam, `core/`, `recipes/`, `expand_keep_fp16`,
  `CalibrationManager`) is kept as-is. QAT is a *consumer* of the engine, not a rewrite of it.
- Sparse encoder stays **FP16** (spconv INT8 was removed in Goal 1 and stays removed).

---

## 0. What "full QAT" means — the definition this spec commits to

Both references converge on the **same** method, so the target is unambiguous:

> **Full QAT = insert fake-quant (per-tensor-histogram activations / per-channel-max-or-mse
> weights) → fold BN → PTQ-calibrate amax → disable the sensitive-layer allowlist → low-LR
> short fine-tune of the *weights* with STE, amax/scales frozen → export the same way as PTQ.**

Evidence that frozen-amax is the production method, not a shortcut:

- **modelopt**: "Typically during QAT, the quantizer states are frozen and the model weights are
  fine-tuned" (`docs/source/guides/_pytorch_quantization.rst:92`); `_amax` is a registered
  **buffer**, not a Parameter (`nn/modules/tensor_quantizer.py:225-230`); `learn_amax` is
  **deprecated and asserted never-True** (`config.py:715-726`). STE gradient: pass-through inside
  the clip range, zero outside (`tensor_quant.py:300-308`).
- **Lidar_AI_Solution**: no `learn_amax` / LSQ / EMA anywhere in the tree; CUDA-V2XFusion's
  from-scratch quantizer stores scale as `requires_grad=False` (`tinyq.py:258`). QAT trains
  weights only, under frozen calibrated scales.

**Consequence:** AWML's existing CenterPoint `QATHook` already implements the *right method*.
This spec is therefore **not** "add real QAT" — it is:

1. **Productionize** CenterPoint QAT (recipe defaults, config surface, checkpoint packaging,
   AMP/DDP correctness, tests) — §4 WP1–WP3.
2. **Port** QAT to BEVFusion without forking the hook — §4 WP4.
3. **Close the loop** deploy-side (`mode="qat"` semantics, QAT deploy configs, e2e gates) — §4 WP3/WP5.

Explicitly **out of scope** (see §7): learnable-amax (LSQ), EMA observers, spconv INT8 QAT,
distillation-assisted QAT (QAD), adopting modelopt as a dependency, multi-GPU QAT in v1.

---

## 1. Current state (verified against the tree, 2026-07-16)

| Piece | CenterPoint | BEVFusion |
|---|---|---|
| QDQ engine / plan / schemes | ✅ shared | ✅ shared |
| PTQ producer CLI + `.calib` cache | ✅ | ✅ |
| Deploy loader for quantized ckpt | ✅ | ✅ (branches on `ptq_checkpoint`) |
| QAT mmengine hook | ✅ `quantization/qat_hook.py` | ❌ |
| QAT producer CLI | ✅ `quantize.py qat` → `run_qat` (`quantize.py:286`) | ❌ (`ptq` only) |
| QAT recipe defaults (lr/epochs) | ❌ bare CLI flags, no reference defaults | ❌ |
| AMP handling during QAT | ❌ AMP stays ON on GPU (`quantize.py:344-351` only downgrades when CUDA absent) | ❌ |
| QAT→deploy checkpoint packaging | ❌ raw mmengine work-dir ckpt (optimizer state, no `.calib`) | ❌ |
| `mode="qat"` deploy config / semantics | ❌ zero configs; loader ignores `mode` | ❌ |
| Resume-QAT support | ❌ broken by construction (see §6 R3) | ❌ |
| Tests | ❌ none | ❌ none |

How the existing CenterPoint QAT works (the part that is already right):

- `run_qat` (`deployment/projects/centerpoint/quantization/quantize.py:286-427`): loads the mm
  train config, injects `QATHook` into `custom_hooks`, overrides `lr` / `max_epochs` /
  `batch_size`, reads `keep_fp16` / `disable_recipes` **from the deploy config**
  (`quantize.py:372-374` — single source of truth for placement), sets `cfg.load_from`, builds an
  mmengine `Runner`, `runner.train()`.
- `QATHook` (`qat_hook.py`): `before_train` → unwrap DDP, build
  `QuantizationConfig(fuse_bn, keep_fp16, disable_recipes)`, `build_centerpoint_plan(config)
  .prepare(model)` (`qat_hook.py:108-113`) — **the same plan as PTQ/deploy**; `before_train_epoch`
  (epoch 0) → `CalibrationManager.calibrate(train_dataloader, method="mse")` or load a `.calib`
  cache (`qat_hook.py:150-163`), then `disable_quantizers_in(expand_keep_fp16(...))`
  (`qat_hook.py:167-172`); `after_train` → log quantizer counts only.
- Ordering is sound for a fresh run: mmengine loads `load_from` (FP32 ckpt, unfused tree) *before*
  `before_train` fires, then the hook fuses BN and inserts Q/DQ, then epoch-0 calibration runs.
- A QAT checkpoint is deploy-loadable **by construction**: `_amax` lives in `state_dict`, and the
  deploy loader rebuilds the identical tree via the same plan then `load_state_dict`.

BEVFusion PTQ facts that constrain the port: `build_bevfusion_plan` =
`DenseQDQScheme(("pts_backbone","pts_neck","bbox_head"))` + `SpconvBnFuseScheme` under `fuse_bn`
(`bevfusion_l/quantization/plan.py:31-58`); PTQ calibration needs a voxel-dtype-normalizing
`forward_fn` (`bevfusion_l/quantization/quantize.py:75`, `_calibrate_dense`); the shipped config
sets `disable_recipes=["add"]` and `keep_fp16=[]`.

---

## 2. Reference methods — what each contributes

### 2.1 Lidar_AI_Solution (CUDA-BEVFusion / CUDA-CenterPoint)

Framing fact: **CUDA-BEVFusion's `qat/` directory contains only the PTQ path** — its README ends
with "work on the QAT part will follow" (`CUDA-BEVFusion/qat/README.md:92-93`). The actual QAT
finetune reference is **CUDA-CenterPoint** (`CUDA-CenterPoint/qat/`). So "reference CUDA-BEVFusion"
for QAT really means: BEVFusion contributes the *placement + export* method, CenterPoint
contributes the *training recipe*.

Pipeline (`CUDA-CenterPoint/qat/README.md:17-31`, `centerpoint_qat_pipline.py`):
fold BN (**before** calibration — stated as a hard requirement) → histogram collect →
`compute_amax(method="mse")` → disable sensitive layers → QAT finetune → export.

Recipe values worth quoting:

| Item | Value | Source |
|---|---|---|
| Calibrator | histogram, amax by MSE | `lean/quantize.py:364` |
| Weight quant | per-channel 8-bit (out-channel axis) | `quantize.py:76-77` |
| Activation quant | per-tensor 8-bit histogram | `quantize.py:76` |
| Calib batches | 300 (BEVFusion) / 400 (CenterPoint), bs=1 | `ptq.py:92`; CP README:85 |
| QAT LR | one-cycle, `lr_max=1e-4`, moms `[0.95,0.85]`, div 10, pct_start 0.4 | CP README:79-81 |
| QAT init | trained FP ckpt + in-process PTQ calibration | CP README:85 |
| amax learned? | **No** — frozen buffers, STE trains weights only | grep: no `learn_amax` in tree |
| AMP during QAT | **forced off** (`fp16_cfg = None`) | `lean/train.py:76` |
| BN | folded before calibration/QAT; QAT runs on folded weights | README:10; `funcs.py:148` |
| FP16-kept layers | spconv `conv_input`; first BEV deconv `neck.deblocks[0][0]`; CP also first block convs + its quant_add | `ptq.py:140-141`; `centerpoint_qat_pipline.py:76-82` |
| QAT gain | PTQ mAP 59.08 → QAT 59.20 (small, real) | CP README table |

AWML already matches the placement/calibration half of this table (histogram + MSE are the engine
defaults in `core/descriptors.py` / `CalibrationManager.calibrate`; sensitive-layer disable =
`keep_fp16`; BN-fold before calibration = `fuse_bn` in the plan). What AWML lacks is the
*training half* (LR schedule, AMP-off, packaging).

### 2.2 TensorRT-Model-Optimizer (modelopt)

- One entry point `mtq.quantize(model, config, forward_loop)`; **QAT = the same quantized model,
  then your normal training loop** (`examples/cnn_qat/torchvision_qat.py:228-261`). No special
  QAT machinery — exactly the mmengine-hook shape AWML already has.
- Recipe guidance: **"QAT for 10% of the original training epochs"**
  (`_pytorch_quantization.rst:117-120`); small LR — SGD ~1e-4 for CNNs, Adam 1e-5 for LLMs
  (`examples/llm_qat/README.md:24`). CNN result: FP32 76.1% → PTQ 75.5% → QAT 75.9%.
- `mto.save`/`mto.restore` persist **weights + quantizer state together** so a quantized ckpt is
  self-describing (`conversion.py:85-134`). AWML's analogue is the plan-rebuild + `state_dict`
  convention (`save_ptq_checkpoint` + `.calib`); the lesson to copy is *the QAT producer must emit
  the same self-consistent artifact PTQ emits*, not a raw training checkpoint.
- Per-layer disable via glob configs (`INT8_DEFAULT_CFG` + `"enable": False` patterns) — AWML's
  `keep_fp16`/`disable_recipes` is already the equivalent surface (Goal 2).
- No spconv support (no plugin exists; custom modules would go through `mtq.register`) — confirms
  keeping the sparse tower out of QAT costs nothing in reference-parity.
- QAD (distillation-assisted QAT) exists for LLMs (`examples/llm_qat/README.md:223-289`) — noted
  as a future option only.

### 2.3 Convergent verdict

The two references disagree on nothing that matters. Both say: frozen scales, STE, weights-only
fine-tune, BN folded first, short/low-LR schedule, calibrate-then-train in one process, export
identical to PTQ. **AWML's engine and QATHook already sit on this method** — the work is
productization and the BEVFusion port, not method change. This also retroactively validates the
Goal-2 decision to keep placement (`keep_fp16`/`disable_recipes`) in the deploy config: the QAT
hook consumes the exact same surface.

---

## 3. Design decisions

### D1 — QAT method: frozen-amax STE fine-tune. No learnable scales. **(settled by §2)**
No alternatives worth carrying: modelopt deprecated `learn_amax`; Lidar_AI_Solution never had it.
Revisit only if a measured mAP gap survives a properly-tuned frozen-amax QAT.

### D2 — Where QAT hyperparameters live

Today: bare CLI flags on `quantize.py qat` (no defaults tied to any recipe). Options:

- **(a) CLI-only, add reference defaults.** Smallest diff; but runs are not reproducible from
  config, and BEVFusion would duplicate the flag set.
- **(b) Typed `qat=dict(...)` sub-block inside the deploy config's `quantization` block**
  *(recommended)*. E.g.:

  ```python
  quantization = dict(
      enabled=True, mode="qat", fuse_bn=True,
      default_precision="int8", keep_fp16=[...], disable_recipes=[...],
      qat=dict(
          train_cfg="projects/CenterPoint/configs/t4dataset/...py",  # base training config
          checkpoint="work_dirs/.../epoch_50.pth",                   # FP init
          epochs=5,                # ~10% of original training (modelopt guidance)
          lr=1e-4,                 # CUDA-CenterPoint lr_max
          calibrate_samples=400,   # CUDA-CenterPoint default (bs=1 batches)
          calib_cache=None,        # optional .calib to skip in-process calibration
      ),
  )
  ```

  Pros: one file reproduces a QAT run; the deploy config is *already* the single source of truth
  for placement (`run_qat` reads `keep_fp16`/`disable_recipes` from it today), so this completes
  an existing pattern rather than inventing one; the `KNOWN_KEYS` typo guard extends naturally
  (nested `QATConfig` dataclass with its own known-keys check). Cons: training knobs in a deploy
  file blurs the deploy/train boundary; mitigated by keeping the block optional and small (CLI
  flags stay as overrides, mirroring `with_overrides`).
- **(c) Separate per-project QAT config file.** Cleanest boundary, but a third config artifact to
  keep in sync with the deploy config's placement block — exactly the drift the sacred invariant
  exists to prevent. Rejected.

Recommendation: **(b)**, parsed into a frozen `QATConfig` dataclass hanging off
`QuantizationConfig` (`deployment/config/schema.py`), `None` when absent. `mode="qat"` does **not**
change deploy-load behavior (see D6); it gates whether the `qat` block may be present.

### D3 — One hook, two projects: how BEVFusion reuses `QATHook`

The hook body is already 95% project-agnostic; the project-specific parts are exactly two:
(1) which plan builder to call, (2) how to push a calibration batch through the model
(BEVFusion needs the voxel-dtype-normalizing `forward_fn` its PTQ path uses,
`bevfusion_l/quantization/quantize.py:75`). Options:

- **(a) Copy the hook into `bevfusion_l/quantization/qat_hook.py`.** Fast, but a 200-line fork of
  training-critical logic — the exact "don't fork the pipeline" failure mode the export seam
  refactor removed. Rejected.
- **(b) Generic hook in the engine + thin per-project registration** *(recommended)*.
  Move the body to `deployment/quantization/qat_hook.py` as `QATHookBase` (not registered).
  Each project registers a ~15-line subclass next to its plan:
  `centerpoint/quantization/qat_hook.py` → `class CenterPointQATHook(QATHookBase)` supplying
  `build_plan = staticmethod(build_centerpoint_plan)` and (BEVFusion) `calib_forward_fn`.
  mmengine's `HOOKS` registry + `custom_imports` wiring stays exactly as today. Pros: single owner
  for the training-loop logic; the subclass surface is precisely the two seams that differ —
  same shape as `SampleExtractor`/`ComponentBuilder` on the export side. Cons: one more base
  class; acceptable because the two variation points are real and already exist.
- **(c) One registered hook taking a dotted-path `plan_builder` string param.** Avoids
  subclasses, but stringly-typed indirection in configs, and `calib_forward_fn` can't be expressed
  as a config value cleanly. Rejected.

Keep the existing name `QATHook` registered by CenterPoint for config back-compat, or rename with
a one-release alias — implementer's choice, note in the PR.

### D4 — QAT checkpoint packaging: the producer must emit the PTQ-shaped artifact

Today `run_qat` ends at "model saved in work_dir" — an mmengine checkpoint carrying optimizer
state, scheduler state, and message hub, *not* the `{"state_dict"}` + `.calib` pair that
`save_ptq_checkpoint` emits and the deploy configs reference. It happens to load (mmengine ckpts
have a `state_dict` key and the loader is `strict=False`), but "happens to load" is not a
contract. Fix (no real alternatives):

- Add `save_qat_checkpoint(model, work_dir_ckpt, output_path)` next to `save_ptq_checkpoint` in
  `deployment/quantization/producer.py`: strip to `{"state_dict"}`, and emit the sibling `.calib`
  via `CalibrationManager.save_calib_cache` (amax is already in the state_dict; the `.calib` keeps
  the PTQ/QAT artifact contract uniform and lets a later QAT run reuse it).
- `run_qat` calls it after `runner.train()`, selecting last (or `save_best` — see §6 R6) epoch.
- Deploy configs then point `checkpoint` at `..._qat.pth` exactly as they point at `_ptq.pth`
  today. This is the AWML analogue of `mto.save`.

### D5 — AMP off during QAT

CenterPoint's train configs use `AmpOptimWrapper`; `run_qat` keeps it on GPU
(`quantize.py:344-351` only downgrades when CUDA is absent). Both references run QAT in fp32
(CUDA: `fp16_cfg = None` forced, `lean/train.py:76`; modelopt examples train un-autocast).
Fake-quant under autocast is a known numerics trap (histogram/amax computed in fp32, forward in
fp16, STE clipping around a half-precision value). Decision: **`run_qat` always downgrades
`AmpOptimWrapper` → `OptimWrapper`** (log it), matching the references. Escape hatch not offered
until someone demonstrates a need — config surface stays minimal.

### D6 — `mode="qat"` semantics at deploy time: provenance only

The loader rebuilds the identical tree and `load_state_dict`s regardless of PTQ/QAT origin — that
is the invariant working as designed, so `mode` must **not** grow a loader branch. Semantics:
`mode` records provenance + gates the `qat` config block; the loader may *log* it. The alternative
(delete `mode` entirely) loses the gate for D2(b) and the self-documentation of configs; keep it.

### D7 — BEVFusion sparse tower during QAT

`SpconvBnFuseScheme` folds sparse BN at `prepare()` — so during QAT the sparse encoder trains
BN-less in fp32, no fake-quant, and deploys FP16 exactly as PTQ does. This matches
CUDA-BEVFusion's own FP16-kept `conv_input`/sparse layers and requires **zero new mechanism**.
One consequence to document: sparse weights *do* continue training during QAT (they are not
frozen). That is what the references do too (CUDA-CenterPoint fine-tunes the whole detector).
Freezing the sparse tower is a possible ablation, not a default.

### D8 — Calibration data inside the training loop

`QATHook` calibrates on `runner.train_dataloader` — i.e. the *augmented* training distribution
(GT-sampling, flips, rotations). CUDA-CenterPoint does the same (calibration runs inside the train
pipeline). PTQ, by contrast, calibrates on the un-augmented val-style loader. Keep the reference
behavior (train loader) as default, but record it in §6 R5 as an open question with a cheap
experiment attached, because it is the one place AWML's PTQ amax and QAT amax can legitimately
differ.

---

## 4. Work packages

Ordering: WP1 → WP2 → WP3 (CenterPoint e2e proof) → WP4 (BEVFusion) → WP5. WP1/WP2 are
prerequisites for both projects; WP3 proves the loop end-to-end on the project that already has
QAT before any porting starts.

### WP1 — Typed QAT config surface (D2)

1. `deployment/config/schema.py`: add frozen `QATConfig` dataclass
   (`train_cfg`, `checkpoint`, `epochs`, `lr`, `calibrate_samples`, `calib_cache`,
   `work_dir=None`) with its own `KNOWN_KEYS` guard; parse from `quantization["qat"]`;
   present only when `mode="qat"` (raise on `qat` block with `mode="ptq"` — config lies are the
   enemy). Defaults: `epochs`/`lr` **required** (no silent recipe), `calibrate_samples=400`.
2. `load_quantization_config` / `with_overrides`: CLI flags override `QATConfig` fields — same
   pattern as the existing overrides.
3. Host-runnable tests: parse round-trip, unknown-key rejection, `mode`/`qat` consistency
   (extend `deployment/tests/` config tests; no torch needed).

### WP2 — Shared QAT hook + producer packaging (D3, D4, D5)

1. Move the hook body to `deployment/quantization/qat_hook.py` (`QATHookBase`): keep the current
   before_train / before_train_epoch / after_train logic verbatim; add the two injection points —
   `build_plan` (abstract) and `calib_forward_fn` (default `None` → `CalibrationManager`'s current
   `model.test_step` path). `CalibrationManager.calibrate` already accepts what it needs; if a
   `forward_fn` parameter is missing there, thread it through (it exists on the BEVFusion PTQ
   path — reuse, don't duplicate).
2. `fuse_bn` flows from the deploy config instead of the hook's hardcoded `freeze_bn=True`
   (today's silent mismatch risk: a `fuse_bn=False` deploy config would still get a fused QAT
   tree).
3. `save_qat_checkpoint` in `deployment/quantization/producer.py` (D4).
4. `run_qat` hardening: AMP always off (D5); refuse `--resume` with a clear message pointing at
   §6 R3 (parked) instead of failing deep inside `load_state_dict`.
5. Keep `after_train` emitting the quantizer-count log; add the packaging call.

### WP3 — CenterPoint QAT completion

1. `centerpoint/quantization/qat_hook.py` shrinks to the registered thin subclass (D3b);
   `run_qat` reads `QATConfig` (WP1) with CLI overrides; recipe defaults documented in `--help`.
2. Add one QAT deploy config, e.g.
   `centerpoint/config/deploy_config_int8_second_qat.py`: `_base_` the existing INT8 base,
   `mode="qat"`, `qat=dict(...)` block, `checkpoint=".../second_qat.pth"`. This is the missing
   "zero configs with mode=qat" artifact.
3. Docker e2e (the acceptance gate, §5): PTQ vs QAT mAP on the same eval pipeline.

### WP4 — BEVFusion QAT port

1. `bevfusion_l/quantization/qat_hook.py`: registered subclass supplying `build_bevfusion_plan` +
   the voxel-dtype `calib_forward_fn` (lifted from `_calibrate_dense`, not copied — hoist the
   `forward_fn` to module level so PTQ and QAT share it).
2. `bevfusion_l/quantization/quantize.py`: add the `qat` subparser + `run_qat`, mirroring
   CenterPoint's but reading the BEVFusion train config
   (`projects/BEVFusion/configs/t4dataset/BEVFusion-L/...`). Shared logic (config surgery, hook
   injection, AMP-off, Runner build, packaging call) goes to a helper in
   `deployment/quantization/producer.py` — the two `run_qat`s should differ only in project
   constants, same policy as the Goal-4 producer dedup.
3. `disable_recipes=["add"]` and `keep_fp16` flow from the shipped deploy config unchanged —
   no BEVFusion-specific placement logic in the hook.
4. Practical constraints to verify in Docker before committing defaults: BEVFusion-L training
   memory/step-time with fake-quant inserted (fake-quant roughly doubles activation memory on
   quantized towers); pick `epochs` default accordingly (start at ~10% of the original schedule).

### WP5 — Tests, gates, docs

1. **Tree-parity test (the invariant test, Docker/pytest):** prepare one model via the PTQ path
   and one via `QATHookBase.before_train`'s path with the same `QuantizationConfig`; assert
   identical `state_dict` key sets and identical enabled-quantizer sets. This turns the sacred
   invariant into an executable check covering the *third* consumer (the hook) — today only
   producer/loader pairs are exercised implicitly.
2. Config tests (host): WP1 item 3; also parse-check the new QAT deploy configs in the existing
   per-project config test files.
3. Docker e2e gates: §5.
4. Docs: extend `deployment/quantization/README.md` §3.4 from "QAT (CenterPoint only)" to both
   projects + the recipe table (§2.1/§2.2 values and their sources); update the
   `docs/quantization_pipeline.md` mermaid to show the shared hook and the BEVFusion branch;
   document D7's "sparse trains but stays FP16" consequence.

---

## 5. Verification gates

Host (no torch — same regime as spec.md):
- `ast.parse` + `pyflakes` on every touched file; config parse tests; grep gates
  (no hook fork: `class QATHook` defined once in the engine; no duplicated `forward_fn`).

Docker (the real gates):
1. **Tree parity** test green (WP5.1).
2. **CenterPoint**: `quantize.py qat` on the release SECOND config → packaged `_qat.pth` →
   deploy + eval through the unchanged pipeline. Gate: **QAT INT8 mAP ≥ PTQ INT8 mAP** on the
   same checkpoint lineage, and PTQ numbers themselves unchanged from today (QAT work must not
   perturb the PTQ path).
3. **BEVFusion**: same shape. Gate: QAT dense-INT8 ≥ PTQ dense-INT8; FP16-sparse baseline
   (0.3228 / 0.3931 reference eval) unchanged.
4. Export/verify: ONNX + TRT export of a QAT checkpoint produces the same node structure as the
   PTQ export (Q/DQ count identical — only amax values and weights may differ).

Expected magnitude, for honesty in review: references show QAT recovering ~0.1–0.4 points over
PTQ (CUDA-CenterPoint 59.08→59.20; modelopt ResNet 75.5→75.9). If PTQ is already at-parity with
FP16 for a given model, QAT may show ~0 gain — that is a valid outcome, and the gate is "≥", not
"materially better."

---

## 6. Risks & open questions

- **R1 — AMP interplay (addressed by D5, verify in Docker).** After forcing fp32, confirm
  BEVFusion-L still fits in memory at a usable batch size; if not, the fallback is grad
  accumulation, not re-enabling AMP.
- **R2 — DDP wrap order.** mmengine wraps the model in DDP at Runner init, *before*
  `before_train` mutates the tree. Replacing submodules inside an already-wrapped DDP model can
  desynchronize reducer buckets. CUDA-CenterPoint quantizes before wrapping. v1 scope: **document
  single-GPU QAT as supported**, add a hard check in the hook (refuse under
  `MMDistributedDataParallel` with >1 ranks) rather than silently training wrong. Multi-GPU is a
  follow-up (likely: quantize in `runner.model` construction via a wrapper cfg, or rebuild
  buckets).
- **R3 — Resume-QAT is structurally broken today**: `load_from`/`resume` restores into the
  *unquantized* tree before the hook runs. Parked in v1 (WP2.4 refuses with a message). Fix
  sketch for v2: hook `before_run` (fires before checkpoint load) performs the plan-prepare, so
  resume checkpoints land in the mutated tree.
- **R4 — EMA / schedule hooks in train configs.** If a base train config carries EMA hooks or
  GT-sampler fade-out (`...fade` configs), the QAT run inherits them. Decide per-project when
  wiring WP3/WP4: default = strip EMA (references have none), keep the pipeline otherwise.
- **R5 — Calibration distribution (D8).** Cheap experiment once WP3 lands: same model, amax from
  train-loader vs val-loader calibration, compare deployed mAP. Pick the winner as default;
  one-line config escape (`calib_cache`) already covers the other.
- **R6 — Which epoch to package.** mmengine `CheckpointHook` with `save_best` on the val metric
  is the natural choice (QAT val loop runs the quantized model, so "best" is measured in the
  deployed numeric regime). Confirm the val loop is enabled in the QAT-modified config; otherwise
  package `last`.
- **R7 — BEVFusion training-config availability.** The deployment tree assumes a trainable
  BEVFusion-L config + dataset access inside the Docker image used for deployment work. Verify
  early (WP4.4) — if training must run in a different image, the producer CLI split
  (train elsewhere, package + deploy here) still works because the artifact contract (D4) is
  just `{"state_dict"}` + `.calib`.

---

## 7. Deliberately not doing (guardrails)

- **No learnable-amax / LSQ / EMA observers** (D1). Both references reject them; revisit only
  with measured evidence.
- **No spconv INT8 QAT.** Goal 1 removed the subsystem for no-win; QAT does not change that
  calculus (CUDA keeps first/sparse layers FP16 too).
- **No QAD (distillation).** modelopt ships it for LLMs only; for these detectors it is
  premature. Noted as the natural next lever *if* QAT gains prove insufficient.
- **No modelopt dependency.** Same verdict as spec.md §1.3 — borrow the workflow shape
  (quantize → normal training loop → self-describing artifact), keep the pytorch-quantization
  engine.
- **No multi-GPU QAT in v1** (R2) and **no resume support in v1** (R3) — both fail loudly
  instead of silently.
- **No new config DSL.** The QAT surface is one nested dict with a typed parser; placement stays
  exclusively `keep_fp16`/`disable_recipes`.

---

## 8. Sequencing summary

| Step | Deliverable | Gate |
|---|---|---|
| WP1 | `QATConfig` in schema + parse tests | host tests green |
| WP2 | `QATHookBase` + `save_qat_checkpoint` + AMP-off | host lint; hook unit-importable |
| WP3 | CenterPoint thin hook + QAT deploy config + e2e | Docker: tree parity; QAT ≥ PTQ mAP (SECOND release cfg) |
| WP4 | BEVFusion hook subclass + `qat` CLI | Docker: QAT ≥ PTQ dense-INT8; FP16 baseline untouched |
| WP5 | tests + docs | all gates in §5 |

Each WP is independently landable; WP3 and WP4 must not start before WP2 lands (no forked hooks,
ever).
