# Spec: Should the quantization config be split from the deploy config?

Status: **evaluation only — no decision, no code change.**
Prompted by: after the PTQ/QAT config-driven unification, the deploy config now carries the whole
quantization run. The question on the table: *"quantization and export have nothing to do with each
other — wouldn't `quant + model config` and `export + model config` as two separate files be
cleaner?"*

This spec evaluates that proposal against the invariants the current design bought, and lays out
the realistic alternatives with their costs. It deliberately does not pick a winner; §7 lists the
questions whose answers decide it.
跑qunat + model config.
---

## 1. Current architecture (facts)

One deploy config file per deployable artifact. Sections and their consumers, as verified in code:

| Config section | producer (`quantize.py ptq/qat`) | deploy: export | deploy: eval / verify |
|---|---|---|---|
| model config (separate CLI arg) | ✓ FP model + calib/train dataloader | ✓ | ✓ |
| `checkpoint_path` | ✓ default `--output` | ✓ input | ✓ input |
| `quantization` placement (`enabled`, `fuse_bn`, `default_precision`, `keep_fp16`, `disable_recipes`) | ✓ builds the Q/DQ tree | ✓ **rebuilds the same tree** to `load_state_dict` (`deployment/projects/*/io/model_loader.py` → `build_*_plan(config).prepare(model)`) | ✓ same (PyTorch backend) |
| `quantization.ptq` / `quantization.qat` | ✓ **only consumer** | ✗ | ✗ |
| `export` / `components` / `onnx_config` / `tensorrt_config` | ✗ | ✓ | partially (engine/model dirs) |
| `evaluation` / `verification` | ✗ | ✗ | ✓ |

Supporting facts:

- A non-quantized deploy config (`deploy_config.py`, BEVFusion FP) has **no quantization section at
  all** — the section is already optional and self-contained.
- In the INT8 BEVFusion config the quantization section is ~20 of ~240 lines; the bulk of the file
  is export/components/eval/verification.
- `_base_` inheritance already factors shared skeletons (`_deploy_config_int8_base.py`) and lets a
  variant config override only the `quantization` block (`deploy_config_int8_second_qat.py` does
  exactly this).

## 2. Invariants the current design bought (what any change must preserve)

These were paid for across spec.md / spec_qat.md and the PTQ unification; losing any of them is a
regression, not a style choice:

- **I1 — Tree parity by construction.** Producer and deploy loader call the *same*
  `build_*_plan(config)` with the *same parsed placement*. The PTQ `state_dict` and the deployed
  module tree line up because there is one source of placement, parsed once
  (`deployment/tests/test_qat_tree_parity.py` guards it).
- **I2 — Artifact pairing.** `checkpoint_path` is both the producer's default output and the deploy
  input. A checkpoint cannot silently be deployed with a different placement than the one that
  produced it, because both live in the same file the two commands share.
- **I3 — One-file reproducibility.** The `ptq` / `qat` block records the producer recipe next to
  the artifact it produced; re-running the producer from the same config reproduces the checkpoint.
- **I4 — Typo safety.** `KNOWN_KEYS` guards on every block; a misspelled key fails loudly instead
  of silently degrading placement (visible only as a Docker-eval mAP drop).

## 3. Challenging the premise: is quantization really unrelated to export?

Half true, and the half matters.

- **True half:** the `ptq`/`qat` producer blocks are consumed by nobody on the deploy side, and
  `export`/`components`/`evaluation`/`verification` are consumed by nobody on the producer side.
  Operationally the producer already ignores the export sections; carrying them costs nothing at
  runtime. The *conceptual* impurity is real: the deploy config carries provenance the deploy step
  never reads.
- **False half:** the quantization **placement** is not producer-only. The deploy loader must
  rebuild the identical Q/DQ tree to load the checkpoint (§1 table, I1), and the exported ONNX
  embeds the Q/DQ nodes that placement created (paired with `tensorrt_config` precision flags and,
  for BEVFusion, plugin libraries). Placement is the *shared kernel* between the two halves.

So the clean cut is **not** "quantization vs export". The severable pieces are the producer
sub-blocks (`ptq`/`qat`); the placement must remain visible to both sides through exactly one
source. Every option below is really an answer to: *where does placement live, and how does the
other side see it?*

## 4. Options

### Option A — status quo (one deploy config; producer reads its slice)

- **Benefits:** one file fully describes a deployable artifact; I1–I4 hold by construction; every
  command takes one config path; variants via `_base_` are already cheap.
- **Drawbacks:** the deploy config carries producer-only blocks (mild conceptual impurity); a
  reader must know which sections belong to which lifecycle phase.
- **Migration cost:** zero. **Maintenance:** unchanged.

### Option B — organizational split via `_base_` (same schema, same CLIs)

Extract the `quantization = dict(...)` section into its own file, e.g.
`quant_recipe_second_2_6.py`; the deploy config inherits it:

```python
_base_ = ["./_deploy_config_int8_base.py", "./quant_recipe_second_2_6.py"]
```

Both CLIs keep taking the deploy config. The split is at the *file* level only; the parsed config
is identical.

- **Benefits:** file-level ownership (edit the recipe file to re-quantize, the deploy file to
  change shapes/backends); one recipe reusable across several deploy targets (same placement,
  different TRT profiles); cleaner diffs/review; **all of I1–I4 preserved**; zero schema/CLI/loader
  change.
- **Drawbacks:** one more file and one indirection hop per config; the recipe file alone is not a
  runnable producer input (it has no `checkpoint_path`) — someone will eventually point
  `--deploy-cfg` at it and get the existing "needs --output" error, which is correct but worth a
  clearer message; `_base_` dict-merge semantics must be understood when a child overrides
  individual recipe keys.
- **Migration cost:** small, config files + docs only. **Maintenance:** slightly lower when recipes
  are shared, slightly higher file count.

### Option C — hard split (the proposal): `quant_cfg + model_cfg` / `export_cfg + model_cfg`

Placement must live somewhere; the sub-options are the real design:

- **C1 — export config references the quant config** (`quantization_cfg = "path/to/quant.py"`):
  deploy loads the referenced file to rebuild the tree. Keeps one placement source (I1), but
  introduces a new *config→config dependency edge* outside `_base_` that mmengine does not manage,
  and the reference must be updated in lockstep with the artifact (I2 now spans two files).
- **C2 — placement duplicated in both files:** rejected outright. Drift between the two copies is a
  silent mAP drop — the exact bug class I1/I4 were built to eliminate.
- **C3 — deploy CLI takes three paths** (`export_cfg model_cfg quant_cfg`): keeps one placement
  source but pushes the pairing problem to every command line: *which* quant cfg produced this
  checkpoint becomes tribal knowledge again — precisely what the unification just fixed.

Common to all C variants:

- **`checkpoint_path` loses its single owner.** If it stays in the export config, the producer no
  longer has an output default (or must read the export config — defeating the split). If the quant
  config gets its own output path, the same artifact has two names in two files (pairing risk, I2
  broken).
- **Benefits:** maximal conceptual purity; a quant run never sees export keys; smaller individual
  files.
- **Drawbacks:** reintroduces mispairing at file granularity; longer commands or fragile
  references; renaming an artifact touches two files (change amplification); docs get longer.
- **Migration cost:** high — schema loader, both producer CLIs, deploy CLI, all configs, docs,
  tests. **Maintenance:** two files whose invisible compatibility constraint (checkpoint ↔
  placement) is enforced by discipline instead of structure.
- **Precedent check:** mmdeploy splits *deploy cfg vs model cfg* — quantization lives **inside**
  the deploy cfg; there is no third file. TensorRT practice binds the calibration recipe to the
  engine-build config. The proposed three-way split has no established precedent in the ecosystems
  this framework imitates.

### Option D — artifact-carried placement (checkpoint becomes self-describing)

The producer embeds placement (+ provenance) into the checkpoint
(`{"state_dict", "quantization": {...}}`); the deploy loader reads placement from the artifact, not
from config. This is the NVIDIA modelopt `mto.save/restore` pattern.

- **Benefits:** *true* decoupling — the deploy config genuinely needs no quantization knowledge;
  mispairing is impossible by construction (placement travels with the weights it built); I1/I2
  become properties of the artifact instead of conventions of the config.
- **Drawbacks:** placement disappears from code review ("what is FP16 in this deployment?" now
  requires loading a binary); checkpoint format change with migration for existing artifacts; the
  QAT hook, tree-parity test, and loaders all rework; config-level placement experiments need an
  override path (complexity returns through the back door).
- **Migration cost:** highest — artifact contract change. **Maintenance:** lowest for the pairing
  class of bugs, higher for tooling/debugging.

## 5. What each option does to the invariants

| | I1 tree parity | I2 pairing | I3 reproducibility | I4 typo guard |
|---|---|---|---|---|
| A status quo | ✓ | ✓ | ✓ | ✓ |
| B `_base_` split | ✓ | ✓ | ✓ | ✓ |
| C1 reference | ✓ | ⚠ spans two files | ✓ | ✓ |
| C2 duplicate | ✗ | ✗ | ⚠ | ⚠ |
| C3 three paths | ✓ | ✗ per-command | ⚠ | ✓ |
| D artifact-carried | ✓✓ (structural) | ✓✓ (structural) | ⚠ (in artifact, not reviewable text) | ✓ |

## 6. Cost/benefit summary

- The runtime/product behavior is identical across A/B/C; only D changes the artifact contract.
  The debate is purely about *where humans read and edit things* — so the deciding factor is
  ownership and change patterns, not machinery.
- The measured impurity is small: ~20 lines of quantization in a ~240-line config, absent entirely
  from FP configs.
- C pays a high migration cost to move the impurity from "producer blocks visible in the deploy
  config" to "pairing constraint invisible between two configs". That is trading a cosmetic cost
  for a correctness risk.

## 7. Decision questions (answers pick the option)

1. **What is the actual pain today?** File length, unclear ownership, or conceptual discomfort?
   (If length: the export/components sections dominate, not quantization. If ownership: B solves it
   for a fraction of C's cost.)
2. **Is recipe reuse across deploy targets a real upcoming need** (same placement, several TRT
   shape/backend configs)? If yes → B pays for itself immediately.
3. **Will a checkpoint ever legitimately deploy with a different placement than produced it?**
   (Today: forbidden by construction, and that is a feature.) If the answer is "never" and the team
   wants real decoupling, the principled endgame is D — placement belongs to the artifact — not C.
4. **Are quant tuning and deployment done by different people/teams?** If the same person (today's
   reality), C's ownership benefit is theoretical while its pairing cost is real.
5. **How stable is the quantization schema?** D freezes placement into artifacts; while
   `keep_fp16`/recipes are still evolving, config-carried placement (A/B) is easier to migrate.

## 8. Reviewer's leaning (not a decision)

- **A or B now.** B if file-level ownership or recipe reuse (Q1/Q2) is a felt need; otherwise A —
  the impurity is 20 lines of provenance, and provenance next to the artifact is a feature, not a
  smell.
- **C in any variant is not recommended:** it cuts through the load-bearing joint (placement) and
  re-creates the checkpoint↔placement pairing problem that the config-driven unification just
  eliminated. The clean-looking split is clean only because the coupling it hides moves into
  people's heads.
- **D is the honest version of the proposal's instinct** — if quantization and deployment should
  truly not know about each other, the knowledge must move into the artifact, not into a second
  config file. Worth revisiting once the quantization schema stabilizes and Docker e2e is green,
  as its own spec.

---

## 9. Follow-up: where should the model-config reference live?

Raised after §1–8: *"a config that embeds a path to the model config feels wrong."* This section
treats that as its own design question, because it is sharper than the §4 split question and can be
decided independently.

### 9.1 The actual inconsistency

The framework currently gives the deploy config **two different identities in two commands**:

| Command | How the model config arrives | Deploy config's identity |
|---|---|---|
| `deployment.cli.main <project> <deploy_cfg> <model_cfg>` | **CLI positional peer** (`deployment/cli/args.py:126`) | *settings file* — pairing is the caller's job |
| `quantize.py ptq --deploy-cfg <deploy_cfg>` | **embedded path** `quantization.ptq.model_cfg` | *manifest* — pairing is recorded in the file |
| `quantize.py qat --deploy-cfg <deploy_cfg>` | **embedded path** `quantization.qat.train_cfg` | *manifest* |

The discomfort is not really "a config referencing another config" — `_base_` does exactly that,
and `checkpoint_path` / `qat.checkpoint` already made this file reference external inputs. The
discomfort is the **hybrid**: the same file is a manifest for the producer and a settings file for
deploy. Whichever identity is chosen, choosing *one* removes the weirdness.

### 9.2 A fact that constrains the design: there is no single "model config"

The lifecycle touches up to three model-config variants of the same architecture (real example,
CenterPoint SECOND 2.6):

- **train** — `second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py` (QAT fine-tune; optimizer,
  AMP, train pipeline),
- **calibration** — the eval-variant config whose `val_dataloader` feeds PTQ,
- **deploy/eval** — `second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py` (deploy CLI docstring).

Calibration and deploy/eval are usually the *same* file; train is genuinely different. Any design
must keep the train config phase-specific while ideally single-sourcing the calib/eval one.

A second constraining fact: the deploy config is **already irreversibly model-specific** — TRT
profile shapes (`1020×1020` grid, 11-channel pillars) and `keep_fp16` module names
(`pts_backbone.blocks.0`) are architecture-bound. There is no "one deploy config, many models"
reuse to protect. The deploy_cfg↔model_cfg pairing is 1:1 (occasionally 1:few eval variants).

### 9.3 Designs

**M1 — commit to the manifest identity: lift `model_cfg` to the top level of the deploy config.**

```python
# deploy config = the artifact's manifest: what it is, how it was made, how it deploys
model_cfg = "projects/CenterPoint/configs/.../second_..._t4metric_v2.py"
checkpoint_path = "work_dirs/.../epoch_29_ptq.pth"
```

- Deploy CLI: `model_cfg` positional becomes an optional override (`nargs="?"`); the common command
  shrinks to `deployment.cli.main centerpoint <deploy_cfg>`. The 1:few eval-variant case is served
  by the override.
- Producer PTQ: reads top-level `model_cfg` (the `ptq.model_cfg` key is deleted — the block shrinks
  to pure calibration recipe: samples/batch/seed/shuffle). `--config` override stays.
- QAT: `qat.train_cfg` **stays** — the training config is genuinely phase-specific (9.2).
- **Benefits:** one identity everywhere; every command becomes `tool + one config path
  (+ overrides)`; kills the *remaining* tribal-knowledge pairing — today nothing stops deploying
  with the wrong `model_cfg` positional; the reference the proposal found "weird" moves from a
  buried `quantization.ptq.model_cfg` to the manifest's header, next to `checkpoint_path`, where
  its meaning ("this artifact is this model") is self-evident.
- **Drawbacks:** the config→config path reference is relocated, not eliminated (it *cannot* be
  eliminated without giving up recorded pairing — see M2); deploy CLI signature changes (backward
  compatible if the positional is kept as override); one more top-level key in the schema.
- **Migration:** small and mechanical — schema + `args.py` positional + `resolve_ptq_settings`
  fallback + configs + docs. No artifact or loader change.

**M2 — commit to the settings-file identity: model config is always a CLI peer, never embedded.**

Delete `ptq.model_cfg` and `qat.train_cfg`; producer requires `--config`, mirroring the deploy
CLI's positional.

- **Benefits:** configs become pure settings, zero config→config references; maximal symmetry with
  mmdeploy's two-argument convention.
- **Drawbacks:** rolls back half of the config-driven unification — QAT returns to multi-flag
  commands; the calibration-dataset and training-config choices (part of the recipe, I3) are no
  longer recorded anywhere; every command re-states a pairing that is 1:1 anyway, which is ceremony
  plus a mispairing opportunity per invocation.

**M3 — merge the model config via `_base_`: rejected.** Merging a training config's namespace
(`model`, `train_pipeline`, `optim_wrapper`, …) into the deploy config pollutes the key space the
KNOWN_KEYS guards protect, bloats the parsed config, and the loaders hand the model config to
mmdet3d APIs as a separate object anyway. `_base_` is for composing *same-kind* configs.

**M4 — a third "run manifest" file binding {model_cfg, deploy_cfg, checkpoint}: rejected for now.**
Pure settings files plus explicit binding is the theoretically clean answer, but since the deploy
config is already 1:1 with the artifact (9.2), the manifest would be a file whose only content is
two pointers — a new file type and CLI change purchasing nothing M1 doesn't. Revisit only if deploy
configs ever become model-agnostic.

### 9.4 Interaction with §4

M1/M2 are orthogonal to the §4 options and can be decided first. Notably, **M1 + B** compose well:
a `quant_recipe_*.py` fragment (placement + recipe) inherited by a manifest-identity deploy config
gives file-level ownership *and* one-command runs. M1 also strengthens the §4-A/B position: once
the deploy config is admitted to be the artifact's manifest, "it references other files" stops
being a smell — that is what manifests are for (Cargo.toml, package.json, a BOM).

### 9.5 Reviewer's leaning → **DECIDED: M1 adopted, implemented 2026-07-16**

Top-level `model_cfg` added to the quant deploy configs; `ptq.model_cfg` deleted from the schema;
deploy CLI `model_cfg` positional is now an optional override; `qat.train_cfg` unchanged
(phase-specific). Non-quant configs keep working via the positional.

Original leaning, kept for the record:

**M1.** The file already crossed the manifest Rubicon when `checkpoint_path` became both producer
output and deploy input — and the TRT shapes bound it to one model long before that. Finishing the
thought (top-level `model_cfg`, optional positional) resolves the hybrid identity, shortens every
command, and removes the last unrecorded pairing. M2 is self-consistent but pays for purity with
reproducibility — the framework's stated goal points the other way.
