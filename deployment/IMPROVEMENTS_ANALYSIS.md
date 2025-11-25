# BaseDeploymentRunner Improvement Analysis

After the refactoring, `BaseDeploymentRunner` is much cleaner (856 → 569 lines, -33.6%). However, there are still opportunities for improvement:

## 🔴 Critical Issues

### 1. **Code Duplication: Model Loading Logic**
**Location:** Lines 472-483 and 492-503

**Problem:**
```python
# This exact block appears twice:
try:
    pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
    results["pytorch_model"] = pytorch_model
    self.artifact_manager.register_artifact(Backend.PYTORCH, Artifact(path=checkpoint_path))
    if hasattr(self.evaluator, "set_pytorch_model"):
        self.evaluator.set_pytorch_model(pytorch_model)
        self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")
except Exception as e:
    self.logger.error(f"Failed to load PyTorch model: {e}")
    return results
```

**Impact:** Violates DRY principle, harder to maintain, error-prone

**Fix:** Extract to `_load_and_register_pytorch_model()` method

---

### 2. **Dead Code: Unused Helper Methods**
**Location:** Lines 342-357

**Problem:**
- `_artifact_from_path()` - Never called
- `_build_model_spec()` - Never called (orchestrators build their own)

**Impact:** Confusing, adds maintenance burden

**Fix:** Remove these methods

---

## 🟡 Major Issues

### 3. **Long `run()` Method (164 lines)**
**Location:** Lines 404-568

**Problem:** The `run()` method handles too many concerns:
- Path resolution
- Model loading coordination
- Export orchestration
- Verification/evaluation delegation
- Result aggregation

**Impact:** Hard to test, hard to understand, violates Single Responsibility

**Fix:** Extract sub-methods:
- `_determine_requirements()` - Lines 433-458
- `_resolve_model_paths()` - Lines 531-549
- `_execute_exports()` - Lines 485-529

---

### 4. **Complex Conditional Logic**
**Location:** Lines 433-458

**Problem:** Nested conditionals determine what's needed:
```python
needs_export_onnx = should_export_onnx
needs_pytorch_eval = False
if eval_config.enabled:
    models_to_eval = eval_config.models
    if self._get_backend_entry(models_to_eval, Backend.PYTORCH):
        needs_pytorch_eval = True
needs_pytorch_for_verification = False
if verification_cfg.enabled:
    export_mode = self.config.export_config.mode
    scenarios = self.config.get_verification_scenarios(export_mode)
    if scenarios:
        needs_pytorch_for_verification = any(...)
requires_pytorch_model = needs_export_onnx or needs_pytorch_eval or needs_pytorch_for_verification
```

**Impact:** Hard to test, hard to reason about

**Fix:** Extract to `_determine_pytorch_requirements()` method with clear return type

---

### 5. **Repetitive Path Resolution**
**Location:** Lines 531-549

**Problem:** Similar logic for ONNX and TensorRT path resolution:
```python
if not results["onnx_path"]:
    onnx_path = self._get_backend_entry(eval_models, Backend.ONNX)
    if onnx_path and os.path.exists(onnx_path):
        results["onnx_path"] = onnx_path
        self.artifact_manager.register_artifact(...)
    elif onnx_path:
        self.logger.warning(...)
# Same pattern for TensorRT
```

**Impact:** Code duplication, error-prone

**Fix:** Extract to `_resolve_and_register_artifact(backend, results_key)` helper

---

### 6. **Magic Strings**
**Location:** Lines 192, 286, 291, 294

**Problem:** Hard-coded directory names:
```python
onnx_dir = os.path.join(self.config.export_config.work_dir, "onnx")
tensorrt_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
output_path = os.path.join(tensorrt_dir, "model.engine")
engine_filename = onnx_filename.replace(".onnx", ".engine")
```

**Impact:** Hard to change, inconsistent naming

**Fix:** Extract to constants or config values

---

## 🟢 Minor Issues

### 7. **Generic Exception Handling**
**Location:** Multiple try-except blocks

**Problem:** All exceptions caught generically:
```python
except Exception:
    self.logger.error("ONNX export failed")
    raise
```

**Impact:** Hides specific error types, harder to debug

**Fix:** Catch specific exceptions or at least log exception type

---

### 8. **Export Path Logic Duplication**
**Location:** Lines 289-295

**Problem:** Logic for determining TensorRT output path:
```python
if os.path.isdir(onnx_path):
    output_path = os.path.join(tensorrt_dir, "model.engine")
else:
    onnx_filename = os.path.basename(onnx_path)
    engine_filename = onnx_filename.replace(".onnx", ".engine")
    output_path = os.path.join(tensorrt_dir, engine_filename)
```

**Impact:** Could be reused elsewhere

**Fix:** Extract to `_get_tensorrt_output_path(onnx_path, tensorrt_dir)` helper

---

### 9. **Inconsistent Error Handling**
**Location:** Some methods return `None` on error, others raise

**Problem:**
- `export_onnx()` returns `None` if skipped
- `export_tensorrt()` returns `None` if skipped
- But `run()` returns partial results dict on error

**Impact:** Unclear error contract

**Fix:** Standardize error handling strategy (either all return None or all raise)

---

## 📊 Summary

**Current State:**
- ✅ Much cleaner than before (856 → 569 lines)
- ✅ Good separation via orchestrators
- ⚠️ Still has some code smells

**Recommended Priority:**
1. **High:** Fix code duplication (#1, #5)
2. **Medium:** Extract long method (#3)
3. **Low:** Remove dead code (#2), extract constants (#6)

**Estimated Impact:**
- **Lines reduced:** ~50-80 lines (9-14% reduction)
- **Cyclomatic complexity:** Reduced by ~15-20%
- **Testability:** Significantly improved
- **Maintainability:** Much better

---

## 🎯 Proposed Refactoring Plan

### Phase 1: Quick Wins (Low Risk)
1. Remove dead code (`_artifact_from_path`, `_build_model_spec`)
2. Extract constants for directory names
3. Extract `_load_and_register_pytorch_model()` helper

### Phase 2: Method Extraction (Medium Risk)
1. Extract `_determine_pytorch_requirements()`
2. Extract `_resolve_and_register_artifact()`
3. Extract `_get_tensorrt_output_path()`

### Phase 3: Major Refactoring (Higher Risk, Higher Reward)
1. Split `run()` into smaller methods:
   - `_prepare_deployment()` - Determine requirements, resolve paths
   - `_execute_exports()` - Handle ONNX/TensorRT exports
   - `_execute_verification_and_evaluation()` - Delegate to orchestrators

---

## 💡 Design Suggestions

### Consider a "Deployment Planner" Class
Instead of having complex logic in `run()`, create a `DeploymentPlanner` that:
- Analyzes config to determine what needs to be done
- Returns a `DeploymentPlan` object with clear steps
- `run()` just executes the plan

**Benefits:**
- Testable planning logic separately
- Can validate plan before execution
- Easier to add new deployment modes

### Consider Result Builder Pattern
Instead of building results dict inline, use a `DeploymentResultBuilder`:
```python
builder = DeploymentResultBuilder()
builder.set_pytorch_model(model)
builder.set_onnx_path(path)
builder.add_error("onnx_export", error)
return builder.build()
```

**Benefits:**
- Clearer result construction
- Type-safe result building
- Easier to extend with new result fields

---

## ✅ Conclusion

The refactoring was excellent! The runner is much cleaner. However, there's still room for improvement:

- **Immediate:** Fix code duplication and remove dead code
- **Short-term:** Extract complex methods
- **Long-term:** Consider architectural patterns (Planner, Builder)

The current code is **production-ready** but could be **even better** with these improvements.
