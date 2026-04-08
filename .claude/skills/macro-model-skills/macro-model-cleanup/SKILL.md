---
name: macro-model-cleanup
description: >
  Orchestrates the diagnostic and cleanup workflow for our macro model codebase.
  Use when user says "clean up the code", "diagnose the model", "refactor",
  "fix the codebase", "improve code quality", or "start the cleanup pipeline".
  Coordinates the architecture and review skills into a phased workflow.
metadata:
  author: TODO
  version: 0.1.0
---

# Macro Model Cleanup Workflow

Phased workflow for diagnosing and cleaning messy macro model code.
Designed to be conservative: diagnose fully before changing anything.

---

## Philosophy

1. **Read everything before changing anything**
2. **One category of change at a time** — don't mix refactors with bug fixes
3. **Verify numerically after every change** — policy functions must not drift
4. **Smallest diff that fixes the problem**

---

## Phase 1: Inventory & Baseline

**Goal:** Understand what exists and establish a numerical baseline.

### Steps

1. **Map the codebase**
   - Read every file; build a dependency graph
   - Identify entry point, solver core, simulation, utilities, plotting
   - Cross-reference with `macro-model-architecture` skill

2. **Establish numerical baseline**
   ```bash
   # TODO: command to run the model and save reference output
   # e.g. python main.py --save-baseline baseline_results.npz
   ```
   - Save: policy functions, value functions, simulated moments
   - These are the ground truth for validating all subsequent changes

3. **Produce inventory report**
   For each file, document:
   - Purpose (one line)
   - Lines of code
   - Dependencies (imports from other project files)
   - Suspected issues (first-pass, using `macro-model-review` skill)

### Output
- `reports/inventory.md` — file-by-file summary
- `baseline_results.npz` (or equivalent) — numerical reference

---

## Phase 2: Diagnose

**Goal:** Comprehensive issue list, categorised and prioritised.

### Steps

1. **Run the review skill** (`macro-model-review`) on each file
2. **Categorise every issue:**

   | Category | Risk of changing | Examples |
   |----------|-----------------|----------|
   | Dead code removal | Very low | Unused imports, commented blocks, unreachable branches |
   | Naming & style | Very low | Inconsistent names, magic numbers → named constants |
   | Deduplication | Low | Copy-pasted grid construction, repeated utility code |
   | Bug fixes | Medium | Off-by-one, wrong interpolation, missing constraints |
   | Structural refactor | Higher | Splitting god-functions, reorganising modules |
   | Algorithmic improvement | Highest | Changing solution method, vectorisation |

3. **Produce diagnosis report**

### Output
- `reports/diagnosis.md` — all issues with severity, category, file, line(s)

---

## Phase 3: Low-Hanging Fruit Cleanup

**Goal:** Safe, high-confidence changes that can't break numerics.

### What qualifies as low-hanging fruit
- Remove clearly dead code (unused imports, commented-out blocks)
- Replace magic numbers with named constants
- Fix naming inconsistencies
- Add type hints to function signatures
- Deduplicate identical code blocks into shared utility functions
- Remove redundant variable copies

### Workflow per change
1. Make the change
2. Run the model
3. Compare output to baseline: `np.allclose(new, baseline, atol=TODO, rtol=TODO)`
   <!-- TODO: define acceptable tolerances for your model -->
4. If output matches → commit. If not → revert and flag for Phase 4.

### Validation script

```python
# scripts/validate_against_baseline.py
# TODO: adapt to your actual output format

import numpy as np
import sys

baseline = np.load('baseline_results.npz')
current = np.load(sys.argv[1])

ATOL = 1e-10  # TODO: set appropriate tolerance
RTOL = 1e-10  # TODO: set appropriate tolerance

all_ok = True
for key in baseline.files:
    if not np.allclose(baseline[key], current[key], atol=ATOL, rtol=RTOL):
        max_diff = np.max(np.abs(baseline[key] - current[key]))
        print(f"MISMATCH: {key}, max diff = {max_diff:.2e}")
        all_ok = False
    else:
        print(f"OK: {key}")

sys.exit(0 if all_ok else 1)
```

### Output
- Cleaned code (committed incrementally)
- `reports/phase3_changes.md` — what was changed and why

---

## Phase 4: Medium-Risk Changes (future)

**Goal:** Bug fixes and structural refactors, one at a time.

<!-- TODO: expand this phase once Phase 3 is battle-tested -->

- Each change gets its own branch
- Numerical validation after every change
- Human review before merge

---

## Phase 5: Algorithmic Improvements (future)

<!-- TODO: expand once earlier phases are complete -->

- Vectorisation of remaining loops
- Interpolation method upgrades
- Potential numba/jax acceleration

---

## Guard Rails

### Never do these without explicit human approval
- Change the economic model (preferences, constraints, income process)
- Modify convergence tolerances
- Alter grid bounds or density
- Change the solution method

### Always do these
- Validate numerically after every change
- Keep changes atomic and reversible
- Document what changed and why

<!-- TODO: add any project-specific guard rails, e.g.
     "never touch calibration.py without running the moment-matching check" -->
