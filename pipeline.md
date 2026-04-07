# Research Code Cleanup Pipeline

## Overview

An agentic pipeline that takes a messy research codebase — typically accumulated over months/years of iterative development — and reorganizes it into clean, documented, reproducible code while preserving exact numerical functionality.

Designed for: macro/finance model code, data pipelines, numerical optimization, econometric estimation. The kind of code where correctness is defined by output, not by a spec.

---

## Human Inputs (required before pipeline starts)

Collected via `/intake` command. The human provides:

| Input | Example | Why needed |
|---|---|---|
| **Project description** | "Solves a heterogeneous-agent DSGE model, calibrates to EU data, produces welfare counterfactuals" | Cartographer uses this to interpret code purpose |
| **Entry points** | `main.py`, or "run scripts 1-5 in order", or "I think `run_all.sh` works" | Needed to trace execution flow |
| **Known outputs** | Paper tables, figures, saved `.csv`/`.npy` files, or "converges to ~4.7%" | Test-writer pins against these |
| **Target end state** | Replication package / co-author handoff / publishable repo / personal cleanup | Shapes the architect's plan |
| **Known landmines** (optional) | "The calibration in `calib_v2.py` is fragile", "ignore the `old/` folder" | Prevents the refactorer from breaking things or wasting time |

---

## Agents

### 1. `cartographer`

**Purpose:** Understand the codebase without changing it.

**Allowed tools:** Read, Glob, Grep, Bash (read-only commands only)

**Inputs:** Project folder + human description + entry points

**Outputs:**
- `artifacts/dependency_graph.json` — module-level import/call graph
- `artifacts/codebase_map.md` — plain-language summary of every file: what it does, what it depends on, what it produces
- `artifacts/entry_points.md` — traced execution order from entry points
- `artifacts/dead_code.md` — files/functions that appear unreachable
- `artifacts/red_flags.md` — hardcoded paths, duplicated functions, global state, missing imports, `import *`, etc.

**Key instruction:** Do not guess. If a file's purpose is unclear, say so. Flag ambiguity rather than inventing explanations.

---

### 2. `test-writer`

**Purpose:** Pin current behavior before any refactoring begins.

**Allowed tools:** Read, Write, Bash

**Inputs:** Cartographer outputs + entry points + known outputs from human

**Outputs:**
- `tests/` folder with regression tests
- `tests/snapshots/` — captured numerical outputs (arrays, dataframes, scalars)
- `tests/conftest.py` — fixtures for seed-fixing, path resolution, tolerance settings

**Key instructions:**
- Use approximate equality for floats (`np.allclose` with documented tolerances)
- Fix random seeds where possible; flag stochastic code that can't be seeded
- If code takes >5 min to run, create a "fast" subset that tests critical intermediate values
- Capture file outputs (saved csvs, figures) as reference artifacts
- Every test must pass on the current messy code before proceeding

**Gate:** Pipeline does not proceed until all pinned tests pass on the unmodified codebase.

---

### 3. `architect`

**Purpose:** Design the target structure. Does not touch code.

**Allowed tools:** Read only

**Inputs:** Cartographer outputs + human's target end state

**Outputs:**
- `artifacts/proposed_structure.md` — target folder layout with rationale
- `artifacts/migration_plan.md` — ordered checklist of moves, renames, merges, splits
- `artifacts/naming_conventions.md` — proposed variable/function/module naming rules
- `artifacts/config_extraction.md` — hardcoded values to extract into config (paths, parameters, calibration values)

**Key instructions:**
- Group by function (data loading, model, estimation, analysis, plotting), not by chronology
- Preserve the execution order — the pipeline must still run in sequence
- Flag files where merge vs. split is a judgment call and ask the human
- Propose a single `config.yaml` or `params.py` for all magic numbers

**Gate: Human reviews and approves the migration plan before proceeding.** This is the critical checkpoint. The human marks each proposed change as approved / modified / rejected.

---

### 4. `refactorer`

**Purpose:** Execute the approved migration plan.

**Allowed tools:** Read, Write, Bash

**Inputs:** Approved migration plan + test suite

**Process:**
1. Work through the migration plan in order
2. After each file change: run the test suite
3. If tests fail: revert and flag the change for human review
4. Commit after each successful step (atomic commits with descriptive messages)

**Scope of changes:**
- Move/rename files and update all imports
- Extract hardcoded values into config
- Rename variables/functions per naming conventions
- Extract duplicated code into shared utilities
- Add type hints to function signatures
- Remove confirmed dead code (cartographer-flagged + human-approved)
- Replace `import *` with explicit imports
- Standardize file I/O (pathlib, relative paths from project root)

**Not in scope** (require human decision):
- Algorithmic changes, even "obvious" improvements
- Changing numerical methods or solver settings
- Removing code the human hasn't confirmed as dead
- Restructuring the model itself

---

### 5. `doc-writer`

**Purpose:** Document the cleaned codebase.

**Allowed tools:** Read, Write

**Inputs:** Cleaned codebase + cartographer map + architect plan

**Outputs:**
- Docstrings on all public functions (Google style)
- `README.md` — project overview, setup instructions, how to run, expected outputs
- `docs/pipeline.md` — execution order with description of each step
- `docs/model.md` — (if applicable) model equations and their code counterparts
- Inline comments for non-obvious numerical logic ("this is the FOC for household problem")

**Key instruction:** Explain *why*, not *what*. `# iterate over agents` is useless. `# solve each agent type's Bellman equation independently before aggregating` is useful.

---

## Skills

### `dependency-mapping`
Uses Python `ast` module to parse imports and function calls across files. Produces a structured JSON graph and a human-readable summary. Handles relative imports, `sys.path` hacks, and dynamic imports (flags them).

### `numerical-validation`
Instructions for building regression tests on numerical code:
- Tolerances by type (scalars: `rtol=1e-6`, arrays: `np.allclose`, dataframes: column-wise comparison)
- Seed-fixing patterns for numpy, scipy, torch
- Handling long-running code (checkpoint intermediate values)
- Comparing saved files (csv diff, image perceptual hash)

### `research-code-patterns`
Catalog of common research code smells and their fixes:
- `final_v3_REAL.py` → identify the canonical version
- Notebook cells pasted into `.py` with dead `In[47]:` markers
- Hardcoded absolute paths (`/Users/marleen/Desktop/data/...`)
- Copy-pasted functions across files (identify canonical, replace with imports)
- Global mutable state used to pass config
- `plt.show()` blocking execution in batch scripts

---

## Commands

### `/intake`
Collects human inputs. Prompts for: project description, entry points, known outputs, target end state, known landmines.
Saves to `artifacts/intake.md`.

### `/map`
Runs the cartographer agent. Produces dependency graph and codebase map.

### `/pin`
Runs the test-writer agent. Produces regression test suite. Verifies all tests pass on current code.

### `/plan`
Runs the architect agent. Produces migration plan for human review.

### `/clean $ARGUMENTS`
Runs the refactorer on a specific file, module, or `all` for the full plan.
Always runs tests after each change.

### `/validate`
Runs the full test suite. Reports pass/fail with diffs on any failures.

### `/document`
Runs the doc-writer agent on the cleaned codebase.

### `/status`
Shows pipeline progress: which stages are complete, which tests pass, which migration plan items are done.

---

## Execution Order

```
1. /intake          ← human provides context
2. /map             ← cartographer reads everything
3. /pin             ← test-writer captures current behavior
   ↓
   GATE: all pinned tests must pass
   ↓
4. /plan            ← architect proposes structure
   ↓
   GATE: human reviews and approves plan
   ↓
5. /clean all       ← refactorer executes (with tests after each step)
6. /validate        ← final check: all tests still pass
7. /document        ← doc-writer adds documentation
```

---

## Artifacts Directory

All pipeline outputs go to `artifacts/` to keep them separate from the project code:

```
artifacts/
├── intake.md
├── dependency_graph.json
├── codebase_map.md
├── entry_points.md
├── dead_code.md
├── red_flags.md
├── proposed_structure.md
├── migration_plan.md        ← human annotates this
├── naming_conventions.md
├── config_extraction.md
└── status.md
```

---

## Failure Modes to Watch For

- **Silent numerical drift:** Tests pass with loose tolerances but results shifted. Mitigation: pin scalar outputs with tight tolerances, arrays with `rtol=1e-6`.
- **Import cycles after reorganization:** Moving files around can create circular imports. Refactorer must check for this after each move.
- **Path-dependent execution:** Some research code relies on being run from a specific directory. Refactorer must standardize to project-root-relative paths.
- **Fragile solver convergence:** Reordering operations or changing float precision can break convergence. If model code exists, refactorer only touches structure (names, file locations), never the numerics.
- **Dead code that isn't dead:** Cartographer flags it, but only the human confirms deletion. Some "unused" functions are called via string dispatch or config-driven execution.