# Research Code Cleanup Pipeline

## Overview

An agentic pipeline that takes a messy research codebase and reorganizes it into clean, documented, reproducible code while preserving exact numerical functionality.

Designed for: macro/finance model code, data pipelines, numerical optimization, econometric estimation. The kind of code where correctness is defined by output, not by a spec.

---

## Pipeline Flowchart

```
┌──────────────────────────────────────────────────────────┐
│                  PHASE 1 — INTAKE                        │
│                                                          │
│  Human provides context.md                               │
│  (model description, architecture, known problems,       │
│   experiential context interview)                        │
│         │                                                │
│         ▼                                                │
│  Cartographer [blue, sonnet]                             │
│  Tools:  Read, Glob, Grep, Bash (read-only), Write      │
│  Inputs: clean_the_mess/ + context.md                    │
│  Outputs:                                                │
│    artifacts/intake.md              (project context)    │
│    artifacts/codebase_map.md        (file-by-file)       │
│    artifacts/dependency_graph.json  (import graph)       │
│    artifacts/entry_points.md        (call chains)        │
│    artifacts/dead_code.md           (unreachable code)   │
│    artifacts/red_flags.md           (issues by severity) │
│    artifacts/fix_proposals.md       (effort-tiered)      │
│    artifacts/task_queue.md          (progress tracker)   │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE 1: Human reviews fix_proposals.md       ║      │
│  ║  Marks each proposal: approved / modified /    ║      │
│  ║  rejected. Confirms dead code is truly dead.   ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: DONE                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────────┐  ┌─────────────────────────────┐
│   PHASE 2 — PLAN        │  │  PHASE 3 — SAFE CLEANUP     │
│   (can run in parallel) │  │  (surely-safe changes only)  │
│                         │  │                              │
│  Architect [cyan,sonnet]│  │  Refactorer [red, opus]      │
│  Inputs: artifacts +    │  │  Inputs: task_queue.md       │
│    context.md           │  │                              │
│  Outputs:               │  │  Only tasks that remove      │
│   proposed_structure.md │  │  dead/unreachable code:      │
│   migration_plan.md     │  │   - Delete duplicate files   │
│   naming_conventions.md │  │   - Remove commented blocks  │
│   config_extraction.md  │  │   - Remove triple-quote      │
│                         │  │     block comments           │
│  ════════════════════   │  │   - Fix duplicate imports    │
│  ║ GATE 2: Human     ║  │  │   - Fix cosmetic typos      │
│  ║ reviews migration ║  │  │   - Delete unused functions  │
│  ║ plan. Only blocks ║  │  │                              │
│  ║ Phase 5c (large   ║  │  │  Each task includes syntax   │
│  ║ refactors).       ║  │  │  check: python -c "import    │
│  ════════════════════   │  │  <module>" after every edit.  │
│                         │  │                              │
│  Status: TODO           │  │  Rule: changes must have     │
│                         │  │  ZERO chance of affecting     │
│  Status: TODO           │  │  runtime behavior. If in     │
│                         │  │  doubt, wait for Phase 4.    │
└─────────────────────────┘  │                              │
                              │  ════════════════════════    │
                              │  ║ GATE 3: Human runs    ║  │
                              │  ║ solve_ss w/ reduced   ║  │
                              │  ║ grids (iXin:3,        ║  │
                              │  ║ iNumStates:3). ~5 min.║  │
                              │  ║ Confirm output same.  ║  │
                              │  ════════════════════════    │
                              │                              │
                              │  Status: TODO                │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 4 — PIN                           │
│                                                          │
│  Test-Writer [green, sonnet]                             │
│  Tools:  Read, Glob, Grep, Bash, Write, Edit             │
│  Inputs: entry_points.md + context.md + post-cleanup code│
│  (Phase 3 only removed dead code; numerics identical)    │
│                                                          │
│  Staged test suite (fast → slow):                        │
│                                                          │
│  Stage A — Smoke tests (< 30s):                          │
│    tests/conftest.py       (session fixture, Numba init) │
│    tests/test_smoke.py     (imports, grids, par_dict)    │
│                                                          │
│  Stage B — VFI checks (~5 min, reduced grids):           │
│    tests/test_vfi.py       (solve_ss, stat_dist_finder)  │
│    tests/snapshots/*.npz   (reference arrays)            │
│                                                          │
│  Stage C — Regression (~15 min):                         │
│    tests/test_regression.py (welfare equivalents,        │
│                              hardcoded coefficients)     │
│                                                          │
│  Stage D — DEFERRED (hours, blocked):                    │
│    Full equilibrium pinning (vCoeff, homeownership,      │
│    median net worth). Blocked until calibration path     │
│    is fixed.                                             │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE 4: Stages A-C pass on post-cleanup code ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              PHASE 5 — CLEAN (iterative)                 │
│                                                          │
│  Refactorer [red, opus]                                  │
│  Tools:  Read, Glob, Grep, Bash, Write, Edit             │
│  Inputs: approved migration_plan.md + tests/ +           │
│          task_queue.md                                    │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  For each task in task_queue.md (dependency     │      │
│  │  order):                                        │      │
│  │                                                 │      │
│  │  1. Mark task IN_PROGRESS                       │      │
│  │  2. Make one atomic change                      │      │
│  │  3. Run test suite                              │      │
│  │  4a. Pass → commit, mark DONE                   │      │
│  │  4b. Fail → revert, mark BLOCKED,               │      │
│  │      log failure details, wait for human        │      │
│  │                                                 │      │
│  │  Phase 5a: Remaining quick wins (need tests)    │      │
│  │  Phase 5b: Medium effort                        │      │
│  │  ──── GATE 5: Human reviews progress ────       │      │
│  │  Phase 5c: Large refactors                      │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  After Phase 5b completes:                               │
│  → Cartographer refreshes codebase_map.md and            │
│    dependency_graph.json from current code state          │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE: Tests pass after each step.            ║      │
│  ║  BLOCKED tasks need human review before retry.║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 6 — VALIDATE                      │
│                                                          │
│  Run full regression suite on cleaned codebase.          │
│  Compare all pinned values to Phase 4 snapshots.         │
│  Verify all @njit functions compile (import all modules).│
│                                                          │
│  Cartographer refreshes all artifacts from final code.   │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE 6: Zero numerical drift + all artifacts ║      │
│  ║  consistent with current code                 ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 7 — DOCUMENT                      │
│                                                          │
│  Doc-Writer [yellow, sonnet]                             │
│  Tools:  Read, Glob, Grep, Write, Edit                   │
│  Inputs: cleaned codebase + all artifacts + context.md   │
│  Outputs:                                                │
│    README.md                        (project overview)   │
│    docs/pipeline.md                 (execution order)    │
│    docs/model.md                    (equations <> code)  │
│    Docstrings on all public functions (Google style)     │
│    Inline comments on non-obvious numerical logic        │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────────────────────────────────────┘
```

---

## Hard Rules

1. **@njit stays.** Never remove `@njit` from any function. All refactoring works within numba's type system. No dataclasses, `**kwargs`, or wrapper layers in the `@njit` call chain. (Tested: removing @njit from orchestration functions causes performance regression.)
2. **Numerical identity.** Every change in Phase 5+ is validated against pinned regression tests. `vCoeff_C`, `vCoeff_NC`, homeownership rate, and median net worth must match to 6 decimal places.
3. **Atomic commits.** Each refactoring step is one commit. If tests fail, revert and mark BLOCKED. No multi-step changes without intermediate validation.
4. **Human gates.** Six gates in the pipeline (see flowchart). No phase proceeds past a gate without human approval. Dead code deletion requires human confirmation.
5. **task_queue.md is the source of truth.** Agents read their next task from it, mark progress in it, and log completion summaries in it.
6. **Surely-safe before tests.** Phase 3 (safe cleanup) runs before Phase 4 (pin tests). Only changes with zero chance of affecting runtime behavior are allowed pre-tests: removing dead code, comments, unreachable files, and cosmetic fixes. If there is any doubt, the task waits for Phase 5.
7. **Git strategy.** All work happens on `main` with atomic commits. Each commit message references the task_queue task ID (e.g., `[3.1] Delete extensionless simulation duplicate`). Commits are individually revertible.
8. **Artifact freshness.** Cartographer refreshes `codebase_map.md` and `dependency_graph.json` after Phase 5b (medium effort) and again in Phase 6 (validate). Stale artifacts are marked with a warning header until refreshed.

---

## Escalation: When a Task Fails

```
Refactorer makes change → tests fail → revert change
  │
  ▼
Mark task BLOCKED in task_queue.md with:
  - Which test(s) failed
  - What numerical drift occurred (if any)
  - What the change was attempting
  │
  ▼
Human reviews the BLOCKED task and decides:
  a) Modify the approach and re-assign to refactorer
  b) Split into smaller sub-tasks
  c) Reject the task (mark REJECTED with reason)
  d) Accept the drift and update test baselines (rare, requires justification)
```

---

## Agents

| Agent | Color | Model | Purpose | Phases |
|-------|-------|-------|---------|--------|
| **cartographer** | blue | sonnet | Read-only codebase exploration and mapping | 1, 5 (refresh), 6 (refresh) |
| **architect** | cyan | sonnet | Design target structure and migration plan | 2 |
| **test-writer** | green | sonnet | Pin current numerical behavior | 4 |
| **refactorer** | red | opus | Execute approved changes, one at a time | 3, 5 |
| **doc-writer** | yellow | sonnet | Add documentation after cleanup | 7 |

---

## Commands

| Command | Agent | What it does |
|---------|-------|-------------|
| `/intake` | -- | Collect human context, save to `artifacts/intake.md` |
| `/map` | cartographer | Produce dependency graph, codebase map, dead code, red flags |
| `/plan` | architect | Propose target structure and migration checklist |
| `/pin` | test-writer | Create regression test suite, verify on post-cleanup code |
| `/clean $ARGS` | refactorer | Execute task(s) from task_queue.md. `$ARGS`: task number, phase, or `all` |
| `/validate` | -- | Run full test suite, report pass/fail with diffs |
| `/refresh` | cartographer | Update codebase_map.md and dependency_graph.json from current code |
| `/document` | doc-writer | Add docstrings, README, model docs |
| `/status` | -- | Show pipeline progress from task_queue.md |

---

## Artifacts

All pipeline outputs go to `artifacts/` to keep them separate from the project code:

```
artifacts/
├── intake.md                 ← project context and landmines
├── codebase_map.md           ← file-by-file inventory (refreshed after Phase 5b, 6)
├── dependency_graph.json     ← import graph with layers (refreshed after Phase 5b, 6)
├── entry_points.md           ← call chains from each entry point
├── dead_code.md              ← unreachable code inventory
├── red_flags.md              ← issues ranked by severity
├── fix_proposals.md          ← effort-tiered fix proposals with rationale
├── task_queue.md             ← SOURCE OF TRUTH for progress
├── proposed_structure.md     ← target folder layout (Phase 2)
├── migration_plan.md         ← ordered checklist, human-annotated (Phase 2)
├── naming_conventions.md     ← naming rules (Phase 2)
└── config_extraction.md      ← magic numbers to extract (Phase 2)
```

---

## Failure Modes to Watch For

- **Silent numerical drift:** Tests pass with loose tolerances but results shifted. Mitigation: pin scalar outputs with tight tolerances, arrays with `rtol=1e-6`.
- **Import cycles after reorganization:** Moving files around can create circular imports. Refactorer must check for this after each move.
- **Path-dependent execution:** Some research code relies on being run from a specific directory. Refactorer must standardize to project-root-relative paths.
- **Fragile solver convergence:** Reordering operations or changing float precision can break convergence. Refactorer only touches structure (names, file locations), never the numerics.
- **Dead code that isn't dead:** Cartographer flags it, but only the human confirms deletion. Some "unused" functions are called via string dispatch or config-driven execution.
- **Numba recompilation:** Moving @njit functions between files triggers recompilation. Verify numba can still compile all functions after each move. After Phase 5c (large refactors), explicitly import all modules and verify no compilation errors.
- **Stale artifacts:** After code changes, `codebase_map.md` and `dependency_graph.json` may not match the code. Cartographer refreshes these at defined points. Between refreshes, treat artifacts as approximate.
