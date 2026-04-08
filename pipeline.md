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
│  (model description, architecture, known problems)       │
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
│  Status: DONE                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 2 — PLAN                          │
│                                                          │
│  Architect [cyan, sonnet]                                │
│  Tools:  Read, Glob, Grep, Write                         │
│  Inputs: all cartographer artifacts + context.md         │
│  Outputs:                                                │
│    artifacts/proposed_structure.md   (target layout)     │
│    artifacts/migration_plan.md      (ordered checklist)  │
│    artifacts/naming_conventions.md  (naming rules)       │
│    artifacts/config_extraction.md   (magic numbers)      │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE: Human reviews migration_plan.md        ║      │
│  ║  Marks each item: approved / modified / reject ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 3 — PIN                           │
│                                                          │
│  Test-Writer [green, sonnet]                             │
│  Tools:  Read, Glob, Grep, Bash, Write, Edit             │
│  Inputs: entry_points.md + intake.md + unmodified code   │
│  Outputs:                                                │
│    tests/conftest.py                (fixtures, seeds)    │
│    tests/test_regression.py         (pinned outputs)     │
│    tests/test_fast.py               (quick subset)       │
│    tests/snapshots/                 (reference values)   │
│                                                          │
│  Pins: vCoeff_C, vCoeff_NC, homeownership rate,         │
│        median net worth (match to 6 decimal places)      │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE: All tests pass on unmodified codebase  ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              PHASE 4 — CLEAN (iterative)                 │
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
│  │  4a. Tests pass → commit, mark DONE             │      │
│  │  4b. Tests fail → revert, flag for human        │      │
│  │                                                 │      │
│  │  Phase 4a: Quick wins     (tasks 2.1–2.9)      │      │
│  │  Phase 4b: Medium effort  (tasks 3.1–3.8)      │      │
│  │  Phase 4c: Large refactor (tasks 4.1–4.2)      │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE: Full test suite passes after each step ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 5 — VALIDATE                      │
│                                                          │
│  Run full regression suite on cleaned codebase.          │
│  Compare all pinned values to Phase 3 snapshots.         │
│                                                          │
│  ══════════════════════════════════════════════════       │
│  ║  GATE: Zero numerical drift                   ║      │
│  ══════════════════════════════════════════════════       │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  PHASE 6 — DOCUMENT                      │
│                                                          │
│  Doc-Writer [yellow, sonnet]                             │
│  Tools:  Read, Glob, Grep, Write, Edit                   │
│  Inputs: cleaned codebase + all artifacts + context.md   │
│  Outputs:                                                │
│    README.md                        (project overview)   │
│    docs/pipeline.md                 (execution order)    │
│    docs/model.md                    (equations ↔ code)   │
│    Docstrings on all public functions (Google style)     │
│    Inline comments on non-obvious numerical logic        │
│                                                          │
│  Status: TODO                                            │
└──────────────────────────────────────────────────────────┘
```

---

## Hard Rules

1. **@njit stays.** Never remove `@njit` from any function. All refactoring works within numba's type system. No dataclasses, `**kwargs`, or wrapper layers in the `@njit` call chain. (Tested: removing @njit from orchestration functions causes performance regression.)
2. **Numerical identity.** Every change is validated against pinned regression tests. `vCoeff_C`, `vCoeff_NC`, homeownership rate, and median net worth must match to 6 decimal places.
3. **Atomic commits.** Each refactoring step is one commit. If tests fail, revert. No multi-step changes without intermediate validation.
4. **Human gates.** The migration plan requires explicit human approval before code changes begin. Dead code deletion requires human confirmation.
5. **task_queue.md is the source of truth.** Agents read their next task from it, mark progress in it, and log completion summaries in it.

---

## Agents

| Agent | Color | Model | Purpose | Phase |
|-------|-------|-------|---------|-------|
| **cartographer** | blue | sonnet | Read-only codebase exploration and mapping | 1 |
| **architect** | cyan | sonnet | Design target structure and migration plan | 2 |
| **test-writer** | green | sonnet | Pin current numerical behavior | 3 |
| **refactorer** | red | opus | Execute approved changes, one at a time | 4 |
| **doc-writer** | yellow | sonnet | Add documentation after cleanup | 6 |

---

## Commands

| Command | Agent | What it does |
|---------|-------|-------------|
| `/intake` | -- | Collect human context, save to `artifacts/intake.md` |
| `/map` | cartographer | Produce dependency graph, codebase map, dead code, red flags |
| `/plan` | architect | Propose target structure and migration checklist |
| `/pin` | test-writer | Create regression test suite, verify on unmodified code |
| `/clean $ARGS` | refactorer | Execute task(s) from task_queue.md. `$ARGS`: task number, phase, or `all` |
| `/validate` | -- | Run full test suite, report pass/fail with diffs |
| `/document` | doc-writer | Add docstrings, README, model docs |
| `/status` | -- | Show pipeline progress from task_queue.md |

---

## Artifacts

All pipeline outputs go to `artifacts/` to keep them separate from the project code:

```
artifacts/
├── intake.md                 ← project context and landmines
├── codebase_map.md           ← file-by-file inventory
├── dependency_graph.json     ← import graph with layers
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
- **Numba recompilation:** Moving @njit functions between files triggers recompilation. Verify numba can still compile all functions after each move.
