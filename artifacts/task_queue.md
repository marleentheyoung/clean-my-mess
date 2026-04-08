# Task Queue

Source of truth for refactoring progress. Agents mark tasks DONE with a one-line summary.

**Status key:** TODO | IN_PROGRESS | DONE | REJECTED

---

## Phase 0: Diagnostics (no code changes)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 0.1 | Produce cartographer artifacts (codebase_map, dependency_graph, entry_points, dead_code, red_flags) | DONE | cartographer | -- | All artifacts present in `artifacts/`, internally consistent | -- |
| 0.2 | Produce fix_proposals.md | DONE | cartographer | 0.1 | Proposals map to context.md sections with rationale | -- |
| 0.3 | Human reviews fix_proposals.md and marks approved/rejected | TODO | human | 0.2 | Annotations in fix_proposals.md or task_queue.md | -- |

## Phase 1: Regression Tests (must complete before any code changes)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 1.1 | Build regression test suite pinning vCoeff_C, vCoeff_NC, homeownership rate, median net worth | TODO | test-writer | 0.3 | All tests pass on unmodified codebase | -- |
| 1.2 | Create fast subset tests (grid creation, single VFI step, single equilibrium iteration) | TODO | test-writer | 1.1 | Fast tests pass in < 60s | -- |

## Phase 2: Quick Wins (near-zero risk, no test suite required but run if available)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 2.1 | Delete extensionless `simulation` duplicate file | TODO | refactorer | 0.3 | No file references `simulation` without `.py`; model runs | -- |
| 2.2 | Remove dead code from `solve_epsilons.py` (~2600 lines) | TODO | refactorer | 0.3 | Active path runs, identical output | -- |
| 2.3 | Remove commented-out `main()` in `tauchen.py` | TODO | refactorer | 0.3 | File imports correctly | -- |
| 2.4 | Remove commented-out grid alternatives in `grid_creation.py` | TODO | refactorer | 0.3 | No runtime change | -- |
| 2.5 | Fix duplicate imports in `solve_epsilons.py` | TODO | refactorer | 0.3 | No runtime change | -- |
| 2.6 | Fix `.xslx` -> `.xlsx` typo | TODO | refactorer | 0.3 | Output files have correct extension | -- |
| 2.7 | Name magic numbers as constants (multiple files) | TODO | refactorer | 0.3 | Identical numerical output | -- |
| 2.8 | Unify `LoM_C`/`LoM_NC` into single `LoM` function | TODO | refactorer | 0.3 | Call with same inputs, identical output | -- |
| 2.9 | Delete unused `DoubleGrid` from `full_calibration.py` | TODO | refactorer | 0.3 | Grep confirms no calls to this copy | -- |

## Phase 3: Medium Effort (low-medium risk, test suite recommended)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 3.1 | Consolidate duplicate interpolation into `interp.py` | TODO | refactorer | 1.1, 2.9 | Full model output identical; all imports resolve | -- |
| 3.2 | Split `misc_functions.py` by purpose | TODO | refactorer | 3.1 | All imports resolve; full model identical | -- |
| 3.3 | Verify and fix `grids.PDF_z` vs `vPDF_z` bug | TODO | refactorer | 1.1 | Either dead code confirmed or attribute fixed | -- |
| 3.4 | Document `fastmath` difference between mortgage_choice variants | TODO | doc-writer | 0.1 | Documentation in code comments or artifacts | -- |
| 3.5 | Extract duplicated rental price calculation into shared helper | TODO | refactorer | 1.1 | Rental prices match pre-refactor | -- |
| 3.6 | Add numerical guards to unguarded divisions | TODO | refactorer | 1.1 | No change when grids valid; graceful failure otherwise | -- |
| 3.7 | Rename `proper_welfare_debug.py` -> `welfare.py` | TODO | refactorer | 1.1 | All imports resolve | -- |
| 3.8 | Remove `_epsilons` suffix from filenames | TODO | refactorer | 3.2, 3.7 | All imports resolve; full model identical | -- |

## Phase 4: Large Refactors (high risk, full test suite required)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 4.1 | Create single config source (`config.py`) | TODO | refactorer | 1.1, 3.8 | All parameter values match; full model identical | -- |
| 4.2 | Full package restructure into `model/` | TODO | refactorer | 4.1 | Full regression suite passes; all imports resolve; numba compiles | -- |

## REJECTED

| # | Task | Status | Reason |
|---|------|--------|--------|
| R1 | Remove @njit from orchestration functions | REJECTED | Performance regression -- njit boundary crossing overhead in equilibrium iteration loop |
| R2 | Introduce Python dataclasses for argument bundling | REJECTED | Depends on R1; dataclasses incompatible with @njit call chain |
