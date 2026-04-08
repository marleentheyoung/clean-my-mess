# Task Queue

Source of truth for refactoring progress. Agents mark tasks DONE with a one-line summary.

**Status key:** TODO | IN_PROGRESS | DONE | BLOCKED | REJECTED

**BLOCKED** means: tests failed or human review needed. The task includes a failure log. Human decides next step (see Escalation in pipeline.md).

Phase numbers match `pipeline.md`.

**Out of scope: `full_calibration.py`** — do not refactor, do not attempt to run. It calls nonexistent functions (`initialise_coefficients_initial`, `solve_initial`) and requires `nlopt` which may not be installed. Only update its imports when other files are renamed (5b.8). The user will fix the calibration path separately.

---

## Phase 1: Intake (cartographer)

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 1.1 | Produce cartographer artifacts | DONE | cartographer | -- | All artifacts in `artifacts/`, internally consistent | Completed 2026-04-08 |
| 1.2 | Produce fix_proposals.md | DONE | cartographer | 1.1 | Proposals map to context.md with rationale | Completed 2026-04-08 |
| 1.3 | Human reviews fix_proposals.md | DONE | human | 1.2 | Each proposal marked approved/modified/rejected | Reviewed throughout session; approved with modifications (sections 3/4 rejected, Phase 4 redesigned as staged) |

---

## Phase 2: Plan (architect) — can run in parallel with Phase 3

GATE 2 (human reviews migration_plan.md) only blocks **Phase 5c** (large refactors). Phases 5a and 5b can proceed without it.

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 2.1 | Produce proposed_structure.md | DONE | architect | 1.1 | Target model/ package with 4 sub-packages, validated against dependency graph | Commit f01533f |
| 2.2 | Produce migration_plan.md | DONE | architect | 2.1 | 8-step, ~20-commit checklist ordered by dependency layer | Commit f01533f |
| 2.3 | Produce naming_conventions.md | DONE | architect | 2.1 | Hungarian notation documented, file/function rename rules, import alias standards | Commit f01533f |
| 2.4 | Produce config_extraction.md | DONE | architect | 2.1 | 48 items across 6 categories; recommends SOLVER_SETTINGS dict + par_dict additions | Commit f01533f |
| 2.5 | Human reviews migration_plan.md | TODO | human | 2.2 | Each item marked approved/modified/rejected. Only blocks Phase 5c. | -- |

---

## Phase 3: Safe Cleanup (refactorer) — surely-safe changes only, pre-tests

**Rule: Only changes with ZERO chance of affecting runtime behavior.** These remove dead code, comments, unreachable files, and cosmetic issues. Nothing that touches active code logic.

**Execution order matters:** Task 3.1 runs first (removes dead code from the largest file, which affects what subsequent tasks need to do). Then 3.2-3.6 in any order.

**Every task includes a syntax check:** After each edit, run `python -c "import <modified_module>"` to catch errors immediately. Don't wait for Gate 3.G.

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 3.1 | **Clean `solve_epsilons.py`**: remove BOTH triple-quote dead blocks (lines 108-1039 and 1040-2725), remove all 16 orphaned unused imports (dead_code.md items #20-35), remove redundant `tauch.tauchen()` call (line 76, return value overwritten on line 77). Keep only imports used by live code: numpy, time, misc_functions, tauchen, par_epsilons, grid_creation, proper_welfare_debug. | DONE | refactorer | 1.3 | 2741 → 53 lines. `python -c "import solve_epsilons"` succeeds. | Removed 2 triple-quote blocks, 16 unused imports, redundant tauchen call, commented profiling |
| 3.2 | Delete extensionless `simulation` duplicate file | DONE | refactorer | 1.3 | No subprocess/exec references. `python -c "import simulation"` succeeds. | Deleted. simulation.py remains as the canonical file. |
| 3.3 | Remove commented-out `main()` in `tauchen.py` (lines 143-167) | DONE | refactorer | 1.3 | `python -c "import tauchen"` succeeds. | Removed 26 lines of commented-out main() and if __name__ block |
| 3.4 | Remove commented-out grid alternatives in `grid_creation.py` (triple-quote block ~lines 56-104) | DONE | refactorer | 1.3 | `python -c "import grid_creation"` succeeds. | Removed 49-line triple-quote block with alternative grid logic |
| 3.5 | Fix `.xslx` -> `.xlsx` typo in **live code only** | DONE | refactorer | 3.1 | No `.xslx` references remain in codebase. | All typos were in dead code already removed by 3.1. Zero live-code typos found. |
| 3.6 | Delete unused `DoubleGrid` from `full_calibration.py` (lines 19-36) | DONE | refactorer | 1.3 | `python -c "compile(...)"` syntax check passes. (nlopt not installed, so full import fails pre-existing.) | Removed 18-line unused function definition |
| 3.G | **GATE 3: Verify Phase 3 didn't change active code.** | DONE | human | 3.1-3.6 | Git diff: 2784 lines deleted, zero content lines added. All 24 modules import successfully. | Verified via git diff (only deletions, no active code changed) + full import smoke test |

---

## Phase 4: Pin (test-writer) — staged regression tests on post-cleanup code

Tests pin the POST-cleanup code (Phase 3 only removed dead code, so numerical behavior is identical to original).

**Important constraints discovered during audit:**
- The live entry point (solve_epsilons.py:main) only produces welfare equivalents, not equilibrium outputs.
- The calibration path (full_calibration.py) is broken — calls nonexistent functions. Cannot run full equilibrium to generate vCoeff_C/vCoeff_NC from scratch.
- Running the full equilibrium solver (find_coefficients) takes hours. Not feasible for routine testing.
- The only available coefficient values are hardcoded initial guesses in solve_epsilons.py.

**Strategy: Pin what we CAN compute now. Defer what needs the equilibrium solver.**

**Prerequisite:** For best results, fill in these context.md TODOs before Phase 4:
- "What reference outputs exist?"
- "Are there intermediate checkpoints worth pinning?"
- "Which entry paths actually work right now?" (solve_epsilons.py:main → ?)
- "What does correct output look like?"

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 4.A | **Stage A — Smoke tests (< 30s).** 8 tests: all modules import, par_dict keys, parameter values, grid creation attributes, Tauchen rows sum to 1, grid shapes, vPDF_z structure, coefficient values pinned. | DONE | test-writer | 3.G | 8/8 passed in 5.6s | Created tests/conftest.py, tests/test_smoke.py |
| 4.B | **Stage B — VFI checks (~47s on reduced grids).** 4 tests: solve_ss returns finite values, output shapes correct, values nonzero, snapshot pinned. Uses iXin:3, iNumStates:3. | DONE | test-writer | 4.A | 4/4 passed in 47s. Snapshot saved to tests/snapshots/solve_ss_reduced.npz | Created tests/test_vfi.py |
| 4.C | **Stage C — Regression (~63s on full grids).** 3 tests: grid creation deterministic (snapshot), solve_ss full-grid finite, solve_ss full-grid snapshot. Slow welfare test (@pytest.mark.slow) deferred to manual runs. | DONE | test-writer | 4.B | 3/3 passed in 63s. Snapshots saved. Slow test deselected. | Created tests/test_regression.py, pytest.ini |
| 4.D | **Deferred: Full equilibrium pinning + welfare equivalents.** find_expenditure_equiv marked @pytest.mark.slow (run with `pytest -m slow`). Full equilibrium pinning blocked until calibration path is fixed. | BLOCKED | test-writer | calibration path fix | Welfare test exists but takes 10-15 min. Equilibrium test needs hours + fixed calibration path. | -- |

---

## Phase 5a: Remaining Quick Wins (refactorer) — need tests

These touch active code (renaming, interface changes) so they require the test suite.

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 5a.1 | **Name magic numbers as constants.** | DONE | refactorer | 4.B | 15/15 tests pass. `grep -rn "-1e12" clean_the_mess/` shows only NEG_INF definitions. | Extracted NEG_INF (-1e12) across 6 files (~30 occurrences), PRICE_TOL/ERROR_TOL/MAX_ITERATIONS/SECANT_STEP in equilibrium.py, N_CONSUMPTION_NODES in stayer_problem.py and stayer_problem_renter.py. Commit ef6fa4d. |
| 5a.2 | **Unify `LoM_C`/`LoM_NC` into single `LoM` function.** | DONE | refactorer | 4.B | 15/15 tests pass. Single LoM function with LoM_C/LoM_NC aliases for backward compatibility. | Unified to `LoM(grids, t_index, vCoeff)`. 52 call sites unchanged via aliases. Commit 0875202. |

---

## Phase 5b: Medium Effort (refactorer) — needs tests, does NOT need GATE 2

**Refactorer rule for rename tasks:** When renaming files, update imports in BOTH source code AND `tests/` files. Verify with `pytest --collect-only` (import check without running tests).

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 5b.1 | **Consolidate duplicate interpolation into `interp.py`.** | DONE | refactorer | 4.B | 15/15 tests pass. misc_functions.py re-exports from interp.py for backward compat. | Moved 2D, 4D interp + binary_search_sim to interp.py. Deleted duplicates from misc_functions.py. Re-export for compat. Commit 9b79747. |
| 5b.2 | **Document misc_functions.py split for Phase 5c.** | DONE | refactorer | 5b.1 | misc_functions.py has re-exports + utility functions only. | Physical move of ols_numba/net_income deferred to 5c.2 (package restructure). Re-exports already in place. Commit b86982a. |
| 5b.3 | **Resolve `grids.PDF_z` vs `vPDF_z` bug — DEAD CODE.** | DONE | refactorer | 4.B | Grep confirmed zero callers for all 4 functions. | Deleted W_bequest_flooddamage, Q_bequest_flooddamage, W_bequest_noflooddamage, Q_bequest_noflooddamage from utility.py. Commit 9b79747. |
| 5b.5 | **Extract rental_price_calc() helper.** | DONE | refactorer | 4.B | 15/15 tests pass. 4 inline formulas in simulation.py replaced. | Added rental_price_calc(par, dP, dP_prime, damage_frac) to utility.py. Left stayer_problem_renter.py/moments.py untouched (variant without max floor). Commit 9b79747. |
| 5b.6 | **Add numerical guards to interpolation.** | DONE | refactorer | 5b.1 | 15/15 tests pass. 4 division guards added. | Added max(denom, 1e-15) to all nom/denom in interp.py. Commit b86982a. |
| 5b.7 | **Rename `proper_welfare_debug.py` → `welfare.py`.** | DONE | refactorer | 4.B | All imports updated in source + tests. | Updated 1 import in solve.py + test files. Commit 9b79747. |
| 5b.8 | **Remove `_epsilons` suffix from 6 files.** | DONE | refactorer | 5b.2, 5b.7 | 15/15 tests pass. Zero old references remain. | Renamed par_epsilons→par, utility_epsilons→utility, buyer_problem_epsilons→buyer_problem, household_problem_epsilons_nolearning→household_problem, solve_epsilons→solve, LoM_epsilons→lom. ~25 imports updated. Commit d63a0e8. |
| 5b.R | **Cartographer refreshes codebase_map.md and dependency_graph.json** | DONE | cartographer | 5b.8 | 25 files, 6221 lines, 73 edges, 7 layers. All renames reflected. | Commit 8f991ea |
| 5b.G | **GATE 5: Human reviews progress before large refactors** | TODO | human | 5b.R | Approves proceeding to Phase 5c | -- |

---

## Phase 5c: Large Refactors (refactorer) — full test suite required, needs GATE 2 for 5c.2

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 5c.1 | **Create single config source (`config.py`).** Merge parameter definitions from `par.py` (was `par_epsilons.py`), `full_calibration.py`, and `grid_creation.py` into one `config.py` with a `create_par_dict()` function. Move the `vPi_S_median` mutation (currently at module level in `par.py` line 17) inside the function. Calibration overrides specific values by modifying the returned dict. Must produce a numba-compatible dict (passed through `construct_jitclass`). | TODO | refactorer | 5b.G | Tests pass; all parameter values identical; no module-level mutation; `par.py` and `full_calibration.py` import from `config.py`. | -- |
| 5c.2 | **Full package restructure into `model/`.** Follow `context.md` section 8 target structure. Create `model/` package with `household/`, `simulation/`, `equilibrium/`, `analysis/` subpackages. Move all files per the architect's `migration_plan.md`. Update ALL imports in source AND tests. Verify: `python -c "import model"` succeeds (Numba compiles all @njit functions across modules). Check for circular imports. | TODO | refactorer | 5c.1, **2.5** | Full test suite passes; all imports resolve; `python -c "import model"` succeeds; no circular imports; `pytest` passes. | -- |

---

## Phase 6: Validate

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 6.1 | Run full regression suite | TODO | -- | 5c.2 | Zero numerical drift from Phase 4 snapshots | -- |
| 6.2 | Verify all @njit functions compile | TODO | -- | 5c.2 | Import all modules; no Numba compilation errors | -- |
| 6.3 | Cartographer refreshes all artifacts | TODO | cartographer | 6.1 | All artifacts match final code state | -- |

---

## Phase 7: Document

| # | Task | Status | Agent | Depends On | Validation | Summary |
|---|------|--------|-------|------------|------------|---------|
| 7.1 | Add docstrings to all public functions | TODO | doc-writer | 6.3 | Google style; Numba types documented | -- |
| 7.2 | Write README.md | TODO | doc-writer | 6.3 | Setup, how to run, expected outputs | -- |
| 7.3 | Write docs/model.md | TODO | doc-writer | 6.3 | Equations mapped to code | -- |

---

## REJECTED

| # | Task | Status | Reason |
|---|------|--------|--------|
| R1 | Remove @njit from orchestration functions | REJECTED | Performance regression -- njit boundary crossing overhead in equilibrium iteration loop |
| R2 | Introduce Python dataclasses for argument bundling | REJECTED | Depends on R1; dataclasses incompatible with @njit call chain |
