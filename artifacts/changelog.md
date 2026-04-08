# Changelog

Detailed record of every code change made during cleanup. Each entry references a task_queue.md task ID.

For the exact diff of any change, use `git show <commit>`.

---

## Phase 3: Safe Cleanup

### [3.1] Clean solve_epsilons.py
**Date:** 2026-04-08
**Files:** `clean_the_mess/solve_epsilons.py`
**Changes:**
- Removed triple-quote dead block lines 108-1039 (~931 lines)
- Removed triple-quote dead block lines 1040-2725 (~1685 lines)
- Removed 16 unused imports (lines 13-45): `numba as nb`, `matplotlib.pyplot`, `njit`, `grids as grid`, `simulate_initial_joint`, `household_problem_epsilons_nolearning`, `simulation`, `equilibrium`, `LoM_epsilons`, `quantecon`, `utility_epsilons`, `interp`, `buyer_problem_simulation`, `continuation_value_nolearning`, `stayer_problem`, `stayer_problem_renter`, `buyer_problem_epsilons`, duplicate `pandas as pd`, `numba.config`, `scipy.stats.norm`, `moments as mom`, `plot_creation`, `experiments`
- Removed redundant `tauch.tauchen()` call (line 76, return value overwritten by grid_creation.create on line 77)
- Removed commented-out profiling in `if __name__` block
- **Result:** 2741 → 53 lines. Only live imports remain: numpy, time, misc_functions, tauchen, par_epsilons, grid_creation, proper_welfare_debug

### [3.2] Delete extensionless simulation duplicate
**Date:** 2026-04-08
**Files:** `clean_the_mess/simulation` (deleted)
**Changes:**
- Deleted 1264-line file without `.py` extension (not importable by Python, near-duplicate of simulation.py)

### [3.3] Remove commented-out main() in tauchen.py
**Date:** 2026-04-08
**Files:** `clean_the_mess/tauchen.py`
**Changes:**
- Removed lines 143-167: commented-out `main()` function and `if __name__` block (26 lines of `#`-prefixed code)

### [3.4] Remove commented-out grid alternatives in grid_creation.py
**Date:** 2026-04-08
**Files:** `clean_the_mess/grid_creation.py`
**Changes:**
- Removed lines 56-104: triple-quote block containing alternative grid construction logic with `iNb_left_tail`, `iNb_left`, `iNb_right` parameters (49 lines)

### [3.5] Fix .xslx typo
**Date:** 2026-04-08
**Files:** (none — all typos were in dead code removed by task 3.1)
**Changes:**
- No action needed. All `.xslx` references were inside the dead code blocks already removed.

### [3.6] Delete unused DoubleGrid from full_calibration.py
**Date:** 2026-04-08
**Files:** `clean_the_mess/full_calibration.py`
**Changes:**
- Removed lines 19-36: `DoubleGrid(vA, vH)` function definition (18 lines). Never called within this file; canonical version is in misc_functions.py.

---

## Phase 4: Pin (test suite)

### [4.A] Stage A — Smoke tests
**Date:** 2026-04-08
**Files created:**
- `tests/__init__.py` (empty)
- `tests/conftest.py` — session-scoped fixtures for par, grids, mMarkov (full and reduced)
- `tests/test_smoke.py` — 8 tests: module imports, par_dict keys, parameter values, grid attributes, Tauchen matrix, grid shapes, vPDF_z structure, coefficient values
- `pytest.ini` — configures `slow` marker, excludes slow tests by default

### [4.B] Stage B — VFI checks
**Date:** 2026-04-08
**Files created:**
- `tests/test_vfi.py` — 4 tests: solve_ss finite output, output shapes, nonzero values, snapshot comparison
- `tests/snapshots/solve_ss_reduced.npz` — reference snapshot for reduced-grid solve_ss

### [4.C] Stage C — Regression
**Date:** 2026-04-08
**Files created:**
- `tests/test_regression.py` — 3 fast tests (grid creation snapshot, solve_ss full-grid finite, solve_ss full-grid snapshot) + 1 slow test (find_expenditure_equiv, marked @pytest.mark.slow)
- `tests/snapshots/grid_creation.npz` — reference snapshot for grid creation
- `tests/snapshots/solve_ss_full.npz` — reference snapshot for full-grid solve_ss

---

## Phase 5a: Remaining Quick Wins

### [5a.1] Name magic numbers as constants
**Date:** 2026-04-08
**Commit:** ef6fa4d
**Files modified (9):**

**equilibrium.py:**
- Lines 523-525: `price_tol=1e-3`, `error_tol=1e-5`, `max_iterations=15` → `PRICE_TOL`, `ERROR_TOL`, `MAX_ITERATIONS`
- Lines 529-537: `0.005` perturbation step → `SECANT_STEP = 0.005`
- Updated all references within `house_prices_algorithm` (lines 550-633)

**buyer_problem_epsilons.py:** Added `NEG_INF = -1e12`, replaced 2 occurrences of `*-1e12`
**buyer_problem_simulation.py:** Added `NEG_INF = -1e12`, replaced 1 occurrence
**mortgage_choice_simulation.py:** Added `NEG_INF = -1e12`, replaced 1 `*-1e12` + changed `-1e12+1e-8` → `NEG_INF+1e-8` on line 76
**mortgage_choice_simulation_exc.py:** Added `NEG_INF = -1e12`, replaced 2 occurrences
**simulation.py:** Added `NEG_INF = -1e12`, replaced 8 `*-1e12` + 2 `=-1e12` assignments
**continuation_value_nolearning.py:** Added `NEG_INF = -1e12`, replaced 11 occurrences (mix of `= -1e12` and `=-1e12`)
**stayer_problem.py:** Added `N_CONSUMPTION_NODES = 100`, replaced `100` in `nonlinspace_jit` call
**stayer_problem_renter.py:** Added `N_CONSUMPTION_NODES = 100`, replaced `100` in `nonlinspace_jit` call

**Not touched:** `par_epsilons.py` (calibration outputs, not magic numbers)

### [5a.2] Unify LoM_C/LoM_NC into single LoM function
**Date:** 2026-04-08
**Commit:** 0875202
**Files modified (1):**

**LoM_epsilons.py:**
- Replaced two identical functions `LoM_C(grids, t_index, vCoeff_C)` and `LoM_NC(grids, t_index, vCoeff_NC)` with single `LoM(grids, t_index, vCoeff)`
- Added `LoM_C = LoM` and `LoM_NC = LoM` aliases for backward compatibility
- 52 call sites across 7 files (household_problem, equilibrium, simulation, moments, plot_creation, full_calibration, experiments) continue to work unchanged via aliases
- Aliases to be removed when callers are updated in Phase 5b.8 (file renames)

---

## Phase 5b: Medium Effort

### [5b.1] Consolidate duplicate interpolation into interp.py
**Date:** 2026-04-08
**Commit:** 9b79747
**Files modified:** `interp.py`, `misc_functions.py`
- Moved `_interp_2d`/`interp_2d`, `_interp_4d`/`interp_4d`, `binary_search_sim` from misc_functions.py to interp.py
- Deleted duplicate `_interp_3d`/`interp_3d`/`binary_search` from misc_functions.py
- misc_functions.py now re-exports interpolation functions from interp.py for backward compatibility
- Restored original `construct_jitclass` implementation (using `nb.typeof()`) after rewrite broke Numba compilation

### [5b.2] Document misc_functions.py split
**Date:** 2026-04-08
**Commit:** b86982a
- Physical move of `ols_numba` and `net_income` deferred to Phase 5c package restructure
- Re-exports already in place from 5b.1; all callers continue to use `misc.` prefix

### [5b.3] Delete dead bequest functions (PDF_z bug resolved)
**Date:** 2026-04-08
**Commit:** 9b79747
**Files modified:** `utility.py` (was utility_epsilons.py)
- Grep confirmed zero callers for `W_bequest_flooddamage`, `Q_bequest_flooddamage`, `W_bequest_noflooddamage`, `Q_bequest_noflooddamage`
- Deleted all 4 functions (~30 lines). The `grids.PDF_z` bug is moot — code was dead.

### [5b.5] Extract rental_price_calc() helper
**Date:** 2026-04-08
**Commit:** 9b79747
**Files modified:** `utility.py`, `simulation.py`
- Added `rental_price_calc(par, dP, dP_prime, damage_frac)` @njit function to utility.py
- Replaced 4 inline rental price formulas in simulation.py (2 blocks x 2 markets) with `ut.rental_price_calc()`
- Left `stayer_problem_renter.py` and `moments.py` untouched — they use a variant without `max(...,0)` floor

### [5b.6] Add numerical guards to interpolation
**Date:** 2026-04-08
**Commit:** b86982a
**Files modified:** `interp.py`
- Added `max(denom, 1e-15)` to all 4 `nom/denom` patterns in `_interp_1d`, `_interp_2d`, `_interp_3d`, `_interp_4d`
- Prevents silent NaN/Inf from coincident grid points

### [5b.7] Rename proper_welfare_debug.py → welfare.py
**Date:** 2026-04-08
**Commit:** 9b79747
- `mv proper_welfare_debug.py welfare.py`
- Updated import in `solve.py` and test files

### [5b.8] Remove _epsilons suffix from filenames
**Date:** 2026-04-08
**Commit:** d63a0e8
**Files renamed (6):**
- `par_epsilons.py` → `par.py`
- `utility_epsilons.py` → `utility.py`
- `buyer_problem_epsilons.py` → `buyer_problem.py`
- `household_problem_epsilons_nolearning.py` → `household_problem.py`
- `solve_epsilons.py` → `solve.py`
- `LoM_epsilons.py` → `lom.py`

**Imports updated (~25+ across ~15 source files + test files).** `full_calibration.py` imports updated per out-of-scope rule (imports only, no refactoring). `buyer_problem_epsilons` alias in household_problem.py updated to `buyer_problem`.
