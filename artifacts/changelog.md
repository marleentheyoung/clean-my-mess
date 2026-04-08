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
