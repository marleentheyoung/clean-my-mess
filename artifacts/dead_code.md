# Dead Code Inventory

## Summary

| # | File | Lines | Type | Description | Confidence |
|---|------|-------|------|-------------|------------|
| 1 | `solve_epsilons.py` | 108-1039 | triple-quote block comment | First `"""..."""` block; contains calls to `household_problem.solve_ss`, `equil.find_coefficients`, `experiments.full_information_experiment`, plotting, `.xslx` file writes, and more. ~931 lines. | HIGH |
| 2 | `solve_epsilons.py` | 1040-2725 | triple-quote block comment | Second `"""..."""` block (line 1040 opens, line 2725 closes). Contains ~1,685 lines of old plotting code, distribution analysis, excess demand checks, welfare comparisons, and simulation loops. References undefined variables (`dPi_L`, `iNj`, `results_vector`, `dPi_S`, `vt_buy_c`, `vt_buy_nc`, `a_index`, `alpha`, `err`, `learning`, `Line2D`). All inside a string literal so Python treats it as a no-op expression. | HIGH |
| 4 | `solve_epsilons.py` | 50-63 | commented-out block | Multiple commented-out `vCoeff_C_initial`/`vCoeff_NC_initial` alternative values (7 lines of `#`-prefixed code) | HIGH |
| 5 | `solve_epsilons.py` | 80-93 | commented-out block | Commented-out Chebyshev coefficient initialization (~14 lines) | HIGH |
| 6 | `solve_epsilons.py` | 199-201 | commented-out block | 3 lines of commented-out welfare calculator call | HIGH |
| 7 | `solve_epsilons.py` | 227-231 | commented-out block | 5 lines of commented-out `find_coefficients` call with `.xslx` output | HIGH |
| 8 | `solve_epsilons.py` | 284-296 | commented-out block | ~13 lines of commented-out simulation and plotting code | HIGH |
| 9 | `solve_epsilons.py` | 340-351 | commented-out block | ~12 lines of commented-out welfare calculator calls | HIGH |
| 10 | `solve_epsilons.py` | 354-398 | commented-out block | ~45 lines of commented-out variable initializations and old code | HIGH |
| 11 | `solve_epsilons.py` | 400-418 | commented-out block | ~19 lines of commented-out equilibrium calls and timing | HIGH |
| 12 | `solve_epsilons.py` | 420-445 | commented-out block | ~26 lines of commented-out VFI solve and plotting loop | HIGH |
| 13 | `solve_epsilons.py` | 650-651 | commented-out block | 2 lines commented-out print statements | LOW |
| 14 | `solve_epsilons.py` | 679-699 | commented-out block | ~21 lines of commented-out density transform code | HIGH |
| 15 | `solve_epsilons.py` | 707-727 | commented-out block | ~21 lines of commented-out density transform code (near-duplicate of #14) | HIGH |
| 16 | `solve_epsilons.py` | 786-793 | commented-out block | ~8 lines of commented-out simulation and equilibrium code | HIGH |
| 17 | `solve_epsilons.py` | 894-899 | commented-out block | ~6 lines of commented-out LoM prints | LOW |
| 18 | `solve_epsilons.py` | 900-986 | commented-out block | ~87 lines of commented-out path simulation loop | HIGH |
| 19 | `solve_epsilons.py` | 988-1038 | commented-out block | ~51 lines of commented-out iteration/equilibrium/plotting code | HIGH |
| 20 | `solve_epsilons.py` | 12-13, 36 | unused import | `import numba as nb` (line 13), `import pandas as pd` (line 14, duplicated on line 36) -- `nb` and duplicate `pd` never used | HIGH |
| 21 | `solve_epsilons.py` | 17 | unused import | `from numba import njit` -- no `@njit` in this file | HIGH |
| 22 | `solve_epsilons.py` | 19 | unused import | `import grids as grid` -- `grid.` never referenced in live code | HIGH |
| 23 | `solve_epsilons.py` | 22 | unused import | `import simulate_initial_joint as initial_joint` -- `initial_joint.` never referenced | HIGH |
| 24 | `solve_epsilons.py` | 28 | unused import | `import quantecon as qe` -- `qe.` never referenced | HIGH |
| 25 | `solve_epsilons.py` | 29 | unused import | `import utility_epsilons as ut` -- `ut.` never referenced in live code | HIGH |
| 26 | `solve_epsilons.py` | 30 | unused import | `import interp as interp` -- only referenced in commented-out code | HIGH |
| 27 | `solve_epsilons.py` | 31 | unused import | `import buyer_problem_simulation as buy_sim` -- `buy_sim.` never referenced | HIGH |
| 28 | `solve_epsilons.py` | 32 | unused import | `import continuation_value_nolearning as continuation_value_epsilons` -- never referenced | HIGH |
| 29 | `solve_epsilons.py` | 33 | unused import | `import stayer_problem as stayer_problem` -- never referenced | HIGH |
| 30 | `solve_epsilons.py` | 34 | unused import | `import stayer_problem_renter as stayer_problem_renter` -- never referenced | HIGH |
| 31 | `solve_epsilons.py` | 35 | unused import | `import buyer_problem_epsilons as buyer_problem_epsilons` -- never referenced | HIGH |
| 32 | `solve_epsilons.py` | 41 | unused import | `from numba import config` -- `config.` never referenced | HIGH |
| 33 | `solve_epsilons.py` | 42 | unused import | `from scipy.stats import norm` -- `norm.` never referenced | HIGH |
| 34 | `solve_epsilons.py` | 39 | unused import | `import moments as mom` -- used once on line 1619 which is in dead code | HIGH |
| 35 | `solve_epsilons.py` | 45 | unused import | `import plot_creation as plot_creation` -- `plot_creation.` never referenced | HIGH |
| 36 | `solve_epsilons.py` | 76 | redundant call | `tauch.tauchen(...)` return value `mMarkov` is immediately overwritten on line 77 by `grid_creation.create(par)` | HIGH |
| 37 | `simulation` (no .py) | 1-1230+ | duplicate file | File without `.py` extension with identical content to `simulation.py`. Confirmed by matching line-for-line content on lines 1-50 and matching grep results (`vPDF_z` references at identical line numbers). Cannot be imported by Python. | HIGH |
| 38 | `tauchen.py` | 143-167 | commented-out block | Commented-out `main()` function and `if __name__` block (~25 lines) | HIGH |
| 39 | `grid_creation.py` | 56-104 | triple-quote block comment | `"""..."""` wrapping ~49 lines of alternative grid logic with `iNb_left_tail`, `iNb_left`, `iNb_right` parameters | HIGH |
| 40 | `full_calibration.py` | 19-36 | duplicate function | `DoubleGrid(vA, vH)` is a non-jitted duplicate of `misc_functions.py:DoubleGrid()` (line 12). The one in `full_calibration.py` is never called anywhere. | HIGH |
| 41 | `par_epsilons.py` | 12-15 | commented-out block | 4 lines of commented-out alternative `vPi_S_median` array | MEDIUM |
| 42 | `moments.py` | 50-57 | triple-quote block comment | `"""..."""` wrapping type-amenity moment computation | HIGH |
| 43 | `moments.py` | 62-109 | triple-quote block comment | `"""..."""` wrapping old moments dict and per-type shares | HIGH |
| 44 | `equilibrium.py` | 85-112 | commented-out block | ~28 lines of commented-out debug prints and timing code | MEDIUM |
| 45 | `grid_creation.py` | 106, 109 | commented-out lines | Alternative grid definitions commented out with `#` | LOW |

## Line Count Estimate

- `solve_epsilons.py`: ~2,680 lines in `main()`, of which ~60 are live. Lines 108-1039 and 1040-2725 are two `"""..."""` block comments = ~2,617 dead lines
- `simulation` (no extension): ~1,230 lines, entire file is a duplicate
- `grid_creation.py`: ~49 lines dead (block comment)
- `tauchen.py`: ~25 lines dead (commented main)
- `full_calibration.py`: ~18 lines dead (unused DoubleGrid)
- `moments.py`: ~55 lines dead (block comments)
- Other scattered: ~100 lines

**Total estimated dead code: ~4,100 lines out of ~9,200 (approximately 45%)**
