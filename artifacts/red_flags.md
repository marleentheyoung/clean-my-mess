# Red Flags

Issues ranked by severity with file, line numbers, and impact.

---

## CRITICAL -- May produce wrong results silently

### 1. `grids.PDF_z` vs `grids.vPDF_z` attribute mismatch in bequest functions
- **File:** `utility_epsilons.py`, lines 54 and 71
- **Description:** `W_bequest_flooddamage()` (line 54) and `Q_bequest_flooddamage()` (line 71) reference `grids.PDF_z[damage_index]`, but the grid object created in `grid_creation.py` (line 133) stores the field as `grids.vPDF_z`. Every other file in the codebase uses `grids.vPDF_z` correctly (confirmed in `continuation_value_nolearning.py`, `simulation.py`, `moments.py`, `stayer_problem_renter.py`, `plot_creation.py`, `proper_welfare_debug.py`).
- **Impact:** Calling `W_bequest_flooddamage` or `Q_bequest_flooddamage` will raise an `AttributeError` at runtime. If Numba caches a stale version, this could silently use wrong probabilities. These functions compute bequest utility under flood damage, which feeds into the continuation value for coastal homeowners.

### 2. `equil.initialise_coefficients_initial()` does not exist
- **File:** `full_calibration.py`, line 138; also `solve_epsilons.py`, line 203
- **Description:** Both files call `equil.initialise_coefficients_initial(par, grids, method, dPi_L, ...)` but `equilibrium.py` only defines `initialise_coefficients_ss()` (line 200). The function signature also differs: the real function takes `(par, grids, method, iNj, mMarkov, vCoeff_C_ss, vCoeff_NC_ss, initial, sceptics)` but the calls pass `dPi_L` as the 4th argument where `iNj` is expected.
- **Impact:** Both the calibration path and the main solve path will crash with `AttributeError` at runtime. This means the calibration cannot currently be run.

### 3. `household_problem.solve_initial()` does not exist
- **File:** `full_calibration.py`, line 139; `solve_epsilons.py`, line 449
- **Description:** Calls `household_problem.solve_initial(grids, par, dPi_L, par.iNj, mMarkov, ...)` but `household_problem_epsilons_nolearning.py` only defines `solve()` (line 27) and `solve_ss()` (line 170). There is no `solve_initial` method.
- **Impact:** Runtime crash. The calibration path is broken at this step too.

### 4. `fastmath=True` vs `fastmath` absent -- inconsistent numerical behavior
- **File:** `mortgage_choice_simulation.py`, line 7 vs `mortgage_choice_simulation_exc.py`, line 7
- **Description:** `mortgage_choice_simulation.py` uses `@njit(fastmath=True)` while `mortgage_choice_simulation_exc.py` uses plain `@njit`. Both files implement the same `solve()` function with near-identical logic for mortgage choice during simulation.
- **Impact:** `fastmath=True` allows the compiler to reorder floating-point operations, which can change results for edge cases (e.g., near-zero denominators, subtractive cancellation in mortgage calculations). The two variants will produce subtly different numerical results on the same inputs. Since `simulation.py` imports both (lines 15-16), the choice of which gets called can silently affect equilibrium outcomes.

---

## HIGH -- Magic numbers, unguarded divisions, mutation hazards

### 5. Module-level array mutation in `par_epsilons.py`
- **File:** `par_epsilons.py`, line 17
- **Description:** `vPi_S_median=1-(1-vPi_S_median)**time_increment` mutates the module-level array. Since `par_epsilons.py` is imported by `solve_epsilons.py` (line 21), any subsequent import or reference to `parfile.vPi_S_median` sees the transformed values. But `full_calibration.py` redefines its own `vPi_S_median` locally (line 58) and applies the same transformation (line 62), so the two entry points may use different values if `par_epsilons` was previously imported.
- **Impact:** Python module caching means the transformation on line 17 runs exactly once per process. But if anyone references `vPi_S_median` before line 17 executes (e.g., in a REPL or test), they get the raw annual probabilities instead of biennial ones. This is a fragile design.

### 6. Unguarded division in interpolation functions
- **File:** `interp.py`, lines 50, 56, 113, 123; `misc_functions.py`, lines 77, 87, 156, 168, 302, 310
- **Description:** All interpolation functions compute `denom = (grid[j+1] - grid[j]) * ...` and then return `nom/denom` without checking for `denom == 0`. If two adjacent grid points coincide, this produces division by zero.
- **Impact:** With `fastmath=True` (set on all these functions), division by zero produces `inf` or `nan` silently rather than raising an exception. The grids are constructed with `nonlinspace_jit` which should produce distinct points, but there is no defensive check. The `fast_interp_all` function in `interp.py` (line 164) does guard against `dx == 0`, showing the author was aware of the risk but did not apply the guard consistently.

### 7. Magic numbers throughout calibration parameters
- **File:** `par_epsilons.py`, lines 23-84; `full_calibration.py`, lines 51-131
- **Description:** Dozens of hardcoded economic parameters with no named constants or documentation of sources. Examples:
  - `0.940074219` (beta, line 23 of par_epsilons.py)
  - `44.5312500` (bequest motive nu, line 35)
  - `3.18164063` (bequest floor b_bar, line 45)
  - `5.014401` (wealth cutoff in simulate_initial_joint.py, line 39)
  - `0.0223437500` (amenity spread, lines 57-58)
  - `0.311` (labor supply, line 71)
  - The `vPi_S_median` array (52 hardcoded flood probabilities)
- **Impact:** These values appear to be calibration outputs pasted back as inputs. Changing one without understanding its origin risks inconsistency. The number of significant digits (e.g., `0.940074219`) suggests machine-precision outputs that should be documented.

### 8. `.xslx` typo (should be `.xlsx`)
- **File:** `solve_epsilons.py`, lines 138, 140, 146, 148, 249, 251, 1623
- **Description:** Multiple calls to `df.to_excel("....xslx")` with `.xslx` instead of `.xlsx`. Pandas `to_excel` determines the engine from the extension and may fail or write a corrupt file.
- **Impact:** On some pandas versions, this will raise `ValueError: No engine for filetype 'xslx'`. On others, it may silently write a file that Excel cannot open. Lines 138, 140, 146, 148 are in dead code (inside the `"""` block) but lines 249, 251 are in the unreachable-but-uncommented section that could become live if dead code is cleaned up. Line 1623 is also in dead code.

### 9. `Line2D` used but never imported
- **File:** `solve_epsilons.py`, lines 883-884
- **Description:** `Line2D([0], [0], ...)` is used in the dead code section to create custom legend handles, but `from matplotlib.lines import Line2D` is never imported.
- **Impact:** Would crash with `NameError` if this code path were reached. Currently in dead code (line >1039), so not an active bug, but would bite when cleaning up.

### 10. Hardcoded `1e12` sentinel values without named constant
- **File:** `mortgage_choice_simulation.py`, line 11; `mortgage_choice_simulation_exc.py`, line 10; `buyer_problem_epsilons.py`, line 16; `buyer_problem_simulation.py`, line 15
- **Description:** Arrays initialized with `np.ones(...)*-1e12` as "negative infinity" sentinel for value function comparisons. The magic number `-1e12` appears in 4 files.
- **Impact:** If any computed utility exceeds `-1e12` in absolute value (extremely unlikely but possible with extreme parameter choices), the sentinel would interfere. More practically, the inconsistent use of a magic number makes reasoning about edge cases harder.

---

## MEDIUM -- Duplication, long functions, naming issues

### 11. `DoubleGrid()` duplicated across files
- **File:** `misc_functions.py`, line 12 (with `@njit`); `full_calibration.py`, line 19 (without `@njit`)
- **Description:** Two implementations of the same function. The version in `full_calibration.py` returns a 2-row matrix (transposed layout), while the version in `misc_functions.py` returns an N-by-2 matrix. Neither version in `full_calibration.py` is actually called anywhere.
- **Impact:** Dead code, but confusing if someone tries to use the wrong version. The different return shapes would cause silent dimension errors.

### 12. `_interp_3d` and `interp_3d` duplicated in two files
- **File:** `interp.py`, lines 80-150; `misc_functions.py`, lines 44-114
- **Description:** Both files contain identical implementations of `_interp_3d` and `interp_3d`. Both use `@njit(fastmath=True)`. Both also contain `binary_search` (interp.py line 9 vs misc_functions.py line 205).
- **Impact:** Numba will JIT-compile both independently, doubling compilation time and memory. Import order could lead to confusion about which version is being called.

### 13. `solve_epsilons.py:main()` is ~2,680 lines
- **File:** `solve_epsilons.py`, lines 48-2725
- **Description:** A single function spanning nearly the entire file. Even the "live" portion (lines 48-107) does 5+ major steps without any helper functions.
- **Impact:** Impossible to test, debug, or understand in isolation. The function mixes parameter setup, grid creation, model solving, moment computation, calibration diagnostics, welfare analysis, plotting, and file I/O.

### 14. `simulation` file without `.py` extension
- **File:** `simulation` (no extension), entire file
- **Description:** An exact copy of `simulation.py` that lacks the `.py` extension, so Python cannot import it. It appears to be an accidental copy or backup.
- **Impact:** Wastes repository space, risks confusion. If someone renames it to `.py`, it would shadow the real file depending on import order.

### 15. Inconsistent parameter passing: `dPi_L` passed where `iNj` expected
- **File:** `full_calibration.py`, line 138; `solve_epsilons.py`, line 203
- **Description:** The calls pass `dPi_L` as the 4th positional argument to `equil.initialise_coefficients_initial(par, grids, method, dPi_L, par.iNj, ...)`. But `initialise_coefficients_ss` (the likely intended function) takes `(par, grids, method, iNj, mMarkov, ...)`. This means `dPi_L` (a float ~0.01) would be used as `iNj` (an integer ~30).
- **Impact:** Even if the function name mismatch is fixed, the parameter order is wrong and would cause the solver to iterate for the wrong number of age periods.

### 16. `vPDF_z` does not sum to 1
- **File:** `grid_creation.py`, line 41
- **Description:** `vPDF_z = np.array([1, 0.4, 0.4, 0.2])` sums to 2.0. The first element (1) corresponds to "no damage" (`vZ[0] = 1`), and elements 1-3 are conditional on flooding occurring. The code uses `vPDF_z[1:]` (summing to 1.0) for flood damage calculations and uses the full vector only in `continuation_value_nolearning.py` where it splits into flood/no-flood branches with separate probability weights.
- **Impact:** The array is intentionally structured this way (conditional probabilities for the damage states), but the naming `PDF_z` is misleading since it is not a proper PDF. This is a documentation/naming issue rather than a bug, but could easily mislead someone modifying the damage distribution.

---

## LOW -- Cosmetic, debug prints

### 17. Excessive debug `print()` statements
- **File:** `solve_epsilons.py`, lines 104-107, 214-225, 566-576, 588-591, 654, 795-796, etc.; `equilibrium.py`, lines 104, 160-165, 227-228, 252-253, 268-269
- **Description:** Print statements left from debugging, including `print(tax_equiv_newborns)`, `print("Time step:", t_index)`, `print('Coefficients C', vCoeff_C)`.
- **Impact:** Pollutes stdout during long-running calibration/equilibrium iterations. No functional harm but makes it hard to find real warnings.

### 18. Redundant `tauch.tauchen()` call
- **File:** `solve_epsilons.py`, line 76
- **Description:** `mMarkov, vE = tauch.tauchen(...)` is called, but both return values are immediately overwritten by `grid_creation.create(par)` on line 77 (which internally calls `tauch.tauchen()` itself).
- **Impact:** Wasted computation (Tauchen discretization is O(N^2) in states). No effect on results.

### 19. Duplicate `import pandas as pd`
- **File:** `solve_epsilons.py`, lines 14 and 36
- **Description:** `pandas` is imported twice under the same alias.
- **Impact:** No functional harm; purely cosmetic.

### 20. `PROBLEM` comment left in production code
- **File:** `simulate_initial_joint.py`, line 31
- **Description:** Comment reads `#PROBLEM - SPREAD OF INITIAL WEALTH SEEMS UNACCEPTABLY HIGH`.
- **Impact:** Suggests an unresolved issue with the initial wealth distribution calibration. Worth investigating whether this was resolved or is still producing inaccurate results.

### 21. Commented-out import `#import schumaker as schum`
- **File:** `utility_epsilons.py`, line 12
- **Description:** Dead import suggesting an alternative interpolation library was considered.
- **Impact:** No functional harm; cosmetic.

### 22. `#import error_statistics as err` and `err.prediction_errors` in dead code
- **File:** `solve_epsilons.py`, line 38 (commented import), line 799 and 1621 (dead code using `err`)
- **Description:** The `error_statistics` module import is commented out, but dead code on lines 799 and 1621 tries to call `err.prediction_errors(...)`.
- **Impact:** Would crash if dead code became live. The `error_statistics` module may not even exist in the codebase.
