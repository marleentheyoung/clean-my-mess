# Fix Proposals

Mapped to `context.md` sections. Each proposal includes a rationale explaining why it's safe and what it improves.

---

## Quick Wins (< 1h each, near-zero risk)

### Q1. Delete extensionless `simulation` duplicate [context.md section 1]

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/simulation` (delete) |
| **Risk** | Very low |
| **Rationale** | Python imports require `.py` extension. The extensionless `simulation` file is never imported by any module -- `import simulation` resolves to `simulation.py`. Deleting it removes confusion without affecting any import chain. |
| **Validation** | Confirm no file references `simulation` without `.py`; run model, verify identical output. |

### Q2. Remove dead code from `solve_epsilons.py` [context.md section 2]

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/solve_epsilons.py` |
| **Risk** | Very low |
| **Rationale** | ~95% of `main()` is inside triple-quoted block comments or `#` comment blocks. This dead code is never executed, but it obscures the ~20 active lines. Removing it makes the entry point readable. The dead code is preserved in git history. |
| **Validation** | Run the active path; confirm output matches pre-edit. |

### Q3. Remove commented-out blocks in `tauchen.py`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/tauchen.py` (commented `main()` at bottom) |
| **Risk** | None |
| **Rationale** | A commented-out `main()` function has no runtime effect. It adds ~24 lines of noise. Git preserves history. |
| **Validation** | No runtime change; verify file still imports correctly. |

### Q4. Remove commented-out grid alternatives in `grid_creation.py`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/grid_creation.py` (~48 lines of commented alternative grid logic) |
| **Risk** | None |
| **Rationale** | Large commented block with no explanation of why it was disabled. Adds confusion about which grid logic is active. Git preserves history. |
| **Validation** | No runtime change. |

### Q5. Fix duplicate imports in `solve_epsilons.py`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/solve_epsilons.py` |
| **Risk** | None |
| **Rationale** | `pandas` is imported twice, `moments` is imported under two aliases. Duplicate imports are harmless but confusing. |
| **Validation** | No runtime change. |

### Q6. Fix `.xslx` typo to `.xlsx`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/solve_epsilons.py` (multiple lines writing Excel output) |
| **Risk** | None |
| **Rationale** | Files are created with wrong extension (`.xslx` instead of `.xlsx`). This doesn't break functionality but the files won't auto-open in Excel. |
| **Validation** | Generated files have correct extension. |

### Q7. Name magic numbers as constants

| Field | Value |
|-------|-------|
| **Files** | Multiple: `misc_functions.py`, `tauchen.py`, `grid_creation.py`, `stayer_problem.py`, `stayer_problem_renter.py`, `equilibrium.py` |
| **Risk** | Very low |
| **Rationale** | Hardcoded values like `0.7` (retirement income replacement), `100` (consumption grid nodes), `1.4` (nonlinear grid parameter), `1e-3`/`1e-5` (tolerances) appear without context. Extracting them to named constants at the top of each file documents their meaning without changing behavior. All constants stay inside @njit-compatible scope. |
| **Validation** | Identical numerical output. |

### Q8. Unify `LoM_C` and `LoM_NC` into single function

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/LoM_epsilons.py` |
| **Risk** | Very low |
| **Rationale** | `LoM_C` and `LoM_NC` are byte-identical except for variable naming. A single `LoM(grids, t_index, vCoeff)` function removes duplication. The merged function stays @njit. |
| **Validation** | Call both old and new with same inputs; confirm identical outputs. |

### Q9. Delete unused `DoubleGrid` from `full_calibration.py`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/full_calibration.py` |
| **Risk** | None |
| **Rationale** | `DoubleGrid` is defined but never called within `full_calibration.py`. The canonical version is in `misc_functions.py`. Removing the dead copy prevents confusion. |
| **Validation** | Grep confirms no calls to this copy. |

---

## Medium Effort (1-4h, low-medium risk)

### M1. Consolidate duplicate interpolation into `interp.py` [context.md section 5]

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/interp.py`, `clean_the_mess/misc_functions.py` |
| **Risk** | Low |
| **Rationale** | `binary_search`, `_interp_2d`/`interp_2d`, `_interp_3d`/`interp_3d` are duplicated between these files. Having two copies means a bug fix in one doesn't propagate to the other. Consolidate all interpolation into `interp.py`, add `interp_4d` from `misc_functions.py`, update all import sites. All functions stay @njit. |
| **Validation** | Run full model; numerical output identical to baseline. |

### M2. Split `misc_functions.py` by purpose [context.md section 7]

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/misc_functions.py` + consumers |
| **Risk** | Low-medium |
| **Rationale** | `misc_functions.py` is a junk drawer mixing interpolation, OLS, income calculation, jitclass construction, and grid utilities. Per context.md section 7: move interpolation to `interp.py`, `ols_numba` to `equilibrium.py` (only consumer), `net_income` to `utility_epsilons.py`, keep `construct_jitclass`/`DoubleGrid`/`maxRow`. This makes each file's purpose clear and reduces coupling. All @njit decorators preserved. |
| **Validation** | All imports resolve; full model output identical. |

### M3. Remove `_epsilons` suffix from filenames [context.md section 9]

| Field | Value |
|-------|-------|
| **Files** | `par_epsilons.py`, `utility_epsilons.py`, `buyer_problem_epsilons.py`, `household_problem_epsilons_nolearning.py`, `solve_epsilons.py`, `LoM_epsilons.py` |
| **Risk** | Medium |
| **Rationale** | The `_epsilons` suffix is vestigial from an earlier model version where epsilon preference shocks were being added. It no longer distinguishes these files from alternatives. Removing it makes file names cleaner. Requires updating all import statements. |
| **Validation** | All imports resolve; full model output identical. |

### M4. Rename `proper_welfare_debug.py` to `welfare.py`

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/proper_welfare_debug.py` + consumers |
| **Risk** | Low |
| **Rationale** | The name suggests a temporary debug file, but it contains the actual welfare analysis code. Renaming clarifies its role. |
| **Validation** | All imports resolve. |

### M5. Verify and fix `grids.PDF_z` vs `grids.vPDF_z` bug

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/utility_epsilons.py` |
| **Risk** | Medium |
| **Rationale** | `W_bequest_flooddamage` references `grids.PDF_z` but the grid object uses `vPDF_z`. If this function is called, it crashes at Numba compile time. If it never crashes, the function is dead code. Either fix the attribute name or confirm dead and remove. Either way, this is a latent bug. |
| **Validation** | If dead: grep confirms no callers. If live: fix attribute, verify bequest computation unchanged. |

### M6. Document `fastmath` difference between mortgage_choice variants

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/mortgage_choice_simulation.py`, `clean_the_mess/mortgage_choice_simulation_exc.py` |
| **Risk** | Low |
| **Rationale** | `mortgage_choice_simulation.py` uses `@njit(fastmath=True)` while `_exc.py` uses plain `@njit`. `fastmath` can change numerical results (reorders floating point operations). Both are imported by `simulation.py`. Understanding when each is called and whether the fastmath difference is intentional prevents silent numerical drift during refactoring. |
| **Validation** | Documentation only; no code change. |

### M7. Extract duplicated rental price calculation

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/simulation.py`, `clean_the_mess/stayer_problem_renter.py` |
| **Risk** | Medium |
| **Rationale** | The rental price formula `dPsi + max(dP_C - (1-dDelta-damage)/(1+r)*dP_C_prime, 0)` appears in at least two places. If one copy is updated and the other isn't, prices become inconsistent. Extract into a shared @njit helper. |
| **Validation** | Rental prices match pre-refactor values. |

### M8. Add numerical guards to unguarded divisions

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/interp.py`, `clean_the_mess/misc_functions.py`, `clean_the_mess/proper_welfare_debug.py`, `clean_the_mess/utility_epsilons.py` |
| **Risk** | Low |
| **Rationale** | Interpolation functions divide by `grid[j+1]-grid[j]` which is zero if grid has duplicate points. Welfare debug divides by `y1-y0` in linear zero-crossing. Utility functions raise values to fractional powers that produce NaN for negative inputs. Adding guards (e.g., `max(denom, 1e-15)` or early-return on zero) prevents silent NaN propagation. Guards work inside @njit. |
| **Validation** | No change to output when grids are valid; graceful failure instead of NaN when they aren't. |

---

## Large Refactors (4h+, need tests first)

### L1. Build staged regression test suite [context.md section 10]

| Field | Value |
|-------|-------|
| **Files** | New `tests/` directory |
| **Risk** | None (additive) |
| **Rationale** | No refactoring should happen without a way to verify numerical output hasn't changed. However, the full equilibrium solver takes hours and the calibration path is broken, so we cannot pin vCoeff_C/vCoeff_NC from a fresh solve. Instead, build a staged suite: Stage A (smoke tests, < 30s) pins grid creation and imports; Stage B (~5 min, reduced grids) pins solve_ss and distribution properties; Stage C (~15 min) pins welfare equivalents and hardcoded coefficient inputs. Full equilibrium pinning (Stage D) is deferred until the calibration path is repaired. |
| **Validation** | Stages A-C pass on post-Phase-3 codebase. |

### L2. Single config source [context.md section 6]

| Field | Value |
|-------|-------|
| **Files** | `clean_the_mess/par_epsilons.py`, `clean_the_mess/full_calibration.py`, `clean_the_mess/grid_creation.py`, new `config.py` |
| **Risk** | Medium-high |
| **Rationale** | Parameters are defined in `par_epsilons.py`, then partially redefined in `full_calibration.py` and `grid_creation.py`. `par_epsilons.py` also mutates `vPi_S_median` at module level (line 17). A single config source with a base dict + calibration overrides eliminates duplication and the mutation hazard. Must stay as numba-compatible dict (passed through `construct_jitclass`). |
| **Validation** | Full model output identical; all parameter values match. |

### L3. Full package restructure [context.md section 8]

| Field | Value |
|-------|-------|
| **Files** | Entire `clean_the_mess/` directory |
| **Risk** | High |
| **Rationale** | context.md section 8 proposes a `model/` package with household/, simulation/, equilibrium/, analysis/ subpackages. This makes the architecture legible and each module's role clear. But it touches every import in the codebase and requires regression tests to validate. |
| **Validation** | Full regression suite must pass. All imports resolve. Numba can compile all @njit functions. |

---

## REJECTED

### ~~R1. Remove @njit from orchestration functions~~ [context.md section 3]

| Field | Value |
|-------|-------|
| **Status** | **REJECTED** |
| **Reason** | Performance regression. The call chain find_coefficients -> generate_pricepath -> house_prices_algorithm -> compute_excess_demand_pair -> excess_demand_continuous runs millions of times. Every Python<->njit boundary crossing has overhead that compounds. Tested and measured. |

### ~~R2. Introduce Python dataclasses for argument bundling~~ [context.md section 4]

| Field | Value |
|-------|-------|
| **Status** | **REJECTED** |
| **Reason** | Depends on R1 (removing @njit from orchestration). Since R1 is rejected, dataclasses cannot be used anywhere in the @njit call chain. Function signature bloat stays as a cost of numba. |
