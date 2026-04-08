# Migration Plan -- Phase 5c: Restructure into `model/` package

Phase 3 (dead code removal) is DONE. Phases 5a-5b (quick wins, medium effort) will execute before this plan. This plan covers Phase 5c only -- the large restructure.

**Pre-condition:** Before starting Phase 5c, ensure:
- All dead code is removed (Phase 3 done)
- Interpolation dedup is done (Phase 5a -- consolidate into `interp.py`)
- `_epsilons` suffixes are removed from filenames (Phase 5b)
- `fastmath` inconsistency between `mortgage_choice_simulation.py` and `_exc` is resolved (Phase 5b)
- `grids.PDF_z` -> `grids.vPDF_z` bug is fixed (Phase 5a)
- Reference outputs are saved for regression testing

**Testing protocol for every step:** After each commit, run:
```python
python -c "from model.<module> import <key_function>; print('OK')"
```
And when applicable, run the steady-state solve + distribution check to verify numerical equivalence.

---

## Step 1: Create package skeleton

Create the `model/` directory with empty `__init__.py` files.

```
model/__init__.py
model/household/__init__.py
model/simulation/__init__.py
model/equilibrium/__init__.py
model/analysis/__init__.py
```

**Test:** `import model` succeeds.

**Commit message:** `create model/ package skeleton with empty __init__.py files`

---

## Step 2: Move leaf modules (layer 0) -- no import changes needed internally

These modules have no local imports, so moving them cannot break anything internally. Only their consumers need updating.

### Step 2a: Move `tauchen.py` -> `model/tauchen.py`

Copy file. Add re-export in `model/__init__.py`. Create a shim at the old location:
```python
# clean_the_mess/tauchen.py (shim)
from model.tauchen import *
```

**Test:** `from model.tauchen import tauchen; print('OK')`

**Commit message:** `move tauchen.py into model/ package`

### Step 2b: Move `interp.py` -> `model/interp.py`

At this point, Phase 5a should have already consolidated all interpolation from `misc_functions.py` into `interp.py`. Move the consolidated file.

Create shim at old location.

**Test:** `from model.interp import interp_1d, interp_3d, binary_search; print('OK')`

**Commit message:** `move consolidated interp.py into model/ package`

### Step 2c: Create `model/utility.py` from `utility_epsilons.py`

By Phase 5b, the `_epsilons` suffix should already be removed. Move to `model/utility.py`.

Create shim at old location.

**Test:** `from model.utility import u, u_c, W_bequest; print('OK')`

**Commit message:** `move utility.py into model/ package`

### Step 2d: Create `model/lom.py` from `LoM_epsilons.py`

By Phase 5b, should already be renamed. Move to `model/lom.py`.

Create shim at old location.

**Test:** `from model.lom import lom_c, lom_nc; print('OK')` (or `LoM_C`, `LoM_NC` if function names are kept for Phase 5b)

**Commit message:** `move lom.py into model/ package`

### Step 2e: Create `model/config.py` from `par_epsilons.py`

Move `par_epsilons.py` content into `model/config.py`. Rename the function to `create_par_dict()`. Add `SOLVER_SETTINGS` dict (extracted in config_extraction.md).

Create shim at old location.

**Test:** `from model.config import create_par_dict; par_dict = create_par_dict(); print(par_dict['dBeta'])`

**Commit message:** `move parameter definitions into model/config.py`

### Step 2f: Create `model/utils.py` from remainder of `misc_functions.py`

After Phase 5a extracts interpolation, the remaining functions are: `construct_jitclass`, `DoubleGrid`, `maxRow`, `lininterp_zero_crossing`. Move these to `model/utils.py`.

Create shim at old location.

**Test:** `from model.utils import construct_jitclass; print('OK')`

**Commit message:** `move utility functions into model/utils.py`

### Step 2g: Create `model/income.py` from `misc_functions.py:net_income`

Extract `net_income` into `model/income.py`.

**Test:** `from model.income import net_income; print('OK')`

**Commit message:** `extract net_income into model/income.py`

---

## Step 3: Move layer-1 modules into sub-packages

### Step 3a: Merge `grids.py` + `grid_creation.py` -> `model/grids.py`

Merge `nonlinspace_jit` and `equilogspace` from the current `grids.py` with the `create()` function from `grid_creation.py`. The merged file imports `model.tauchen` and `model.utils` instead of the old paths.

Create shims at both old locations.

**Test:** `from model.grids import create, nonlinspace_jit; print('OK')`

**Commit message:** `merge grid_creation.py and grids.py into model/grids.py`

### Step 3b: Move `simulate_initial_joint.py` -> `model/simulation/initial_joint.py`

No internal import changes (leaf module).

Create shim at old location.

**Test:** `from model.simulation.initial_joint import initial_joint; print('OK')`

**Commit message:** `move simulate_initial_joint.py into model/simulation/`

### Step 3c: Move household VFI sub-modules

Move in dependency order (leaves first):

1. `stayer_problem.py` -> `model/household/stayer.py`
2. `stayer_problem_renter.py` -> `model/household/renter.py`
3. `buyer_problem_epsilons.py` -> `model/household/buyer.py`
4. `continuation_value_nolearning.py` -> `model/household/continuation.py`

Update imports in each file to use `model.utility`, `model.interp`, `model.grids`, `model.utils`.

Create shims at old locations.

**Test per file:** Import the module and verify key function exists.

**Commit message:** `move household VFI sub-modules into model/household/`

### Step 3d: Move simulation sub-modules (non-split files)

1. `buyer_problem_simulation.py` -> `model/simulation/buyer_sim.py`
2. `mortgage_choice_simulation.py` + `mortgage_choice_simulation_exc.py` -> `model/simulation/mortgage_sim.py`

Update imports to use `model.utils`, `model.interp`.

Create shims at old locations.

**Test:** Import each and verify functions exist.

**Commit message:** `move simulation sub-modules into model/simulation/`

### Step 3e: Move `moments.py` -> `model/analysis/moments.py`

Update imports to use `model.interp`, `model.lom`, `model.utils`.

Create shim.

**Test:** `from model.analysis.moments import calc_moments; print('OK')`

**Commit message:** `move moments.py into model/analysis/`

---

## Step 4: Move layer-2 modules

### Step 4a: Move `household_problem_epsilons_nolearning.py` -> `model/household/vfi.py`

Update imports to use `model.household.continuation`, `model.household.stayer`, `model.household.renter`, `model.household.buyer`, `model.lom`, `model.utils`.

Update `model/household/__init__.py` to re-export `solve` and `solve_ss`.

Create shim at old location.

**Test:** `from model.household import solve, solve_ss; print('OK')`

**Commit message:** `move VFI orchestrator into model/household/vfi.py`

### Step 4b: Split `simulation.py` into `model/simulation/` sub-modules

This is the most complex step. Split the 1,263-line file into:

1. `model/simulation/transitions.py` -- `simulate_buy`, `simulate_buy_ret`, `simulate_rent`, `simulate_rent_ret`, `simulate_stay`, `simulate_stay_ret`, `simulate_rent_outer`, `simulate_buy_outer`
2. `model/simulation/decisions.py` -- `continuous_decide`, `continuous_decide_renter`, `renter_sim`, `renter_sim_demand`, `renter_solve`, `compute_p_left`
3. `model/simulation/excess_demand.py` -- `excess_demand_continuous`
4. `model/simulation/distribution.py` -- `stat_dist_finder`, `update_dist_continuous`, `construct_m1`, `mortgage_matrix_solve`

Update all internal imports. Create shim at old `simulation.py` location.

Update `model/simulation/__init__.py` to re-export key functions.

**Test:** Run steady-state solve + `stat_dist_finder` and compare distribution output against reference.

**Commit message:** `split simulation.py into model/simulation/ sub-modules`

---

## Step 5: Move layer-3+ modules

### Step 5a: Split `equilibrium.py` into `model/equilibrium/` sub-modules

1. `model/equilibrium/market_clearing.py` -- `house_prices_algorithm`, `precompute_market_data`, `compute_excess_demand_pair`, `bisection_root_finding`, `secant_method_system_2d`, `check_convergence`, `ols_numba` (moved from utils)
2. `model/equilibrium/steady_state.py` -- `initialise_coefficients_ss`
3. `model/equilibrium/solver.py` -- `find_coefficients`, `generate_pricepath`, `coeff_updater`, `flatten_third_dim`

Update imports. Create shim. Update `model/equilibrium/__init__.py`.

**Test:** `from model.equilibrium import find_coefficients, initialise_coefficients_ss; print('OK')`

**Commit message:** `split equilibrium.py into model/equilibrium/ sub-modules`

### Step 5b: Move `experiments.py` -> `model/analysis/experiments.py`

Update imports. Create shim.

**Test:** `from model.analysis.experiments import full_information_experiment; print('OK')`

**Commit message:** `move experiments.py into model/analysis/`

### Step 5c: Move `proper_welfare_debug.py` -> `model/analysis/welfare.py`

Update imports. Create shim.

**Test:** `from model.analysis.welfare import find_expenditure_equiv; print('OK')`

**Commit message:** `move welfare analysis into model/analysis/welfare.py`

---

## Step 6: Move top-level entry points

### Step 6a: Create `model/run.py` from live portion of `solve_epsilons.py`

The live portion (~54 lines) becomes the clean entry point. All imports updated to `model.*` paths.

**Test:** `python model/run.py` produces same welfare output as before.

**Commit message:** `create clean entry point model/run.py from solve_epsilons.py`

### Step 6b: Move `plot_creation.py` -> `model/plots.py`

Update imports. Create shim.

**Test:** `from model.plots import plot_pricepaths; print('OK')`

**Commit message:** `move plot_creation.py into model/plots.py`

### Step 6c: Update `full_calibration.py` imports (OUT OF SCOPE for refactoring)

Only update import statements to point to `model.*` paths. Do not refactor any logic.

**Test:** `python -c "import full_calibration; print('OK')"`

**Commit message:** `update full_calibration.py imports to use model/ package`

---

## Step 7: Remove shims

Once all consumers are updated, remove the backward-compatibility shims in the old locations.

**Test:** Full steady-state run + distribution comparison against reference.

**Commit message:** `remove backward-compatibility shims after migration`

---

## Step 8: Final validation

1. Run full steady-state solve (`solve_ss`) and compare VFI output against reference
2. Run `stat_dist_finder` and compare distribution against reference
3. If equilibrium reference exists, run `initialise_coefficients_ss` and compare
4. Verify no old file paths remain in any import statements:
   ```
   grep -r "import.*_epsilons" model/
   grep -r "import misc_functions" model/
   grep -r "import grid_creation" model/
   ```

**Commit message:** `verify numerical equivalence after model/ restructure`

---

## Summary

| Step | Description | Files touched | Risk |
|---|---|---|---|
| 1 | Package skeleton | 5 new `__init__.py` | None |
| 2a-2g | Move leaf modules (7 steps) | 7 moves + 7 shims | Low -- no internal imports |
| 3a | Merge grids | 2 files -> 1 | Medium -- merge logic |
| 3b-3e | Move layer-1 modules (4 steps) | 8 moves | Low -- simple renames |
| 4a | Move VFI orchestrator | 1 move | Low |
| 4b | Split simulation.py | 1 file -> 4 | **High** -- largest refactor |
| 5a | Split equilibrium.py | 1 file -> 3 | Medium |
| 5b-5c | Move analysis modules | 2 moves | Low |
| 6a-6c | Entry points | 3 files | Low |
| 7 | Remove shims | ~15 deletions | Low |
| 8 | Final validation | 0 | None |

**Total commits:** ~20 atomic commits
**Highest risk step:** 4b (simulation.py split) -- recommend extra caution and intermediate testing
