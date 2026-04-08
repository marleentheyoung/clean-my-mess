# Proposed Structure

Target folder layout for the `model/` package, based on `context.md` section 8 and validated against the dependency graph.

## Layout

```
model/
├── __init__.py
├── config.py
├── utils.py
├── grids.py
├── tauchen.py
├── interp.py
├── utility.py
├── lom.py
├── income.py
├── household/
│   ├── __init__.py
│   ├── vfi.py
│   ├── continuation.py
│   ├── stayer.py
│   ├── renter.py
│   └── buyer.py
├── simulation/
│   ├── __init__.py
│   ├── distribution.py
│   ├── excess_demand.py
│   ├── buyer_sim.py
│   ├── mortgage_sim.py
│   ├── decisions.py
│   ├── transitions.py
│   └── initial_joint.py
├── equilibrium/
│   ├── __init__.py
│   ├── solver.py
│   ├── steady_state.py
│   └── market_clearing.py
├── analysis/
│   ├── __init__.py
│   ├── moments.py
│   ├── welfare.py
│   └── experiments.py
├── calibration.py
├── plots.py
└── run.py
```

## Rationale for each file/directory

### Top-level modules (layer 0 -- no local imports)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `config.py` | `par_epsilons.py` + magic numbers extracted from `grid_creation.py`, `equilibrium.py`, `simulate_initial_joint.py` | Single source of truth for all parameters and solver settings. Contains `create_par_dict()` returning a plain dict, plus a `SOLVER_SETTINGS` dict for equilibrium tolerances. Numba-compatible: dict is converted to jitclass by `utils.construct_jitclass()`. |
| `utils.py` | `misc_functions.py` (only: `construct_jitclass`, `DoubleGrid`, `maxRow`, `lininterp_zero_crossing`) | General-purpose utilities that don't belong to any domain module. `construct_jitclass` is used everywhere. `DoubleGrid` and `maxRow` are pure helpers. |
| `grids.py` | `grid_creation.py` + `grids.py` (current) | Merge: the current `grids.py` provides `nonlinspace_jit` and `equilogspace`, which are only used by `grid_creation.py`. Combining them removes one layer of indirection. The merged module exports `create()` (grid construction) and `nonlinspace_jit()`. |
| `tauchen.py` | `tauchen.py` | Standalone income-process discretization. No changes except removing commented-out `main()`. Already a clean leaf module. |
| `interp.py` | `interp.py` + interpolation functions from `misc_functions.py` (`_interp_2d`, `interp_2d`, `_interp_3d`, `interp_3d`, `_interp_4d`, `interp_4d`, `binary_search`, `binary_search_sim`) | Consolidates ALL interpolation and binary search into one module, eliminating the duplication flagged in red_flags #12. The `misc_functions.py` versions are identical to `interp.py` versions for 3D; 2D and 4D only exist in `misc_functions.py`. Move them here. |
| `utility.py` | `utility_epsilons.py` | Rename only (drop `_epsilons` vestigial suffix). Contains `u`, `u_c`, `W_bequest`, `Q_bequest`, bequest variants, `renter_solve`. Fix `grids.PDF_z` -> `grids.vPDF_z` (red flag #1) during this move. |
| `lom.py` | `LoM_epsilons.py` | Rename only (drop `LoM_` prefix and `_epsilons` suffix, use snake_case). Contains `lom_c`, `lom_nc`. |
| `income.py` | `misc_functions.py:net_income` | Extract `net_income` into its own small module. It's the only domain-specific function in `misc_functions.py` and logically belongs near income/tax computation. Only 12 lines. |

### Household sub-package (layers 1-2 -- VFI solver components)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `household/__init__.py` | *(new)* | Re-exports `vfi.solve`, `vfi.solve_ss` for backward compatibility. |
| `household/vfi.py` | `household_problem_epsilons_nolearning.py` | The VFI orchestrator. Imports all other household sub-modules. Rename drops the unwieldy original name. |
| `household/continuation.py` | `continuation_value_nolearning.py` | Continuation value computation (586 lines). Imports from `interp`, `utility`, `utils`. No circular risk: it does not import from `household/vfi.py`. |
| `household/stayer.py` | `stayer_problem.py` | EGM for homeowner-stayers. Imports `utility`, `grids` (for `nonlinspace_jit`). |
| `household/renter.py` | `stayer_problem_renter.py` | EGM for renters. Imports `utility`, `grids`, `interp`. |
| `household/buyer.py` | `buyer_problem_epsilons.py` | Buyer discrete optimization during VFI. Imports `utility`, `interp`. |

**Dependency direction within `household/`:** `vfi.py` imports from `continuation.py`, `stayer.py`, `renter.py`, `buyer.py`. None of those import from `vfi.py`. No circular risk.

### Simulation sub-package (layers 1-2 -- forward simulation components)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `simulation/__init__.py` | *(new)* | Re-exports key functions: `stat_dist_finder`, `excess_demand_continuous`, `update_dist_continuous`. |
| `simulation/distribution.py` | `simulation.py:stat_dist_finder`, `update_dist_continuous`, `construct_m1`, `mortgage_matrix_solve` | Stationary distribution and distribution updating. Core simulation loop functions. |
| `simulation/excess_demand.py` | `simulation.py:excess_demand_continuous` | The excess demand computation, called by equilibrium solver. Separated because it is the interface between simulation and equilibrium. |
| `simulation/buyer_sim.py` | `buyer_problem_simulation.py` | Buyer choices on the finer simulation grid. Already a standalone module. |
| `simulation/mortgage_sim.py` | `mortgage_choice_simulation.py` + `mortgage_choice_simulation_exc.py` | **Merge**: these two files implement the same mortgage repayment logic. `_exc` adds coastal damage and NC variants. Merge into one module with `solve_coastal`, `solve_noncoastal` (from `_exc`) and `solve` (from base). Fix fastmath inconsistency (red flag #4) during merge. |
| `simulation/decisions.py` | `simulation.py:continuous_decide`, `continuous_decide_renter`, `renter_sim`, `renter_sim_demand`, `renter_solve`, `compute_p_left` | Decision functions that determine agent choices during forward simulation. |
| `simulation/transitions.py` | `simulation.py:simulate_buy`, `simulate_buy_ret`, `simulate_rent`, `simulate_rent_ret`, `simulate_stay`, `simulate_stay_ret`, `simulate_rent_outer`, `simulate_buy_outer` | State transition functions. Grouped because they share the same interface pattern (take distribution + policy functions, output updated distribution). |
| `simulation/initial_joint.py` | `simulate_initial_joint.py` | Initial joint distribution of wealth and income. Already standalone. |

**Splitting `simulation.py` (1,263 lines):** This is the largest module after `solve_epsilons.py`. The split groups functions by their role in the simulation pipeline. The dependency flow is: `distribution.py` calls `transitions.py` and `decisions.py`; `excess_demand.py` calls `distribution.py`. No circularity.

**JUDGMENT CALL -- simulation split granularity:** The proposed 7-file split may be too fine-grained. An alternative is a 3-file split: `distribution.py` (stat_dist + update_dist), `forward.py` (all simulate_* + decide + renter functions), `excess_demand.py`. The 7-file version is easier to navigate but creates more import boilerplate. **Recommend the user decide.**

### Equilibrium sub-package (layer 3)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `equilibrium/__init__.py` | *(new)* | Re-exports `solver.find_coefficients`, `steady_state.initialise_coefficients_ss`, `market_clearing.house_prices_algorithm`. |
| `equilibrium/solver.py` | `equilibrium.py:find_coefficients`, `generate_pricepath`, `coeff_updater`, `flatten_third_dim` | The outer coefficient iteration loop. Imports `household.vfi` and `simulation`. |
| `equilibrium/steady_state.py` | `equilibrium.py:initialise_coefficients_ss` | Steady-state price finder. Same imports as `solver.py`. Separated because it has distinct convergence logic (rho adaptation, oscillation detection). |
| `equilibrium/market_clearing.py` | `equilibrium.py:house_prices_algorithm`, `precompute_market_data`, `compute_excess_demand_pair`, `bisection_root_finding`, `secant_method_system_2d`, `check_convergence` | Market-clearing algorithms. Pure numerical methods that don't need to know about the economic model beyond excess demand. Also move `ols_numba` here from `misc_functions.py` (it's only used by `coeff_updater`). |

**Dependency direction:** `solver.py` and `steady_state.py` both import from `market_clearing.py`. `market_clearing.py` imports from `simulation.excess_demand` only via the `compute_excess_demand_pair` wrapper. No circular risk.

### Analysis sub-package (layer 4)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `analysis/__init__.py` | *(new)* | Minimal. |
| `analysis/moments.py` | `moments.py` | Moment computation from simulated distributions. Clean leaf at layer 1 in the dependency graph (only imports `interp`, `lom`, `utils`). |
| `analysis/welfare.py` | `proper_welfare_debug.py` | Welfare analysis. Rename drops the `_debug` suffix. Imports `household.vfi`, `simulation`, `equilibrium`. |
| `analysis/experiments.py` | `experiments.py` | Counterfactual experiments. Same dependency layer as welfare. |

### Top-level scripts (layers 5-6)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `calibration.py` | `full_calibration.py` | **OUT OF SCOPE for refactoring.** Only update import paths. Keep at top level because it's an entry point. |
| `plots.py` | `plot_creation.py` | Rename. Non-njit plotting code. Keep at top level because it's presentation layer, not model logic. |
| `run.py` | `solve_epsilons.py` (live portion only: ~54 lines) | Clean entry point. The ~2,680 lines of dead code will already be removed in Phase 3. The remaining live code orchestrates the coefficient chain and welfare analysis. |

## Circular import analysis

Checked all proposed import paths against the dependency graph:

1. **`household/` sub-package:** `vfi.py` -> `{continuation, stayer, renter, buyer}.py`. All one-directional. No cycles.
2. **`simulation/` sub-package:** `distribution.py` -> `{transitions, decisions, buyer_sim, mortgage_sim}.py`. All one-directional. No cycles.
3. **`equilibrium/` -> `household/` + `simulation/`:** One-directional (higher layer imports lower layer). No cycles.
4. **`analysis/` -> `equilibrium/` + `simulation/` + `household/`:** One-directional. No cycles.
5. **Cross-package at same layer:** `household/vfi.py` (layer 2) does NOT import `simulation` (layer 2). `simulation` does NOT import `household/vfi.py`. They are independent at the same layer. No cycles.
6. **`interp.py` and `utils.py`:** Both are layer 0. Neither imports the other. No cycles.
7. **`ols_numba` move:** Currently in `misc_functions.py` (layer 0), proposed for `equilibrium/market_clearing.py` (layer 3). Since `ols_numba` is only called by `coeff_updater` (in `equilibrium/solver.py`), this move is safe. However, if `ols_numba` stays in `utils.py`, that also works. **Recommend moving to `equilibrium/market_clearing.py`** to avoid a misleading dependency from equilibrium to utils just for OLS.

## Files deleted

| File | Reason |
|---|---|
| `simulation` (no .py extension) | Near-duplicate of `simulation.py`. Already flagged for deletion in Phase 3 (dead code removal, DONE). |
| `par_epsilons.py` | Replaced by `config.py`. |
| `grids.py` (current standalone) | Merged into the new `grids.py` (which also absorbs `grid_creation.py`). |
| `grid_creation.py` | Merged into the new `grids.py`. |
| `misc_functions.py` | Split into `utils.py`, `interp.py`, `income.py`, and `equilibrium/market_clearing.py`. |
| `utility_epsilons.py` | Renamed to `utility.py`. |
| `LoM_epsilons.py` | Renamed to `lom.py`. |
| `household_problem_epsilons_nolearning.py` | Renamed to `household/vfi.py`. |
| `continuation_value_nolearning.py` | Renamed to `household/continuation.py`. |
| `stayer_problem.py` | Renamed to `household/stayer.py`. |
| `stayer_problem_renter.py` | Renamed to `household/renter.py`. |
| `buyer_problem_epsilons.py` | Renamed to `household/buyer.py`. |
| `buyer_problem_simulation.py` | Renamed to `simulation/buyer_sim.py`. |
| `mortgage_choice_simulation.py` | Merged into `simulation/mortgage_sim.py`. |
| `mortgage_choice_simulation_exc.py` | Merged into `simulation/mortgage_sim.py`. |
| `simulate_initial_joint.py` | Renamed to `simulation/initial_joint.py`. |
| `proper_welfare_debug.py` | Renamed to `analysis/welfare.py`. |
| `experiments.py` | Moved to `analysis/experiments.py`. |
| `moments.py` | Moved to `analysis/moments.py`. |
| `plot_creation.py` | Renamed to `plots.py`. |
| `solve_epsilons.py` | Replaced by `run.py` (live portion only). |
