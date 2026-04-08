# Proposed Structure

Target folder layout for the `model/` package, based on `context.md` section 8, validated against the dependency graph, and revised per human review.

## Human review decisions applied

- **`income.py` eliminated** — `net_income` (12 lines) moves into `utility.py` alongside other economic primitives
- **Mortgage files stay separate** — `mortgage_sim.py` and `mortgage_sim_exc.py` (different `fastmath` settings, not resolved yet)
- **Simulation split reduced** — 3 files instead of 4/7: `distribution.py` (stat_dist + update_dist + all helpers they call), `excess_demand.py` (equilibrium interface), standalone files for buyer_sim, mortgage_sim, mortgage_sim_exc, initial_joint
- **`__init__.py` files kept empty** — use explicit imports (`from model.household.vfi import solve_ss`) not re-exports. Less maintenance burden for a research codebase.
- **Shims have kill date** — removed in Step 7, same session as the moves. Deprecation comments in every shim.
- **config.py creation follows task_queue 5c.1** — not duplicated in the migration plan

## Layout

```
model/
├── __init__.py                    (empty)
├── config.py                      (from par.py + mutation fix, per task 5c.1)
├── utils.py                       (construct_jitclass, DoubleGrid, maxRow, lininterp_zero_crossing)
├── grids.py                       (merged grid_creation.py + grids.py)
├── tauchen.py                     (unchanged)
├── interp.py                      (consolidated: 1D/2D/3D/4D + binary_search + binary_search_sim)
├── utility.py                     (u, u_c, bequest, renter_solve, rental_price_calc, net_income)
├── lom.py                         (unified LoM with LoM_C/LoM_NC aliases)
├── household/
│   ├── __init__.py                (empty)
│   ├── vfi.py                     (VFI orchestrator: solve, solve_ss)
│   ├── continuation.py            (continuation values)
│   ├── stayer.py                  (EGM for homeowners)
│   ├── renter.py                  (EGM for renters)
│   └── buyer.py                   (buyer discrete optimization)
├── simulation/
│   ├── __init__.py                (empty)
│   ├── distribution.py            (stat_dist_finder, update_dist_continuous + all
│   │                               helpers they call: simulate_buy/stay/rent/*_ret,
│   │                               *_outer, continuous_decide, renter_sim, etc.)
│   ├── excess_demand.py           (excess_demand_continuous — equilibrium interface)
│   ├── buyer_sim.py               (buyer choices on simulation grid)
│   ├── mortgage_sim.py            (@njit(fastmath=True) — standard mortgage choice)
│   ├── mortgage_sim_exc.py        (@njit — extended mortgage choice, NO fastmath)
│   └── initial_joint.py           (initial wealth/income distribution)
├── equilibrium/
│   ├── __init__.py                (empty)
│   ├── solver.py                  (find_coefficients, generate_pricepath, coeff_updater)
│   ├── steady_state.py            (initialise_coefficients_ss)
│   └── market_clearing.py         (house_prices_algorithm, secant, bisection, ols_numba)
├── analysis/
│   ├── __init__.py                (empty)
│   ├── moments.py                 (moment computation)
│   ├── welfare.py                 (consumption equivalents)
│   └── experiments.py             (counterfactual experiments)
├── calibration.py                 (OUT OF SCOPE — imports only)
├── plots.py                       (plotting functions)
└── run.py                         (clean entry point)
```

## Rationale for each file/directory

### Top-level modules (layer 0 — no local imports)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `config.py` | `par.py` + extracted magic numbers | Single source of truth for parameters. Contains `create_par_dict()` (no module-level mutation). Created by task 5c.1. |
| `utils.py` | `misc_functions.py` remainder | `construct_jitclass`, `DoubleGrid`, `maxRow`, `lininterp_zero_crossing`. General utilities. |
| `grids.py` | `grid_creation.py` + `grids.py` | Merge: `nonlinspace_jit`/`equilogspace` only used by `grid_creation.create()`. Removes indirection. |
| `tauchen.py` | `tauchen.py` | Standalone income-process discretization. Clean leaf module. |
| `interp.py` | `interp.py` (already consolidated in 5b.1) | All interpolation: 1D/2D/3D/4D + binary search. Numerical guards added (5b.6). |
| `utility.py` | `utility.py` + `net_income` from misc_functions | Economic primitives: utility, marginal utility, bequest, renter intratemporal solution, rental price calculation, after-tax income. |
| `lom.py` | `lom.py` | Unified Chebyshev law of motion with LoM_C/LoM_NC backward-compat aliases. |

### Household sub-package (layers 1-2)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `household/vfi.py` | `household_problem.py` | VFI orchestrator. Imports all other household sub-modules. |
| `household/continuation.py` | `continuation_value_nolearning.py` | Continuation values (586 lines). Imports interp, utility, utils. Does NOT import vfi.py — no circular risk. |
| `household/stayer.py` | `stayer_problem.py` | EGM for homeowner-stayers. Imports utility, grids. |
| `household/renter.py` | `stayer_problem_renter.py` | EGM for renters. Imports utility, grids, interp. |
| `household/buyer.py` | `buyer_problem.py` | Buyer discrete optimization. Imports utility, interp. |

### Simulation sub-package (layers 1-2)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `simulation/distribution.py` | `simulation.py`: stat_dist_finder, update_dist_continuous, construct_m1, mortgage_matrix_solve, + all simulate_*/decide/renter helpers | **Keeps tightly-coupled functions together.** update_dist_continuous calls continuous_decide, simulate_stay, simulate_buy_outer, simulate_rent_outer, renter_sim — all in one file to avoid cross-module @njit call overhead risk. |
| `simulation/excess_demand.py` | `simulation.py`: excess_demand_continuous | Equilibrium interface. Separated because it's the API boundary between simulation and equilibrium. |
| `simulation/buyer_sim.py` | `buyer_problem_simulation.py` | Already standalone. |
| `simulation/mortgage_sim.py` | `mortgage_choice_simulation.py` | Keeps `@njit(fastmath=True)`. NOT merged with _exc. |
| `simulation/mortgage_sim_exc.py` | `mortgage_choice_simulation_exc.py` | Keeps plain `@njit`. Different fastmath setting — documented but not resolved. |
| `simulation/initial_joint.py` | `simulate_initial_joint.py` | Already standalone. |

### Equilibrium sub-package (layer 3)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `equilibrium/solver.py` | `equilibrium.py`: find_coefficients, generate_pricepath, coeff_updater, flatten_third_dim | Outer coefficient iteration loop. |
| `equilibrium/steady_state.py` | `equilibrium.py`: initialise_coefficients_ss | Steady-state price finder. Distinct convergence logic. |
| `equilibrium/market_clearing.py` | `equilibrium.py`: house_prices_algorithm, precompute_market_data, compute_excess_demand_pair, bisection, secant, check_convergence + `ols_numba` from misc_functions | Market-clearing algorithms. ols_numba moved here (only consumer is coeff_updater). |

### Analysis sub-package (layer 4)

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `analysis/moments.py` | `moments.py` | Moment computation from distributions. |
| `analysis/welfare.py` | `welfare.py` | Consumption-equivalent welfare analysis. |
| `analysis/experiments.py` | `experiments.py` | Counterfactual experiments. |

### Top-level scripts

| Proposed file | Source file(s) | Why grouped here |
|---|---|---|
| `calibration.py` | `full_calibration.py` | OUT OF SCOPE — update imports only. |
| `plots.py` | `plot_creation.py` | Plotting. Non-@njit. |
| `run.py` | `solve.py` (53 lines) | Clean entry point. |

## Circular import analysis

All import paths validated against dependency graph:

1. **household/**: vfi.py → {continuation, stayer, renter, buyer}. All one-directional. No cycles.
2. **simulation/**: distribution.py → {buyer_sim, mortgage_sim, mortgage_sim_exc, initial_joint}. excess_demand.py → distribution.py. All one-directional. No cycles.
3. **equilibrium/ → household/ + simulation/**: Higher layer imports lower. No cycles.
4. **analysis/ → equilibrium/ + simulation/ + household/**: One-directional. No cycles.
5. **Cross-package same layer**: household (layer 2) and simulation (layer 2) are independent. No cycles.

## Files deleted after migration

| Old file | Replacement |
|---|---|
| `par.py` | `model/config.py` |
| `grids.py` + `grid_creation.py` | `model/grids.py` |
| `misc_functions.py` | `model/utils.py` + `model/interp.py` + `model/utility.py` + `model/equilibrium/market_clearing.py` |
| `household_problem.py` | `model/household/vfi.py` |
| `continuation_value_nolearning.py` | `model/household/continuation.py` |
| `stayer_problem.py` | `model/household/stayer.py` |
| `stayer_problem_renter.py` | `model/household/renter.py` |
| `buyer_problem.py` | `model/household/buyer.py` |
| `simulation.py` | `model/simulation/distribution.py` + `excess_demand.py` |
| `buyer_problem_simulation.py` | `model/simulation/buyer_sim.py` |
| `mortgage_choice_simulation.py` | `model/simulation/mortgage_sim.py` |
| `mortgage_choice_simulation_exc.py` | `model/simulation/mortgage_sim_exc.py` |
| `simulate_initial_joint.py` | `model/simulation/initial_joint.py` |
| `equilibrium.py` | `model/equilibrium/solver.py` + `steady_state.py` + `market_clearing.py` |
| `moments.py` | `model/analysis/moments.py` |
| `welfare.py` | `model/analysis/welfare.py` |
| `experiments.py` | `model/analysis/experiments.py` |
| `plot_creation.py` | `model/plots.py` |
| `solve.py` | `model/run.py` |
| `lom.py` | `model/lom.py` |
| `interp.py` | `model/interp.py` |
| `tauchen.py` | `model/tauchen.py` |
| `utility.py` | `model/utility.py` |
