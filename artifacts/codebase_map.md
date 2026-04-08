# Codebase Map -- Post-Restructure Snapshot

> Generated 2026-04-08.  Definitive inventory after the flat-file-to-package
> migration.  The old shim layer has been removed; every module now lives
> under `model/`.

---

## 1. Package layout

```
clean_the_mess/
  full_calibration.py          (out of scope -- legacy flat-import calibration runner)
  model/
    __init__.py                (empty)
    config.py
    grids.py
    grids_util.py
    interp.py
    lom.py
    plots.py
    run.py
    tauchen.py
    utility.py
    utils.py
    analysis/
      __init__.py              (empty)
      experiments.py
      moments.py
      welfare.py
    equilibrium/
      __init__.py              (empty)
      solver.py
    household/
      __init__.py              (empty)
      buyer.py
      continuation.py
      renter.py
      stayer.py
      vfi.py
    simulation/
      __init__.py              (empty)
      buyer_sim.py
      distribution.py
      excess_demand.py
      initial_joint.py
      mortgage_sim.py
      mortgage_sim_exc.py
```

---

## 2. File inventory

### 2.1 Top-level model modules

| File | Lines | Purpose | Key functions / classes | Imports (internal) |
|---|---|---|---|---|
| `model/__init__.py` | 0 | Package marker | -- | -- |
| `model/config.py` | 151 | Single source of truth for all calibrated model parameters | `create_par_dict()`, module-level `par_dict` | `numpy` |
| `model/grids.py` | 200 | Grid construction for the full model: state-space grids, Tauchen discretisation, PTI mortgage limits | `create(par, experiment)`, `nonlinspace_jit()`, `equilogspace()`, `net_payment_frac()`, `max_mortgage_size()` | `model.tauchen`, `model.utils`, `scipy.optimize.brentq` |
| `model/grids_util.py` | 64 | Standalone grid-spacing utilities (no model dependencies) | `nonlinspace_jit()`, `equilogspace()` | `numpy`, `numba` |
| `model/interp.py` | 278 | Numba-compiled interpolation routines (1D-4D) plus binary search helpers | `binary_search()`, `interp_1d()`, `interp_2d()`, `interp_3d()`, `interp_4d()`, `fast_interp_all()`, `binary_search_sim()` | `numpy`, `numba` |
| `model/lom.py` | 37 | Law of Motion for house prices -- evaluates Chebyshev polynomial forecasting rule | `LoM(grids, t_index, vCoeff)`, aliases `LoM_C`, `LoM_NC` | `numba` |
| `model/plots.py` | 612 | Visualisation: price paths, rental price paths, distribution snapshots, stacked-area stock trajectories | `plot_pricepaths()`, `plot_distribution_2026()`, `plot_rentalpricepaths()`, `plot_stock_trajectories()` | `model.lom`, `model.tauchen`, `model.config`, `model.utils`, `model.grids`, `model.analysis.experiments`, `model.equilibrium.solver`, `model.household.vfi`, `model.simulation.distribution`, `matplotlib` |
| `model/run.py` | 53 | CLI entry point -- loads parameters, creates grids, runs welfare analysis | `main()` | `model.utils`, `model.tauchen`, `model.config`, `model.grids`, `model.analysis.welfare` |
| `model/tauchen.py` | 142 | Tauchen method for AR(1) discretisation; lifecycle income profile; invariant distribution utilities | `tauchen()`, `initial_dist()`, `weight_matrix()`, `lifecycle()`, `median_inc()`, `invar_dist()`, `combine_vectors()` | `numpy`, `scipy.stats.norm` |
| `model/utility.py` | 66 | CRRA utility, marginal utility, bequest functions, rental price calculation, renter optimal shares | `u()`, `u_c()`, `W_bequest()`, `Q_bequest()`, `rental_price_calc()`, `renter_solve()` | `numpy`, `numba` |
| `model/utils.py` | 123 | Miscellaneous helpers: jitclass builder, OLS, net income, DoubleGrid; re-exports interp functions for backward compatibility | `construct_jitclass()`, `DoubleGrid()`, `lininterp_zero_crossing()`, `maxRow()`, `ols_numba()`, `net_income()` | `model.interp` (re-exports `binary_search`, `binary_search_sim`, `interp_2d`-`interp_4d`), `numpy`, `numba` |

### 2.2 analysis/

| File | Lines | Purpose | Key functions | Imports (internal) |
|---|---|---|---|---|
| `analysis/__init__.py` | 0 | Sub-package marker | -- | -- |
| `analysis/experiments.py` | 50 | Policy experiment runners: full-information shock, generate 2026 distribution, full-information experiment wrapper | `full_information_shock()`, `gen_distribution_now()`, `full_information_experiment()` | `model.household.vfi`, `model.simulation.distribution`, `model.lom`, `model.equilibrium.solver`, `model.grids` |
| `analysis/moments.py` | 539 | Computes model moments for calibration: homeownership shares, net-worth distributions, housing wealth percentiles | `calc_moments()`, `homeowner_renter_shares()`, `end_of_life_NW()` | `model.utils`, `model.interp`, `model.lom` |
| `analysis/welfare.py` | 357 | Welfare analysis: certainty-equivalent welfare losses for alive generations and newborns; expenditure-equivalent tax calculation | `initial_welfare()`, `solve()`, `find_expenditure_equiv()`, `grid_adjust()`, `grid_adjust_rentshape()`, `find_zero_linear()`, `compute_p_left()` | `model.utils`, `model.household.vfi`, `model.simulation.distribution`, `model.equilibrium.solver`, `model.simulation.initial_joint` |

### 2.3 equilibrium/

| File | Lines | Purpose | Key functions | Imports (internal) |
|---|---|---|---|---|
| `equilibrium/__init__.py` | 0 | Sub-package marker | -- | -- |
| `equilibrium/solver.py` | 637 | General-equilibrium solver: iterates on Chebyshev price coefficients until forecasting-rule convergence; nested price-clearing via secant/bisection | `generate_pricepath()`, `find_coefficients()`, `coeff_updater()`, `initialise_coefficients_ss()`, `house_prices_algorithm()`, `secant_method_system_2d()`, `bisection_root_finding()`, `check_convergence()`, `precompute_market_data()`, `compute_excess_demand_pair()` | `model.household.vfi`, `model.simulation.distribution`, `model.lom`, `model.utils` |

### 2.4 household/

| File | Lines | Purpose | Key functions | Imports (internal) |
|---|---|---|---|---|
| `household/__init__.py` | 0 | Sub-package marker | -- | -- |
| `household/buyer.py` | 63 | Buyer problem: VFI sub-problem choosing house size and LTV for a household transitioning from renter/seller to owner | `solve()` | `model.utility`, `model.interp` |
| `household/continuation.py` | 588 | Continuation-value computation for owners (coastal/non-coastal) and renters; integrates over flood damage, income shocks, and Markov transitions | `solve_last_period_owners_C()`, `solve_last_period_owners_NC()`, `solve_last_period_renters()`, `solve_owners_C()`, `solve_owners_NC()`, `solve_renters()`, `compute_p_left()` | `model.utils`, `model.utility`, `model.interp` |
| `household/renter.py` | 117 | Renter sub-problem: endogenous grid method for renter's consumption-savings with location choice | `solve()` | `model.utility`, `model.grids` (for `nonlinspace_jit`), `model.interp` |
| `household/stayer.py` | 97 | Stayer sub-problem: endogenous grid method for owner staying in current house with upper-envelope to handle non-concavities | `solve()` | `model.utility`, `model.grids` (for `nonlinspace_jit`) |
| `household/vfi.py` | 373 | Top-level VFI orchestrator: backward induction over time, age, belief type, and amenity preference; dispatches to buyer/stayer/renter/continuation sub-solvers | `solve()`, `solve_ss()`, `precompute_coastal_stayer_inputs()`, `precompute_noncoastal_stayer_inputs()`, `precompute_mover_inputs()` | `model.household.continuation`, `model.household.stayer`, `model.household.renter`, `model.household.buyer`, `model.lom`, `model.utils` |

### 2.5 simulation/

| File | Lines | Purpose | Key functions | Imports (internal) |
|---|---|---|---|---|
| `simulation/__init__.py` | 0 | Sub-package marker | -- | -- |
| `simulation/buyer_sim.py` | 68 | Simulation-time buyer decision: evaluates value of buying each house-size at each savings grid point | `solve()` | `model.utils` |
| `simulation/distribution.py` | 1035 | Distribution dynamics: stationary-distribution finder (bequest loop); forward simulation of joint distribution over age, savings, housing, LTV, income | `stat_dist_finder()`, `update_dist_continuous()` (+ numerous inline helper blocks) | `model.interp`, `model.lom`, `model.utils`, `model.simulation.buyer_sim`, `model.utility`, `model.simulation.mortgage_sim`, `model.simulation.mortgage_sim_exc`, `model.simulation.initial_joint`, `model.tauchen` |
| `simulation/excess_demand.py` | 253 | Excess-demand computation for market clearing: given candidate prices, simulates one-period transitions and returns excess housing demand for coastal and non-coastal markets | `excess_demand_continuous()` | `model.interp`, `model.lom`, `model.utils`, `model.simulation.buyer_sim`, `model.utility`, `model.simulation.mortgage_sim`, `model.simulation.mortgage_sim_exc`, `model.simulation.initial_joint`, `model.tauchen` |
| `simulation/initial_joint.py` | 63 | Generates the initial joint distribution over savings and income for newborn cohorts, using log-normal wealth and logit no-wealth probabilities | `initial_joint()`, `norm_cdf()` | `numpy`, `math` |
| `simulation/mortgage_sim.py` | 86 | Simulation-time mortgage stayer decision with full LTV tracking (returns m_out and ltv_out for distribution updates) | `solve()` | `model.utils`, `model.interp` |
| `simulation/mortgage_sim_exc.py` | 132 | Simulation-time mortgage stayer decision for excess-demand computation (returns only value, no allocation tracking) | `solve()` | `model.utils`, `model.interp` |

### 2.6 Out-of-scope file

| File | Lines | Purpose | Key functions | Imports (internal) |
|---|---|---|---|---|
| `full_calibration.py` | 192 | Legacy calibration runner using nlopt (G_MLSL_LDS + Nelder-Mead); matches 7 model moments to data. Uses **old flat imports** (not yet migrated to `model.*`) | `f(x, grad)`, `main()` | `equilibrium`, `misc_functions`, `tauchen`, `grid_creation`, `household_problem`, `simulation`, `moments`, `lom`, `nlopt` |

---

## 3. Totals

| Metric | Value |
|---|---|
| Python files | 31 (30 in `model/` + 1 out-of-scope) |
| Total lines | 6,376 |
| Lines in `model/` only | 6,184 |
| Non-empty `__init__.py` | 0 |
| Sub-packages | 4 (`analysis`, `equilibrium`, `household`, `simulation`) |
