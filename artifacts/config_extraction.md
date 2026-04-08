# Config Extraction

Every hardcoded magic number that should be extracted into `config.py`, categorized as either:
- **Calibration output** -- keep in `par_dict` (these are calibrated parameter values)
- **Solver setting** -- extract to a `SOLVER_SETTINGS` dict or module-level constant
- **Grid construction** -- extract to `par_dict` or a `GRID_SETTINGS` dict
- **Domain constant** -- extract to a named constant

## Sentinel values

| # | File | Line(s) | Current value | Proposed name | Type | Notes |
|---|------|---------|---------------|---------------|------|-------|
| 1 | `buyer_problem_epsilons.py` | 16-17 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Used as "negative infinity" for value function initialization in 4 files |
| 2 | `buyer_problem_simulation.py` | 15 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Same sentinel |
| 3 | `mortgage_choice_simulation.py` | 11 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Same sentinel |
| 4 | `mortgage_choice_simulation_exc.py` | 10, 72 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Same sentinel, appears twice |
| 5 | `continuation_value_nolearning.py` | 187-188, 190, 224, 229, 379-380, 384, 418, 423 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Same sentinel, appears 10 times |
| 6 | `simulation.py` | 201, 242, 459, 517, 1127, 1158, 1164, 1195 | `-1e12` | `NEG_INF_SENTINEL` | Domain constant | Same sentinel, appears 8 times |
| 7 | `mortgage_choice_simulation.py` | 74 | `-1e12+1e-8` | `NEG_INF_SENTINEL + 1e-8` | Domain constant | Comparison threshold derived from sentinel |

**Implementation:** Define `NEG_INF_SENTINEL = -1e12` in `config.py`. Import it in all 6 files. This gives a single point of control and makes the intent explicit.

---

## Retirement income fraction

| # | File | Line | Current value | Proposed name | Type | Notes |
|---|------|------|---------------|---------------|------|-------|
| 8 | `grid_creation.py` | 109 | `0.7` | `RETIREMENT_INCOME_FRACTION` | Calibration output -> `par_dict` | Retirement income = 70% of pre-retirement peak. Used in `net_payment_frac`. |
| 9 | `misc_functions.py` | 371 | `0.7` | `RETIREMENT_INCOME_FRACTION` | Calibration output -> `par_dict` | Same 0.7 in `net_income`. Must stay in sync with #8. |

**Implementation:** Add `'retirement_income_fraction': 0.7` to `par_dict`. Reference as `par.retirement_income_fraction` in both locations.

---

## Year constants

| # | File | Line(s) | Current value | Proposed name | Type | Notes |
|---|------|---------|---------------|---------------|------|-------|
| 10 | `grid_creation.py` | 37 | `2026`, `1998` | `EXPERIMENT_YEAR`, `MODEL_START_YEAR` | Domain constant | `int((2026-1998)/par.time_increment)` computes the experiment start period |
| 11 | `equilibrium.py` | 39-40 | `2026`, `1998` | `EXPERIMENT_YEAR`, `MODEL_START_YEAR` | Domain constant | Same computation in `generate_pricepath` |
| 12 | `plot_creation.py` | 28-29, 49, 331-332, 352, 420, 515 | `1998` | `MODEL_START_YEAR` | Domain constant | Used to convert time indices to calendar years |

**Implementation:** Define `MODEL_START_YEAR = 1998` and `EXPERIMENT_YEAR = 2026` in `config.py`. These are domain constants, not calibration outputs.

---

## Damage states and flood distribution

| # | File | Line | Current value | Proposed name | Type | Notes |
|---|------|------|---------------|---------------|------|-------|
| 13 | `grid_creation.py` | 40 | `np.array([1, 0.9, 0.7, 0.3])` | Already in grids as `vZ` | Grid construction | Damage multipliers. Element 0 = no damage (1.0). Elements 1-3 = conditional damage states. Already stored in `grids.vZ`. |
| 14 | `grid_creation.py` | 41 | `np.array([1, 0.4, 0.4, 0.2])` | Already in grids as `vPDF_z` | Grid construction | Conditional probabilities. Element 0 = P(no damage\|no flood) = 1. Elements 1-3 = P(damage\|flood). Already stored in `grids.vPDF_z`. Intentionally does not sum to 1 (see red_flags.md #16). |

**Implementation:** These are already stored in the grids object but are hardcoded inline in `grid_creation.py`. Move the definitions to `par_dict` so they can be changed from one location:
```python
'vZ': np.array([1.0, 0.9, 0.7, 0.3]),
'vPDF_z': np.array([1.0, 0.4, 0.4, 0.2]),
```
Then in `grid_creation.create()`, read from `par.vZ` and `par.vPDF_z` instead of defining inline.

---

## Grid construction magic numbers

| # | File | Line | Current value | Proposed name | Type | Notes |
|---|------|------|---------------|---------------|------|-------|
| 15 | `grid_creation.py` | 44 | `1.5` (vL_sim upper bound) | Add to `par_dict` as `dL_sim_max` | Grid construction | Maximum LTV on simulation grid |
| 16 | `grid_creation.py` | 44 | `35` (vL_sim points) | Add to `par_dict` as `iNl_sim` | Grid construction | Number of LTV simulation grid points |
| 17 | `grid_creation.py` | 46 | `1.50` (vH lower bound) | Already `vH=np.linspace(1.50, par.h_max, 3)` | Grid construction | Minimum house size. Should be `par.h_min`. |
| 18 | `grid_creation.py` | 46 | `3` (vH points) | Add to `par_dict` as `iNh` | Grid construction | Number of house sizes. Currently hardcoded after commenting out the 6-point version. |
| 19 | `grid_creation.py` | 47 | `np.array([1.17, 1.92])` | Add to `par_dict` as `vH_renter` | Grid construction | Rental housing sizes |
| 20 | `grid_creation.py` | 76 | `1.3` (vL upper bound) | Add to `par_dict` as `dL_max` | Grid construction | Maximum LTV on VFI grid |
| 21 | `grid_creation.py` | 76 | `20` (vL points) | Add to `par_dict` as `iNl` | Grid construction | Number of LTV grid points |
| 22 | `grid_creation.py` | 57, 59, 61, 62, 63 | `1.4` (nonlinspace curvature) | Add to `par_dict` as `dGrid_curvature` | Grid construction | Non-linear grid spacing parameter. Used identically in 5 calls. |
| 23 | `grid_creation.py` | 61 | `0.01` (vM lower bound) | Add to `par_dict` as `dM_min` | Grid construction | Minimum cash-in-hand |
| 24 | `grid_creation.py` | 93 | `np.linspace(0,5,2)` | Add to `par_dict` as `vLkeps` | Grid construction | Learning kernel epsilon grid |
| 25 | `grid_creation.py` | 96 | `np.array([0.58, 0.42])` | Add to `par_dict` as `vTypes` | Calibration output | Population fractions: 58% realists, 42% optimists |

**Implementation:** Add all of these to `par_dict` in `config.py`. The grid construction function should read from `par` rather than using inline literals.

---

## Equilibrium solver settings

| # | File | Line(s) | Current value | Proposed name | Type | Notes |
|---|------|---------|---------------|---------------|------|-------|
| 26 | `equilibrium.py` | 137 | `15` | `TRANSITION_MAX_ITER` | Solver setting | Max iterations for `find_coefficients` outer loop |
| 27 | `equilibrium.py` | 167 | `0.001` | `COEFF_CONVERGENCE_TOL_TRANSITION` | Solver setting | Convergence tolerance for transition coefficients (multiplied by `rho`) |
| 28 | `equilibrium.py` | 182 | `0.5` | `TRANSITION_DAMPING` | Solver setting | Damping factor (rho) for coefficient updating in `coeff_updater` |
| 29 | `equilibrium.py` | 202 | `25` | `SS_MAX_ITER` | Solver setting | Max iterations for `initialise_coefficients_ss` |
| 30 | `equilibrium.py` | 204 | `0.4` | `SS_INITIAL_DAMPING` | Solver setting | Initial damping factor for steady-state iteration |
| 31 | `equilibrium.py` | 237-240 | `0.25` | `SS_BISECTION_BOUND_WIDTH` | Solver setting | Bisection search bound width around LoM price in SS |
| 32 | `equilibrium.py` | 275 | `0.0005` | `COEFF_CONVERGENCE_TOL_SS` | Solver setting | Convergence tolerance for SS coefficients (multiplied by `rho`) |
| 33 | `equilibrium.py` | 257 | `0.1` | `MIN_DAMPING` | Solver setting | Minimum damping factor (floor for rho reduction) |
| 34 | `equilibrium.py` | 354 | `1e-5` (tol) | `BISECTION_TOL` | Solver setting | Bisection root-finding tolerance |
| 35 | `equilibrium.py` | 354 | `50` (max_iter) | `BISECTION_MAX_ITER` | Solver setting | Bisection max iterations |
| 36 | `equilibrium.py` | 421 | `1e-5` (tol) | `SECANT_TOL` | Solver setting | Secant method tolerance |
| 37 | `equilibrium.py` | 421 | `1e-3` (tol_wider) | `SECANT_TOL_WIDER` | Solver setting | Wider tolerance for early exit after 9 iterations |
| 38 | `equilibrium.py` | 421 | `30` (max_iter) | `SECANT_MAX_ITER` | Solver setting | Secant method max iterations |
| 39 | `equilibrium.py` | 447, 459 | `1e-15` | `SINGULARITY_TOL` | Solver setting | Tolerance for detecting singular matrices in secant method |
| 40 | `equilibrium.py` | 523 | `1e-3` (price_tol) | `MC_PRICE_TOL` | Solver setting | Market clearing price tolerance |
| 41 | `equilibrium.py` | 524 | `1e-5` (error_tol) | `MC_ERROR_TOL` | Solver setting | Market clearing error tolerance |
| 42 | `equilibrium.py` | 525 | `15` | `MC_MAX_ITER` | Solver setting | Market clearing max iterations |
| 43 | `equilibrium.py` | 529-537 | `0.005` | `SECANT_INITIAL_SPREAD` | Solver setting | Initial triangle spread for 2D secant starting points |
| 44 | `equilibrium.py` | 77-78 | `0.1` (bound_c_l, bound_nc_l) | `PRICE_LOWER_BOUND` | Solver setting | Lower bound for house prices in generate_pricepath |
| 45 | `equilibrium.py` | 80-83 | `0.1` (bound offsets) | `BISECTION_BOUND_OFFSET` | Solver setting | Offset for bisection bounds around guess |
| 46 | `equilibrium.py` | 587 | `5e-4` | `EARLY_EXIT_PRICE_TOL` | Solver setting | Early exit threshold for small price changes |

**Implementation:** Create a `SOLVER_SETTINGS` dict in `config.py`:
```python
SOLVER_SETTINGS = {
    # Transition path (find_coefficients)
    'transition_max_iter': 15,
    'coeff_convergence_tol_transition': 0.001,
    'transition_damping': 0.5,
    
    # Steady state (initialise_coefficients_ss)
    'ss_max_iter': 25,
    'ss_initial_damping': 0.4,
    'ss_bisection_bound_width': 0.25,
    'coeff_convergence_tol_ss': 0.0005,
    'min_damping': 0.1,
    
    # Market clearing
    'mc_price_tol': 1e-3,
    'mc_error_tol': 1e-5,
    'mc_max_iter': 15,
    'price_lower_bound': 0.1,
    'bisection_bound_offset': 0.1,
    'early_exit_price_tol': 5e-4,
    
    # Root finding
    'bisection_tol': 1e-5,
    'bisection_max_iter': 50,
    'secant_tol': 1e-5,
    'secant_tol_wider': 1e-3,
    'secant_max_iter': 30,
    'secant_initial_spread': 0.005,
    'singularity_tol': 1e-15,
}
```

Since these settings are used inside `@njit` functions, they should be passed as arguments or accessed from a numba-compatible structure. Two options:
1. **Pass as arguments** to the solver functions (cleanest but adds arguments)
2. **Convert to a jitclass** using `construct_jitclass(SOLVER_SETTINGS)` and pass that object

Recommend option 1 for settings that vary between calls (tolerances) and option 2 for settings that are fixed for the entire run.

---

## Initial wealth distribution

| # | File | Line | Current value | Proposed name | Type | Notes |
|---|------|------|---------------|---------------|------|-------|
| 47 | `simulate_initial_joint.py` | 39 | `0.95` | `INITIAL_WEALTH_CDF_CUTOFF` | Domain constant | CDF cutoff for right tail of initial wealth distribution |
| 48 | `simulate_initial_joint.py` | 39 | `5.014401` | `INITIAL_WEALTH_RATIO_CUTOFF` | Calibration output | 95th percentile of empirical ratio initial wealth / initial median income. Add to `par_dict`. |

---

## Calibration parameters already in `par_dict` -- KEEP THERE

These are hardcoded in `par_epsilons.py` but they ARE the calibration output. They should stay in `par_dict`, not be extracted further. They are listed here for completeness.

| `par_dict` key | Value | Economic meaning |
|---|---|---|
| `dBeta` | `0.940074219**time_increment` | Discount factor (biennial) |
| `dDelta` | `1-(1-0.015)**time_increment` | Housing depreciation rate |
| `dPsi` | `0.00481015625` | Minimum consumption floor |
| `r` | `1.03**time_increment-1` | Risk-free interest rate |
| `r_m` | `1.04**time_increment-1` | Mortgage interest rate |
| `dKappa_sell` | `0.07` | Selling transaction cost |
| `dKappa_buy` | `0` | Buying transaction cost |
| `dXi_foreclosure` | `0.8` | Foreclosure recovery rate |
| `dNu` | `44.5312500` | Bequest motive strength |
| `dZeta` | `0.01` | Mortgage origination fee (proportional) |
| `dZeta_fixed` | `1/26` | Mortgage origination fee (fixed) |
| `lambda_pti` | `0.25` | Payment-to-income ratio limit |
| `max_ltv` | `0.95` | Maximum loan-to-value ratio |
| `damage_states` | `3` | Number of flood damage states |
| `dLambda` | `0.8` | Housing service weight |
| `dGamma` | `1/1.25` | Intratemporal elasticity of substitution |
| `dSigma` | `2` | Risk aversion |
| `b_bar` | `3.18164063` | Bequest floor |
| `dPhi` | `0.18` | Minimum mortgage payment fraction |
| `iNb` | `60` | Number of savings grid points |
| `iBmin` | `0` | Minimum savings |
| `iBmax` | `27` | Maximum savings |
| `dZ` | `0.8` | Flood damage fraction (used as default) |
| `h_max` | `5.15` | Maximum house size |
| `dXi_min` | `1-0.0223437500` | Minimum amenity preference |
| `dXi_max` | `1+0.0223437500` | Maximum amenity preference |
| `iXin` | `7` | Number of amenity grid points |
| `alpha_0` | `0.4` | Housing weight in utility |
| `dRho` | `0.97` | AR(1) income persistence |
| `dSigmaeps` | `0.20` | AR(1) income shock std dev |
| `iNumStates` | `5` | Number of income states (MUST BE ODD) |
| `iNj` | `30` | Number of age periods |
| `j_ret` | `23` | Retirement age period |
| `dNC_frac` | `0.5` | Non-coastal housing stock fraction |
| `dC_frac` | `0.5` | Coastal housing stock fraction |
| `dTheta` | `1.5/2.5` | Relative coastal amenity |
| `dL` | `0.311` | Labor supply |
| `dOmega` | `0.010156250` | Amenity constant |
| `tau_0` | `0.75` | Tax function parameter |
| `tau_1` | `0.151` | Tax function progressivity |

All of these should stay in `par_dict` in `config.py`. The significant-digit precision (e.g., `0.940074219`, `44.5312500`, `3.18164063`, `0.0223437500`, `0.010156250`) indicates these are machine-precision calibration outputs. Add a comment:
```python
# Values below are calibrated outputs -- do not round or modify without re-running calibration
```

---

## Summary

| Category | Count | Action |
|---|---|---|
| Sentinel values (`-1e12`) | 25+ occurrences across 6 files | Extract to `NEG_INF_SENTINEL` constant |
| Retirement income fraction (`0.7`) | 2 files | Add to `par_dict` |
| Year constants (`1998`, `2026`) | 3 files, 12 occurrences | Extract to named constants |
| Grid construction literals | 11 values in `grid_creation.py` | Add to `par_dict` |
| Solver settings | 21 values in `equilibrium.py` | Extract to `SOLVER_SETTINGS` dict |
| Initial wealth cutoffs | 2 values in `simulate_initial_joint.py` | 1 constant + 1 to `par_dict` |
| Calibration parameters | ~40 values | KEEP in `par_dict` (already there) |
