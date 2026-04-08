# Model Documentation

## Model overview

This is a heterogeneous-agent overlapping-generations (OLG) housing model designed to answer: **what is the welfare cost of rising flood risk in the housing market, given that some households underestimate flood risk?**

### Agent types

- **Realists** (type 0): correctly perceive time-varying flood probabilities.
- **Sceptics** (type 1): underestimate flood risk.
- Population shares are set by `vTypes` (58% sceptics, 42% realists by default).

### Two housing markets

- **Coastal**: exposed to flood damage. House values depreciate stochastically via the damage vector `vZ` with conditional probabilities `vPDF_z`. Flood probability rises over time via the `vPi_S_median` vector (biennial probabilities from 1998 to 2100+).
- **Non-coastal (inland)**: no flood risk. Serves as a safe alternative. Amenity preference parameter `g` (the `vG` grid) differentiates valuations.

### Household decisions

Each period (2-year increment), households choose:

1. **Renters**: consumption, savings, and whether to buy coastal or non-coastal housing.
2. **Homeowners**: consumption, savings, mortgage payment (minimum vs. extra), and whether to stay, sell, or default.
3. **Buyers**: house size (from `vH` grid), mortgage LTV (from `vL` grid), and location.

### State space

The full state is 7-dimensional: age (j) x belief type (k) x amenity preference (g) x savings/cash-on-hand (b/x/m) x house size (h) x mortgage LTV (l) x persistent income (e).

## Solution method

### Value Function Iteration (VFI)

The model solves household problems via backward induction from the terminal period (age `iNj`). This is implemented in `household/vfi.py`.

- **`solve_ss()`**: Solves for a single time period (steady state). Prices are constant. Time dimension collapses to 1.
- **`solve()`**: Solves over the full transition path. Iterates backward over both age and time, using the Chebyshev law of motion to forecast future prices.

### Endogenous Grid Method (EGM)

For continuous choices (consumption and savings), the stayer and renter problems use the Endogenous Grid Method:

1. Start from a grid of *end-of-period savings* (`vB`).
2. Use the Euler equation to back out optimal consumption given continuation values.
3. Invert the budget constraint to find the *beginning-of-period cash-on-hand* that rationalises each savings choice.
4. Interpolate back onto the exogenous cash-on-hand grid (`vM` or `vX`).

This avoids root-finding for the Euler equation. Implemented in `household/stayer.py` (homeowners) and `household/renter.py` (renters).

### Discrete optimisation for buyers

Buyers face a discrete choice over house size and LTV. The buyer problem (`household/buyer.py`) evaluates all feasible (house size, LTV) combinations on the cash-on-hand grid and picks the maximum. No EGM here because the choice variables are discrete.

### Continuation values

`household/continuation.py` computes expected continuation values by integrating over:

- Income transitions (Markov chain `mMarkov`)
- Flood damage realisations (states `vZ` with probabilities `vPDF_z`)
- Discrete choice outcomes (stay, sell, buy, rent, default)

This produces the `w` (expected value), `q` (expected action-value for discrete choices), and `w_wf` (welfare-adjusted value) arrays that feed into the EGM and buyer problems.

## Equilibrium

### Chebyshev law of motion

Prices are forecast using a 4th-order Chebyshev polynomial in time:

```
P(t) = c_0 + c_1*T_1(t) + c_2*T_2(t) + c_3*T_3(t) + c_4*T_4(t)
```

where `T_k` are Chebyshev polynomials of the first kind, and `t` is normalised to [-1, 1] over the model's time horizon. Implemented in `lom.py`.

The coefficient vectors `vCoeff_C` (coastal) and `vCoeff_NC` (non-coastal) are the key equilibrium objects. In steady state, only `vCoeff[0]` (the constant) is nonzero.

### Market clearing

At each time step, prices must clear both housing markets simultaneously. The algorithm (`house_prices_algorithm` in `equilibrium/solver.py`) works as:

1. **Secant method** (primary): 2D secant iteration on (P_C, P_NC) to find prices where excess demand equals zero in both markets. This is fast and used for the vast majority of iterations.
2. **Bisection fallback**: If secant fails to converge, falls back to sequential bisection on each market. Reliable but slow. If the model repeatedly falls back to bisection, the price guess or coefficients are likely poor.

### Coefficient iteration

**Steady state** (`initialise_coefficients_ss`):

1. Guess `vCoeff_C[0]`, `vCoeff_NC[0]` (constant price).
2. Solve VFI at that price. Find stationary distribution.
3. Clear the market: find the price where excess demand = 0.
4. Update guess with dampening: `vCoeff_new = rho * P_mc + (1 - rho) * P_old`.
5. Repeat until convergence. Adaptive step size: `rho` is halved when oscillation is detected.

**Transition path** (`find_coefficients`):

1. Start from converged steady-state coefficients.
2. Solve full VFI over the transition path.
3. Forward-simulate the distribution, clearing markets at each time step.
4. Regress market-clearing prices on Chebyshev basis (OLS) to get new coefficients.
5. Update with dampening: `vCoeff_new = rho * beta_OLS + (1 - rho) * vCoeff_old`.
6. Repeat until coefficient vectors converge. Typically 10-15 iterations.

## Key code-to-equation mapping

| Economic concept | Code location | Key function |
|---|---|---|
| Utility function (Cobb-Douglas over consumption and housing services, CRRA) | `utility.py` | `u()`, `u_c()` |
| Budget constraint / cash-on-hand | `household/stayer.py`, `household/renter.py` | `solve()` (EGM inversion) |
| Bellman equation (stayers) | `household/stayer.py` | `solve()` |
| Bellman equation (renters) | `household/renter.py` | `solve()` |
| Buyer's discrete choice | `household/buyer.py` | `solve()` |
| Expected continuation value (E[V']) | `household/continuation.py` | `solve_owners_C()`, `solve_renters()` |
| Income process (AR(1) + lifecycle) | `tauchen.py` | `tauchen()`, `lifecycle()` |
| Chebyshev price law of motion | `lom.py` | `LoM()` |
| Excess demand | `simulation/excess_demand.py` | `excess_demand_continuous()` |
| Stationary distribution | `simulation/distribution.py` | `stat_dist_finder()` |
| Forward distribution update | `simulation/distribution.py` | `update_dist_continuous()` |
| Market clearing (2D) | `equilibrium/solver.py` | `house_prices_algorithm()` |
| Steady-state price finding | `equilibrium/solver.py` | `initialise_coefficients_ss()` |
| Transition-path coefficients | `equilibrium/solver.py` | `find_coefficients()` |
| Welfare (consumption equivalents) | `analysis/welfare.py` | `solve()` |
| Welfare (expenditure equivalents) | `analysis/welfare.py` | `find_expenditure_equiv()` |
| Model moments | `analysis/moments.py` | `calc_moments()` |
| Full information counterfactual | `analysis/experiments.py` | `full_information_shock()` |

## Coefficient chain

The order of function calls to go from parameters to equilibrium:

```
1. config.create_par_dict()
   -> plain dict of all model parameters

2. utils.construct_jitclass(par_dict)
   -> numba jitclass `par` object

3. grids.create(par)
   -> numba jitclass `grids` object + Markov transition matrix mMarkov
   (internally calls tauchen.tauchen() for income discretisation)

4. solver.initialise_coefficients_ss(par, grids, ..., initial=True)
   -> converged vCoeff_C_initial, vCoeff_NC_initial (steady-state prices)
   For each iteration:
     a. vfi.solve_ss(grids, par, ..., dCoeff_C, dCoeff_NC)
        -> policy functions and value functions at guessed prices
     b. distribution.stat_dist_finder(...)
        -> stationary wealth distribution at guessed prices
     c. house_prices_algorithm(...)
        -> market-clearing price given the distribution
     d. Update coefficient guess with dampening

5. solver.initialise_coefficients_ss(par, grids, ..., initial=False)
   -> terminal steady-state coefficients (for terminal condition)

6. solver.find_coefficients(par, grids, ..., vCoeff_C, vCoeff_NC, ...)
   -> converged transition-path coefficients
   For each iteration:
     a. generate_pricepath(...)
        i.  vfi.solve(grids, par, ..., vCoeff_C, vCoeff_NC)
            -> full transition-path policy functions
        ii. For each t: house_prices_algorithm(...)
            -> market-clearing prices
        iii. For each t: distribution.update_dist_continuous(...)
             -> forward-simulate the distribution
     b. OLS regression of market-clearing prices on Chebyshev basis
     c. Update coefficients with dampening

7. welfare.find_expenditure_equiv(par, grids, ..., vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC)
   -> expenditure-equivalent welfare losses by agent type
```

### Calling conventions

- **`func` parameter**: `False` = solve for market-clearing prices (default for most uses). `True` = use the LoM to compute prices from coefficients (used inside `initialise_coefficients_ss`).
- **`dCoeff_C` / `dCoeff_NC` (scalar)**: When a function takes a scalar price, pass `vCoeff_C[0]` or `vCoeff_NC[0]` (the first Chebyshev coefficient = the constant term = the price level). The `d` prefix denotes scalar.
- **`sceptics` parameter**: `True` = include both belief types (k_dim=2). `False` = realists only (k_dim=1), used for the rational-expectations counterfactual.
