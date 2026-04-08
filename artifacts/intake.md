# Intake

Collected 2026-04-08. Updated 2026-04-08 with experiential context from user.

## Project description

Heterogeneous-agent overlapping-generations housing model with:
- Two housing markets: coastal (flood-exposed) and inland (non-coastal)
- Households choose: rent, buy coastal, buy inland, stay, sell, default on mortgage
- Heterogeneous beliefs about flood risk ("sceptics" vs "realists")
- Endogenous house prices cleared via excess demand in both markets
- Chebyshev polynomial law of motion for prices over time
- Mortgage choice with LTV constraints, refinancing, minimum payments
- Welfare analysis comparing steady state vs transition paths
- Calibration via method of simulated moments (NLOpt)

Solved via backward induction (VFI) over a 7-dimensional state space (age x belief type x amenity preference x savings x house size x mortgage LTV x income), then forward simulation of the wealth distribution, iterated to find equilibrium price coefficients.

**Paper goal:** Welfare cost of rising flood risk in the housing market, given that some households underestimate flood risk. Future: analyse which policies reduce this welfare cost (not coded yet).

## Entry points

1. `solve_epsilons.py:main()` -- master entry point, but ~95% is dead/commented code. Live path only runs welfare analysis.
2. `full_calibration.py:f()` -- NLOpt calibration wrapper. **BROKEN** (calls nonexistent functions).
3. `experiments.py` -- counterfactual experiments (full information shock).

**Status:** "Not too much" works right now (user's words). The main working path is: grid creation → solve_ss → stat_dist_finder → visual inspection of distributions.

## Coefficient chain (from user)

The model produces coefficients in this order:
1. `initialise_coefficients_ss(sceptics=True)` → vCoeff_C_initial, vCoeff_NC_initial
2. `initialise_coefficients_ss(sceptics=False)` → vCoeff_C_initial_RE, vCoeff_NC_initial_RE (**user forgot to save these**)
3. `initialise_coefficients_ss(initial=False, sceptics=False)` → vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE
4. `initialise_coefficients_ss(initial=False, sceptics=True)` → vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE
5. `find_coefficients(sceptics=True, dP_C_initial=vCoeff_C_initial[0], ...)` → vCoeff_C, vCoeff_NC
6. `find_coefficients(sceptics=False, dP_C_initial=vCoeff_C_initial_RE[0], ...)` → vCoeff_C_RE, vCoeff_NC_RE

User notes: "This is a really messy, user-unfriendly way of gathering outputs. If you can propose a better way, feel free to adjust."

## Known outputs

- Price coefficients: `vCoeff_C`, `vCoeff_NC` (Chebyshev) — from converged equilibrium
- Welfare equivalents (consumption equivalents) — from find_expenditure_equiv
- Calibration moments: homeownership rate, median net worth, housing wealth
- VFI solutions and stationary distribution (key intermediate checkpoints)
- Parameters in `par_epsilons.py` are **final** calibration

## Runtime characteristics

- `solve_ss`: a few minutes (faster with iXin:3, iNumStates:3)
- `stat_dist_finder`: a few minutes
- `initialise_coefficients_ss`: 15-60 minutes depending on initial guess quality
- `find_coefficients`: several hours at least
- Numba first-compilation: a few minutes
- Runs on a laptop. No cluster needed.
- Equilibrium convergence: 10-15 iterations typically. Fragile if parameters change much.

## Known landmines

- **Numba boundary is load-bearing**: @njit cannot be removed from ANY function. All refactoring must work within numba's type system.
- `simulation` file without `.py` extension is a near-duplicate of `simulation.py`
- `full_calibration.py` calls nonexistent functions (`initialise_coefficients_initial`, `solve_initial`)
- `construct_jitclass` pattern must be preserved for `par` and `grids`
- `@njit` functions cannot accept Python classes, dataclasses, **kwargs, or default arguments
- `vPDF_z` intentionally doesn't sum to 1: element 0 is P(no damage|no flood)=1, elements 1-3 are P(damage|flood) conditional probabilities
- `dBeta` is especially sensitive — changing it can push savings beyond grid limits
- `iBmax` is a key grid limit — too low breaks convergence
- Bisection fallback in equilibrium solver is very slow — if it triggers repeatedly, better to stop
- User forgot to save vCoeff_C_initial_RE / vCoeff_NC_initial_RE in solve_epsilons.py
- Calibration: user will fix themselves. Don't worry about the calibration path.

## Target end state

Clean, documented, reproducible code for journal submission. Preserve exact numerical functionality. No model extensions planned.
