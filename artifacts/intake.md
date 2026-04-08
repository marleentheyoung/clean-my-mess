# Intake

Collected 2026-04-08. Source: user description + `context.md`.

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

## Entry points

1. `solve_epsilons.py:main()` -- master entry point, but ~95% is dead/commented code
2. `full_calibration.py:f()` -- NLOpt calibration wrapper
3. `experiments.py` -- counterfactual experiments

## Known outputs

- Price coefficients: `vCoeff_C`, `vCoeff_NC` (Chebyshev)
- Welfare equivalents (consumption equivalents)
- Calibration moments: homeownership rate, median net worth, housing wealth
- Excel files (written with `.xslx` typo in extension)

## Target end state

Diagnostic report with effort-graded fix proposals. No migrations yet. Preserve exact numerical functionality.

## Known landmines

- **Numba boundary is load-bearing**: @njit cannot be removed from ANY function. Tested: removing @njit from orchestration functions causes measurable performance regression due to njit boundary crossing overhead in equilibrium iteration loop. All refactoring must work within numba's type system.
- `simulation` file without `.py` extension is a near-duplicate of `simulation.py`
- `full_calibration.py` uses `nlopt` (external optimizer)
- `construct_jitclass` pattern in `misc_functions.py` creates numba jitclass instances -- must be preserved for `par` and `grids`
- `@njit` functions cannot accept Python classes, dataclasses, **kwargs, or default arguments
- context.md sections 3 (remove @njit from orchestration) and 4 (introduce dataclasses) are **REJECTED**
