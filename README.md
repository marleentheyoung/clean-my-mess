# Heterogeneous-Agent Housing Model with Flood Risk

A heterogeneous-agent overlapping-generations model with two housing markets (coastal and inland), where households choose to rent, buy, sell, stay, or default on mortgages. Agents have heterogeneous beliefs about flood risk ("sceptics" vs "realists"), and house prices are endogenously determined via excess demand in both markets. The model uses Chebyshev polynomial laws of motion for price forecasting, and computes welfare costs of rising flood risk through expenditure-equivalent analysis.

## Setup

Requires Python 3.11+. Install dependencies manually (no `setup.py` yet):

- numpy
- numba
- scipy
- pandas
- matplotlib
- quantecon

## How to run

```bash
cd clean_the_mess
python -c "from model.run import main; main()"
```

This loads calibrated parameters, creates discretised grids and Markov chains, and runs the welfare expenditure-equivalent analysis comparing steady-state vs transition-path welfare under sea-level rise. The first run takes approximately 3 minutes for Numba JIT compilation; subsequent runs within the same session are faster.

## Project structure

```
clean_the_mess/model/
├── config.py               # All model parameters (calibrated). Single source of truth.
├── utils.py                # Jitclass constructor, DoubleGrid, OLS, re-exports from interp.
├── grids.py                # Grid creation: savings, housing, income, LTV, and Markov chains.
├── tauchen.py              # Tauchen method for AR(1) income process discretisation.
├── interp.py               # 1D/2D/3D/4D interpolation with binary search (@njit, fastmath).
├── utility.py              # CRRA utility, marginal utility, bequest utility, rental price.
├── lom.py                  # Chebyshev law of motion for coastal and non-coastal prices.
├── run.py                  # Entry point: loads parameters, grids, runs welfare analysis.
├── plots.py                # Plotting functions for price paths and model output.
│
├── household/              # Value function iteration (backward induction)
│   ├── vfi.py              # VFI orchestrator: solve() for transition, solve_ss() for steady state.
│   ├── continuation.py     # Expected continuation values integrating over income, flood, choices.
│   ├── stayer.py           # EGM solution for homeowners staying in current house.
│   ├── renter.py           # EGM solution for renters choosing consumption and savings.
│   └── buyer.py            # Discrete optimisation over house size and LTV for buyers.
│
├── simulation/             # Forward simulation of the wealth distribution
│   ├── distribution.py     # Stationary distribution finder and forward distribution update.
│   ├── excess_demand.py    # Excess demand computation for market clearing.
│   ├── buyer_sim.py        # Buyer problem on the simulation grid.
│   ├── mortgage_sim.py     # Mortgage choice on the simulation grid.
│   ├── mortgage_sim_exc.py # Mortgage choice with excess payment handling.
│   └── initial_joint.py    # Initial joint distribution of wealth and income for newborns.
│
├── equilibrium/            # Outer equilibrium loop
│   └── solver.py           # find_coefficients (transition), initialise_coefficients_ss (steady state),
│                           # generate_pricepath, house_prices_algorithm (secant + bisection).
│
└── analysis/               # Post-solution analysis
    ├── moments.py          # Model moments from the simulated distribution.
    ├── welfare.py          # Welfare analysis: consumption equivalents and expenditure equivalents.
    └── experiments.py      # Counterfactual experiments (e.g., full information shock).
```

`full_calibration.py` (top level) is broken and out of scope.

## Testing

```bash
pytest tests/
```

Runs 16 tests in approximately 75 seconds (after Numba compilation). Includes import smoke tests, parameter pinning, grid creation regression, and VFI snapshot tests.

```bash
pytest tests/ -m slow
```

Includes the `find_expenditure_equiv` welfare equivalents test, which takes approximately 15 minutes.

Snapshot files (`.npz`) in `tests/snapshots/` are auto-generated on first run and used for regression testing.

## Naming conventions

The codebase uses a Hungarian-notation prefix convention inherited from Dutch econometrics:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `m` | Matrix (2D+ array) | `mMarkov`, `mDist1_c` |
| `v` | Vector (1D array) | `vCoeff_C`, `vE` |
| `d` | Scalar (float) | `dP_C`, `dBeta` |
| `i` | Integer scalar | `iNj`, `iNb` |

This convention is applied consistently in function signatures and module-level variables, but inconsistently in some local variables within @njit functions.

## Key constraints

- **All `@njit` decorators must stay.** The entire call chain from `find_coefficients` through `generate_pricepath` to `excess_demand_continuous` runs millions of iterations. Removing `@njit` from any function in the chain causes measurable performance regression.
- **Numba type system.** All data passing is limited to numba-compatible types: numpy arrays, scalars, tuples, and numba typed dicts. No Python dataclasses, keyword arguments, or wrapper layers in the `@njit` call chain.
- **`full_calibration.py` is broken.** It calls functions that no longer exist. Calibration is out of scope for now.
