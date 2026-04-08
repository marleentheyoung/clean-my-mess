# Refactoring Context for Claude Code

## What this codebase is

This is a heterogeneous-agent overlapping-generations housing model with:
- Two housing markets: coastal (flood-exposed) and inland (non-coastal)
- Households choose: rent, buy coastal, buy inland, stay, sell, default on mortgage
- Heterogeneous beliefs about flood risk ("sceptics" vs "realists")
- Endogenous house prices cleared via excess demand in both markets
- Chebyshev polynomial law of motion for prices over time
- Mortgage choice with LTV constraints, refinancing, minimum payments
- Welfare analysis comparing steady state vs transition paths
- Calibration via method of simulated moments (NLOpt)

The model is solved via backward induction (VFI) over a 7-dimensional state space (age × belief type × amenity preference × savings × house size × mortgage LTV × income), then forward simulation of the wealth distribution, iterated to find equilibrium price coefficients.

## Architecture overview

The solve flow is:
1. `par_epsilons.py` → parameters
2. `grid_creation.py` → grids and Markov chains
3. `household_problem_epsilons_nolearning.py` → VFI orchestrator, calls:
   - `continuation_value_nolearning.py` → expected continuation values integrating over income, flood damage, and discrete choices
   - `stayer_problem.py` → EGM for homeowners staying in current house
   - `stayer_problem_renter.py` → EGM for renters
   - `buyer_problem_epsilons.py` → discrete optimization over house size and LTV for buyers
4. `simulation.py` → forward simulation of wealth distribution given policy functions
   - `buyer_problem_simulation.py` → buyer choices on simulation grid
   - `mortgage_choice_simulation.py` / `mortgage_choice_simulation_exc.py` → mortgage choice on simulation grid
   - `simulate_initial_joint.py` → initial joint distribution of wealth and income for newborns
5. `equilibrium.py` → outer loop: find Chebyshev coefficients for price law of motion
   - `find_coefficients` → iterate: solve VFI → simulate price path → regress prices → update coefficients
   - `initialise_coefficients_ss` → find steady state prices
   - `house_prices_algorithm` → market clearing at each time step (secant method on 2D system, fallback to bisection)
6. `moments.py` → compute model moments from distribution
7. `proper_welfare_debug.py` → welfare analysis (consumption equivalents)
8. `experiments.py` → counterfactual experiments (full information shock)
9. `full_calibration.py` → NLOpt calibration wrapper
10. `plot_creation.py` → plotting functions
11. `solve_epsilons.py` → main entry point (but 90% is dead commented-out code)

Supporting modules:
- `utility_epsilons.py` → utility function, marginal utility, bequest utility, renter intratemporal solution
- `interp.py` → 1D/3D interpolation with binary search
- `misc_functions.py` → 2D/3D/4D interpolation, binary search (DUPLICATED from interp.py), OLS, net income, jitclass constructor, DoubleGrid
- `grids.py` → nonlinear grid constructor
- `tauchen.py` → Tauchen method for AR(1) discretization, lifecycle income, initial distribution
- `LoM_epsilons.py` → Chebyshev law of motion for coastal and non-coastal prices

## Key problems to fix

### 1. DUPLICATE FILE: `simulation` (no .py extension)
There is a file called `simulation` (no extension) that is a near-complete duplicate of `simulation.py`. Delete the one without the .py extension.

### 2. DEAD CODE: `solve_epsilons.py`
The `main()` function is ~3000 lines, of which ~95% is commented out or inside triple-quoted strings used as block comments. The actual active code is roughly 20 lines that set coefficients and call welfare_stats. Extract the active path and archive the rest.

### 3. KEEP ALL @njit — DO NOT REMOVE
All `@njit` decorators must stay. The call chain from `find_coefficients` → `generate_pricepath` → `house_prices_algorithm` → `compute_excess_demand_pair` → `excess_demand_continuous` runs millions of times during equilibrium iteration. Every Python↔njit boundary crossing has overhead. Removing @njit from orchestration functions causes measurable slowdowns even though they "just" call other njit functions.

**Strategy**: Keep @njit everywhere. Accept that this constrains data passing to numba-compatible types (typed dicts, arrays, scalars, tuples). The existing pattern of numba typed dicts for `precompute_coastal_stayer_inputs` etc. is fine — keep it. The ugly 20+ argument function signatures are the cost of performance. Do NOT try to introduce Python dataclasses or plain-Python wrapper layers in the call chain.

Improvements within the @njit constraint:
- Where functions currently pass 10+ separate arrays that always travel together, consider grouping them into a numba typed dict at the call site (this is already done in several places and works)
- When splitting modules (section 8), @njit functions can import and call other @njit functions across modules — this works fine
- Keep `construct_jitclass` for `par` and `grids` objects — these are numba jitclass instances that work inside @njit

### 4. FUNCTION SIGNATURE BLOAT (ACCEPT AS COST OF NUMBA)
Many functions take 20-30 positional arguments. Because @njit must stay on all functions in the call chain, we CANNOT use Python dataclasses or keyword arguments to clean this up. 

What we CAN do:
- Add clear comments grouping arguments by purpose (e.g. `# --- price state ---`, `# --- policy functions ---`, `# --- distribution ---`)
- Where the codebase already uses numba typed dicts to bundle arguments (e.g. `coastal_stayer_inputs`, `mover_inputs`), keep and extend this pattern
- When reorganizing into modules, maintain consistent argument ordering across related functions
- Do NOT attempt to introduce Python wrapper layers — the performance cost is real

### 5. DUPLICATED INTERPOLATION CODE
`interp.py` and `misc_functions.py` both define `binary_search`, `_interp_2d`/`interp_2d`, `_interp_3d`/`interp_3d`. Consolidate all interpolation into `interp.py`. Have `misc_functions.py` import from `interp.py`. Also move `interp_4d` from `misc_functions.py` to `interp.py`.

### 6. PARAMETER DUPLICATION
`par_epsilons.py` defines a parameter dict. `full_calibration.py` defines a nearly identical dict with some values replaced by calibration variables. `grid_creation.py` also sets some parameters inline. Create a single `config.py` with a base parameter function, and have calibration override specific values.

### 7. `misc_functions.py` IS A JUNK DRAWER
It contains: interpolation (duplicate), OLS, binary search (duplicate), DoubleGrid, construct_jitclass, net_income, maxRow. Split by purpose:
- Interpolation → `interp.py`
- `construct_jitclass`, `DoubleGrid` → keep in `misc_functions.py` or rename to `utils.py`
- `net_income` → `income.py` or into `utility_epsilons.py`
- `ols_numba` → `equilibrium.py` (only used there)

### 8. MODULE ORGANIZATION
Target structure:
```
model/
├── config.py                    # parameters, create_par_dict()
├── utils.py                     # construct_jitclass, DoubleGrid
├── grids.py                     # grid creation (merge grid_creation.py + grids.py)
├── tauchen.py                   # income process discretization
├── interp.py                    # ALL interpolation functions
├── utility.py                   # utility, marginal utility, bequest, renter_solve
├── lom.py                       # law of motion (Chebyshev price forecasting)
├── household/
│   ├── __init__.py
│   ├── vfi.py                   # main VFI orchestrator (from household_problem_epsilons_nolearning.py)
│   ├── continuation.py          # continuation value computation
│   ├── stayer.py                # stayer problem (EGM)
│   ├── renter.py                # renter problem (EGM)
│   └── buyer.py                 # buyer problem (discrete optimization)
├── simulation/
│   ├── __init__.py
│   ├── distribution.py          # stat_dist_finder, update_dist_continuous
│   ├── excess_demand.py         # excess_demand_continuous
│   ├── buyer_sim.py             # buyer_problem_simulation
│   ├── mortgage_sim.py          # mortgage_choice_simulation + _exc
│   ├── decisions.py             # continuous_decide, continuous_decide_renter
│   ├── transitions.py           # simulate_stay, simulate_buy, simulate_rent and variants
│   └── initial_joint.py         # initial joint distribution
├── equilibrium/
│   ├── __init__.py
│   ├── solver.py                # find_coefficients, generate_pricepath
│   ├── steady_state.py          # initialise_coefficients_ss
│   └── market_clearing.py       # house_prices_algorithm, secant, bisection
├── analysis/
│   ├── __init__.py
│   ├── moments.py               # moment computation
│   ├── welfare.py               # welfare analysis (from proper_welfare_debug.py)
│   └── experiments.py           # counterfactual experiments
├── calibration.py               # NLOpt calibration (from full_calibration.py)
├── plots.py                     # all plotting (from plot_creation.py)
└── run.py                       # clean entry point
```

### 9. NAMING CONVENTIONS
Current names are inconsistent. Adopt:
- Files: `snake_case.py` (already mostly done)
- Functions: `snake_case` (already done)
- Variables: the `m` prefix for matrices, `v` for vectors, `d` for scalars, `i` for integers is a Dutch/Econometrics convention. It's fine to keep for internal consistency but document it.
- Remove `_epsilons` suffix from filenames — it's a vestige of an earlier model version

### 10. TESTING STRATEGY
There is no test suite. Before any refactoring:
1. Run the current model to steady state and save the output (price coefficients, distribution moments) as reference values
2. After each refactoring step, re-run and compare against reference
3. Convergence output IS the test: if `vCoeff_C`, `vCoeff_NC`, homeownership rate, and median net worth match to 6 decimal places, the refactor is correct

## Numba-specific constraints to remember
- `@njit` functions cannot accept or return Python classes, dataclasses, or most containers
- `@njit` functions CAN accept and return numba typed dicts (created inside @njit), numpy arrays, scalars, and tuples
- You cannot use `**kwargs` or default arguments in @njit functions
- `assert` statements work inside @njit
- `print` works inside @njit (with limitations)
- `np.where`, `np.dot`, `np.sum`, `np.zeros`, `np.ones`, `np.empty`, `np.linspace`, `np.arange` all work
- `np.interp` works
- `np.linalg.solve` works
- List comprehensions do NOT work inside @njit
- String formatting does NOT work inside @njit (f-strings fail)
- The `construct_jitclass` pattern in `misc_functions.py` creates numba jitclass instances that are passed as opaque objects to @njit functions — this works and should be preserved for `par` and `grids`

---

# Experiential Context

The sections below capture knowledge from running and debugging the model that agents cannot discover from reading the source code. Each section is tagged with which agents should read it.

Fill in answers below each question. Leave `TODO` for questions you're unsure about — agents will work around missing answers.

---

## Runtime & Performance
<!-- agents: test-writer, refactorer -->

- **How long does a full model run take (steady state + transition)?**
  TODO

- **How long does Numba first-compilation take?**
  TODO

- **What's the approximate memory footprint of a full solve?**
  TODO

- **Any known performance bottlenecks beyond the equilibrium iteration loop?**
  TODO

- **How many equilibrium iterations does convergence typically require?**
  TODO

- **Does the model run on a laptop, or does it need a cluster/server?**
  TODO

---

## Numerical Stability
<!-- agents: test-writer, refactorer, cartographer -->

- **Known parameter ranges that cause convergence failure?**
  TODO

- **Has the model ever produced NaN/Inf during normal runs? If so, where?**
  TODO

- **Any known sensitivity to grid density (e.g., does reducing `iBmax` from 27 to 15 break convergence)?**
  TODO

- **Does the equilibrium solver always converge, or does it sometimes fail? What happens when it fails?**
  TODO

- **Are there specific time periods (early vs late in the 52-period transition) where the solver is more fragile?**
  TODO

- **Is the secant method or the bisection fallback used more often in practice?**
  TODO

---

## Testing & Validation
<!-- agents: test-writer -->

- **What reference outputs exist? (saved coefficient files, known vCoeff_C/vCoeff_NC values, Excel outputs, figures)**
  TODO

- **Can the model be run in a "fast" mode? (fewer grid points, fewer iterations, single time period)**
  TODO

- **Are there intermediate checkpoints worth pinning? (e.g., steady-state prices before transition, grid creation output, single VFI step)**
  TODO

- **Which entry paths actually work right now?**
  - `solve_epsilons.py:main()` → TODO (works / broken / partially working?)
  - `full_calibration.py:main()` → Broken (calls nonexistent `initialise_coefficients_initial` and `solve_initial`)
  - `experiments.py` → TODO

- **Is there any seed-dependent behavior? (random number generation, stochastic shocks)**
  TODO

- **What does "correct" output look like? (describe what a successful run produces)**
  TODO

- **How do you currently verify that the model is working correctly after a change?**
  TODO

---

## Calibration History
<!-- agents: test-writer, architect, refactorer -->

- **Where did the hardcoded vCoeff_C/vCoeff_NC values in `solve_epsilons.py` come from?**
  TODO (e.g., "from a calibration run on [date]", "copied from a previous version", "manually tuned")

- **Are the current `par_epsilons.py` values the "final" calibration, or are they intermediate?**
  TODO

- **What moments is the model calibrated to match? What are the data targets?**
  TODO (e.g., homeownership rate = 0.65, median LTV = 0.80, ...)

- **Is the NLOpt calibration path currently functional?**
  No — calls `equil.initialise_coefficients_initial()` and `household_problem.solve_initial()` which don't exist.

- **Has the calibration ever been run to completion? What were the results?**
  TODO

- **Are the 52 `vPi_S_median` flood probability values from data or assumed?**
  TODO

---

## Model Economics
<!-- agents: doc-writer, architect -->

- **What paper/project is this model for?**
  TODO

- **One-paragraph economic intuition: what question does this model answer?**
  TODO

- **What are the key experiments/counterfactuals? (e.g., "full information shock" in experiments.py)**
  TODO

- **What policy questions does the welfare analysis address?**
  TODO

- **Are there planned model extensions that should influence the refactoring? (e.g., adding a third region, endogenous beliefs, rental market regulation)**
  TODO

- **Who else works with this code? (co-authors, RAs, referees wanting replication)**
  TODO

---

## Known Gotchas
<!-- agents: all -->

Things you've learned the hard way that aren't visible in the source code.

- **Fragile parts of the code that look safe but aren't:**
  TODO

- **Implicit assumptions that aren't documented anywhere:**
  - vPDF_z doesn't sum to 1 — it's conditional probabilities, not a proper PDF (element 0 is the no-damage weight, elements 1-3 are conditional on flooding)
  - TODO: add more

- **Things that have broken in the past and what caused them:**
  - Removing @njit from orchestration functions causes measurable performance regression
  - TODO: add more

- **Order-of-operations dependencies that aren't obvious from imports:**
  - `par_epsilons.py` line 17 mutates `vPi_S_median` at module import time — import order matters
  - `grid_creation.create()` internally calls `tauchen()`, so calling `tauchen()` before `create()` is redundant (solve_epsilons.py line 76)
  - TODO: add more

- **Anything else an agent should know before touching this code:**
  TODO