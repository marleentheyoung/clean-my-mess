# Naming Conventions

## Current patterns

### File names
- Mostly `snake_case.py` (good)
- **Vestigial `_epsilons` suffix** on 4 files: `par_epsilons.py`, `utility_epsilons.py`, `LoM_epsilons.py`, `buyer_problem_epsilons.py`, `household_problem_epsilons_nolearning.py`, `solve_epsilons.py`, `mortgage_choice_simulation_exc.py`. The `_epsilons` suffix comes from an earlier model version with an epsilon-based discretization. It has no current meaning.
- **`LoM_epsilons.py`** uses `PascalCase` abbreviation prefix, inconsistent with other files
- **`proper_welfare_debug.py`** has a `_debug` suffix suggesting it's temporary, but it's the production version

### Function names
- Consistently `snake_case` (good): `solve`, `create`, `tauchen`, `stat_dist_finder`, `excess_demand_continuous`, etc.
- Some abbreviations: `u` (utility), `u_c` (marginal utility of consumption), `LoM_C` / `LoM_NC` (law of motion coastal/non-coastal)

### Variable names -- Hungarian notation
The codebase uses a **Dutch/econometrics Hungarian notation** system for variable prefixes:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `m` | Matrix (2D+ numpy array) | `mMarkov`, `mDist0_c`, `mPTI`, `mVt`, `mC_pol_stayer` |
| `v` | Vector (1D numpy array) | `vCoeff_C`, `vE`, `vH`, `vX`, `vPi_S_median`, `vTypes` |
| `d` | Scalar (double/float) | `dBeta`, `dP_C`, `dSigma`, `dP_C_lag`, `dRho` |
| `i` | Integer scalar or count | `iNj`, `iNb`, `iNumStates`, `iXin`, `iM` |
| *(none)* | Mixed/unclear | `par`, `grids`, `method`, `sceptics`, `func` |

This convention is applied **inconsistently**:
- `rho` in equilibrium.py is a scalar but has no `d` prefix
- `max_it`, `iteration`, `counter` are integers without `i` prefix
- `nperiods` is an integer without `i` prefix
- `price_history` is a matrix without `m` prefix
- `bequest_guess` is a vector without `v` prefix
- Grid arrays inside `grids_dict` use `v` prefix (`vB`, `vH`, `vX`) but some don't (`median_inc`, `min_inc`, `max_ltv`)

### Import aliases
Current pattern is inconsistent:

| Import | Alias | Notes |
|--------|-------|-------|
| `misc_functions` | `misc` | Short, good |
| `tauchen` | `tauch` | Abbreviated |
| `utility_epsilons` | `ut` | Very short |
| `interp` | `interpfun` or `interp` | Two different aliases for same module |
| `LoM_epsilons` | `lom` | Good |
| `grids` | `grid` or `gridsfun` | Two different aliases |
| `buyer_problem_epsilons` | `buyer_problem_epsilons` | No alias (redundant) |
| `stayer_problem` | `stayer_problem` | No alias (redundant) |
| `household_problem_epsilons_nolearning` | `household_problem` | Good abbreviation |
| `simulation` | `sim` | Good |
| `equilibrium` | `equil` | Good |

---

## Proposed rules

### 1. File names

**Rule:** `snake_case.py`, no vestigial suffixes, no abbreviations that aren't universally understood.

| Current | Proposed | Reason |
|---------|----------|--------|
| `par_epsilons.py` | `config.py` | Descriptive; drop `_epsilons` |
| `utility_epsilons.py` | `utility.py` | Drop `_epsilons` |
| `LoM_epsilons.py` | `lom.py` | snake_case; drop `_epsilons` |
| `buyer_problem_epsilons.py` | `buyer.py` (in `household/`) | Shorter; context provided by directory |
| `household_problem_epsilons_nolearning.py` | `vfi.py` (in `household/`) | Descriptive of what it does |
| `continuation_value_nolearning.py` | `continuation.py` | Drop `_nolearning` (there is no learning version) |
| `stayer_problem.py` | `stayer.py` | Drop `_problem` (context provided by directory) |
| `stayer_problem_renter.py` | `renter.py` | Drop `_problem` |
| `buyer_problem_simulation.py` | `buyer_sim.py` | Shorter; clarifies it's simulation-grid version |
| `mortgage_choice_simulation.py` | `mortgage_sim.py` | Merge with `_exc` variant |
| `mortgage_choice_simulation_exc.py` | *(merged into `mortgage_sim.py`)* | |
| `simulate_initial_joint.py` | `initial_joint.py` | Drop `simulate_` prefix (directory provides context) |
| `misc_functions.py` | `utils.py` | Standard name for utility module |
| `proper_welfare_debug.py` | `welfare.py` | Drop `proper_` and `_debug` |
| `solve_epsilons.py` | `run.py` | Descriptive of role |
| `plot_creation.py` | `plots.py` | Shorter |
| `full_calibration.py` | `calibration.py` | Drop `full_` |

### 2. Function names

**Rule:** `snake_case`. Use full words except for universally understood abbreviations (`ss` for steady state, `sim` for simulation, `dist` for distribution).

Current function names are already good. Specific adjustments:

| Current | Proposed | Reason |
|---------|----------|--------|
| `LoM_C` | `lom_c` | Consistent snake_case |
| `LoM_NC` | `lom_nc` | Consistent snake_case |
| `DoubleGrid` | `double_grid` | snake_case |
| `maxRow` | `max_row` | snake_case |

All other function names (`solve`, `create`, `tauchen`, `stat_dist_finder`, `excess_demand_continuous`, `find_coefficients`, etc.) are already good and should be kept.

### 3. Variable names -- Hungarian notation

**Recommendation: KEEP the prefix convention for array/matrix variables. Document it. Extend it consistently.**

Rationale:
- The codebase manipulates arrays of 1-7 dimensions. Knowing the dimensionality from the variable name is genuinely useful.
- Dropping the convention would require a massive rename that adds risk with no functional benefit.
- The convention is common in computational economics codebases and will be familiar to referees.
- The inconsistencies should be fixed, but the system itself is sound.

**Rules:**

| Prefix | Use for | Examples |
|--------|---------|---------|
| `m` | 2D+ numpy arrays (matrices, tensors) | `mMarkov`, `mDist`, `mPTI` |
| `v` | 1D numpy arrays (vectors) | `vCoeff_C`, `vE`, `vH` |
| `d` | Float scalars (doubles) | `dBeta`, `dP_C`, `dRho` |
| `i` | Integer scalars and counts | `iNj`, `iNb`, `iT` |
| *(no prefix)* | Booleans, strings, objects, loop indices | `sceptics`, `method`, `par`, `grids`, `j`, `k` |

**Fixes for current inconsistencies** (to be applied during Phase 5b):

| Current | Proposed | Reason |
|---------|----------|--------|
| `rho` (in equilibrium.py) | `dRho_step` | Float scalar, distinguish from `par.dRho` (AR(1) persistence) |
| `max_it` | `iMax_it` | Integer |
| `iteration` | `iIteration` or just keep `iteration` | Loop counter -- either convention works, just be consistent |
| `nperiods` | `iNperiods` | Integer count |
| `price_history` | `mPrice_history` | 2D array |
| `bequest_guess` | `vBequest_guess` | 1D array |
| `median_inc` (in grids_dict) | `dMedian_inc` | Scalar in the dict |
| `min_inc` (in grids_dict) | `dMin_inc` | Scalar in the dict |
| `max_ltv` (in grids_dict) | `dMax_ltv` | Scalar in the dict |

**Exception:** Loop index variables (`j`, `k`, `t_index`, `e_index`, `h_index`, etc.) do NOT need the `i` prefix. They are universally understood as integers from context.

### 4. Module import aliases

**Rule:** Use short, consistent aliases. Each module gets exactly one alias used everywhere.

| Module (new name) | Standard alias | Rationale |
|---|---|---|
| `model.config` | `cfg` | Short, unambiguous |
| `model.utils` | `utils` | Already clear |
| `model.grids` | `grids` | No alias needed |
| `model.tauchen` | `tauch` | Keep existing convention |
| `model.interp` | `interp` | No alias needed, already short |
| `model.utility` | `ut` | Keep existing short alias |
| `model.lom` | `lom` | Keep existing alias |
| `model.income` | `inc` | Short |
| `model.household.vfi` | `vfi` | Short, descriptive |
| `model.household.continuation` | `cont` | Short |
| `model.household.stayer` | `stayer` | No alias needed |
| `model.household.renter` | `renter` | No alias needed |
| `model.household.buyer` | `buyer` | No alias needed |
| `model.simulation.distribution` | `dist` | Short |
| `model.simulation.excess_demand` | `exd` | Short |
| `model.simulation.buyer_sim` | `buy_sim` | Keep existing alias |
| `model.simulation.mortgage_sim` | `mort_sim` | Short |
| `model.simulation.decisions` | `dec` | Short |
| `model.simulation.transitions` | `trans` | Short |
| `model.simulation.initial_joint` | `init_joint` | Short |
| `model.equilibrium.solver` | `eq_solver` | Distinguishes from household solver |
| `model.equilibrium.steady_state` | `eq_ss` | Short |
| `model.equilibrium.market_clearing` | `mc` | Short |
| `model.analysis.moments` | `mom` | Keep existing alias |
| `model.analysis.welfare` | `welfare` | No alias needed |
| `model.analysis.experiments` | `exp` | Short |

**Anti-pattern to avoid:** Importing a module with the same name as its alias (`import interp as interp`). Either use a different alias or don't use `as`.

### 5. Constant names

**Rule:** `UPPER_SNAKE_CASE` for solver settings and magic numbers extracted into `config.py`.

Examples:
- `NEG_INF_SENTINEL = -1e12`
- `RETIREMENT_INCOME_FRACTION = 0.7`
- `MODEL_START_YEAR = 1998`
- `EXPERIMENT_YEAR = 2026`
- `COEFF_CONVERGENCE_TOL_TRANSITION = 0.001`

See `config_extraction.md` for the full list.

---

## Summary of changes by phase

| Phase | Naming changes |
|---|---|
| 5a (quick wins) | Fix `grids.PDF_z` -> `grids.vPDF_z` in utility_epsilons.py |
| 5b (medium effort) | Rename files (drop `_epsilons`, `_nolearning`, `_debug`). Rename `LoM_C` -> `lom_c`, `DoubleGrid` -> `double_grid`, `maxRow` -> `max_row`. Standardize import aliases. |
| 5c (restructure) | Directory moves per migration_plan.md. No additional renames beyond what 5b already did. |
| Post-5c (optional) | Fix Hungarian notation inconsistencies in variable names. This is lowest priority and highest risk (many changes). |
