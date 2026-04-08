# Naming Conventions

Revised per human review 2026-04-08.

## Current patterns

### File names
- Mostly `snake_case.py` (good)
- `_epsilons` suffixes removed in Phase 5b
- `proper_welfare_debug.py` renamed to `welfare.py` in Phase 5b
- `LoM_epsilons.py` renamed to `lom.py` in Phase 5b

### Function names
- Consistently `snake_case` (good)
- Three functions still use non-snake_case: `LoM_C`/`LoM_NC` (aliased to `LoM`), `DoubleGrid`, `maxRow`

### Variable names -- Hungarian notation
The codebase uses a **Dutch/econometrics Hungarian notation** system:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `m` | Matrix (2D+ numpy array) | `mMarkov`, `mDist0_c`, `mPTI` |
| `v` | Vector (1D numpy array) | `vCoeff_C`, `vE`, `vH` |
| `d` | Scalar (double/float) | `dBeta`, `dP_C`, `dSigma` |
| `i` | Integer scalar or count | `iNj`, `iNb`, `iNumStates` |
| *(none)* | Booleans, strings, objects, loop indices | `sceptics`, `method`, `par`, `grids` |

Applied inconsistently (e.g., `rho`, `price_history`, `nperiods` lack prefixes). **These inconsistencies are documented, not fixed.** See below.

### Import aliases
Currently inconsistent (e.g., `interp` aliased as both `interpfun` and `interp`; `grids` aliased as both `grid` and `gridsfun`).

---

## Proposed rules

### 1. File names

**Rule:** `snake_case.py`, no vestigial suffixes. Already applied in Phase 5b for all files in `clean_the_mess/`. Phase 5c moves them into the `model/` package structure.

### 2. Function names

**Rule:** `snake_case`. Rename these three during Phase 5c (same commit as the file moves):

| Current | Proposed | Reason |
|---------|----------|--------|
| `LoM_C` / `LoM_NC` / `LoM` | `lom` (keep `LoM_C`/`LoM_NC` as aliases during transition) | Already unified in 5a.2; final rename happens in 5c |
| `DoubleGrid` | `double_grid` | snake_case |
| `maxRow` | `max_row` | snake_case |

All other function names are already correct.

### 3. Variable names -- Hungarian notation

**Decision: KEEP the convention. Document it. DO NOT fix inconsistencies.**

Rationale:
- Renaming `rho` to `dRho_step`, `price_history` to `mPrice_history`, etc. touches dozens of lines inside @njit functions for zero functional benefit
- It's pure cosmetic risk with no payoff — a referee cares about correctness and reproducibility, not prefix consistency
- Document the convention in the README, note that inconsistencies exist, leave the variables alone

**What to document in README:**
- The prefix system (`m`/`v`/`d`/`i`)
- That it's applied inconsistently in some local variables
- That this is intentional and not a cleanup target

### 4. Module import aliases

**Rule:** Use readable, consistent aliases. Each module gets exactly one alias used everywhere.

| Module (new name) | Standard alias | Notes |
|---|---|---|
| `model.config` | `cfg` | |
| `model.utils` | `utils` | |
| `model.grids` | `grids` | No alias needed |
| `model.tauchen` | `tauch` | Keep existing |
| `model.interp` | `interp` | No alias needed |
| `model.utility` | `ut` | Keep existing |
| `model.lom` | `lom` | Keep existing |
| `model.household.vfi` | `vfi` | |
| `model.household.continuation` | `cont` | |
| `model.household.stayer` | `stayer` | No alias needed |
| `model.household.renter` | `renter` | No alias needed |
| `model.household.buyer` | `buyer` | No alias needed |
| `model.simulation.distribution` | `dist` | |
| `model.simulation.excess_demand` | `excess_demand` | Full name — readable in 6 months |
| `model.simulation.buyer_sim` | `buy_sim` | |
| `model.simulation.mortgage_sim` | `mort_sim` | |
| `model.simulation.mortgage_sim_exc` | `mort_sim_exc` | Separate file, separate alias |
| `model.simulation.initial_joint` | `init_joint` | |
| `model.equilibrium.solver` | `eq_solver` | |
| `model.equilibrium.steady_state` | `eq_ss` | |
| `model.equilibrium.market_clearing` | `market_clearing` | Full name — `mc` is unreadable |
| `model.analysis.moments` | `mom` | Keep existing |
| `model.analysis.welfare` | `welfare` | No alias needed |
| `model.analysis.experiments` | `experiments` | Full name — `exp` conflicts with math |

### 5. Constant names

**Rule:** `UPPER_SNAKE_CASE` for extracted magic numbers and solver settings.

Already applied in Phase 5a.1: `NEG_INF`, `PRICE_TOL`, `ERROR_TOL`, `MAX_ITERATIONS`, `SECANT_STEP`, `N_CONSUMPTION_NODES`.

See `config_extraction.md` for the full list of values to extract during Phase 5c.1.

---

## Summary of changes by phase

| Phase | Naming changes |
|---|---|
| 5a (DONE) | Named magic numbers as constants. Unified LoM functions. |
| 5b (DONE) | Renamed files (dropped `_epsilons`, `_nolearning`, `_debug`). Renamed `proper_welfare_debug` → `welfare`. |
| 5c (TODO) | Directory moves. Rename `DoubleGrid` → `double_grid`, `maxRow` → `max_row` in same commit as file moves. |
| Hungarian fixes | **DEFERRED INDEFINITELY.** Document in README, leave variables alone. |
