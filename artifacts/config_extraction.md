# Config Extraction

Revised per human review 2026-04-08.

**Principle: Only extract constants that are either (a) used in multiple files and must stay in sync, or (b) whose meaning isn't obvious from context. Single-use values with a comment are fine inline.**

Categories:
- **Calibration output** — keep in `par_dict` (calibrated parameter values)
- **Multi-file constant** — extract to `config.py` or `par_dict`
- **Single-file constant** — define at top of the file that uses it, not in config.py

---

## Constants to extract (multi-file, must stay in sync)

### NEG_INF sentinel

| Status | Current value | Name | Files | Implementation |
|--------|---------------|------|-------|----------------|
| **DONE (5a.1)** | `-1e12` | `NEG_INF` | 6 files, ~30 occurrences | Currently defined as module-level constant in each file. In Phase 5c, move definition to `config.py` and import: `from config import NEG_INF`. Numba handles module-level imported constants inside @njit. **Verify with a quick test before doing all replacements.** |

### Retirement income fraction

| # | File | Current | Name | Implementation |
|---|------|---------|------|----------------|
| 8 | `grid_creation.py:109` | `0.7` | `retirement_income_fraction` | Add to `par_dict`. Reference as `par.retirement_income_fraction`. |
| 9 | `misc_functions.py:371` (now utility.py after 5c) | `0.7` | Same | Same `par.retirement_income_fraction`. |

**Note:** `full_calibration.py` also hardcodes `0.7` on line ~139 (inside the broken calibration path). Add a comment in config.py: `# Also used in full_calibration.py — update there when calibration path is fixed.`

### Year constants

| # | Files | Current | Name | Implementation |
|---|-------|---------|------|----------------|
| 10-12 | `grid_creation.py`, `equilibrium.py`, `plot_creation.py` | `1998`, `2026` | `MODEL_START_YEAR`, `EXPERIMENT_YEAR` | Define in `config.py`. Domain constants, not calibration outputs. |

### Main tolerance triplet (already done in 5a.1)

| Status | File | Name | Implementation |
|--------|------|------|----------------|
| **DONE (5a.1)** | `equilibrium.py` | `PRICE_TOL`, `ERROR_TOL`, `MAX_ITERATIONS`, `SECANT_STEP` | Defined as constants at top of `house_prices_algorithm` in equilibrium.py. **Keep in equilibrium.py** — only used there. Do NOT move to config.py. |

---

## Constants to add to `par_dict` (economic parameters)

### Damage states and flood distribution

| # | File | Current | Key | Notes |
|---|------|---------|-----|-------|
| 13 | `grid_creation.py:40` | `np.array([1, 0.9, 0.7, 0.3])` | `'vZ'` | Damage multipliers. Already in grids object but hardcoded inline. Move definition to par_dict. |
| 14 | `grid_creation.py:41` | `np.array([1, 0.4, 0.4, 0.2])` | `'vPDF_z'` | Conditional probabilities. **Add comment: "Element 0 = P(no damage\|no flood) = 1. Elements 1-3 = P(damage_level\|flood). Intentionally does not sum to 1 — these are conditional probabilities."** |

### Population fractions

| # | File | Current | Key | Notes |
|---|------|---------|-----|-------|
| 25 | `grid_creation.py:96` | `np.array([0.58, 0.42])` | `'vTypes'` | 58% realists, 42% optimists. Calibration output — belongs in par_dict. |

### Initial wealth cutoff

| # | File | Current | Key | Notes |
|---|------|---------|-----|-------|
| 48 | `simulate_initial_joint.py:39` | `5.014401` | `'initial_wealth_ratio_cutoff'` | 95th percentile of empirical ratio. Calibration output — belongs in par_dict. |

---

## Constants to keep inline (single-file, define at top of file)

### Grid construction literals — keep in grid_creation.py (or merged grids.py)

These are grid resolution details, not economic parameters. They don't belong alongside `dBeta` and `dSigma` in par_dict. Define as constants at top of `grid_creation.py`:

| # | Current value | Where to define | Notes |
|---|---------------|-----------------|-------|
| 15 | `1.5` (vL_sim upper) | Top of grids.py | `_L_SIM_MAX = 1.5` |
| 16 | `35` (vL_sim points) | Top of grids.py | `_NL_SIM = 35` |
| 17 | `1.50` (vH lower) | Top of grids.py | `_H_MIN = 1.50` |
| 18 | `3` (vH points) | Top of grids.py | `_NH = 3` |
| 19 | `np.array([1.17, 1.92])` (vH_renter) | Top of grids.py | `_VH_RENTER = np.array([1.17, 1.92])` |
| 20 | `1.3` (vL upper) | Top of grids.py | `_L_MAX = 1.3` |
| 21 | `20` (vL points) | Top of grids.py | `_NL = 20` |
| 22 | `1.4` (nonlinspace curvature) | Top of grids.py | `_GRID_CURVATURE = 1.4` |
| 23 | `0.01` (vM lower) | Top of grids.py | `_M_MIN = 0.01` |
| 24 | `np.linspace(0,5,2)` (vLkeps) | Top of grids.py | Keep inline with comment |

**Rationale:** If someone wants to change grid resolution, they edit grids.py — which is where they'd look anyway. Only `iNb`, `iBmax`, `iXin` (which already ARE in par_dict) need to vary across calibration runs.

### Equilibrium solver settings — keep in equilibrium.py

**Do NOT move to config.py.** These are only used in equilibrium.py. Moving them creates a dependency from equilibrium back to config for values no other module needs.

Define as module-level constants at the top of equilibrium.py. @njit functions can read module-level constants defined in the same file.

**Extract only the constants already done (5a.1):** `PRICE_TOL`, `ERROR_TOL`, `MAX_ITERATIONS`, `SECANT_STEP`.

**Keep the rest inline with comments.** Single-use values defined 3 lines above where they're used don't benefit from naming:

| # | Value | Decision |
|---|-------|----------|
| 26 | `15` (transition max iter) | Keep inline: `for iteration in range(15):  # max transition iterations` |
| 27 | `0.001` (coeff convergence) | Keep inline with comment |
| 28 | `0.5` (transition damping) | Keep inline: `rho=0.5  # damping factor for coefficient update` |
| 29 | `25` (SS max iter) | Keep inline with comment |
| 30 | `0.4` (SS initial damping) | Keep inline with comment |
| 31 | `0.25` (SS bisection bound) | Keep inline with comment |
| 32 | `0.0005` (SS convergence) | Keep inline with comment |
| 33 | `0.1` (min damping) | Keep inline with comment |
| 34-35 | Bisection tol/max_iter | Keep inline |
| 36-38 | Secant tol/tol_wider/max_iter | Keep inline |
| 39 | `1e-15` (singularity) | Keep inline |
| 44-46 | Bounds and offsets | Keep inline |

### Initial wealth CDF cutoff

| # | Value | Decision |
|---|-------|----------|
| 47 | `0.95` | Keep inline in simulate_initial_joint.py with comment |

---

## Calibration parameters already in `par_dict` — KEEP THERE

All ~40 parameters in `par_epsilons.py` (now `par.py`) are calibration outputs. They stay in `par_dict`. Add this comment at the top of `create_par_dict()`:

```python
# Values below are calibrated outputs — do not round or modify without re-running calibration.
# Significant-digit precision (e.g., 0.940074219) indicates machine-precision calibrated values.
```

---

## Summary

| Category | Count | Action |
|---|---|---|
| `NEG_INF` sentinel | 6 files | DONE (5a.1). Centralize in config.py during 5c. |
| Retirement income fraction | 2 files | Add to `par_dict`. Flag full_calibration.py. |
| Year constants | 3 files | Define in `config.py`. |
| Damage states (vZ, vPDF_z) | 1 file | Move to `par_dict` with conditional probability comment. |
| Population fractions (vTypes) | 1 file | Move to `par_dict`. |
| Initial wealth cutoff | 1 file | Move to `par_dict`. |
| Grid construction literals | 11 values | Keep at top of grids.py as local constants. NOT in par_dict. |
| Solver settings | 21 values | Keep in equilibrium.py. Only PRICE_TOL/ERROR_TOL/MAX_ITERATIONS/SECANT_STEP already named (5a.1). |
| Calibration parameters | ~40 | KEEP in par_dict (already there). |
